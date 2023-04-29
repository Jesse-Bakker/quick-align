mod audioreader;
mod dtw;
mod tts;

use std::{iter, thread};

use audioreader::AudioReader;
use mfcc::{mfcc, FrameExtractionOpts, FrameSupplier, MelBanksOpts, MfccOptions};

use crate::dtw::find_subsequence;

const MFCC_WINDOW_SHIFT: f32 = 40. /* milliseconds */;
const MFCC_WINDOW_LENGTH: f32 = 100. /* milliseconds */;

const SILENCE_MIN_LENGTH: f32 = 0.6 /* seconds */;
const SILENCE_MAX_ENERGY: f32 = 0.699; // log10(5)
const START_DETECTION_MAX_SKIP: f32 = 60. /* seconds */;
const CMN_WINDOW: f32 = 3000. /* milliseconds */;

#[derive(PartialEq)]
struct CmpF32(f32);

impl Eq for CmpF32 {}

impl PartialOrd for CmpF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for CmpF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl From<f32> for CmpF32 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

struct Silences<I>
where
    I: Iterator<Item = (usize, [f32; 13])>,
{
    inner: I,
    current_silence: Option<(usize, usize)>,
    minimum_length: usize,
    max_energy: f32,
}

impl<I> Iterator for Silences<I>
where
    I: Iterator<Item = (usize, [f32; 13])>,
{
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        const MIN_ENERGY: f32 = -11.512_925; // ln(0.0001)
        loop {
            let (index, mfcc) = self.inner.next()?;
            match (
                self.current_silence,
                MIN_ENERGY.max(mfcc[0]) < MIN_ENERGY + self.max_energy,
            ) {
                (Some((start, _)), true) => {
                    self.current_silence = Some((start, index));
                }
                (Some(interval @ (start, end)), false) => {
                    self.current_silence = None;
                    if end - start > self.minimum_length {
                        return Some(interval);
                    }
                }
                (None, true) => {
                    self.current_silence = Some((index, index));
                }
                (None, false) => {
                    continue;
                }
            }
        }
    }
}

pub fn silences(
    mfcc: impl Iterator<Item = [f32; 13]>,
    minimum_length: usize,
    max_energy: f32,
) -> impl Iterator<Item = (usize, usize)> {
    Silences {
        inner: mfcc.enumerate(),
        current_silence: None,
        minimum_length,
        max_energy,
    }
}

fn normalize_mfcc(window: usize, mfcc: &[[f32; 13]]) -> Vec<[f32; 13]> {
    let n_frames = mfcc.len();
    let mut last_window = None;
    let mut cur_sum = [0.; 13];

    let mut out = Vec::with_capacity(mfcc.len());
    for t in 0..n_frames {
        let (window_start, window_end) = if t < (window / 2) {
            (0, window.min(n_frames))
        } else {
            let end = (t + (window / 2)).min(n_frames);
            (end.saturating_sub(window), end)
        };
        if let Some((last_window_start, last_window_end)) = last_window {
            if window_start > last_window_start {
                for (s, v) in iter::zip(cur_sum.iter_mut(), mfcc[last_window_start].iter()) {
                    *s -= v;
                }
            }
            if window_end > last_window_end {
                for (s, v) in iter::zip(cur_sum.iter_mut(), mfcc[last_window_end].iter()) {
                    *s += v;
                }
            }
        } else {
            for row in &mfcc[window_start..window_end] {
                for (s, v) in iter::zip(cur_sum.iter_mut(), row.iter()) {
                    *s += v;
                }
            }
        }
        last_window = Some((window_start, window_end));
        let window_frames = window_end - window_start;
        let mut out_frame = mfcc[t];
        for (v, s) in iter::zip(out_frame.iter_mut(), cur_sum) {
            *v -= s / window_frames as f32;
        }
        out.push(out_frame);
    }
    out
}

fn compute_mfcc<T: FrameSupplier>(wave: &mut T, frame_opts: FrameExtractionOpts) -> Vec<[f32; 13]> {
    let m: Vec<_> = mfcc(
        MfccOptions {
            mel_opts: MelBanksOpts {
                n_bins: 40,
                low_freq: Some(133.3333.into()),
                high_freq: Some(6855.4976.into()),
            },
            frame_opts,
            n_ceps: 13,
        },
        wave,
    )
    .collect();
    //normalize_mfcc(300, &m)
    m
}

fn search_sorted_right(a: Vec<usize>, v: impl ExactSizeIterator<Item = usize>) -> Vec<usize> {
    let mut ret = Vec::with_capacity(v.len());
    let mut i = 0;
    for val in v {
        while i < a.len() {
            if (i == 0 || a[i - 1] <= val) && val < a[i] {
                ret.push(i);
                break;
            }
            i += 1;
        }
    }
    ret
}

fn find_boundaries(
    real_indices: Vec<usize>,
    synth_indices: Vec<usize>,
    anchors: Vec<usize>,
) -> Vec<usize> {
    // These are the indices of the anchors corresponding to the indices in the synth_indices array
    let anchor_indices = anchors
        .into_iter()
        .map(|anchor| /* time in ms */ anchor / MFCC_WINDOW_SHIFT as usize);
    // Now, find where we should insert these anchors into our synthetic indices array, to find
    // where they would occur in real_indices, thus where they would occur in our real audio
    let begin_indices = search_sorted_right(synth_indices, anchor_indices);
    begin_indices.into_iter().map(|i| real_indices[i]).collect()
}

macro_rules! time {
    ($s:expr, $m:expr) => {{
        #[cfg(timing)]
        {
            let instant = std::time::Instant::now();
            let result = $s;
            eprintln!("{}: {}", $m, instant.elapsed().as_millis());
            result
        }
        #[cfg(not(timing))]
        {
            $s
        }
    }};
}

pub fn find_start(audio_mfcc: &[[f32; 13]], synth_mfcc: &[[f32; 13]]) -> usize {
    const SILENCE_MIN_FRAMES: usize = (SILENCE_MIN_LENGTH / (MFCC_WINDOW_SHIFT / 1000.)) as usize;
    const START_DETECTION_MAX_SKIP_FRAMES: usize =
        (START_DETECTION_MAX_SKIP / (MFCC_WINDOW_SHIFT / 1000.)) as usize;
    let start_detection_synth_frames: usize =
        (3 * START_DETECTION_MAX_SKIP_FRAMES).min(synth_mfcc.len());
    let silences = silences(
        audio_mfcc.iter().copied(),
        SILENCE_MIN_FRAMES,
        SILENCE_MAX_ENERGY,
    );
    let candidates: Vec<usize> = std::iter::once((0, 0))
        .chain(silences)
        .map(|(_, end)| end)
        .take_while(|&end| end < START_DETECTION_MAX_SKIP_FRAMES)
        .collect();
    let Some(last) = candidates.last() else {
        return 0;
    };

    let max_start_detection_audio_frames =
        (1.2 * start_detection_synth_frames as f64) as usize;
    let max_start_detection_audio_frames =
        max_start_detection_audio_frames.min(audio_mfcc.len() - last);
    let min_start_detection_audio_frames = max_start_detection_audio_frames - (0.2 * start_detection_synth_frames as f64) as usize;
    let psi = max_start_detection_audio_frames - min_start_detection_audio_frames;

    let synth_frames = &synth_mfcc[0..start_detection_synth_frames];

    let candidate_samples = candidates
        .iter()
        .map(|&start| &audio_mfcc[start..(start + max_start_detection_audio_frames)]);
    let (start, _cost) = find_subsequence(synth_frames, candidate_samples, psi);
    candidates[start]
}

pub fn align(audio_file: &str, text_fragments: &[String]) -> Vec<f32> {
    let frame_opts = FrameExtractionOpts {
        sample_freq: 22050,
        frame_length_ms: MFCC_WINDOW_LENGTH,
        frame_shift_ms: MFCC_WINDOW_SHIFT,
        emphasis_factor: 0.97,
    };
    let (audio_mfcc, synth_mfcc, anchors) = thread::scope(|s| {
        let t_audio = s.spawn(|| {
            let reader = AudioReader::new();
            let mut audio_samples = time!(
                reader
                    .read_and_transcode_file(audio_file, frame_opts)
                    .unwrap(),
                "Read and transcode audio"
            );
            time!(compute_mfcc(&mut audio_samples, frame_opts), "Audio mfcc")
        });

        let t_synth = s.spawn(|| {
            let mut synth_samples = time!(
                tts::speak_multiple(text_fragments.into()).unwrap(),
                "Synthesize audio"
            );

            let mfcc = compute_mfcc(&mut synth_samples, frame_opts);
            let anchors = synth_samples.anchors();
            (anchors, mfcc)
        });

        let audio_mfcc = t_audio.join().unwrap();
        let (anchors, synth_mfcc) = t_synth.join().unwrap();
        (audio_mfcc, synth_mfcc, anchors)
    });

    let start = find_start(&audio_mfcc, &synth_mfcc);
    let res = dtw::fastdtw::fast_dtw(&audio_mfcc[start..], &synth_mfcc, Some(100));
    let (real_indices, synth_indices): (Vec<_>, Vec<_>) = res.path.into_iter().unzip();
    let boundaries = find_boundaries(real_indices, synth_indices, anchors);

    boundaries
        .into_iter()
        .map(|index| index + start)
        .map(|index| index as f32 * MFCC_WINDOW_SHIFT / 1000.0)
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_sorted() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let v = vec![0, 2, 5, 6, 8, 8];
        assert_eq!(
            search_sorted_right(a, v.into_iter()),
            vec![0, 2, 5, 6, 8, 8],
        );
    }
}
