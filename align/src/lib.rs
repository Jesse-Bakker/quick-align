mod audioreader;
mod dtw;
mod dtw_striped;
mod tts;

use std::thread;

use dtw::Dtw;
use mfcc::{
    FrameExtractionOpts, FrameSupplier, MelBanksOpts, Mfcc, MfccComputer, MfccOptions,
};
use ndarray::Array2;

use crate::{
    audioreader::AudioReader,
    dtw::{DTWExact, DtwAlgorithm},
    dtw_striped::DTWStriped,
};

const MFCC_WINDOW_SHIFT: f32 = 40. /* milliseconds */;
const MFCC_WINDOW_LENGTH: f32 = 100. /* milliseconds */;

const DTW_MARGIN: f32 = 60. /* seconds */;
const DTW_DELTA: usize = (2. * DTW_MARGIN / (MFCC_WINDOW_SHIFT * 0.001)) as usize;
const DTW_SKIP_PENALTY: f32 = 0.70;

struct PreloadedFrameSupplier {
    frame_length: usize,
    frame_shift: usize,
    i: usize,
    wave: Vec<f32>,
}

impl PreloadedFrameSupplier {
    fn new(wave: Vec<f32>, frame_opts: FrameExtractionOpts) -> Self {
        Self {
            frame_length: frame_opts.win_size_padded(),
            frame_shift: frame_opts.win_shift(),
            i: 0,
            wave,
        }
    }
}

impl FrameSupplier for PreloadedFrameSupplier {
    fn n_samples_est(&self) -> usize {
        self.wave.len()
    }

    fn fill_next(&mut self, output: &mut [f32]) -> usize {
        let start = self.i * self.frame_shift;
        if start > self.n_samples_est() {
            return 0;
        }

        let end = (start + self.frame_length).min(self.n_samples_est());
        let len = end - start;
        output[..len].copy_from_slice(&self.wave[start..end]);
        self.i += 1;
        len
    }
}

fn compute_mfcc(wave: impl FrameSupplier, frame_opts: FrameExtractionOpts) -> Array2<f32> {
    let mut computer = MfccComputer::new(Mfcc::new(MfccOptions {
        mel_opts: MelBanksOpts {
            n_bins: 40,
            low_freq: Some(133.3333.into()),
            high_freq: Some(6855.4976.into()),
        },
        frame_opts,
        n_ceps: 13,
    }));
    computer.compute(wave)
}

fn search_sorted_right(a: Vec<usize>, v: impl ExactSizeIterator<Item = usize>) -> Vec<usize> {
    let mut ret = Vec::with_capacity(v.len());
    let mut i = 0;
    for val in v {
        while i < a.len() {
            if (i == 0 || a[i - 1] <= val) && val < a[i] {
                ret.push(i + 1);
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
        let instant = std::time::Instant::now();
        let result = $s;
        eprintln!("{}: {}", $m, instant.elapsed().as_millis());
        result
    }};
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
            let audio_samples = time!(
                reader
                    .read_and_transcode_file(audio_file, frame_opts)
                    .unwrap(),
                "Read and transcode audio"
            );
            time!(compute_mfcc(audio_samples, frame_opts), "Audio mfcc")
        });

        let t_synth = s.spawn(|| {
            let synth_samples = time!(
                tts::speak_multiple(text_fragments.iter().map(|s| s.as_str()).collect()).unwrap(),
                "Synthesize audio"
            );

            let float_samples: Vec<f32> = synth_samples
                .wav
                .into_iter()
                .map(|sample| sample as f32 / u16::MAX as f32)
                .collect();
            let frame_supplier = PreloadedFrameSupplier::new(float_samples, frame_opts);

            let mut anchors = synth_samples.anchors;
            // The first fragment starts at 0, so this converts from end-timings to begin-timings
            anchors.insert(0, 0);

            (
                anchors,
                compute_mfcc(frame_supplier, frame_opts),
            )
        });

        let audio_mfcc = t_audio.join().unwrap();
        let (anchors, synth_mfcc) = t_synth.join().unwrap();
        (audio_mfcc, synth_mfcc, anchors)
    });

    // Exact algorithm is faster (and more accurate) for smaller inputs
    let dtw = if false
    /*n_samples < 2 * DTW_DELTA*/
    {
        DtwAlgorithm::Exact(DTWExact)
    } else {
        DtwAlgorithm::Striped(DTWStriped::new(DTW_DELTA, Some(DTW_SKIP_PENALTY)))
    };

    let path = time!(dtw.path(&audio_mfcc, &synth_mfcc), "DTW");
    let (real_indices, synth_indices): (Vec<_>, Vec<_>) = path.into_iter().unzip();
    let boundaries = find_boundaries(real_indices, synth_indices, anchors);

    boundaries
        .into_iter()
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
            vec![1, 3, 6, 7, 9, 9]
        );
    }
}
