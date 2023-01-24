mod audioreader;
mod dtw;
mod dtw_striped;
mod tts;

use dtw::Dtw;
use mfcc::{freq::Freq, FrameExtractionOpts, MelBanksOpts, Mfcc, MfccOptions, OfflineFeature};
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
const DTW_SKIP_PENALTY: f32 = 0.75;

fn compute_mfcc(wave: Vec<f32>, sample_freq: u32) -> Array2<f32> {
    let mut computer = OfflineFeature::new(Mfcc::new(MfccOptions {
        mel_opts: MelBanksOpts {
            n_bins: 40,
            low_freq: Some(133.3333.into()),
            high_freq: Some(6855.4976.into()),
        },
        frame_opts: FrameExtractionOpts {
            sample_freq: Freq::from(sample_freq as f32),
            frame_length_ms: MFCC_WINDOW_LENGTH,
            frame_shift_ms: MFCC_WINDOW_SHIFT,
            emphasis_factor: 0.97,
        },
        n_ceps: 13,
    }));
    computer.compute(wave.as_slice())
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

#[cfg(debug_assertions)]
mod timer {
    use std::{cell::Cell, time::Instant};

    pub(crate) struct Timer {
        last: Cell<Instant>,
    }

    impl Timer {
        pub(crate) fn new() -> Self {
            Self {
                last: Cell::new(Instant::now()),
            }
        }

        pub(crate) fn print_duration(&self, what: &'static str) {
            let last = self.last.replace(Instant::now());
            let duration = last.elapsed().as_millis();
            println!("{}: {} ms", what, duration);
        }
    }
}
#[cfg(not(debug_assertions))]
mod timer {
    pub(crate) struct Timer;

    impl Timer {
        pub(crate) fn new() -> Self {
            Self
        }
        pub(crate) fn print_duration(&self, what: &'static str) {}
    }
}

pub fn align(audio_file: &str, text_fragments: &[String]) -> Vec<f32> {
    let timer = timer::Timer::new();
    let reader = AudioReader::new();
    let audio_samples = reader.read_and_transcode_file(audio_file, 22050).unwrap();
    timer.print_duration("Reading and transcoding audio");
    let real_mfcc = compute_mfcc(audio_samples, 22050);
    timer.print_duration("Calculating audio mfcc");

    timer.print_duration("Extracting fragments");
    let synth_samples =
        tts::speak_multiple(text_fragments.iter().map(|s| s.as_str()).collect()).unwrap();
    timer.print_duration("Synthesizing samples");
    let float_samples: Vec<f32> = synth_samples
        .wav
        .into_iter()
        .map(|sample| sample as f32 / u16::MAX as f32)
        .collect();
    timer.print_duration("Scaling samples");

    let mut anchors = synth_samples.anchors;
    // The first fragment starts at 0, so this converts from end-timings to begin-timings
    anchors.insert(0, 0);

    let synth_mfcc = compute_mfcc(float_samples, synth_samples.sample_rate as u32);
    timer.print_duration("Calculating synthesized mfcc");

    // Exact algorithm is faster (and more accurate) for smaller inputs
    let dtw = if false
    /*n_samples < 2 * DTW_DELTA*/
    {
        DtwAlgorithm::Exact(DTWExact)
    } else {
        DtwAlgorithm::Striped(DTWStriped::new(DTW_DELTA, Some(DTW_SKIP_PENALTY)))
    };

    let path = dtw.path(&real_mfcc, &synth_mfcc);
    timer.print_duration("Calculating DTW");
    let (real_indices, synth_indices): (Vec<_>, Vec<_>) = path.into_iter().unzip();
    let boundaries = find_boundaries(real_indices, synth_indices, anchors);
    let ret = boundaries
        .into_iter()
        .map(|index| index as f32 * MFCC_WINDOW_SHIFT / 1000.0)
        .collect::<Vec<_>>();
    timer.print_duration("Finding boundaries");
    ret
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
