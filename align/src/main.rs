use clap::Parser;
use dtw::{DTWExact, Dtw, DtwAlgorithm};
use dtw_striped::DTWStriped;
use ndarray::prelude::*;

use std::{cell::Cell, io::Read, path::Path, time::Instant};

mod audioreader;

use mfcc::{freq::Freq, FrameExtractionOpts, MelBanksOpts, Mfcc, MfccOptions, OfflineFeature};

use crate::audioreader::AudioReader;

const MFCC_WINDOW_SHIFT: f32 = 40. /* milliseconds */;
const MFCC_WINDOW_LENGTH: f32 = 100. /* milliseconds */;

const DTW_MARGIN: f32 = 60. /* seconds */;
const DTW_DELTA: usize = (2. * DTW_MARGIN / (MFCC_WINDOW_SHIFT * 0.001)) as usize;
const DTW_SKIP_PENALTY: f32 = 0.75;

mod dtw;
mod dtw_striped;
mod tts;

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

#[derive(Parser, Debug)]
struct Args {
    #[clap(value_parser)]
    audio_file: String,

    #[clap(value_parser)]
    text_file: String,
}

fn extract_fragments<P>(path: P) -> Vec<String>
where
    P: AsRef<Path>,
{
    let mut file = std::fs::File::open(path.as_ref()).expect("Could not open file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    contents.lines().map(|s| s.to_owned()).collect()
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

struct Timer {
    last: Cell<Instant>,
}

impl Timer {
    fn new() -> Self {
        Self {
            last: Cell::new(Instant::now()),
        }
    }

    fn print_duration(&self, what: &'static str) {
        let last = self.last.replace(Instant::now());
        let duration = last.elapsed().as_millis();
        println!("{}: {} ms", what, duration);
    }
}

fn main() {
    let timer = Timer::new();
    let args = Args::parse();
    timer.print_duration("Parsing arguments");
    let audio_file = args.audio_file;
    let text_file = args.text_file;
    let reader = AudioReader::new();
    let audio_samples = reader.read_and_transcode_file(&audio_file, 22050).unwrap();
    timer.print_duration("Reading and transcoding audio");
    let real_mfcc = compute_mfcc(audio_samples, 22050);
    timer.print_duration("Calculating audio mfcc");

    let fragments = extract_fragments(text_file);
    timer.print_duration("Extracting fragments");
    let synth_samples =
        tts::speak_multiple(fragments.iter().map(|s| s.as_str()).collect()).unwrap();
    timer.print_duration("Synthesizing samples");
    let float_samples: Vec<f32> = synth_samples
        .wav
        .into_iter()
        .map(|sample| sample as f32 / u16::MAX as f32)
        .collect();
    let n_samples = float_samples.len();
    timer.print_duration("Scaling samples");

    let mut anchors = synth_samples.anchors;
    // The first fragment starts at 0, so this converts from end-timings to begin-timings
    anchors.insert(0, 0);

    let synth_mfcc = compute_mfcc(float_samples, synth_samples.sample_rate as u32);
    timer.print_duration("Calculating synthesized mfcc");

    // Exact algorithm is faster (and more accurate) for smaller inputs
    let dtw = if false /*n_samples < 2 * DTW_DELTA*/ {
        DtwAlgorithm::Exact(DTWExact)
    } else {
        DtwAlgorithm::Striped(DTWStriped::new(DTW_DELTA, Some(DTW_SKIP_PENALTY)))
    };

    let path = dtw.path(&real_mfcc, &synth_mfcc);
    timer.print_duration("Calculating DTW");
    let (real_indices, synth_indices): (Vec<_>, Vec<_>) = path.into_iter().unzip();
    let boundaries = find_boundaries(real_indices, synth_indices, anchors);
    let time_boundaries = boundaries
        .into_iter()
        .map(|index| index as f32 * MFCC_WINDOW_SHIFT / 1000.0)
        .collect::<Vec<_>>();
    timer.print_duration("Finding boundaries");
    for boundary in time_boundaries {
        print!("{boundary}, ")
    }
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
