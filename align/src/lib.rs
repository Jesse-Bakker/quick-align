mod audioreader;
mod dtw;
mod dtw_striped;
mod fast_dtw;
mod tts;

use std::thread;

use audioreader::AudioReader;
use mfcc::{mfcc, FrameExtractionOpts, FrameSupplier, MelBanksOpts, MfccIter, MfccOptions};

const MFCC_WINDOW_SHIFT: f32 = 40. /* milliseconds */;
const MFCC_WINDOW_LENGTH: f32 = 100. /* milliseconds */;

fn compute_mfcc<T: FrameSupplier>(
    wave: &mut T,
    frame_opts: FrameExtractionOpts,
) -> MfccIter<'_, T> {
    mfcc(
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
            time!(
                compute_mfcc(&mut audio_samples, frame_opts).collect::<Vec<_>>(),
                "Audio mfcc"
            )
        });

        let t_synth = s.spawn(|| {
            let mut synth_samples = time!(
                tts::speak_multiple(text_fragments.into()).unwrap(),
                "Synthesize audio"
            );

            let mfcc = compute_mfcc(&mut synth_samples, frame_opts).collect::<Vec<_>>();
            let anchors = synth_samples.anchors();
            (anchors, mfcc)
        });

        let audio_mfcc = t_audio.join().unwrap();
        let (anchors, synth_mfcc) = t_synth.join().unwrap();
        (audio_mfcc, synth_mfcc, anchors)
    });

    let path = fast_dtw::fast_dtw(&audio_mfcc, &synth_mfcc, Some(100));
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
