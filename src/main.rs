use quick_align::{AudioDuration, FeatBuf, FeatureOptions};
use clap::Parser;

use std::{io::Read, path::Path, thread, time::Duration};

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

    contents
        .lines()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}
fn align(audio_file: &str, text_fragments: &[String]) -> Vec<AudioDuration> {
    let feature_options = FeatureOptions {
        frame_length: 0.1,
        frame_shift: 0.04,
        emphasis_factor: 0.97,
    };

    let (audio_mfcc, synth_mfcc, anchors) = thread::scope(|s| {
        let t_audio = s.spawn(|| FeatBuf::from_audio_file(audio_file, feature_options));

        let t_synth =
            s.spawn(|| FeatBuf::from_text_fragments(text_fragments.iter(), feature_options));

        let audio_mfcc = t_audio.join().unwrap().unwrap();
        let (synth_mfcc, anchors) = t_synth.join().unwrap().unwrap();
        (audio_mfcc, synth_mfcc, anchors)
    });

    let audio_mfcc = audio_mfcc.as_segment();
    let max_start_skip = audio_mfcc.duration(Duration::from_secs(60));
    let min_silence_length = audio_mfcc.duration(Duration::from_millis(400));
    let audio_mfcc = audio_mfcc.find_overlap(&synth_mfcc.as_segment(), max_start_skip, min_silence_length);
    let alignment = audio_mfcc.align_with(&synth_mfcc.as_segment(), &anchors);
    alignment.into_iter().map(|(audio, _synth)| audio).collect()
}
fn main() {
    let args = Args::parse();
    let audio_file = args.audio_file;
    let text_file = args.text_file;
    let fragments = extract_fragments(&text_file);
    let time_boundaries = align(&audio_file, &fragments);
    for (boundary, fragment) in time_boundaries.iter().zip(fragments) {
        let seconds = boundary.seconds();
        println!("{seconds} - {fragment}")
    }
}
