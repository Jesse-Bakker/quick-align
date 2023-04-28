use align::align;
use std::{io::BufRead, path::Path, time::Instant};

fn main() {
    const CORPUS_DIR: &str = "tests/corpus";
    let files = (0..)
        .map(|n| {
            let base_path = Path::new(CORPUS_DIR).join(n.to_string());
            (
                base_path.with_extension("flac"),
                base_path.with_extension("csv"),
            )
        })
        .take_while(|(a, b)| a.exists() && b.exists());

    let start = Instant::now();
    let mut total_large_errors = 0;
    for (i, (audio_file, transcription_file)) in files.enumerate() {
        let reader = std::io::BufReader::new(std::fs::File::open(transcription_file).unwrap());
        let (timestamps, fragments): (Vec<_>, Vec<_>) = reader
            .lines()
            .map(|line| {
                let line = line.unwrap();
                let (a, b) = line.split_once(',').unwrap().to_owned();
                (a.to_owned(), b.to_owned())
            })
            .unzip();
        let instant = std::time::Instant::now();
        let calculated_timestamps = align(audio_file.to_str().unwrap(), &fragments);
        let elapsed = instant.elapsed().as_millis();
        let errors = timestamps
            .iter()
            .zip(calculated_timestamps.iter())
            .map(|(truth, guess)| (truth.parse::<f32>().unwrap() - guess).abs());
        /*
        timestamps
            .iter()
            .zip(calculated_timestamps.iter())
            .for_each(|(a,b)| eprintln!("{a} - {b}"));
        */

        let mut errors_1s = 0;
        let mut errors_2s = 0;
        let mut errors_5s = 0;
        for error in errors {
            if error > 5. {
                errors_5s += 1;
            }
            if error > 2. {
                errors_2s += 1;
            }
            if error > 1. {
                errors_1s += 1;
            }
        }
        eprintln!("Aligning {i} took {elapsed} ms, with {errors_5s}, {errors_2s}, {errors_1s} errors over 5, 2, 1 seconds respectively.");
        total_large_errors += errors_5s;
    }
    let elapsed = start.elapsed().as_secs();
    eprintln!("Aligning all took {elapsed} seconds, with {total_large_errors} over 1 second");
}
