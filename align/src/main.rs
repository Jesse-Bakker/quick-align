use clap::Parser;

use align::align;
use std::{io::Read, path::Path};

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

fn main() {
    let args = Args::parse();
    let audio_file = args.audio_file;
    let text_file = args.text_file;
    let fragments = extract_fragments(&text_file);
    let time_boundaries = align(&audio_file, &fragments);
    for (boundary, fragment) in time_boundaries.iter().zip(fragments) {
        println!("{boundary} - {fragment}")
    }
}
