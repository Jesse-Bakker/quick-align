use clap::Parser;
use ndarray::prelude::*;

use std::{io::Read, path::Path};

use symphonia::{
    core::{
        audio::SampleBuffer,
        codecs::{Decoder, DecoderOptions},
        errors::Error,
        formats::{FormatOptions, FormatReader, Packet},
        io::{MediaSourceStream, MediaSourceStreamOptions},
        meta::MetadataOptions,
        probe::Hint,
        units::Duration,
    },
    default::*,
};

use mfcc::{freq::Freq, FrameExtractionOpts, MelBanksOpts, Mfcc, MfccOptions, OfflineFeature};

const MFCC_WINDOW_SHIFT: f32 = 20.;
const MFCC_WINDOW_LENGTH: f32 = 50.;

mod dtw;
mod tts;

struct SampleIterator<'a> {
    format_reader: &'a mut dyn FormatReader,
    decoder: &'a mut dyn Decoder,
    buffer: Option<SampleBuffer<f32>>,
    buf_idx: usize,
}

impl<'a> SampleIterator<'a> {
    fn new(format_reader: &'a mut dyn FormatReader, decoder: &'a mut dyn Decoder) -> Self {
        Self {
            format_reader,
            decoder,
            buffer: None,
            buf_idx: 0,
        }
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, Error> {
        let packet = self.format_reader.next_packet();
        match packet {
            Ok(packet) => Ok(Some(packet)),
            Err(Error::IoError(_)) => Ok(None),
            Err(err) => Err(err),
        }
    }
}

impl Iterator for SampleIterator<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self
            .buffer
            .as_ref()
            .map_or(true, |buf| buf.len() == self.buf_idx)
        {
            let packet = self.read_packet().unwrap()?;
            loop {
                match self.decoder.decode(&packet) {
                    Ok(audio_buf) => {
                        let buffer = self.buffer.get_or_insert_with(|| {
                            SampleBuffer::new(audio_buf.capacity() as Duration, *audio_buf.spec())
                        });
                        buffer.copy_planar_ref(audio_buf);
                        self.buf_idx = 0;
                        break;
                    }
                    Err(Error::IoError(_)) => return None,
                    Err(Error::DecodeError(_)) => continue,
                    Err(err) => panic!("{}", err),
                }
            }
        }

        let buf_idx = self.buf_idx;
        self.buf_idx += 1;
        Some(self.buffer.as_ref().unwrap().samples()[buf_idx])
    }
}

fn read_audio_samples<P>(path: P) -> (u32, Vec<f32>)
where
    P: AsRef<Path>,
{
    let codecs = get_codecs();
    let probe = get_probe();
    let file = std::fs::File::open(path.as_ref()).expect("Could not open file");
    let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut format_hint = Hint::new();
    if let Some(extension) = path.as_ref().extension() {
        format_hint.with_extension(extension.to_str().unwrap());
    }

    let format_options = FormatOptions::default();
    let metadata_options = MetadataOptions::default();

    let probe_result = probe
        .format(&format_hint, stream, &format_options, &metadata_options)
        .expect("Could not determine stream format");

    let mut format = probe_result.format;
    let track = format.default_track().expect("No tracks in file").clone();
    let mut decoder = codecs
        .make(&track.codec_params, &DecoderOptions::default())
        .unwrap();

    let iterator = SampleIterator::new(&mut *format, &mut *decoder);
    (track.codec_params.sample_rate.unwrap(), iterator.collect())
}

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

fn main() {
    let args = Args::parse();
    let audio_file = args.audio_file;
    let text_file = args.text_file;
    let (real_sample_rate, real_samples) = read_audio_samples(audio_file);
    let real_mfcc = compute_mfcc(real_samples, real_sample_rate);

    let fragments = extract_fragments(text_file);
    let synth_samples =
        tts::speak_multiple(fragments.iter().map(|s| s.as_str()).collect()).unwrap();
    let float_samples = synth_samples
        .wav
        .into_iter()
        .map(|sample| sample as f32 / u16::MAX as f32)
        .collect();

    let mut anchors = synth_samples.anchors;
    // The first fragment starts at 0, so this converts from end-timings to begin-timings
    anchors.insert(0, 0);

    let synth_mfcc = compute_mfcc(float_samples, synth_samples.sample_rate as u32);

    let path = dtw::path(&real_mfcc, &synth_mfcc);
    let (real_indices, synth_indices): (Vec<_>, Vec<_>) = path.into_iter().unzip();
    let boundaries = find_boundaries(real_indices, synth_indices, anchors);
    let time_boundaries = boundaries
        .into_iter()
        .map(|index| index as f32 * MFCC_WINDOW_SHIFT / 1000.0)
        .collect::<Vec<_>>();
    dbg!(time_boundaries);
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
