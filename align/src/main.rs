use csv::Writer;
use ndarray_csv::Array2Writer;

use std::path::Path;

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

fn main() {
    let codecs = get_codecs();
    let probe = get_probe();

    let args = std::env::args().collect::<Vec<_>>();
    let filename = &args[1];
    let path = Path::new(filename);
    let file = std::fs::File::open(filename).expect("Could not open file");
    let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut format_hint = Hint::new();
    if let Some(extension) = path.extension() {
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
    let wave: Vec<_> = iterator.collect();
    println!("Collected {} samples", wave.len());
    let mut computer = OfflineFeature::new(Mfcc::new(MfccOptions {
        mel_opts: MelBanksOpts {
            n_bins: 40,
            low_freq: None,
            high_freq: None,
        },
        frame_opts: FrameExtractionOpts {
            sample_freq: Freq::from(16_000.),
            frame_length_ms: 20.,
            frame_shift_ms: 5.,
            emphasis_factor: 0.97,
        },
        n_ceps: 13,
    }));
    let mfcc = computer.compute(wave.as_slice());
    {
        let mut writer = Writer::from_path("out.csv").unwrap();
        writer.serialize_array2(&mfcc).unwrap();
    }
}
