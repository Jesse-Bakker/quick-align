use std::slice;

use ffmpeg::format::context::Input;
use ffmpeg::format::sample::Type;
use ffmpeg::format::Sample;
use ffmpeg::frame;
use ffmpeg::software::resampling::Context;
use ffmpeg::{media, ChannelLayout, Packet};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format;
use mfcc::{FrameExtractionOpts, FrameSupplier};
use std::sync::Once;

static FFMPEG_INITIALIZED: Once = Once::new();

pub(crate) struct AudioReader;

pub(crate) struct StreamingFrameSupplier {
    ctx: Input,
    decoded_frame: frame::Audio,
    resampled_frame: frame::Audio,
    stream_idx: usize,
    n_samples_est: usize,
    decoder: ffmpeg::decoder::Audio,
    resampler: Context,
    draining: bool,
}

impl StreamingFrameSupplier {
    fn new(
        ctx: Input,
        decoder: ffmpeg::decoder::Audio,
        resampler: Context,
        stream_idx: usize,
        n_samples_est: usize,
    ) -> Self {
        Self {
            ctx,
            decoded_frame: frame::Audio::empty(),
            resampled_frame: frame::Audio::empty(),
            stream_idx,
            n_samples_est,
            decoder,
            resampler,
            draining: false,
        }
    }
}

impl StreamingFrameSupplier {
    fn read_packet(&mut self) -> Option<Packet> {
        let mut packet = Packet::empty();

        loop {
            match packet.read(&mut self.ctx) {
                Ok(..) => return Some(packet),

                Err(ffmpeg::Error::Eof) => return None,

                Err(..) => (),
            }
        }
    }
    fn fill_internal_buf(&mut self) -> Result<Option<usize>, ffmpeg::Error> {
        // First, try to receive a frame from the decoder
        if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
            self.resampler
                .run(&self.decoded_frame, &mut self.resampled_frame)?;
        } else if self.resampler.delay().is_some() {
            // If the decoder is out of frames and we have a delay, flush the resampler
            self.resampler.flush(&mut self.resampled_frame)?;
        } else if !self.draining {
            // Decoder and resampler are empty. Send a new packet
            let packet = loop {
                if let Some(packet) = self.read_packet() {
                    if packet.stream() == self.stream_idx {
                        break Some(packet);
                    } else {
                        continue;
                    }
                } else {
                    break None;
                }
            };
            if let Some(packet) = packet {
                self.decoder.send_packet(&packet)?;
            } else {
                dbg!("No more packets, start draining");
                // We are out of packets. Drain the decoder
                self.draining = true;
                self.decoder.send_eof()?;
            };
            return self.fill_internal_buf();
        } else {
            dbg!("Done");
            // Done
            return Ok(None);
        }
        Ok(Some(self.resampled_frame.samples()))
    }
}

impl FrameSupplier for StreamingFrameSupplier {
    fn n_samples_est(&self) -> usize {
        self.n_samples_est
    }

    fn fill_next(&mut self, output: &mut [f32]) -> usize {
        match self.fill_internal_buf().unwrap() {
            Some(n) if n > 0 => {
                let slice = unsafe {
                    // Cast to float
                    slice::from_raw_parts(self.resampled_frame.data(0).as_ptr() as *const _, n)
                };
                output[..n].copy_from_slice(slice);
                n
            }
            _ => 0,
        }
    }
}

impl AudioReader {
    pub(crate) fn new() -> Self {
        FFMPEG_INITIALIZED.call_once(|| {
            ffmpeg::init().unwrap();
            ffmpeg::log::set_level(ffmpeg::log::Level::Fatal);
        });
        Self
    }

    pub(crate) fn read_and_transcode_file(
        &self,
        filename: &str,
        frame_opts: FrameExtractionOpts,
    ) -> Result<StreamingFrameSupplier, ffmpeg::Error> {
        let ictx = format::input(&filename)?;
        let input = ictx
            .streams()
            .best(media::Type::Audio)
            .expect("Could not find best audio stream");
        let input_index = input.index();

        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;

        let mut decoder = context.decoder().audio()?;

        decoder.set_parameters(input.parameters())?;
        let samples_est = {
            let exact = input.frames();
            if exact != 0 {
                exact as usize
            } else {
                let duration = input.duration();
                let time_base = input.time_base();
                let frames =
                    (duration * frame_opts.sample_freq as i64 * time_base.numerator() as i64)
                        / (time_base.denominator() as i64);
                frames as usize
            }
        };

        let resampler = ffmpeg::software::resampler(
            (decoder.format(), decoder.channel_layout(), decoder.rate()),
            (
                Sample::F32(Type::Planar), // packed or planar does not matter, as we're going mono
                ChannelLayout::MONO,
                frame_opts.sample_freq,
            ),
        )?;

        Ok(StreamingFrameSupplier::new(
            ictx,
            decoder,
            resampler,
            input_index,
            samples_est,
        ))
    }
}
