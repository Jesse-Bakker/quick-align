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
    buf_idx: usize,
    stream_idx: usize,
    frame_shift: usize,
    n_samples_est: usize,
    decoder: ffmpeg::decoder::Audio,
    resampler: Context,
    draining: bool,
    last_output: Vec<f32>,
}

impl StreamingFrameSupplier {
    fn new(
        ctx: Input,
        decoder: ffmpeg::decoder::Audio,
        resampler: Context,
        stream_idx: usize,
        n_samples_est: usize,
        frame_opts: FrameExtractionOpts,
    ) -> Self {
        Self {
            ctx,
            decoded_frame: frame::Audio::empty(),
            resampled_frame: frame::Audio::empty(),
            buf_idx: 0,
            stream_idx,
            last_output: Vec::new(),
            frame_shift: frame_opts.win_shift(),
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
                // We are out of packets. Drain the decoder
                self.draining = true;
                self.decoder.send_eof()?;
            };
            return self.fill_internal_buf();
        } else {
            // Done
            return Ok(None);
        }
        Ok(Some(self.resampled_frame.samples()))
    }

    fn init_last_output(&mut self, len: usize) -> usize {
        assert!(self.last_output.is_empty());
        self.last_output.reserve_exact(len);
        let mut filled = 0;
        while let Some(n) = self.fill_internal_buf().unwrap() {
            if n == 0 {
                return filled;
            }
            let from_this_buf = usize::min(n, len - filled);
            unsafe {
                let buf =
                    slice::from_raw_parts(self.resampled_frame.data(0).as_ptr() as *const f32, n);
                self.last_output.extend_from_slice(&buf[..from_this_buf]);
            };
            filled += n;
            if filled > len {
                self.buf_idx = from_this_buf;
                return len;
            }
        }
        0
    }
}

impl FrameSupplier for StreamingFrameSupplier {
    fn n_samples_est(&self) -> usize {
        self.n_samples_est
    }

    fn fill_next(&mut self, output: &mut [f32]) -> usize {
        assert!(output.len() > self.frame_shift);
        if self.last_output.is_empty() {
            let len = self.init_last_output(output.len());
            output[..len].copy_from_slice(&self.last_output);
            return len;
        }

        let mut len = output.len() - self.frame_shift;
        output[..len].copy_from_slice(&self.last_output[self.frame_shift..]);

        let mut remaining = self.frame_shift;
        while remaining != 0 {
            let left_in_buf = {
                let left_in_buf = self.resampled_frame.samples() - self.buf_idx;
                if left_in_buf != 0 {
                    left_in_buf
                } else {
                    self.buf_idx = 0;
                    match self.fill_internal_buf().unwrap() {
                        Some(0) | None => return len,
                        Some(n) => n,
                    }
                }
            };
            let from_this_buf = usize::min(remaining, left_in_buf);
            unsafe {
                let buf = slice::from_raw_parts(
                    self.resampled_frame.data(0).as_ptr() as *const f32,
                    self.resampled_frame.samples(),
                );
                output[len..len + from_this_buf]
                    .copy_from_slice(&buf[self.buf_idx..(self.buf_idx + from_this_buf)]);
                self.buf_idx += from_this_buf;
                len += from_this_buf;
                remaining -= from_this_buf;
            }
        }
        self.last_output = output.into();
        len
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
            frame_opts,
        ))
    }
}
