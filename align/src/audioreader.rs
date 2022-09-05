use std::slice;

use ffmpeg::format::sample::Type;
use ffmpeg::format::Sample;
use ffmpeg::frame::Audio;
use ffmpeg::software::resampling::Context;
use ffmpeg::{media, ChannelLayout};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format;
use std::sync::Once;

static FFMPEG_INITIALIZED: Once = Once::new();

pub(crate) struct AudioReader;

impl AudioReader {
    pub(crate) fn new() -> Self {
        FFMPEG_INITIALIZED.call_once(|| {
            ffmpeg::init().unwrap();
            ffmpeg::log::set_level(ffmpeg::log::Level::Fatal);
        });
        Self
    }

    fn convert_and_push_frame(frame: &Audio, data: &mut Vec<f32>) {
        let samples = frame.data(0);
        let ptr = samples.as_ptr();
        data.extend_from_slice(unsafe {
            slice::from_raw_parts(ptr as *const f32, frame.samples())
        });
    }
    fn receive_and_process_decoded_frames(
        decoder: &mut ffmpeg::decoder::Audio,
        resampler: &mut Context,
        data: &mut Vec<f32>,
    ) -> Result<(), ffmpeg::Error> {
        let mut decoded = Audio::empty();
        let mut new_frame = Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            resampler.run(&decoded, &mut new_frame)?;
            Self::convert_and_push_frame(&new_frame, data);
        }
        while resampler.delay().is_some() {
            resampler.flush(&mut new_frame)?;
            Self::convert_and_push_frame(&new_frame, data);
        }
        Ok(())
    }

    pub(crate) fn read_and_transcode_file(
        &self,
        filename: &str,
        rate: u32,
    ) -> Result<Vec<f32>, ffmpeg::Error> {
        let mut ictx = format::input(&filename)?;
        let input = ictx
            .streams()
            .best(media::Type::Audio)
            .expect("Could not find best audio stream");
        let input_index = input.index();

        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;

        let mut decoder = context.decoder().audio()?;

        decoder.set_parameters(input.parameters())?;

        let mut resampler = ffmpeg::software::resampler(
            (decoder.format(), decoder.channel_layout(), decoder.rate()),
            (
                Sample::F32(Type::Planar), // packed or planar does not matter, as we're going mono
                ChannelLayout::MONO,
                rate,
            ),
        )?;

        let mut data: Vec<f32> = Vec::new();
        for (stream, packet) in ictx.packets() {
            if stream.index() == input_index {
                decoder.send_packet(&packet)?;
                Self::receive_and_process_decoded_frames(&mut decoder, &mut resampler, &mut data)?;
            }
        }
        decoder.send_eof()?;
        Self::receive_and_process_decoded_frames(&mut decoder, &mut resampler, &mut data)?;
        Ok(data)
    }
}
