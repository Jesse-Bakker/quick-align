mod audioreader;
mod dtw;
mod freq;
mod tts;

use std::{path::Path, time::Duration};

use audioreader::AudioReader;
mod mfcc;
use dtw::{fastdtw::fast_dtw, DtwResult};
use mfcc::{mfcc, FrameExtractionOpts, MelBanksOpts, MfccOptions};

use crate::dtw::find_subsequence;

const SILENCE_MAX_ENERGY: f32 = 0.699; // log10(5)

#[derive(Copy, Clone)]
pub struct AudioDuration {
    frames: usize,
    frame_shift: f32,
}

impl AudioDuration {
    pub fn seconds(&self) -> f32 {
        self.frame_shift * self.frames as f32
    }
}

pub struct FeatBuf {
    feature: Vec<[f32; 13]>,
    options: FeatureOptions,
}

#[derive(Debug)]
pub enum Error {
    FFMpeg(ffmpeg_next::Error),
    Espeak(espeakng::Error),
}

impl From<ffmpeg_next::Error> for Error {
    fn from(err: ffmpeg_next::Error) -> Self {
        Self::FFMpeg(err)
    }
}
impl From<espeakng::Error> for Error {
    fn from(err: espeakng::Error) -> Self {
        Self::Espeak(err)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FeatureOptions {
    pub frame_length: f32,
    pub frame_shift: f32,
    pub emphasis_factor: f32,
}

impl FeatBuf {
    pub fn as_segment(&self) -> FeatBufSegment<'_> {
        FeatBufSegment {
            segment: AudioSegment {
                start: 0,
                end: self.feature.len(),
                _frame_shift: self.options.frame_shift,
            },
            buf: self,
        }
    }

    fn create_options(feat_opts: FeatureOptions) -> MfccOptions {
        let FeatureOptions {
            frame_length,
            frame_shift,
            emphasis_factor,
        } = feat_opts;

        MfccOptions {
            mel_opts: MelBanksOpts {
                n_bins: 40,
                low_freq: Some(133.3333.into()),
                high_freq: Some(6855.4976.into()),
            },
            frame_opts: FrameExtractionOpts {
                sample_freq: 22050,
                frame_length_ms: frame_length * 1000.,
                frame_shift_ms: frame_shift * 1000.,
                emphasis_factor,
            },
            n_ceps: 13,
        }
    }

    pub fn from_audio_file(path: impl AsRef<Path>, options: FeatureOptions) -> Result<Self, Error> {
        let mfcc_options = Self::create_options(options);
        let reader = AudioReader::new();
        let mut audio_samples =
            reader.read_and_transcode_file(path, mfcc_options.frame_opts.sample_freq)?;
        Ok(Self {
            feature: mfcc(Self::create_options(options), &mut audio_samples).collect(),
            options,
        })
    }

    pub fn from_text_fragments<S: AsRef<str>>(
        fragments: impl Iterator<Item = S>,
        options: FeatureOptions,
    ) -> Result<(Self, Vec<AudioDuration>), Error> {
        let mfcc_options = Self::create_options(options);
        let mut spoken = tts::speak_multiple(fragments)?;
        let feature = mfcc(mfcc_options, &mut spoken).collect();
        let anchors = spoken.anchors();
        let anchors = anchors
            .iter()
            .map(|duration| AudioDuration {
                frames: *duration / (options.frame_shift * 1000.) as usize,
                frame_shift: options.frame_shift,
            })
            .collect();
        Ok((Self { feature, options }, anchors))
    }
}

#[derive(Clone, Copy)]
pub struct FeatBufSegment<'a> {
    segment: AudioSegment,
    buf: &'a FeatBuf,
}

impl FeatBufSegment<'_> {
    pub fn as_slice(&self) -> &[[f32; 13]] {
        &self.buf.feature[self.segment.start..self.segment.end]
    }
    pub fn silences(
        &self,
        minimum_length: AudioDuration,
        max_energy: Option<f32>,
    ) -> impl Iterator<Item = AudioSegment> + '_ {
        silences(
            self.as_slice().iter().copied(),
            minimum_length.frames,
            max_energy.unwrap_or(SILENCE_MAX_ENERGY),
        )
        .map(|(start, end)| AudioSegment {
            start,
            end,
            _frame_shift: self.buf.options.frame_shift,
        })
    }

    pub fn duration(&self, duration: Duration) -> AudioDuration {
        let frames = duration.as_secs_f32() / self.buf.options.frame_shift;
        AudioDuration {
            frames: frames as usize,
            frame_shift: self.buf.options.frame_shift,
        }
    }

    pub fn find_overlap(
        &self,
        other: &FeatBufSegment<'_>,
        max_start_skip: AudioDuration,
        min_silence_length: AudioDuration,
    ) -> FeatBufSegment {
        assert_eq!(self.buf.options, other.buf.options);
        let this = self.as_slice();
        let other = other.as_slice();
        let window_shift = self.buf.options.frame_shift;

        let silence_min_frames: usize = min_silence_length.frames;
        let start_detection_max_skip_frames: usize = max_start_skip.frames;
        let start_detection_other_frames: usize = start_detection_max_skip_frames.min(other.len());

        let silences = self
            .silences(
                AudioDuration {
                    frames: silence_min_frames,
                    frame_shift: window_shift,
                },
                Some(SILENCE_MAX_ENERGY),
            )
            .map(|duration| duration.end);
        let candidates: Vec<usize> = std::iter::once(0)
            .chain(silences)
            .take_while(|&end| end < start_detection_max_skip_frames)
            .collect();
        let Some(last) = candidates.last() else {
            return *self;
        };

        let max_start_detection_audio_frames = (1.2 * start_detection_other_frames as f64) as usize;
        let max_start_detection_audio_frames =
            max_start_detection_audio_frames.min(this.len() - last);
        let min_start_detection_audio_frames =
            max_start_detection_audio_frames - (0.2 * start_detection_other_frames as f64) as usize;
        let psi = max_start_detection_audio_frames - min_start_detection_audio_frames;

        let other_frames = &other[0..start_detection_other_frames];

        let candidate_samples = candidates
            .iter()
            .map(|&start| &this[start..(start + max_start_detection_audio_frames)]);
        let (start, _cost) = find_subsequence(other_frames, candidate_samples, Some(psi), None);
        let mut seg = self.segment;
        seg.start += candidates[start];
        FeatBufSegment {
            segment: seg,
            buf: self.buf,
        }
    }

    fn find_boundaries(
        &self,
        this_indices: &[usize],
        other_indices: &[usize],
        anchors: &[usize],
    ) -> Vec<usize> {
        // Now, find where we should insert these anchors into our synthetic indices array, to find
        // where they would occur in real_indices, thus where they would occur in our real audio
        let begin_indices = search_sorted_right(other_indices, anchors.iter().copied());
        begin_indices.into_iter().map(|i| this_indices[i]).collect()
    }
    pub fn align_with(
        &self,
        other: &FeatBufSegment,
        anchors: &[AudioDuration],
    ) -> Vec<(AudioDuration, AudioDuration)> {
        assert_eq!(self.buf.options, other.buf.options);
        let DtwResult { path, cost: _ } = fast_dtw(self.as_slice(), other.as_slice(), Some(100));

        let frame_anchors = anchors.iter().map(|a| a.frames).collect::<Vec<_>>();
        let (this_indices, other_indices): (Vec<_>, Vec<_>) = path
            .into_iter()
            .map(|(a, b)| (a + self.segment.start, b + other.segment.start))
            .unzip();
        let boundaries = self.find_boundaries(&this_indices, &other_indices, &frame_anchors);
        boundaries
            .into_iter()
            .map(|boundary| AudioDuration {
                frames: boundary,
                frame_shift: self.buf.options.frame_shift,
            })
            .zip(anchors.iter().copied())
            .collect()
    }
}

#[derive(Clone, Copy)]
pub struct AudioSegment {
    start: usize,
    end: usize,
    _frame_shift: f32,
}

impl AudioSegment {
    fn new(start: AudioDuration, end: AudioDuration) -> Self {
        assert_eq!(start.frame_shift, end.frame_shift);
        Self {
            start: start.frames,
            end: end.frames,
            _frame_shift: start.frame_shift,
        }
    }
    fn start(&self) -> AudioDuration {
        AudioDuration {
            frames: self.start,
            frame_shift: self._frame_shift,
        }
    }

    fn end(&self) -> AudioDuration {
        AudioDuration {
            frames: self.end,
            frame_shift: self._frame_shift,
        }
    }
}

#[derive(PartialEq)]
struct CmpF32(f32);

impl Eq for CmpF32 {}

impl PartialOrd for CmpF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for CmpF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl From<f32> for CmpF32 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

struct Silences<I>
where
    I: Iterator<Item = (usize, [f32; 13])>,
{
    inner: I,
    current_silence: Option<(usize, usize)>,
    minimum_length: usize,
    max_energy: f32,
}

impl<I> Iterator for Silences<I>
where
    I: Iterator<Item = (usize, [f32; 13])>,
{
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        const MIN_ENERGY: f32 = -11.512_925; // ln(0.0001)
        loop {
            let (index, mfcc) = self.inner.next()?;
            match (
                self.current_silence,
                MIN_ENERGY.max(mfcc[0]) < MIN_ENERGY + self.max_energy,
            ) {
                (Some((start, _)), true) => {
                    self.current_silence = Some((start, index));
                }
                (Some(interval @ (start, end)), false) => {
                    self.current_silence = None;
                    if end - start > self.minimum_length {
                        return Some(interval);
                    }
                }
                (None, true) => {
                    self.current_silence = Some((index, index));
                }
                (None, false) => {
                    continue;
                }
            }
        }
    }
}

pub fn silences(
    mfcc: impl Iterator<Item = [f32; 13]>,
    minimum_length: usize,
    max_energy: f32,
) -> impl Iterator<Item = (usize, usize)> {
    Silences {
        inner: mfcc.enumerate(),
        current_silence: None,
        minimum_length,
        max_energy,
    }
}

fn search_sorted_right(a: &[usize], v: impl ExactSizeIterator<Item = usize>) -> Vec<usize> {
    let mut ret = Vec::with_capacity(v.len());
    let mut i = 0;
    for val in v {
        while i < a.len() {
            if (i == 0 || a[i - 1] <= val) && val < a[i] {
                ret.push(i);
                break;
            }
            i += 1;
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_sorted() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let v = vec![0, 2, 5, 6, 8, 8];
        assert_eq!(
            search_sorted_right(&a, v.into_iter()),
            vec![0, 2, 5, 6, 8, 8],
        );
    }
}
