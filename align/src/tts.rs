use espeakng::*;
use crate::mfcc::FrameSupplier;

pub(crate) struct StreamingFrameSupplier<I>
where
    I: Iterator<Item = Fragment>,
{
    spoken_utterances: I,
    cur: Fragment,
    anchors: Vec<usize>,
    idx: usize,
    duration: usize,
}

impl<I> StreamingFrameSupplier<I>
where
    I: Iterator<Item = Fragment>,
{
    fn new(utterances: I) -> Self {
        Self {
            spoken_utterances: utterances,
            cur: Fragment::default(),
            anchors: vec![0],
            idx: 0,
            duration: 0,
        }
    }
    pub(crate) fn anchors(&self) -> &[usize] {
        &self.anchors
    }
}

impl<I> FrameSupplier for StreamingFrameSupplier<I>
where
    I: Iterator<Item = Fragment>,
{
    fn n_samples_est(&self) -> usize {
        0
    }

    fn fill_next(&mut self, output: &mut [f32]) -> usize {
        if self.idx >= self.cur.data.len() {
            let Some(frag)= self.spoken_utterances.next() else {
                return 0;
            };
            self.duration += frag.duration;
            self.anchors.push(self.duration);
            self.cur = frag;
            self.idx = 0;
        }

        let from_buf = usize::min(output.len(), self.cur.data.len() - self.idx);
        for (o, i) in output[..from_buf]
            .iter_mut()
            .zip(self.cur.data[self.idx..].iter())
        {
            *o = *i as f32;
        }
        self.idx += from_buf;
        from_buf
    }
}

pub(crate) struct Spoken {
    pub(crate) wav: Vec<i16>,
    pub(crate) sample_rate: u32,
    pub(crate) anchors: Vec<usize>,
}

/// Perform Text-To-Speech
pub(crate) fn speak_multiple<S: AsRef<str>>(
    utterances: impl Iterator<Item = S>,
) -> Result<StreamingFrameSupplier<impl Iterator<Item = Fragment>>, espeakng::Error> {
    let es = EspeakNg::new()?;
    let _sample_rate = es.sample_rate();
    let spoken = es
        .synthesize_multiple(Voice::default(), utterances.into_iter())?;
    let s = StreamingFrameSupplier::new(spoken);
    Ok(s)
}
