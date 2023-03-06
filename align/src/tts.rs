use espeakng::*;
use mfcc::FrameSupplier;

pub(crate) struct StreamingFrameSupplier<I> {
    spoken_utterances: I,
    cur: Fragment,
    achors: Vec<usize>,
    idx: usize,
}

impl<I> FrameSupplier for StreamingFrameSupplier<I>
where
    I: Iterator<Item = Fragment>,
{
    fn n_samples_est(&self) -> usize {
        0
    }

    fn fill_next(&mut self, output: &mut [Float]) -> usize {
        if self.idx >= self.cur.data.len() {
            let Some(frag)= self.spoken_utterances.next() else {
                return 0;
            };
            self.cur = frag;
            self.idx = 0;
        }

        let from_buf = usize::min(output.len(), self.cur.data.len() - self.idx);
    }
}

pub(crate) struct Spoken {
    pub(crate) wav: Vec<i16>,
    pub(crate) sample_rate: u32,
    pub(crate) anchors: Vec<usize>,
}

/// Perform Text-To-Speech
pub(crate) fn speak_multiple(utterances: Vec<&str>) -> Result<Spoken, ()> {
    let es = EspeakNg::new().map_err(|_| ())?;
    let sample_rate = es.sample_rate();
    let out: Vec<_> = es
        .synthesize_multiple(Voice::default(), utterances.into_iter())
        .map_err(|_| ())?
        .collect();

    let size = out.iter().fold(0, |acc, e| acc + e.data.len());

    let mut sum = 0;
    let anchors = out
        .iter()
        .map(|b| {
            sum += b.duration;
            sum
        })
        .collect();

    let wav = out
        .into_iter()
        .fold(Vec::with_capacity(size), |mut v, mut w| {
            v.append(&mut w.data);
            v
        });

    Ok(Spoken {
        wav,
        sample_rate,
        anchors,
    })
}
