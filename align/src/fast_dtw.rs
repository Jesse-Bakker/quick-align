use std::{iter, ops::Index};

fn coarsen<T>(x: &[T]) -> Vec<T>
where
    T: Sample + Clone,
{
    let reduced_len = x.len() / 2 + x.len() % 2;
    let mut ret = Vec::with_capacity(reduced_len);
    for row in x.chunks_exact(2) {
        let [row0, row1] = row  else {
            unreachable!()
        };
        ret.push(row0.mean(row1));
    }
    ret
}

struct Window {
    window: Vec<(usize, usize)>,
    len_x: usize,
    len_y: usize,
}

impl Window {
    fn new(len_x: usize, len_y: usize) -> Self {
        Self {
            window: vec![(usize::MAX, usize::MIN); len_x],
            len_x,
            len_y,
        }
    }

    fn mark_visited(&mut self, x: usize, y: usize) {
        let (left, right) = self.window[x];
        self.window[x] = (left.min(y), right.max(y + 1).min(self.len_y));
    }
}

fn create_window(
    path: &[(usize, usize)],
    len_x: usize,
    len_y: usize,
    radius: usize,
) -> Vec<(usize, usize)> {
    let mut w = Window::new(len_x, len_y);
    let (mut last_i, mut last_j) = (usize::MAX, usize::MAX);
    for (i, j) in path {
        let (i, j) = (*i, *j);
        w.mark_visited(i * 2, j * 2);
        w.mark_visited(i * 2, j * 2 + 1);
        w.mark_visited(i * 2 + 1, j * 2);
        w.mark_visited(i * 2 + 1, j * 2 + 1);
        if i > last_i && j > last_j {
            w.mark_visited(i * 2 - 1, j * 2);
            w.mark_visited(i * 2, j * 2 - 1);
        }
        (last_i, last_j) = (i, j);
    }

    // For odd numbered series, add the last point
    for i in [1, 2] {
        for j in [1, 2] {
            w.mark_visited(len_x.saturating_sub(i), len_y.saturating_sub(j));
        }
    }

    let window = w.window.clone();
    for (i, (left, right)) in window.into_iter().enumerate() {
        let left = left.saturating_sub(radius);
        let right = right.saturating_add(radius).min(len_y);

        w.mark_visited(i, left);
        w.mark_visited(i, right);
    }
    w.window
}

pub(crate) fn fast_dtw<T>(x: &[T], y: &[T], radius: Option<usize>) -> Vec<(usize, usize)>
where
    T: Sample + Clone,
{
    let radius = radius.unwrap_or(0);
    if x.len() < 2 || y.len() < 2 {
        return dtw(x, y, None);
    }

    let reduced_x = coarsen(x);
    let reduced_y = coarsen(y);

    let path = fast_dtw(&reduced_x, &reduced_y, Some(radius));
    let window = create_window(&path, x.len(), y.len(), radius);
    dtw(x, y, Some(window))
}

macro_rules! min {
        ($($args:expr),*) => {{
        let result = f32::INFINITY;
        $(
            let result = result.min($args);
        )*
        result
    }}
}

pub(crate) struct SparseCostMatrix {
    indices: Vec<usize>,
    cost_matrix: Vec<f32>,
    windows: Vec<(usize, usize)>,
    size: (usize, usize),
}

impl SparseCostMatrix {
    fn size(&self) -> (usize, usize) {
        self.size
    }

    fn materialize(&self) -> Vec<Vec<f32>> {
        (0..self.size().0)
            .map(|i| (0..self.size().1).map(|j| self[(i, j)]).collect())
            .collect()
    }
}

impl Index<(usize, usize)> for SparseCostMatrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.indices.len());
        let (i, j) = index;
        if j < self.windows[i].0 || j >= self.windows[i].1 {
            return &f32::INFINITY;
        }
        let row_start = self.indices[i];
        &self.cost_matrix[row_start + j - self.windows[i].0]
    }
}

pub(crate) trait Sample: Sized {
    fn mean(&self, other: &Self) -> Self;

    fn dist(&self, other: &Self) -> f32;
}

impl Sample for [f32; 13] {
    fn mean(&self, other: &Self) -> Self {
        let mut ret = [0.; 13];
        for i in 0..13 {
            ret[i] = (self[i] + other[i]) / 2.;
        }
        ret
    }
    fn dist(&self, other: &Self) -> f32 {
        // Options:
        // Inner product distance (this)
        // or
        // Euclidean distance
        let norm0 = self[1..].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm1 = other[1..].iter().map(|x| x * x).sum::<f32>().sqrt();
        let dot = self[1..]
            .iter()
            .zip(other[1..].iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        (1. - dot / (norm0 * norm1)).abs()
    }
}

pub(crate) fn dtw<T>(x: &[T], y: &[T], window: Option<Vec<(usize, usize)>>) -> Vec<(usize, usize)>
where
    T: Sample,
{
    let cost_matrix = cost_matrix(x, y, window);
    best_path(&cost_matrix)
}

pub(crate) fn window_chiboe(len_x: usize, len_y: usize, delta: usize) -> Vec<(usize, usize)> {
    let (n, m) = (len_x, len_y);
    (0..n)
        .map(|i| {
            let diag_j = (m * i) / n;
            let range_start = diag_j.saturating_sub(delta / 2);
            let range_end = range_start + delta;
            if range_end < m {
                (range_start, range_end)
            } else {
                (m.saturating_sub(delta), m)
            }
        })
        .collect()
}
pub(crate) fn dtw_chiboe<T>(x: &[T], y: &[T], delta: usize) -> Vec<(usize, usize)>
where
    T: Sample,
{
    let window = window_chiboe(x.len(), y.len(), delta);
    dtw(x, y, Some(window))
}

pub(crate) fn cost_matrix<T>(
    mfcc1: &[T],
    mfcc2: &[T],
    window: Option<Vec<(usize, usize)>>,
) -> SparseCostMatrix
where
    T: Sample,
{
    let window = window.unwrap_or_else(|| {
        (0..mfcc1.len())
            .map(|_| (0, mfcc2.len()))
            .collect::<Vec<_>>()
    });

    let indices: Vec<usize> = window
        .iter()
        .scan(0, |acc, win| {
            let ret = *acc;
            *acc += win.1 - win.0;
            Some(ret)
        })
        .collect();

    let len_cost_matrix = *indices.last().unwrap();
    let mut cost_matrix = SparseCostMatrix {
        indices,
        cost_matrix: Vec::with_capacity(len_cost_matrix),
        windows: window.clone(),
        size: (mfcc1.len(), mfcc2.len()),
    };

    for ((range_start, range_end), (i, row)) in window.iter().zip(mfcc1.iter().enumerate()) {
        for (offset, column) in mfcc2[*range_start..*range_end].iter().enumerate() {
            let j = range_start + offset;
            let dist = row.dist(column);
            let min_prev = match (i, j) {
                (0, 0) => 0.,
                (0, _) => cost_matrix[(i, j - 1)],
                (_, 0) => cost_matrix[(i - 1, j)],
                (_, _) => {
                    min!(
                        cost_matrix[(i - 1, j)],
                        cost_matrix[(i, j - 1)],
                        cost_matrix[(i - 1, j - 1)]
                    )
                }
            };
            cost_matrix.cost_matrix.push(min_prev + dist);
        }
    }

    cost_matrix
}

fn best_path(accumulated_cost_matrix: &SparseCostMatrix) -> Vec<(usize, usize)> {
    let (n, m) = accumulated_cost_matrix.size();
    let mut path = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n - 1, m - 1);

    path.push((i, j));
    while i > 0 || j > 0 {
        let min_move = match (i, j) {
            (0, _) => (i, j - 1),
            (_, 0) => (i - 1, j),
            _ => [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
                .into_iter()
                .min_by(|&idx0, &idx1| {
                    accumulated_cost_matrix[idx0].total_cmp(&accumulated_cost_matrix[idx1])
                })
                .unwrap(),
        };
        path.push(min_move);
        (i, j) = min_move;
    }
    path.shrink_to_fit();
    path.reverse();
    path
}

macro_rules! time {
    ($s:expr, $m:expr) => {{
        let instant = std::time::Instant::now();
        let result = $s;
        eprintln!("{}: {}", $m, instant.elapsed().as_millis());
        result
    }};
}

#[cfg(test)]
mod tests {
    use std::{borrow::Borrow, io::BufRead};

    use espeakng::EspeakNg;
    use ndarray::{Array2, ArrayView1, ArrayView2};

    use crate::{
        audioreader::AudioReader,
        compute_mfcc,
        dtw::{DTWExact, Dtw},
        dtw_striped::DTWStriped,
        tts::{self, Spoken},
        PreloadedFrameSupplier,
    };

    use super::*;

    impl Sample for u32 {
        fn dist(&self, other: &Self) -> f32 {
            self.abs_diff(*other) as f32
        }

        fn mean(&self, other: &Self) -> Self {
            (self + other) / 2
        }
    }

    #[test]
    fn test_coarsen() {
        let fine = &[1, 3, 5, 7, 9];
        let coarse = &[2, 6];
        assert_eq!(coarsen(fine), coarse);
    }

    #[test]
    fn test_dtw() {
        let x = &[1, 2, 3];
        let y = &[1, 1, 2, 3, 4];
        let cost_matrix = cost_matrix(x, y, None);
        assert_eq!(
            &cost_matrix.materialize(),
            &vec![
                vec![0., 0., 1., 3., 6.,],
                vec![1., 1., 0., 1., 3.,],
                vec![3., 3., 1., 0., 1.],
            ]
        );
        let path = best_path(&cost_matrix);
        assert_eq!(path, vec![(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)]);
        assert_eq!(dtw(x, y, None), fast_dtw(x, y, None));
    }

    #[test]
    fn test_window() {
        // | | |x|
        // | | |x|
        // |x|x| |
        //
        // ->
        //
        // | | | | |x|x|
        // | | | | |x|x|
        // | | | | |x|x|
        // | | | |x|x|x|
        // |x|x|x|x|x| |
        // |x|x|x|x| | |
        let path = &[(0, 0), (0, 1), (1, 2), (2, 2)];
        let window = vec![(0, 4), (0, 5), (3, 6), (4, 6), (4, 6), (4, 6)];
        assert_eq!(create_window(path, 6, 6, 0), window);
    }

    #[test]
    fn test_cost_matrix() {
        let mut a = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let mut mfcc1 = Vec::new();
        let mut mfcc2 = Vec::new();

        for i in 0..10 {
            mfcc1.push(a);
            for n in a.iter_mut() {
                if i % 2 == 0 {
                    *n += 1.;
                } else {
                    *n -= 1.;
                }
            }
            mfcc2.push(a);
        }
        let c = cost_matrix(&mfcc1, &mfcc2, None);
        dbg!(c.materialize());
        let dtws = DTWExact;
        let mut cm = dtws.cost_matrix(
            &Array2::from(mfcc1.clone()).view().t(),
            &Array2::from(mfcc2.clone()).view().t(),
        );
        dtws.accumulated_cost_matrix(&mut cm);
        assert_eq!(dtws.best_path(&cm), dtw_chiboe(&mfcc1, &mfcc2, 3));
    }

    #[test]
    fn test_sample() {
        let frame_opts = mfcc::FrameExtractionOpts {
            sample_freq: 22050,
            frame_length_ms: crate::MFCC_WINDOW_LENGTH,
            frame_shift_ms: crate::MFCC_WINDOW_SHIFT,
            emphasis_factor: 0.97,
        };
        let audio_file = "tests/corpus/0.flac";
        let fragments = std::io::BufReader::new(std::fs::File::open("tests/corpus/0.csv").unwrap())
            .lines()
            .map(|l| l.unwrap().split_once(',').unwrap().1.to_owned())
            .collect::<Vec<_>>();
        let spoken = tts::speak_multiple(fragments.iter().map(|s| s.as_str()).collect()).unwrap();
        let float_samples: Vec<f32> = spoken
            .wav
            .into_iter()
            .map(|sample| sample as f32 / u16::MAX as f32)
            .collect();
        let synth_frame_supplier = PreloadedFrameSupplier::new(float_samples, frame_opts);
        let synth_mfcc = compute_mfcc(synth_frame_supplier, frame_opts);
        let reader = AudioReader::new();
        let audio_samples = reader
            .read_and_transcode_file(audio_file, frame_opts)
            .unwrap();
        let audio_mfcc = compute_mfcc(audio_samples, frame_opts);

        let window = window_chiboe(audio_mfcc.len(), synth_mfcc.len(), crate::DTW_DELTA);
        let cost_matrix = cost_matrix(&audio_mfcc, &synth_mfcc, Some(window));

        let sdtw = DTWStriped::new(crate::DTW_DELTA, None);
        let audio_mfcc_arr = Array2::from(audio_mfcc);
        let synth_mfcc_arr = Array2::from(synth_mfcc);
        let (mut striped_cm, c) = sdtw.cost_matrix(&audio_mfcc_arr.view(), &synth_mfcc_arr.view());
        sdtw.accumulated_cost_matrix(&mut striped_cm, &c);
        let raw_striped = striped_cm.into_raw_vec();

        for (i, (a, b)) in cost_matrix
            .cost_matrix
            .iter()
            .zip(raw_striped.iter())
            .enumerate()
        {
            if (a - b).abs() > 1e-3 {
                eprintln!("{i}: {a} - {b}")
            }
        }
        //assert_eq!(cost_matrix.cost_matrix, raw_striped);
        panic!();
    }
}
