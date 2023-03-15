use std::ops::Index;

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

    let start = path[0];
    {
        for i in start.0.saturating_sub(radius)..start.0.saturating_add(radius).min(len_x) {
            let y = start.1;
            w.window[i] = (
                y.saturating_sub(radius),
                y.saturating_add(radius).min(len_y),
            );
        }
    }

    for (i, (left, right)) in window
        .into_iter()
        .enumerate()
        .filter(|(_, (left, right))| right.saturating_sub(*left) != 0)
    {
        w.mark_visited(i.saturating_sub(radius), left.saturating_sub(radius));
        w.mark_visited(
            i.saturating_add(radius).min(len_x - 1),
            right.saturating_add(radius).min(len_y - 1),
        );
    }

    w.window
}

pub fn fast_dtw<T>(
    x: &[T],
    y: &[T],
    radius: Option<usize>,
    invariance_radius: Option<usize>,
) -> Vec<(usize, usize)>
where
    T: Sample + Clone,
{
    let invariance_radius = invariance_radius.unwrap_or(0);
    let radius = radius.unwrap_or(0);
    fast_dtw_impl(x, y, radius, invariance_radius, 1)
}

fn fast_dtw_impl<T>(
    x: &[T],
    y: &[T],
    radius: usize,
    invariance_radius: usize,
    resolution_ratio: usize,
) -> Vec<(usize, usize)>
where
    T: Sample + Clone,
{
    if x.len() < 2 || y.len() < 2 {
        return dtw(x, y, None, None);
    }

    let reduced_x = coarsen(x);
    let reduced_y = coarsen(y);

    let r = if resolution_ratio < 8 {
        Some(invariance_radius / resolution_ratio)
    } else {
        None
    };
    let path = fast_dtw_impl(
        &reduced_x,
        &reduced_y,
        radius,
        invariance_radius,
        resolution_ratio * 2,
    );
    let window = create_window(&path, x.len(), y.len(), radius);
    //let r = Some(invariance_radius / resolution_ratio);
    dtw(x, y, Some(window), r)
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

pub trait Sample: Sized {
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

        //self[1..].iter().zip(other[1..].iter()).map(|(a,b)| (a - b).powi(2)).sum()
    }
}

pub fn dtw<T>(
    x: &[T],
    y: &[T],
    window: Option<Vec<(usize, usize)>>,
    invariance_radius: Option<usize>,
) -> Vec<(usize, usize)>
where
    T: Sample,
{
    let radius = invariance_radius.unwrap_or(0);
    let cost_matrix = cost_matrix(x, y, window, radius);
    best_path(&cost_matrix, radius)
}

pub fn window_sakoe_chuba(len_x: usize, len_y: usize, delta: usize) -> Vec<(usize, usize)> {
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
pub fn dtw_sakoe_chuba<T>(x: &[T], y: &[T], delta: usize) -> Vec<(usize, usize)>
where
    T: Sample,
{
    let window = window_sakoe_chuba(x.len(), y.len(), delta);
    dtw(x, y, Some(window), None)
}

pub(crate) fn cost_matrix<T>(
    mfcc1: &[T],
    mfcc2: &[T],
    window: Option<Vec<(usize, usize)>>,
    radius: usize,
) -> SparseCostMatrix
where
    T: Sample,
{
    dbg!("cost_matrix");
    let window = window.unwrap_or_else(|| {
        (0..mfcc1.len())
            .map(|_| (0, mfcc2.len()))
            .collect::<Vec<_>>()
    });

    let indices: Vec<usize> = window
        .iter()
        .scan(0, |acc, win| {
            let ret = *acc;
            *acc += win.1.saturating_sub(win.0);
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
        if range_end.saturating_sub(*range_start) == 0 {
            continue;
        }
        for (offset, column) in mfcc2[*range_start..*range_end].iter().enumerate() {
            let j = range_start + offset;
            let dist = row.dist(column);
            let min_prev = match (i, j) {
                // Start point relaxation
                (0, j) if j <= radius => 0.,
                (i, 0) if i <= radius => 0.,
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

fn best_path(accumulated_cost_matrix: &SparseCostMatrix, radius: usize) -> Vec<(usize, usize)> {
    dbg!("path");
    let (n, m) = accumulated_cost_matrix.size();
    let mut path = Vec::with_capacity(n + m);
    let min_x = (n.saturating_sub(radius + 1)..n)
        .min_by(|a, b| {
            accumulated_cost_matrix[(*a, m - 1)].total_cmp(&accumulated_cost_matrix[(*b, m - 1)])
        })
        .unwrap();
    let min_y = (m.saturating_sub(radius + 1)..m)
        .min_by(|a, b| {
            accumulated_cost_matrix[(n - 1, *a)].total_cmp(&accumulated_cost_matrix[(n - 1, *b)])
        })
        .unwrap();

    let (mut i, mut j) =
        if accumulated_cost_matrix[(min_x, m - 1)] < accumulated_cost_matrix[(n - 1, min_y)] {
            (min_x, m - 1)
        } else {
            (n - 1, min_y)
        };

    path.push((i, j));
    while (i > 0 || j > radius) && (j > 0 || i > radius) {
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

#[cfg(test)]
mod tests {
    use super::*;

    impl Sample for u32 {
        fn dist(&self, other: &Self) -> f32 {
            self.abs_diff(*other) as f32
        }

        fn mean(&self, other: &Self) -> Self {
            (self + other) / 2
        }
    }
    impl Sample for f32 {
        fn dist(&self, other: &Self) -> f32 {
            (self - *other).abs()
        }

        fn mean(&self, other: &Self) -> Self {
            (self + other) / 2.
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
        let cost_matrix = cost_matrix(x, y, None, 0);
        assert_eq!(
            &cost_matrix.materialize(),
            &vec![
                vec![0., 0., 1., 3., 6.,],
                vec![1., 1., 0., 1., 3.,],
                vec![3., 3., 1., 0., 1.],
            ]
        );
        let path = best_path(&cost_matrix, 0);
        assert_eq!(path, vec![(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)]);
        assert_eq!(dtw(x, y, None, None), fast_dtw(x, y, None, None));
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
        // |x|x|x|x|x||
        // |x|x|x|x|| |
        let path = &[(0, 0), (0, 1), (1, 2), (2, 2)];
        let window = vec![(0, 4), (0, 5), (3, 6), (4, 6), (4, 6), (4, 6)];
        assert_eq!(create_window(path, 6, 6, 0), window);
    }

    #[test]
    fn test_start_invariant() {
        let x = [1., 2., 3., 4., 5., 8., 9.];
        let y = [3., 4., 5., 7., 1., 2., 3., 4., 5.];

        let window = window_sakoe_chuba(x.len(), y.len(), 2);
        assert_eq!(
            fast_dtw(&x, &y, Some(2), Some(5)),
            vec![(4, 0), (5, 1), (6, 2), (7, 3), (8, 4)]
        );
    }
}
