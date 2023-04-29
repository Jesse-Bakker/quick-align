use super::Psi;

use super::dtw;

use super::DtwResult;

use super::window::Window;
use super::Sample;

pub(crate) fn coarsen<T>(x: &[T]) -> Vec<T>
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

pub(crate) fn expand_window(
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
        w.mark_visited(i, right - 1);
    }
    w.window
}

pub(crate) fn fast_dtw<T>(x: &[T], y: &[T], radius: Option<usize>) -> DtwResult
where
    T: Sample + Clone,
{
    let radius = radius.unwrap_or(0);
    if x.len() < 2 || y.len() < 2 {
        return dtw(x, y, None, Psi::default());
    }

    let reduced_x = coarsen(x);
    let reduced_y = coarsen(y);

    let res = fast_dtw(&reduced_x, &reduced_y, Some(radius));
    let window = expand_window(&res.path, x.len(), y.len(), radius);
    dtw(x, y, Some(window), Psi::default())
}

#[cfg(test)]
mod tests {
    use crate::dtw::{best_path, cost_matrix, SparseCostMatrix};

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

    fn materialize(m: &SparseCostMatrix) -> Vec<Vec<f32>> {
        (0..m.size().0)
            .map(|i| (0..m.size().1).map(|j| m[(i, j)]).collect())
            .collect()
    }

    #[test]
    fn test_dtw() {
        let x = &[1, 2, 3];
        let y = &[1, 1, 2, 3, 4];
        let cost_matrix = cost_matrix(x, y, None, Psi::default());
        assert_eq!(
            &materialize(&cost_matrix),
            &vec![
                vec![0., 0., 1., 3., 6.,],
                vec![1., 1., 0., 1., 3.,],
                vec![3., 3., 1., 0., 1.],
            ]
        );
        let res = best_path(&cost_matrix, Psi::default());
        assert_eq!(res.path, vec![(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)]);
        assert_eq!(
            dtw(x, y, None, Psi::default()).path,
            fast_dtw(x, y, None).path
        );
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
        assert_eq!(expand_window(path, 6, 6, 0), window);
    }
}
