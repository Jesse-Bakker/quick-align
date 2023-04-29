pub(crate) mod fastdtw;
mod window;

use std::ops::Index;

use crate::CmpF32;

pub struct DtwResult {
    pub path: Vec<(usize, usize)>,
    pub cost: f32,
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

        //let norm0 = self[1..].iter().map(|x| x * x).sum::<f32>().sqrt();
        //let norm1 = other[1..].iter().map(|x| x * x).sum::<f32>().sqrt();

        //let dot = self[1..]
        //    .iter()
        //    .zip(other[1..].iter())
        //    .map(|(a, b)| a * b)
        //    .sum::<f32>();
        //(1. - dot / (norm0 * norm1)).abs()
        self[1..]
            .iter()
            .zip(other[1..].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }
}

pub(crate) fn dtw<T>(x: &[T], y: &[T], window: Option<Vec<(usize, usize)>>, psi: Psi) -> DtwResult
where
    T: Sample,
{
    let cost_matrix = cost_matrix(x, y, window, psi);
    best_path(&cost_matrix, psi)
}

pub(crate) fn find_subsequence<'a>(
    needle: &[[f32; 13]],
    haystack: impl Iterator<Item = &'a [[f32; 13]]>,
    psi: usize,
) -> (usize, f32) {
    //let radius: usize = needle.len() / 2;
    let radius: usize = (3 * psi) / 2;
    // Calculate bound using only guaranteed part of needle
    haystack
        .scan(f32::INFINITY, |ub, series| {
            let new_cost = cost_eapruned(series, needle, *ub, psi, radius);
            *ub = ub.min(new_cost);
            Some(new_cost)
        })
        .enumerate()
        .min_by_key(|(_, cost)| CmpF32::from(*cost))
        .unwrap()
}

#[derive(Default, Clone, Copy)]
pub(crate) struct Psi {
    pub(crate) a_start: usize,
    pub(crate) a_end: usize,
    pub(crate) b_start: usize,
    pub(crate) b_end: usize,
}

pub(crate) fn cost_eapruned<T>(
    mfcc1: &[T],
    mfcc2: &[T],
    bsf: f32,
    psi: usize,
    radius: usize,
) -> f32
where
    T: Sample,
{
    let ub = bsf;
    let (co, li) = if mfcc1.len() < mfcc2.len() {
        (mfcc1, mfcc2)
    } else {
        (mfcc2, mfcc1)
    };

    let mut buffers = vec![f32::INFINITY; (1 + co.len()) * 2];
    let mut costs = vec![f32::INFINITY; li.len()];
    let mut c = 1;
    let mut p = co.len() + 2;
    buffers[c - 1] = 0.;
    let mut next_start = 0;
    let mut pruning_point = 0;

    for i in 0..li.len() {
        std::mem::swap(&mut c, &mut p);

        let li = &li[i];

        let j_stop = if radius < co.len() && i + 1 < co.len() - radius {
            i + radius + 1
        } else {
            co.len()
        };
        let j_start = (if i > radius { i - radius } else { 0 }).max(next_start);
        let mut next_pruning_point = j_start;
        let mut j = j_start;
        next_start = j_start;

        // Init the first column
        buffers[c + j - 1] = f32::INFINITY;

        let mut cost = f32::INFINITY;

        // Compute DTW up to the pruning point while advancing next_start: diag and top
        while j == next_start && j < pruning_point {
            let d = Sample::dist(li, &co[j]);
            cost = f32::min(buffers[p + j - 1], buffers[p + j]) + d;
            buffers[c + j] = cost;
            if cost <= ub {
                next_pruning_point = j + 1;
            } else {
                next_start += 1;
            }
            j += 1;
        }

        // Compute DTW up to the pruning point without advancing next_start: prev, diag, top
        while j < pruning_point {
            let d = Sample::dist(li, &co[j]);
            cost = min!(cost, buffers[p + j - 1], buffers[p + j]) + d;
            buffers[c + j] = cost;
            if cost <= ub {
                next_pruning_point = j + 1;
            }
            j += 1;
        }

        // Compute DTW at pruning point
        if j < j_stop {
            let d = Sample::dist(li, &co[j]);
            if j == next_start {
                // Advancing next start: only diag. Done if v>UB.
                cost = buffers[p + j - 1] + d;
                buffers[c + j] = cost;
                if cost <= ub {
                    next_pruning_point = j + 1;
                } else {
                    break;
                }
            } else {
                cost = min!(cost, buffers[p + j - 1]) + d;
                buffers[c + j] = cost;
                if cost <= ub {
                    next_pruning_point = j + 1;
                }
            }
            j += 1;
        } else if j == next_start {
            break;
        }

        // Compute DTW after pruning point: prev. Go on while we advance the next pruning point
        while j == next_pruning_point && j < j_stop {
            let d = Sample::dist(li, &co[j]);
            cost += d;
            buffers[c + j] = cost;
            if cost <= ub {
                next_pruning_point += 1;
            }
            j += 1;
        }

        if j == co.len() {
            costs[i] = cost;
        }
        pruning_point = next_pruning_point;
        /*
        if i + radius + 1 < co.len() - psi {
            ub = bsf - cb[i + radius + 1];
        }
        */
    }

    costs[costs.len() - psi..]
        .iter()
        .copied()
        .min_by_key(|cost| CmpF32::from(*cost)).unwrap()
}

pub(crate) fn cost_matrix<T>(
    mfcc1: &[T],
    mfcc2: &[T],
    window: Option<Vec<(usize, usize)>>,
    psi: Psi,
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
                (0, j) if j < psi.b_start => dist,
                (0, _) => cost_matrix[(i, j - 1)],
                (i, 0) if i < psi.a_start => dist,
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

struct PsiResult {
    path_end: (usize, usize),
    cost: f32,
}

fn psi(accumulated_cost_matrix: &SparseCostMatrix, psi: Psi) -> PsiResult {
    let (n, m) = accumulated_cost_matrix.size();
    let i_min = ((n - psi.a_end)..n)
        .min_by(|&a, &b| {
            accumulated_cost_matrix[(a, m - 1)].total_cmp(&accumulated_cost_matrix[(b, m - 1)])
        })
        .unwrap_or(n - 1);

    let j_min = ((m - psi.b_end)..m)
        .min_by(|&a, &b| {
            accumulated_cost_matrix[(m - 1, a)].total_cmp(&accumulated_cost_matrix[(m - 1, b)])
        })
        .unwrap_or(m - 1);

    let (i, j) = [(i_min, m - 1), (n - 1, j_min)]
        .into_iter()
        .min_by(|&a, &b| accumulated_cost_matrix[a].total_cmp(&accumulated_cost_matrix[b]))
        .unwrap();

    let cost = accumulated_cost_matrix[(i, j)];
    PsiResult {
        path_end: (i, j),
        cost,
    }
}

fn best_path(accumulated_cost_matrix: &SparseCostMatrix, psi_: Psi) -> DtwResult {
    let (n, m) = accumulated_cost_matrix.size();
    let mut path = Vec::with_capacity(n + m);

    let PsiResult {
        path_end: (mut i, mut j),
        cost,
    } = psi(accumulated_cost_matrix, psi_);

    path.push((i, j));
    loop {
        // One we hit the "left" or "bottom" of the matrix and we're within the
        // psi bounds, we're done
        if (i == 0 && j <= psi_.b_start) || (j == 0 && i <= psi_.a_start) {
            break;
        }
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
    DtwResult { path, cost }
}
