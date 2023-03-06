use std::cmp::Ordering;

use ndarray::prelude::*;

use crate::dtw_striped::DTWStriped;

pub(crate) enum DtwAlgorithm {
    Exact(DTWExact),
    Striped(DTWStriped),
}

impl Dtw for DtwAlgorithm {
    fn path(&self, mfcc1: &Array2<f32>, mfcc2: &Array2<f32>) -> Vec<(usize, usize)> {
        match self {
            DtwAlgorithm::Exact(dtw) => dtw.path(mfcc1, mfcc2),
            DtwAlgorithm::Striped(dtw) => dtw.path(mfcc1, mfcc2),
        }
    }
}

pub(crate) trait Dtw {
    fn path(&self, mfcc1: &Array2<f32>, mfcc2: &Array2<f32>) -> Vec<(usize, usize)>;
}

pub(crate) struct DTWExact;

impl DTWExact {
    pub(crate) fn best_path(
        &self,
        accumulated_cost_matrix: &Array2<f32>,
    ) -> std::vec::Vec<(usize, usize)> {
        let (n, m) = accumulated_cost_matrix.dim();
        let mut i = n - 1;
        let mut j = m - 1;
        let mut path = vec![(i, j)];

        while (i > 0) || (j > 0) {
            if i == 0 {
                path.push((0, j - 1));
                j -= 1;
            } else if j == 0 {
                path.push((i - 1, 0));
                i -= 1;
            } else {
                let moves = [(i - 1, j), (i, j - 1), (i - 1, j - 1)];
                let move_ = moves
                    .iter()
                    .min_by(|first, second| {
                        let first = accumulated_cost_matrix[**first];
                        let second = accumulated_cost_matrix[**second];
                        if first > second {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    })
                    .unwrap();
                path.push(*move_);
                (i, j) = *move_;
            }
        }
        path.reverse();
        path
    }
    pub(crate) fn accumulated_cost_matrix(&self, cost_matrix: &mut Array2<f32>) {
        let (n, m) = cost_matrix.dim();
        for j in 1..m {
            cost_matrix[(0, j)] += cost_matrix[(0, j - 1)];
        }

        for i in 1..n {
            cost_matrix[(i, 0)] += cost_matrix[(i - 1, 0)];
            for j in 1..m {
                cost_matrix[(i, j)] += cost_matrix[(i - 1, j)]
                    .min(cost_matrix[(i, j - 1)])
                    .min(cost_matrix[(i - 1, j - 1)]);
            }
        }
    }
    pub(crate) fn cost_matrix(
        &self,
        mfcc1: &ArrayView2<f32>,
        mfcc2: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // Discard the first component
        let mfcc1 = mfcc1.slice(s![.., 1..]);
        let mfcc2 = mfcc2.slice(s![.., 1..]);

        let normsq_1 = (&mfcc1 * &mfcc1).sum_axis(Axis(1)).map(|elem| elem.sqrt());
        let normsq_2 = (&mfcc2 * &mfcc2).sum_axis(Axis(1)).map(|elem| elem.sqrt());

        let cost_matrix = &mfcc1.dot(&mfcc2.t());
        let norm_matrix = &normsq_1 * &normsq_2;
        1. - (cost_matrix / norm_matrix)
    }
}

impl Dtw for DTWExact {
    fn path(&self, mfcc1: &Array2<f32>, mfcc2: &Array2<f32>) -> Vec<(usize, usize)> {
        let mut cost_matrix = self.cost_matrix(&mfcc1.view(), &mfcc2.view());
        self.accumulated_cost_matrix(&mut cost_matrix);
        self.best_path(&cost_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw() {
        let dtw = DTWExact;
        let arr = arr2(&[[1., 2., 3., 4.]; 4]);
        let arr = arr.view();
        // A cost matrix between two identical matrices should be zero
        let mut cost_matrix = dtw.cost_matrix(&arr, &arr);
        assert!(cost_matrix.abs_diff_eq(&Array2::zeros(cost_matrix.dim()), 1e-6));

        dtw.accumulated_cost_matrix(&mut cost_matrix);
        assert!(cost_matrix.abs_diff_eq(&Array2::zeros(cost_matrix.dim()), 1e-6));
    }

    #[test]
    fn test_acm() {
        let dtw = DTWExact;
        let arr = [
            [0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
        ];
        let mut cm = arr2(&arr);
        dtw.accumulated_cost_matrix(&mut cm);
        assert_eq!(
            cm,
            arr2(&[
                [0., 1., 3., 6., 10.],
                [5., 6., 8., 11., 15.],
                [15., 16., 18., 21., 25.],
                [30., 31., 33., 36., 40.]
            ])
        );
    }

    #[test]
    fn test_cm() {
        let dtw = DTWExact;
        let a = arr2(&[
            [0.43123915, 0.14245438, 0.05888148, 0.9096692, 0.54578147],
            [0.36400301, 0.24935615, 0.88831442, 0.63916911, 0.16509406],
            [0.0823209, 0.29505345, 0.88290994, 0.41310553, 0.03780696],
            [0.32343827, 0.46915656, 0.44493509, 0.68262621, 0.2717263],
        ]);

        let b = arr2(&[
            [
                0.68057345, 0.95531281, 0.07671825, 0.5245585, 0.45179126, 0.06398397, 0.30966451,
                0.7299773,
            ],
            [
                0.0821018, 0.00303263, 0.69903472, 0.39051233, 0.11193357, 0.86482746, 0.51732433,
                0.60862085,
            ],
            [
                0.94818144, 0.71147869, 0.99692792, 0.09728868, 0.06965384, 0.45353979, 0.58017138,
                0.81655581,
            ],
            [
                0.54442928, 0.91329335, 0.02092644, 0.80129873, 0.09143143, 0.38208321, 0.22788604,
                0.85899155,
            ],
        ]);

        assert!(dtw.cost_matrix(&a.view(), &b.view()).abs_diff_eq(
            &arr2(&[
                [
                    0.47545551, 0.37896973, 0.42917976, 0.0756509, 0.04015874, 0.08138541,
                    0.22563027, 0.13886885
                ],
                [
                    0.1661137, 0.09157354, 0.35375425, 0.07878056, 0.06302912, 0.17034899,
                    0.17305567, 0.01729046
                ],
                [
                    0.20928561, 0.32593934, 0.06678934, 0.33771581, 0.05455241, 0.03952407,
                    0.00321784, 0.07173629
                ],
                [
                    0.27219956, 0.223295, 0.29883809, 0.08719482, 0.00786294, 0.06631786,
                    0.12345533, 0.03624143
                ],
                [
                    0.4378936, 0.25661218, 0.59277324, 0.0042661, 0.10537615, 0.21447146,
                    0.34738579, 0.14493613
                ]
            ]),
            1e-6
        ));
    }

    #[test]
    fn test_path() {
        let dtw = DTWExact;
        let acm = arr2(&[
            [0.43123915, 0.14245438, 0.05888148, 0.9096692, 0.54578147],
            [0.36400301, 0.24935615, 0.88831442, 0.63916911, 0.16509406],
            [0.0823209, 0.29505345, 0.88290994, 0.41310553, 0.03780696],
            [0.32343827, 0.46915656, 0.44493509, 0.68262621, 0.2717263],
        ]);

        assert_eq!(
            dtw.best_path(&acm),
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 4),
                (2, 4),
                (3, 4)
            ]
        )
    }
}
