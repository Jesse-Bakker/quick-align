use ndarray::prelude::*;

macro_rules! min {
        ($($args:expr),*) => {{
        let result = f32::INFINITY;
        $(
            let result = result.min($args);
        )*
        result
    }}
}

use crate::dtw::Dtw;

pub(crate) struct DTWStriped {
    delta: usize,
    skip_penalty: f32,
}

impl DTWStriped {
    pub(crate) fn new(delta: usize, skip_penalty: Option<f32>) -> Self {
        Self {
            delta,
            skip_penalty: skip_penalty.unwrap_or(f32::INFINITY),
        }
    }
}

impl DTWStriped {
    pub(crate) fn cost_matrix(
        &self,
        mfcc1: &ArrayView2<f32>,
        mfcc2: &ArrayView2<f32>,
    ) -> (Array2<f32>, Array1<usize>) {
        let mfcc1 = mfcc1.slice(s![.., 1..]);
        let mfcc2 = mfcc2.slice(s![.., 1..]);

        let normsq_1 = (&mfcc1 * &mfcc1).sum_axis(Axis(1)).map(|elem| elem.sqrt());
        let normsq_2 = (&mfcc2 * &mfcc2).sum_axis(Axis(1)).map(|elem| elem.sqrt());

        let delta = self.delta;

        let n = mfcc1.shape()[0];
        let m = mfcc2.shape()[0];

        let delta = delta.min(m);

        let mut cost_matrix: Array2<f32> = Array2::zeros((n, delta));

        let mut centers: Array1<usize> = Array1::zeros(n);

        for i in 0..n {
            let diag_j = (m * i) / n;

            let (range_start, range_end) = {
                let range_start = diag_j.saturating_sub(delta / 2);
                let range_end = range_start + delta;
                if range_end < m {
                    (range_start, range_end)
                } else {
                    (m - delta, m)
                }
            };

            let tmp = mfcc1
                .slice(s![i, ..])
                .dot(&mfcc2.slice(s![range_start..range_end, ..]).t());
            cost_matrix.slice_mut(s![i, ..]).assign(
                &(1. - (tmp / (normsq_1[i] * &normsq_2.slice(s![range_start..range_end])))),
            );
            cost_matrix.slice_mut(s![i, ..]).mapv_inplace(|x| x.abs());
            centers[i] = range_start;
        }
        (cost_matrix, centers)
    }

    pub(crate) fn accumulated_cost_matrix(
        &self,
        cost_matrix: &mut ndarray::Array2<f32>,
        centers: &Array1<usize>,
    ) {
        let (n, delta) = (cost_matrix.shape()[0], cost_matrix.shape()[1]);

        let current_row = cost_matrix.slice(s![0, ..]).to_owned();

        for j in 1..delta {
            let i = 0;
            let cost = current_row[j] + cost_matrix[(0, j - 1)];
            cost_matrix[(i, j)] = cost;
        }

        for i in 1..n {
            let current_row = cost_matrix.slice(s![i, ..]).to_owned();
            let offset = centers[i] - centers[i - 1];

            for j in 0..delta {
                let cost0 = if (j + offset) < delta {
                    cost_matrix[(i - 1, j + offset)]
                } else {
                    f32::INFINITY
                };

                let cost1 = if j > 0 {
                    cost_matrix[(i, j - 1)]
                } else {
                    f32::INFINITY
                };

                let cost2 =
                    if (j + offset).saturating_sub(1) < delta && ((j + offset) as isize - 1) >= 0 {
                        cost_matrix[(i - 1, j + offset - 1)]
                    } else {
                        f32::INFINITY
                    };
                let min_cost = current_row[j] + min!(cost0, cost1, cost2);

                cost_matrix[(i, j)] = min_cost;
            }
        }
    }

    pub(crate) fn best_path(
        &self,
        accumulated_cost_matrix: &ndarray::Array2<f32>,
        centers: &Array1<usize>,
    ) -> std::vec::Vec<(usize, usize)> {
        let (n, delta) = (
            accumulated_cost_matrix.shape()[0],
            accumulated_cost_matrix.shape()[1],
        );

        let mut i = n - 1;
        let mut j = delta + centers[i];
        let mut path = vec![(i, j)];

        while i > 0 || j > 0 {
            if i == 0 {
                path.push((0, j - 1));
                j -= 1;
            } else if j == 0 {
                path.push((i - 1, 0));
                i -= 1;
            } else {
                let offset = centers[i] - centers[i - 1];
                let r_j = j - centers[i];
                let cost0 = if (r_j + offset) < delta {
                    accumulated_cost_matrix[(i - 1, r_j + offset)]
                } else {
                    f32::INFINITY
                };

                let cost1 = if r_j > 0 {
                    accumulated_cost_matrix[(i, r_j - 1)]
                } else {
                    f32::INFINITY
                };

                let cost2 = if r_j > 0
                    && (r_j + offset - 1) < delta
                    && ((r_j + offset) as isize - 1) >= 0
                {
                    accumulated_cost_matrix[(i - 1, r_j + offset - 1)]
                } else {
                    f32::INFINITY
                };

                let moves = [(i - 1, j), (i, j - 1), (i - 1, j - 1)];
                let min_move = moves
                    .into_iter()
                    .zip([cost0, cost1, cost2])
                    .min_by(|(_, cost0), (_, cost1)| cost0.total_cmp(cost1))
                    .unwrap();
                // If the cost of the previous move is larger than the cost of skipping to here,
                // we skip to here
                if min_move.1 > self.skip_penalty * (i * j) as f32 {}
                path.push(min_move.0);
                (i, j) = min_move.0;
            }
        }

        path.reverse();
        path
    }
}

macro_rules! time {
    ($s:expr, $m:expr) => {{
        #[cfg(timing)]
        {
            let instant = std::time::Instant::now();
            let result = $s;
            eprintln!("{}: {}", $m, instant.elapsed().as_millis());
            result
        }
        #[cfg(not(timing))]
        {
            $s
        }
    }};
}

impl Dtw for DTWStriped {
    fn path(
        &self,
        mfcc1: &ndarray::Array2<f32>,
        mfcc2: &ndarray::Array2<f32>,
    ) -> Vec<(usize, usize)> {
        let (mut cost_matrix, centers) = time!(
            self.cost_matrix(&mfcc1.view(), &mfcc2.view()),
            "Cost matrix"
        );
        time!(
            self.accumulated_cost_matrix(&mut cost_matrix, &centers),
            "Acc cost matrix"
        );
        time!(self.best_path(&cost_matrix, &centers), "Best path")
    }
}

#[cfg(test)]
mod tests {
    use crate::dtw::DTWExact;

    use super::*;

    #[test]
    fn test_cm() {
        let dtw = DTWStriped::new(usize::MAX, None);
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

        assert!(dtw.cost_matrix(&a.view(), &b.view()).0.abs_diff_eq(
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
    fn test_same() {
        let dtw_exact = DTWExact;
        let dtw_striped = DTWStriped::new(usize::MAX, None);

        let mfcc1 = arr2(&[
            [
                0.37636151, 0.14506918, 0.8932821, 0.92543346, 0.23635581, 0.25896058, 0.92898,
                0.71220063, 0.43744414, 0.98328252,
            ],
            [
                0.03383205, 0.76348745, 0.19178, 0.98500093, 0.72055624, 0.71200318, 0.38637115,
                0.42476922, 0.32790205, 0.47557637,
            ],
            [
                0.14489869, 0.90298677, 0.73323956, 0.49643983, 0.26897165, 0.496066, 0.39834606,
                0.22232316, 0.14362444, 0.5231061,
            ],
            [
                0.41127344, 0.46031603, 0.23253356, 0.58537823, 0.96874476, 0.95636916, 0.65355122,
                0.22307358, 0.82437432, 0.48911197,
            ],
            [
                0.2004087, 0.03940889, 0.29363431, 0.21232599, 0.66705253, 0.84501386, 0.22834149,
                0.57098241, 0.12643724, 0.24974357,
            ],
            [
                0.40989683, 0.1943823, 0.94144313, 0.9068272, 0.29952486, 0.86217115, 0.04364901,
                0.27979906, 0.42811192, 0.23678015,
            ],
            [
                0.34069564, 0.84682527, 0.66846854, 0.06843905, 0.44969427, 0.42957777, 0.42580523,
                0.48954953, 0.15901968, 0.63884968,
            ],
            [
                0.91385204, 0.96092659, 0.93621975, 0.39778401, 0.64186873, 0.40142074, 0.02772388,
                0.387148, 0.41829588, 0.60714988,
            ],
            [
                0.1077559, 0.67411534, 0.66407206, 0.99029164, 0.20685406, 0.85178851, 0.7010098,
                0.60823241, 0.1324144, 0.75049885,
            ],
            [
                0.60380668, 0.70954641, 0.01544768, 0.96384268, 0.19111892, 0.07140014, 0.94806062,
                0.28257138, 0.48400238, 0.07699753,
            ],
        ]);

        let mfcc2 = arr2(&[
            [
                0.93941517, 0.49843187, 0.16496294, 0.4663985, 0.1746724, 0.19566175, 0.76420513,
                0.82940415, 0.22907134, 0.71552757,
            ],
            [
                0.80833703, 0.37406641, 0.15544167, 0.33168377, 0.59163366, 0.34831899, 0.33757537,
                0.2785199, 0.83435067, 0.16383171,
            ],
            [
                0.09357219, 0.57907527, 0.23678246, 0.03339791, 0.34044008, 0.92099126, 0.97719918,
                0.2513343, 0.29441095, 0.23797827,
            ],
            [
                0.37729483, 0.00993517, 0.94755076, 0.15405061, 0.21993193, 0.94196146, 0.86653768,
                0.5701422, 0.02289981, 0.36448615,
            ],
            [
                0.20253352, 0.55320022, 0.1979752, 0.84172472, 0.85201954, 0.40361247, 0.56016404,
                0.20728524, 0.30553487, 0.289864,
            ],
            [
                0.22247105, 0.6659428, 0.02187813, 0.51186015, 0.56407003, 0.60471216, 0.54282203,
                0.78300484, 0.72691429, 0.43545271,
            ],
            [
                0.12788746, 0.21785043, 0.47284899, 0.87153311, 0.53470655, 0.40151344, 0.54236609,
                0.5043277, 0.72755376, 0.80037257,
            ],
            [
                0.65201688, 0.81356688, 0.19773543, 0.54143646, 0.30342681, 0.99423742, 0.67744176,
                0.88364136, 0.75568995, 0.56409944,
            ],
            [
                0.67549221, 0.1552834, 0.59847431, 0.85322785, 0.02614456, 0.90590926, 0.86790947,
                0.17135048, 0.76815286, 0.34127661,
            ],
            [
                0.47908972, 0.08155047, 0.66176262, 0.20602723, 0.45578483, 0.90109742, 0.47987615,
                0.70847402, 0.14886114, 0.75502171,
            ],
        ]);

        let mut cm_exact = dtw_exact.cost_matrix(&mfcc1.view(), &mfcc2.view());
        let mut cm_striped = dtw_striped.cost_matrix(&mfcc1.view(), &mfcc2.view());

        assert!(cm_exact.abs_diff_eq(&cm_striped.0, 1e-6));

        dtw_exact.accumulated_cost_matrix(&mut cm_exact);
        dtw_striped.accumulated_cost_matrix(&mut cm_striped.0, &cm_striped.1);

        assert!(cm_exact.abs_diff_eq(&cm_striped.0, 1e-6));
    }
}
