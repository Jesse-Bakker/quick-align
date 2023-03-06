use ndarray::prelude::*;

fn coarsen(x: &ArrayView2<f32>) -> Array2<f32> {
    let shape = x.shape();
    let reduced_shape = (shape[0] / 2 + shape[0] % 2, shape[1]);
    let mut ret = Array2::zeros(reduced_shape);
    for (i, chunk) in x.axis_chunks_iter(Axis(0), 2).enumerate() {
        ret.row_mut(i).assign(&chunk.mean_axis(Axis(0)).unwrap());
    }
    ret
}
fn fast_dtw(x: &[[f32; 13]], y: &[[f32; 13]]) -> impl Iterator<Item=(usize, usize)> {
    [].into_iter()
}

fn dtw(x: &[[f32; 13]], y: &[[f32; 13]], window: &[(usize,usize)] {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coarsen() {
        let fine = arr2(&[[1., 2., 3., 4.], [2., 4., 6., 8.], [3., 6., 9., 12.]]);
        let coarse = arr2(&[[1.5, 3., 4.5, 6.], [3., 6., 9., 12.]]);
        assert_eq!(coarsen(&fine.view()), coarse);
    }
}

