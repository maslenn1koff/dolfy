#[cfg(feature = "parallel")]
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use reviero::SimdFn;
use reviero::dispatch;

macro_rules! define_dot {
    ($ty:ty) => {
        paste::paste! {
            struct [<Dot $ty>]<'a>(&'a [$ty], &'a [$ty]);

            impl SimdFn for [<Dot $ty>]<'_> {
                type Result = $ty;

                #[inline(always)]
                fn call<S: reviero::traits::Simd>(self, simd: S) -> Self::Result {
                    assert_eq!(self.0.len(), self.1.len());

                    let (x_head, x_tail) = reviero::utils::[<as_simd_ $ty s>]::<S>(self.0);
                    let (y_head, y_tail) = reviero::utils::[<as_simd_ $ty s>]::<S>(self.1);
                    let mut accumulators = [simd.[<splat_ $ty s>](0.0); 8];

                    let mut vectors = x_head.zip(y_head);
                    while vectors.len() >= 8 {
                        let (x0, y0) = vectors.next().unwrap();
                        let (x1, y1) = vectors.next().unwrap();
                        let (x2, y2) = vectors.next().unwrap();
                        let (x3, y3) = vectors.next().unwrap();
                        let (x4, y4) = vectors.next().unwrap();
                        let (x5, y5) = vectors.next().unwrap();
                        let (x6, y6) = vectors.next().unwrap();
                        let (x7, y7) = vectors.next().unwrap();

                        accumulators[0] = simd.[<mul_add_ $ty s>](x0, y0, accumulators[0]);
                        accumulators[1] = simd.[<mul_add_ $ty s>](x1, y1, accumulators[1]);
                        accumulators[2] = simd.[<mul_add_ $ty s>](x2, y2, accumulators[2]);
                        accumulators[3] = simd.[<mul_add_ $ty s>](x3, y3, accumulators[3]);
                        accumulators[4] = simd.[<mul_add_ $ty s>](x4, y4, accumulators[4]);
                        accumulators[5] = simd.[<mul_add_ $ty s>](x5, y5, accumulators[5]);
                        accumulators[6] = simd.[<mul_add_ $ty s>](x6, y6, accumulators[6]);
                        accumulators[7] = simd.[<mul_add_ $ty s>](x7, y7, accumulators[7]);
                    }

                    for (x, y) in vectors {
                        accumulators[0] = simd.[<mul_add_ $ty s>](x, y, accumulators[0]);
                    }

                    accumulators[0] = simd.[<add_ $ty s>](accumulators[0], accumulators[1]);
                    accumulators[1] = simd.[<add_ $ty s>](accumulators[2], accumulators[3]);
                    accumulators[2] = simd.[<add_ $ty s>](accumulators[4], accumulators[5]);
                    accumulators[3] = simd.[<add_ $ty s>](accumulators[6], accumulators[7]);

                    accumulators[0] = simd.[<add_ $ty s>](accumulators[0], accumulators[1]);
                    accumulators[1] = simd.[<add_ $ty s>](accumulators[2], accumulators[3]);
                    accumulators[0] = simd.[<add_ $ty s>](accumulators[0], accumulators[1]);

                    let mut product = simd.[<reduce_sum_ $ty s>](accumulators[0]);

                    for (x, y) in x_tail.iter().zip(y_tail.iter()) {
                        product += x * y;
                    }

                    product
                }
            }
        }
    };
}

define_dot!(f32);
define_dot!(f64);

#[inline(always)]
/// Computes the dot product of two slices of `f32` values.
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    dispatch(Dotf32(x, y))
}

#[inline(always)]
/// Computes the dot product of two slices of `f64` values.
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    dispatch(Dotf64(x, y))
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the dot product of two slices of `f32` values.
pub fn par_dot_f32(x: &[f32], y: &[f32], chunk_size: usize) -> f32 {
    assert!(chunk_size > 0);
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| dot_f32(x, y)).sum()
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the dot product of two slices of `f64` values.
pub fn par_dot_f64(x: &[f64], y: &[f64], chunk_size: usize) -> f64 {
    assert!(chunk_size > 0);
    assert_eq!(x.len(), y.len());
    x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).map(|(x, y)| dot_f64(x, y)).sum()
}
