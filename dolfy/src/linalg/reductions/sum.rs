#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use reviero::{SimdFn, dispatch, traits::Simd, utils};

macro_rules! define_sum {
    ($ty:ident) => {
        paste::paste! {
            struct [<Sum $ty>]<'a>(&'a [$ty]);

            impl SimdFn for [<Sum $ty>]<'_> {
                type Result = $ty;

                #[inline(always)]
                fn call<S: Simd>(self, simd: S) -> Self::Result {
                    let (mut head, tail) = utils::[<as_simd_ $ty s>]::<S>(self.0);
                    let mut acc = [simd.[<splat_ $ty s>](0.0); 8];

                    while head.len() >= 8 {
                        let x0 = head.next().unwrap();
                        let x1 = head.next().unwrap();
                        let x2 = head.next().unwrap();
                        let x3 = head.next().unwrap();
                        let x4 = head.next().unwrap();
                        let x5 = head.next().unwrap();
                        let x6 = head.next().unwrap();
                        let x7 = head.next().unwrap();

                        acc[0] = simd.[<add_ $ty s>](acc[0], x0);
                        acc[1] = simd.[<add_ $ty s>](acc[1], x1);
                        acc[2] = simd.[<add_ $ty s>](acc[2], x2);
                        acc[3] = simd.[<add_ $ty s>](acc[3], x3);
                        acc[4] = simd.[<add_ $ty s>](acc[4], x4);
                        acc[5] = simd.[<add_ $ty s>](acc[5], x5);
                        acc[6] = simd.[<add_ $ty s>](acc[6], x6);
                        acc[7] = simd.[<add_ $ty s>](acc[7], x7);
                    }

                    for vector in head {
                        acc[0] = simd.[<add_ $ty s>](acc[0], vector);
                    }

                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<add_ $ty s>](acc[2], acc[3]);
                    acc[2] = simd.[<add_ $ty s>](acc[4], acc[5]);
                    acc[3] = simd.[<add_ $ty s>](acc[6], acc[7]);
                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<add_ $ty s>](acc[2], acc[3]);
                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);

                    let mut result = simd.[<reduce_sum_ $ty s>](acc[0]);
                    for value in tail {
                        result += value;
                    }
                    result
                }
            }
        }
    };
}

define_sum!(f32);
define_sum!(f64);

#[inline(always)]
/// Computes the sum of a slice of `f32` values.
pub fn sum_f32(x: &[f32]) -> f32 {
    dispatch(Sumf32(x))
}

#[inline(always)]
/// Computes the sum of a slice of `f64` values.
pub fn sum_f64(x: &[f64]) -> f64 {
    dispatch(Sumf64(x))
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the sum of a slice of `f32` values.
pub fn par_sum_f32(x: &[f32], chunk_size: usize) -> f32 {
    assert!(chunk_size > 0);
    x.par_chunks(chunk_size).map(sum_f32).sum()
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the sum of a slice of `f64` values.
pub fn par_sum_f64(x: &[f64], chunk_size: usize) -> f64 {
    assert!(chunk_size > 0);
    x.par_chunks(chunk_size).map(sum_f64).sum()
}
