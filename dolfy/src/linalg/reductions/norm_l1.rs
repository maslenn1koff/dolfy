#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use reviero::{SimdFn, dispatch, traits::Simd, utils};

macro_rules! define_norm_l1 {
    ($ty:ident) => {
        paste::paste! {
            struct [<NormL1 $ty>]<'a>(&'a [$ty]);

            impl <'a> SimdFn for [<NormL1 $ty>]<'a> {
                type Result = $ty;
                fn call<S: Simd>(self, simd: S) -> Self::Result {
                    let (mut head, tall) = utils::[<as_simd_ $ty s>]::<S>(self.0);
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

                        acc[0] = simd.[<add_ $ty s>](acc[0], simd.[<abs_ $ty s>](x0));
                        acc[1] = simd.[<add_ $ty s>](acc[1], simd.[<abs_ $ty s>](x1));
                        acc[2] = simd.[<add_ $ty s>](acc[2], simd.[<abs_ $ty s>](x2));
                        acc[3] = simd.[<add_ $ty s>](acc[3], simd.[<abs_ $ty s>](x3));
                        acc[4] = simd.[<add_ $ty s>](acc[4], simd.[<abs_ $ty s>](x4));
                        acc[5] = simd.[<add_ $ty s>](acc[5], simd.[<abs_ $ty s>](x5));
                        acc[6] = simd.[<add_ $ty s>](acc[6], simd.[<abs_ $ty s>](x6));
                        acc[7] = simd.[<add_ $ty s>](acc[7], simd.[<abs_ $ty s>](x7));
                    }

                    for vector in head {
                        acc[0] = simd.[<add_ $ty s>](acc[0], simd.[<abs_ $ty s>](vector));
                    }

                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<add_ $ty s>](acc[2], acc[3]);
                    acc[2] = simd.[<add_ $ty s>](acc[4], acc[5]);
                    acc[3] = simd.[<add_ $ty s>](acc[6], acc[7]);

                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<add_ $ty s>](acc[2], acc[3]);
                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);

                    let mut result = simd.[<reduce_sum_ $ty s>](acc[0]);

                    for x in tall {
                        result += x.abs();
                    }

                    result
                }
            }
        }
    };
}

define_norm_l1!(f32);
define_norm_l1!(f64);

#[inline(always)]
// Computes the L1 norm of a slice of `f32` values.
pub fn norm_l1_f32(x: &[f32]) -> f32 {
    dispatch(NormL1f32(x))
}

#[inline(always)]
// Computes the L1 norm of a slice of `f64` values.
pub fn norm_l1_f64(x: &[f64]) -> f64 {
    dispatch(NormL1f64(x))
}

#[cfg(feature = "parallel")]
#[inline(always)]
// Parallel computes the L1 norm of a slice of `f32` values.
pub fn par_norm_l1_f32(x: &[f32], chunk_size: usize) -> f32 {
    x.par_chunks(chunk_size).map(norm_l1_f32).sum()
}

#[cfg(feature = "parallel")]
#[inline(always)]
// Parallel computes the L1 norm of a slice of `f64` values.
pub fn par_norm_l1_f64(x: &[f64], chunk_size: usize) -> f64 {
    x.par_chunks(chunk_size).map(norm_l1_f64).sum()
}
