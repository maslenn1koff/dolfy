#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use reviero::{SimdFn, dispatch, traits::Simd, utils};

macro_rules! define_min {
    ($ty:ident) => {
        paste::paste! {
            struct [<Min $ty>]<'a>(&'a [$ty]);

            impl SimdFn for [<Min $ty>]<'_> {
                type Result = $ty;

                #[inline(always)]
                fn call<S: Simd>(self, simd: S) -> Self::Result {
                    let (mut head, tail) = utils::[<as_simd_ $ty s>]::<S>(self.0);
                    let mut acc = [simd.[<splat_ $ty s>]($ty::INFINITY); 8];

                    while head.len() >= 8 {
                        let x0 = head.next().unwrap();
                        let x1 = head.next().unwrap();
                        let x2 = head.next().unwrap();
                        let x3 = head.next().unwrap();
                        let x4 = head.next().unwrap();
                        let x5 = head.next().unwrap();
                        let x6 = head.next().unwrap();
                        let x7 = head.next().unwrap();

                        acc[0] = simd.[<min_ $ty s>](acc[0], x0);
                        acc[1] = simd.[<min_ $ty s>](acc[1], x1);
                        acc[2] = simd.[<min_ $ty s>](acc[2], x2);
                        acc[3] = simd.[<min_ $ty s>](acc[3], x3);
                        acc[4] = simd.[<min_ $ty s>](acc[4], x4);
                        acc[5] = simd.[<min_ $ty s>](acc[5], x5);
                        acc[6] = simd.[<min_ $ty s>](acc[6], x6);
                        acc[7] = simd.[<min_ $ty s>](acc[7], x7);
                    }

                    for vector in head {
                        acc[0] = simd.[<min_ $ty s>](acc[0], vector);
                    }

                    acc[0] = simd.[<min_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<min_ $ty s>](acc[2], acc[3]);
                    acc[2] = simd.[<min_ $ty s>](acc[4], acc[5]);
                    acc[3] = simd.[<min_ $ty s>](acc[6], acc[7]);
                    acc[0] = simd.[<min_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<min_ $ty s>](acc[2], acc[3]);
                    acc[0] = simd.[<min_ $ty s>](acc[0], acc[1]);

                    let mut result = simd.[<reduce_min_ $ty s>](acc[0]);
                    for value in tail {
                        result = result.min(*value);
                    }
                    result
                }
            }
        }
    };
}

define_min!(f32);
define_min!(f64);

#[inline(always)]
/// Computes the minimum of a slice of `f32` values.
///
/// Returns infinity for an empty slice.
pub fn min_f32(x: &[f32]) -> f32 {
    dispatch(Minf32(x))
}

#[inline(always)]
/// Computes the minimum of a slice of `f64` values.
///
/// Returns infinity for an empty slice.
pub fn min_f64(x: &[f64]) -> f64 {
    dispatch(Minf64(x))
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the minimum of a slice of `f32` values.
pub fn par_min_f32(x: &[f32], chunk_size: usize) -> f32 {
    assert!(chunk_size > 0);
    x.par_chunks(chunk_size).map(min_f32).reduce(|| f32::INFINITY, f32::min)
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the minimum of a slice of `f64` values.
pub fn par_min_f64(x: &[f64], chunk_size: usize) -> f64 {
    assert!(chunk_size > 0);
    x.par_chunks(chunk_size).map(min_f64).reduce(|| f64::INFINITY, f64::min)
}
