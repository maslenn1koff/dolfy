#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use reviero::{SimdFn, dispatch, traits::Simd, utils};

macro_rules! define_max {
    ($ty:ident) => {
        paste::paste! {
            struct [<Max $ty>]<'a>(&'a [$ty]);

            impl SimdFn for [<Max $ty>]<'_> {
                type Result = $ty;

                #[inline(always)]
                fn call<S: Simd>(self, simd: S) -> Self::Result {
                    let (mut head, tail) = utils::[<as_simd_ $ty s>]::<S>(self.0);
                    let mut acc = [simd.[<splat_ $ty s>]($ty::NEG_INFINITY); 8];

                    while head.len() >= 8 {
                        let x0 = head.next().unwrap();
                        let x1 = head.next().unwrap();
                        let x2 = head.next().unwrap();
                        let x3 = head.next().unwrap();
                        let x4 = head.next().unwrap();
                        let x5 = head.next().unwrap();
                        let x6 = head.next().unwrap();
                        let x7 = head.next().unwrap();

                        acc[0] = simd.[<max_ $ty s>](acc[0], x0);
                        acc[1] = simd.[<max_ $ty s>](acc[1], x1);
                        acc[2] = simd.[<max_ $ty s>](acc[2], x2);
                        acc[3] = simd.[<max_ $ty s>](acc[3], x3);
                        acc[4] = simd.[<max_ $ty s>](acc[4], x4);
                        acc[5] = simd.[<max_ $ty s>](acc[5], x5);
                        acc[6] = simd.[<max_ $ty s>](acc[6], x6);
                        acc[7] = simd.[<max_ $ty s>](acc[7], x7);
                    }

                    for vector in head {
                        acc[0] = simd.[<max_ $ty s>](acc[0], vector);
                    }

                    acc[0] = simd.[<max_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<max_ $ty s>](acc[2], acc[3]);
                    acc[2] = simd.[<max_ $ty s>](acc[4], acc[5]);
                    acc[3] = simd.[<max_ $ty s>](acc[6], acc[7]);
                    acc[0] = simd.[<max_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<max_ $ty s>](acc[2], acc[3]);
                    acc[0] = simd.[<max_ $ty s>](acc[0], acc[1]);

                    let mut result = simd.[<reduce_max_ $ty s>](acc[0]);
                    for value in tail {
                        result = result.max(*value);
                    }
                    result
                }
            }
        }
    };
}

define_max!(f32);
define_max!(f64);

#[inline(always)]
/// Computes the maximum of a slice of `f32` values.
///
/// Returns negative infinity for an empty slice.
pub fn max_f32(x: &[f32]) -> f32 {
    dispatch(Maxf32(x))
}

#[inline(always)]
/// Computes the maximum of a slice of `f64` values.
///
/// Returns negative infinity for an empty slice.
pub fn max_f64(x: &[f64]) -> f64 {
    dispatch(Maxf64(x))
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the maximum of a slice of `f32` values.
pub fn par_max_f32(x: &[f32], chunk_size: usize) -> f32 {
    assert!(chunk_size > 0);
    x.par_chunks(chunk_size).map(max_f32).reduce(|| f32::NEG_INFINITY, f32::max)
}

#[cfg(feature = "parallel")]
#[inline(always)]
/// Parallel computes the maximum of a slice of `f64` values.
pub fn par_max_f64(x: &[f64], chunk_size: usize) -> f64 {
    assert!(chunk_size > 0);
    x.par_chunks(chunk_size).map(max_f64).reduce(|| f64::NEG_INFINITY, f64::max)
}
