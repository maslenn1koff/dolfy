#[cfg(feature = "parallel")]
use rayon::prelude::*;
use reviero::{traits::Simd, *};
use std::ops::Add;

mod macros;
use macros::*;

define_unary_reduction!(f32, sum, add, 0.0, reduce_sum);
define_unary_reduction!(f32, max, max, f32::NEG_INFINITY, reduce_max);
define_unary_reduction!(f32, min, min, f32::INFINITY, reduce_min);

define_unary_reduction!(f64, sum, add, 0.0, reduce_sum);
define_unary_reduction!(f64, max, max, f64::NEG_INFINITY, reduce_max);
define_unary_reduction!(f64, min, min, f64::INFINITY, reduce_min);

define_dot!(f32);
define_dot!(f64);

define_norm_l1!(f32);
define_norm_l1!(f64);

define_stable_norm_l2!(f32);
define_stable_norm_l2!(f64);

define_argmax!(f32);
define_argmax!(f64);

define_argmin!(f32);
define_argmin!(f64);

/// Computes the L2 norm of a slice of `f32` values.
pub fn norm_l2_f32(x: &[f32]) -> f32 {
    let product = dot_f32(x, x);
    product.sqrt()
}

/// Computes the L2 norm of a slice of `f64` values.
pub fn norm_l2_f64(x: &[f64]) -> f64 {
    let product = dot_f64(x, x);
    product.sqrt()
}

/// Parallel computes the L2 norm of a slice of `f32` values.
///
/// # Panics
/// Panics if `chunk_size` is zero.
#[cfg(feature = "parallel")]
pub fn par_norm_l2_f32(x: &[f32], chunk_size: usize) -> f32 {
    par_dot_f32(x, x, chunk_size).sqrt()
}

/// Parallel computes the L2 norm of a slice of `f64` values.
///
/// # Panics
/// Panics if `chunk_size` is zero.
#[cfg(feature = "parallel")]
pub fn par_norm_l2_f64(x: &[f64], chunk_size: usize) -> f64 {
    par_dot_f64(x, x, chunk_size).sqrt()
}
