#[cfg(feature = "parallel")]
use rayon::prelude::*;
use reviero::{traits::Simd, *};
use std::ops::Add;

mod macros;
mod macros_axis;
use macros::*;
use macros_axis::*;

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

define_stable_norm_l2!(f32);
define_stable_norm_l2!(f64);

define_argmax!(f32);
define_argmax!(f64);

define_argmin!(f32);
define_argmin!(f64);

fn validate_axis_metadata(input_len: usize, shape: &[usize], strides: &[usize], axis: usize, output_len: usize) -> (usize, usize) {
    assert_eq!(shape.len(), strides.len(), "shape and strides must have equal rank");
    assert!(axis < shape.len(), "axis is out of bounds");

    let expected_output_len = shape
        .iter()
        .enumerate()
        .filter(|&(dimension, _)| dimension != axis)
        .try_fold(1usize, |size, (_, &length)| size.checked_mul(length))
        .expect("output shape is too large");
    assert_eq!(output_len, expected_output_len, "output length does not match the reduced shape");

    let required_input_len = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .zip(strides)
            .try_fold(0usize, |offset, (&length, &stride)| {
                (length - 1).checked_mul(stride).and_then(|span| offset.checked_add(span))
            })
            .and_then(|max_offset| max_offset.checked_add(1))
            .expect("input layout is too large")
    };
    assert!(input_len >= required_input_len, "input slice is too short for shape and strides");

    (shape[axis], strides[axis])
}

fn axis_output_offset(output_index: usize, shape: &[usize], strides: &[usize], axis: usize) -> usize {
    let mut remaining = output_index;
    let mut offset = 0;
    for dimension in (0..shape.len()).rev() {
        if dimension != axis {
            let coordinate = remaining % shape[dimension];
            remaining /= shape[dimension];
            offset += coordinate * strides[dimension];
        }
    }
    offset
}

define_axis_value_reduction!(f32, sum, 0.0, |left, right| left + right);
define_axis_value_reduction!(f32, max, f32::NEG_INFINITY, f32::max);
define_axis_value_reduction!(f32, min, f32::INFINITY, f32::min);

define_axis_value_reduction!(f64, sum, 0.0, |left, right| left + right);
define_axis_value_reduction!(f64, max, f64::NEG_INFINITY, f64::max);
define_axis_value_reduction!(f64, min, f64::INFINITY, f64::min);

define_axis_arg_reduction!(f32, argmax, >);
define_axis_arg_reduction!(f32, argmin, <);

define_axis_arg_reduction!(f64, argmax, >);
define_axis_arg_reduction!(f64, argmin, <);
