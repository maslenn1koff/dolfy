#[cfg(feature = "parallel")]
use rayon::prelude::*;
use reviero::{SimdFn, dispatch};
use std::ops::{Add, Div, Mul, Neg, Sub};

mod macros;
use macros::*;

define_ternary_elementwise_op!(f32, mul_add);
define_ternary_elementwise_op!(f64, mul_add);

define_binary_elementwise_op!(f32, add);
define_binary_elementwise_op!(f32, sub);
define_binary_elementwise_op!(f32, mul);
define_binary_elementwise_op!(f32, div);
define_binary_elementwise_op!(f32, max);
define_binary_elementwise_op!(f32, min);

define_binary_elementwise_op!(f64, add);
define_binary_elementwise_op!(f64, sub);
define_binary_elementwise_op!(f64, mul);
define_binary_elementwise_op!(f64, div);
define_binary_elementwise_op!(f64, max);
define_binary_elementwise_op!(f64, min);

define_unary_elementwise_op!(f32, sqrt);
define_unary_elementwise_op!(f32, abs);
define_unary_elementwise_op!(f32, neg);
define_unary_elementwise_op!(f64, sqrt);
define_unary_elementwise_op!(f64, abs);
define_unary_elementwise_op!(f64, neg);

define_scalar_op!(f32, ln);
define_scalar_op!(f32, exp);
define_scalar_op!(f32, tanh);
define_scalar_op!(f32, log, f32);

define_scalar_op!(f64, ln);
define_scalar_op!(f64, exp);
define_scalar_op!(f64, tanh);
define_scalar_op!(f64, log, f64);
