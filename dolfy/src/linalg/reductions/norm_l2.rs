use crate::linalg::reductions::dot;
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

macro_rules! define_stable_norm_l2 {
    ($ty:ident) => {
        paste::paste! {
            #[derive(Clone, Copy)]
            struct [<ScaledSquares $ty>] {
                scale: $ty,
                sum_squares: $ty,
                has_nan: bool,
                has_infinity: bool,
            }

            impl [<ScaledSquares $ty>] {
                fn new() -> Self {
                    Self {
                        scale: 0.0,
                        sum_squares: 0.0,
                        has_nan: false,
                        has_infinity: false,
                    }
                }

                fn add(&mut self, value: $ty) {
                    let value = value.abs();
                    if value.is_nan() {
                        self.has_nan = true;
                    } else if value.is_infinite() {
                        self.has_infinity = true;
                    } else if value != 0.0 {
                        if self.scale < value {
                            let ratio = self.scale / value;
                            self.sum_squares = 1.0 + self.sum_squares * ratio * ratio;
                            self.scale = value;
                        } else {
                            let ratio = value / self.scale;
                            self.sum_squares += ratio * ratio;
                        }
                    }
                }

                #[cfg(feature = "parallel")]
                fn combine(self, other: Self) -> Self {
                    let (scale, sum_squares) = if self.scale == 0.0 {
                        (other.scale, other.sum_squares)
                    } else if other.scale == 0.0 {
                        (self.scale, self.sum_squares)
                    } else if self.scale < other.scale {
                        let ratio = self.scale / other.scale;
                        (other.scale, other.sum_squares + self.sum_squares * ratio * ratio)
                    } else {
                        let ratio = other.scale / self.scale;
                        (self.scale, self.sum_squares + other.sum_squares * ratio * ratio)
                    };

                    Self {
                        scale,
                        sum_squares,
                        has_nan: self.has_nan || other.has_nan,
                        has_infinity: self.has_infinity || other.has_infinity,
                    }
                }

                fn norm(self) -> $ty {
                    if self.has_nan {
                        <$ty>::NAN
                    } else if self.has_infinity {
                        <$ty>::INFINITY
                    } else {
                        self.scale * self.sum_squares.sqrt()
                    }
                }
            }

            /// Computes a numerically stable L2 norm using scaled sum-of-squares.
            pub fn [<stable_norm_l2_ $ty>](x: &[$ty]) -> $ty {
                let mut squares = [<ScaledSquares $ty>]::new();
                for &value in x {
                    squares.add(value);
                }
                squares.norm()
            }

            #[cfg(feature = "parallel")]
            /// Parallel computes a numerically stable L2 norm using scaled sum-of-squares.
            ///
            /// # Panics
            /// Panics if `chunk_size` is zero.
            pub fn [<par_stable_norm_l2_ $ty>](x: &[$ty], chunk_size: usize) -> $ty {
                assert!(chunk_size > 0);
                x.par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut squares = [<ScaledSquares $ty>]::new();
                        for &value in chunk {
                            squares.add(value);
                        }
                        squares
                    })
                    .reduce([<ScaledSquares $ty>]::new, |left, right| left.combine(right))
                    .norm()
            }
        }
    };
}

define_stable_norm_l2!(f32);
define_stable_norm_l2!(f64);

// Computes the L2 norm of a slice of `f32` values.
pub fn norm_l2_f32(x: &[f32]) -> f32 {
    let product = dot::dot_f32(x, x);
    product.sqrt()
}

// Computes the L2 norm of a slice of `f64` values.
pub fn norm_l2_f64(x: &[f64]) -> f64 {
    let product = dot::dot_f64(x, x);
    product.sqrt()
}

// Parallel computes the L2 norm of a slice of `f32` values.
#[cfg(feature = "parallel")]
pub fn par_norm_l2_f32(x: &[f32], chunk_size: usize) -> f32 {
    dot::par_dot_f32(x, x, chunk_size).sqrt()
}

// Parallel computes the L2 norm of a slice of `f64` values.
#[cfg(feature = "parallel")]
pub fn par_norm_l2_f64(x: &[f64], chunk_size: usize) -> f64 {
    dot::par_dot_f64(x, x, chunk_size).sqrt()
}
