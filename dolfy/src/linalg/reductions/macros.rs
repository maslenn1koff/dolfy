macro_rules! define_unary_reduction {
    ($ty:ident, $name:ident, $op:ident, $splat:expr, $reduce:ident) => {
        paste::paste! {
            /// Reduction operation.
            #[inline(always)]
            pub fn [<$name _ $ty>](a: &[$ty]) -> $ty {
                struct Impl<'a>(&'a [$ty]);

                impl SimdFn for Impl<'_> {
                    type Result = $ty;

                    #[inline(always)]
                    fn call<S: Simd>(self, simd: S) -> Self::Result {
                        let (mut head, tail) = utils::[<as_simd_ $ty s>]::<S>(self.0);
                        let mut acc = [simd.[<splat_ $ty s>]($splat); 8];

                        while head.len() >= 8 {
                            let x0 = head.next().unwrap();
                            let x1 = head.next().unwrap();
                            let x2 = head.next().unwrap();
                            let x3 = head.next().unwrap();
                            let x4 = head.next().unwrap();
                            let x5 = head.next().unwrap();
                            let x6 = head.next().unwrap();
                            let x7 = head.next().unwrap();

                            acc[0] = simd.[<$op _ $ty s>](acc[0], x0);
                            acc[1] = simd.[<$op _ $ty s>](acc[1], x1);
                            acc[2] = simd.[<$op _ $ty s>](acc[2], x2);
                            acc[3] = simd.[<$op _ $ty s>](acc[3], x3);
                            acc[4] = simd.[<$op _ $ty s>](acc[4], x4);
                            acc[5] = simd.[<$op _ $ty s>](acc[5], x5);
                            acc[6] = simd.[<$op _ $ty s>](acc[6], x6);
                            acc[7] = simd.[<$op _ $ty s>](acc[7], x7);
                        }

                        for vector in head {
                            acc[0] = simd.[<$op _ $ty s>](acc[0], vector);
                        }

                        acc[0] = simd.[<$op _ $ty s>](acc[0], acc[1]);
                        acc[1] = simd.[<$op _ $ty s>](acc[2], acc[3]);
                        acc[2] = simd.[<$op _ $ty s>](acc[4], acc[5]);
                        acc[3] = simd.[<$op _ $ty s>](acc[6], acc[7]);

                        acc[0] = simd.[<$op _ $ty s>](acc[0], acc[1]);
                        acc[1] = simd.[<$op _ $ty s>](acc[2], acc[3]);
                        acc[0] = simd.[<$op _ $ty s>](acc[0], acc[1]);
                        let mut result = simd.[<$reduce _ $ty s>](acc[0]);

                        for value in tail {
                            result = result.[<$op>](*value);
                        }

                        result
                    }
                }

                dispatch(Impl(a))
            }

            /// Parallel reduction operation.
            ///
            /// # Panics
            /// Panics if `chunk_size` is zero.
            #[cfg(feature = "parallel")]
            #[inline(always)]
            pub fn [<par_ $name _ $ty>](a: &[$ty], chunk_size: usize) -> $ty {
                assert!(chunk_size > 0);

                a.par_chunks(chunk_size)
                    .map(|chunk| [<$name _ $ty>](chunk))
                    .reduce(|| $splat, |x, y| x.[<$op>](y))
            }
        }
    };
}

macro_rules! define_dot {
    ($ty:ty) => {
        paste::paste! {
            /// Dot product of two slices of values.
            ///
            /// # Panics
            /// Panics if the lengths of `x` and `y` are not equal.
            #[inline(always)]
            pub fn [<dot_ $ty>](x: &[$ty], y: &[$ty]) -> $ty {
                assert_eq!(x.len(), y.len());

                struct Impl<'a>(&'a [$ty], &'a [$ty]);

                impl SimdFn for Impl<'_> {
                    type Result = $ty;

                    #[inline(always)]
                    fn call<S: reviero::traits::Simd>(self, simd: S) -> Self::Result {
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

                dispatch(Impl(x, y))
            }

            /// Parallel dot product of two slices of values.
            ///
            /// # Panics
            /// Panics if `chunk_size` is zero or the lengths of `x` and `y` are not equal.
            #[cfg(feature = "parallel")]
            #[inline(always)]
            pub fn [<par_dot_ $ty>](x: &[$ty], y: &[$ty], chunk_size: usize) -> $ty {
                assert_eq!(x.len(), y.len());
                assert!(chunk_size > 0);

                x.par_chunks(chunk_size)
                    .zip(y.par_chunks(chunk_size))
                    .map(|(x_chunk, y_chunk)| [<dot_ $ty>](x_chunk, y_chunk))
                    .sum()
            }
        }
    };
}

macro_rules! define_norm_l1 {
    ($ty:ident) => {
        paste::paste! {
            /// Computes the L1 norm of a slice of values.
            #[inline(always)]
            pub fn [<norm_l1_ $ty>](a: &[$ty]) -> $ty {
                struct Impl<'a>(&'a [$ty]);

                impl <'a> SimdFn for Impl<'a> {
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
                        };

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

                dispatch(Impl(a))
            }

            #[cfg(feature = "parallel")]
            #[inline(always)]
            /// Parallel computes the L1 norm of a slice of values.
            pub fn [<par_ norm_l1_ $ty>](a: &[$ty], chunk_size: usize) -> $ty {
                assert!(chunk_size > 0);
                a.par_chunks(chunk_size).map([<norm_l1_ $ty>]).sum()
            }
        }
    };
}

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

macro_rules! define_argmax {
    ($ty:ident) => {
        paste::paste! {
            /// Returns the index of the first maximum value in a non-empty slice.
            ///
            /// NaN values are ignored when the slice contains a numeric value; if all values
            /// are NaN, returns the index of the first value.
            ///
            /// # Panics
            /// Panics if `x` is empty.
            #[inline(always)]
            pub fn [<argmax_ $ty>](x: &[$ty]) -> usize {
                assert!(!x.is_empty(), "argmax requires a non-empty slice");

                let mut best_index = 0;
                let mut best_value = x[0];
                for (index, &value) in x.iter().enumerate().skip(1) {
                    if best_value.is_nan() {
                        if !value.is_nan() {
                            best_index = index;
                            best_value = value;
                        }
                    } else if !value.is_nan() && value > best_value {
                        best_index = index;
                        best_value = value;
                    }
                }
                best_index
            }

            /// Parallel computes the index of the first maximum value.
            ///
            /// # Panics
            /// Panics if `chunk_size` is zero or `x` is empty.
            #[cfg(feature = "parallel")]
            #[inline(always)]
            pub fn [<par_argmax_ $ty>](x: &[$ty], chunk_size: usize) -> usize {
                assert!(chunk_size > 0);
                assert!(!x.is_empty(), "argmax requires a non-empty slice");

                x.par_chunks(chunk_size)
                    .enumerate()
                    .map(|(chunk_index, chunk)| {
                        let index = [<argmax_ $ty>](chunk);
                        (chunk_index * chunk_size + index, chunk[index])
                    })
                    .reduce_with(|left, right| {
                        let (left_index, left_value) = left;
                        let (right_index, right_value) = right;
                        if left_value.is_nan() {
                            if right_value.is_nan() && left_index < right_index {
                                left
                            } else {
                                (right_index, right_value)
                            }
                        } else if right_value.is_nan() {
                            (left_index, left_value)
                        } else if left_value > right_value
                            || (left_value == right_value && left_index < right_index)
                        {
                            (left_index, left_value)
                        } else {
                            (right_index, right_value)
                        }
                    })
                    .unwrap()
                    .0
            }
        }
    };
}

macro_rules! define_argmin {
    ($ty:ident) => {
        paste::paste! {
            /// Returns the index of the first minimum value in a non-empty slice.
            ///
            /// NaN values are ignored when the slice contains a numeric value; if all values
            /// are NaN, returns the index of the first value.
            ///
            /// # Panics
            /// Panics if `x` is empty.
            #[inline(always)]
            pub fn [<argmin_ $ty>](x: &[$ty]) -> usize {
                assert!(!x.is_empty(), "argmin requires a non-empty slice");

                let mut best_index = 0;
                let mut best_value = x[0];
                for (index, &value) in x.iter().enumerate().skip(1) {
                    if best_value.is_nan() {
                        if !value.is_nan() {
                            best_index = index;
                            best_value = value;
                        }
                    } else if !value.is_nan() && value < best_value {
                        best_index = index;
                        best_value = value;
                    }
                }
                best_index
            }

            /// Parallel computes the index of the first minimum value.
            ///
            /// # Panics
            /// Panics if `chunk_size` is zero or `x` is empty.
            #[cfg(feature = "parallel")]
            #[inline(always)]
            pub fn [<par_argmin_ $ty>](x: &[$ty], chunk_size: usize) -> usize {
                assert!(chunk_size > 0);
                assert!(!x.is_empty(), "argmin requires a non-empty slice");

                x.par_chunks(chunk_size)
                    .enumerate()
                    .map(|(chunk_index, chunk)| {
                        let index = [<argmin_ $ty>](chunk);
                        (chunk_index * chunk_size + index, chunk[index])
                    })
                    .reduce_with(|left, right| {
                        let (left_index, left_value) = left;
                        let (right_index, right_value) = right;
                        if left_value.is_nan() {
                            if right_value.is_nan() && left_index < right_index {
                                left
                            } else {
                                (right_index, right_value)
                            }
                        } else if right_value.is_nan() {
                            (left_index, left_value)
                        } else if left_value < right_value
                            || (left_value == right_value && left_index < right_index)
                        {
                            (left_index, left_value)
                        } else {
                            (right_index, right_value)
                        }
                    })
                    .unwrap()
                    .0
            }
        }
    };
}

pub(crate) use define_argmax;
pub(crate) use define_argmin;
pub(crate) use define_dot;
pub(crate) use define_norm_l1;
pub(crate) use define_stable_norm_l2;
pub(crate) use define_unary_reduction;
