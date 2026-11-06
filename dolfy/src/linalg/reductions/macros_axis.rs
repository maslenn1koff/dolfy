macro_rules! define_axis_value_reduction {
    ($ty:ident, $name:ident, $identity:expr, $combine:expr) => {
        paste::paste! {
            /// Reduces `x` along `axis` and writes the result in row-major output order.
            ///
            /// `shape` and `strides` describe the logical input layout.
            ///
            /// # Panics
            /// Panics if the axis, shape, strides, input length, or output length is invalid.
            pub fn [<$name _axis_ $ty>](
                x: &[$ty],
                shape: &[usize],
                strides: &[usize],
                axis: usize,
                output: &mut [$ty],
            ) {
                let (axis_len, axis_stride) =
                    validate_axis_metadata(x.len(), shape, strides, axis, output.len());
                let combine = $combine;

                for (output_index, result) in output.iter_mut().enumerate() {
                    let base = axis_output_offset(output_index, shape, strides, axis);
                    let mut accumulator = $identity;
                    for index in 0..axis_len {
                        accumulator =
                            combine(accumulator, x[base + index * axis_stride]);
                    }
                    *result = accumulator;
                }
            }
        }
    };
}

macro_rules! define_axis_arg_reduction {
    ($ty:ident, $name:ident, $comparison:tt) => {
        paste::paste! {
            /// Returns the first index of the selected value along `axis`.
            ///
            /// NaN values are ignored if the axis contains a numeric value; if all values
            /// are NaN, returns zero.
            ///
            /// # Panics
            /// Panics if the axis, shape, strides, input length, or output length is invalid.
            pub fn [<$name _axis_ $ty>](
                x: &[$ty],
                shape: &[usize],
                strides: &[usize],
                axis: usize,
                output: &mut [usize],
            ) {
                let (axis_len, axis_stride) =
                    validate_axis_metadata(x.len(), shape, strides, axis, output.len());

                for (output_index, result) in output.iter_mut().enumerate() {
                    let base = axis_output_offset(output_index, shape, strides, axis);
                    let mut best_index = 0;
                    if axis_len > 0 {
                        let mut best_value = x[base];
                        for index in 1..axis_len {
                            let value = x[base + index * axis_stride];
                            if best_value.is_nan() {
                                if !value.is_nan() {
                                    best_index = index;
                                    best_value = value;
                                }
                            } else if !value.is_nan() && value $comparison best_value {
                                best_index = index;
                                best_value = value;
                            }
                        }
                    }
                    *result = best_index;
                }
            }
        }
    };
}

pub(crate) use define_axis_arg_reduction;
pub(crate) use define_axis_value_reduction;
