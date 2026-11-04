#[cfg(feature = "parallel")]
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

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
            #[inline]
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

            #[cfg(feature = "parallel")]
            /// Parallel computes the index of the first maximum value.
            ///
            /// # Panics
            /// Panics if `chunk_size` is zero or `x` is empty.
            #[inline]
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

define_argmax!(f32);
define_argmax!(f64);
