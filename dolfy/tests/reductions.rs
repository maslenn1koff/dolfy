use dolfy::linalg::reductions::*;

macro_rules! complete_reduction_tests {
    ($name:ident, $ty:ty, $suffix:ident, $tolerance:expr) => {
        paste::paste! {
            #[test]
            fn $name() {
                let x: Vec<$ty> = (0..129).map(|i| (i % 23) as $ty - 11.0).collect();
                let y: Vec<$ty> = (0..129).map(|i| (i % 7) as $ty - 3.0).collect();
                let expected_sum = x.iter().sum::<$ty>();
                let expected_dot = x.iter().zip(&y).map(|(&a, &b)| a * b).sum::<$ty>();
                let expected_l1 = x.iter().map(|value| value.abs()).sum::<$ty>();
                let expected_l2 = x.iter().map(|value| value * value).sum::<$ty>().sqrt();

                macro_rules! assert_close {
                    ($actual:expr, $expected:expr) => {
                        let actual: $ty = $actual;
                        let expected: $ty = $expected;
                        let tolerance: $ty = $tolerance;
                        assert!(
                            (actual - expected).abs()
                                <= tolerance * expected.abs().max(1.0),
                            "actual={actual}, expected={expected}"
                        );
                    };
                }

                assert_close!([<sum_ $suffix>](&x), expected_sum);
                assert_close!([<dot_ $suffix>](&x, &y), expected_dot);
                assert_close!([<norm_l1_ $suffix>](&x), expected_l1);
                assert_close!([<norm_l2_ $suffix>](&x), expected_l2);
                assert_close!([<stable_norm_l2_ $suffix>](&x), expected_l2);
                assert_eq!([<max_ $suffix>](&x), 11.0);
                assert_eq!([<min_ $suffix>](&x), -11.0);
                assert_eq!([<argmax_ $suffix>](&x), 22);
                assert_eq!([<argmin_ $suffix>](&x), 0);

                let no_values: [$ty; 0] = [];
                assert_eq!([<sum_ $suffix>](&no_values), 0.0);
                assert_eq!([<dot_ $suffix>](&no_values, &no_values), 0.0);
                assert_eq!([<norm_l1_ $suffix>](&no_values), 0.0);
                assert_eq!([<norm_l2_ $suffix>](&no_values), 0.0);
                assert_eq!([<stable_norm_l2_ $suffix>](&no_values), 0.0);
                assert_eq!([<max_ $suffix>](&no_values), <$ty>::NEG_INFINITY);
                assert_eq!([<min_ $suffix>](&no_values), <$ty>::INFINITY);

                let with_nan = [<$ty>::NAN, 4.0, <$ty>::NAN, -3.0];
                assert_eq!([<argmax_ $suffix>](&with_nan), 1);
                assert_eq!([<argmin_ $suffix>](&with_nan), 3);
                assert_eq!([<argmax_ $suffix>](&[<$ty>::NAN, <$ty>::NAN]), 0);
                assert_eq!([<argmin_ $suffix>](&[<$ty>::NAN, <$ty>::NAN]), 0);
                assert!([<stable_norm_l2_ $suffix>](&[<$ty>::NAN, <$ty>::INFINITY]).is_nan());
                assert_eq!([<stable_norm_l2_ $suffix>](&[<$ty>::INFINITY]), <$ty>::INFINITY);
            }

            #[cfg(feature = "parallel")]
            #[test]
            fn [<$name _parallel>]() {
                let x: Vec<$ty> = (0..129).map(|i| (i % 23) as $ty - 11.0).collect();
                let y: Vec<$ty> = (0..129).map(|i| (i % 7) as $ty - 3.0).collect();
                let expected_sum = x.iter().sum::<$ty>();
                let expected_dot = x.iter().zip(&y).map(|(&a, &b)| a * b).sum::<$ty>();
                let expected_l1 = x.iter().map(|value| value.abs()).sum::<$ty>();
                let expected_l2 = x.iter().map(|value| value * value).sum::<$ty>().sqrt();
                let nan_values = [<$ty>::NAN, 4.0, <$ty>::NAN, -3.0];

                macro_rules! assert_close {
                    ($actual:expr, $expected:expr) => {
                        let actual: $ty = $actual;
                        let expected: $ty = $expected;
                        let tolerance: $ty = $tolerance;
                        assert!(
                            (actual - expected).abs()
                                <= tolerance * expected.abs().max(1.0),
                            "actual={actual}, expected={expected}"
                        );
                    };
                }

                for chunk_size in [1, 7, 16, 33, 256] {
                    assert_close!([<par_sum_ $suffix>](&x, chunk_size), expected_sum);
                    assert_close!([<par_dot_ $suffix>](&x, &y, chunk_size), expected_dot);
                    assert_close!([<par_norm_l1_ $suffix>](&x, chunk_size), expected_l1);
                    assert_close!([<par_norm_l2_ $suffix>](&x, chunk_size), expected_l2);
                    assert_close!([<par_stable_norm_l2_ $suffix>](&x, chunk_size), expected_l2);
                    assert_eq!([<par_max_ $suffix>](&x, chunk_size), 11.0);
                    assert_eq!([<par_min_ $suffix>](&x, chunk_size), -11.0);
                    assert_eq!([<par_argmax_ $suffix>](&x, chunk_size), 22);
                    assert_eq!([<par_argmin_ $suffix>](&x, chunk_size), 0);
                }

                assert_eq!([<par_argmax_ $suffix>](&nan_values, 2), 1);
                assert_eq!([<par_argmin_ $suffix>](&nan_values, 2), 3);
                let no_values: [$ty; 0] = [];
                assert_eq!([<par_sum_ $suffix>](&no_values, 1), 0.0);
                assert_eq!([<par_dot_ $suffix>](&no_values, &no_values, 1), 0.0);
                assert_eq!([<par_norm_l1_ $suffix>](&no_values, 1), 0.0);
                assert_eq!([<par_norm_l2_ $suffix>](&no_values, 1), 0.0);
                assert_eq!([<par_stable_norm_l2_ $suffix>](&no_values, 1), 0.0);
                assert_eq!([<par_max_ $suffix>](&no_values, 1), <$ty>::NEG_INFINITY);
                assert_eq!([<par_min_ $suffix>](&no_values, 1), <$ty>::INFINITY);
            }
        }
    };
}

complete_reduction_tests!(complete_f32_reduction_coverage, f32, f32, 1e-5);
complete_reduction_tests!(complete_f64_reduction_coverage, f64, f64, 1e-12);

#[test]
fn reductions_f32_handle_vector_chunks_and_tails() {
    let values: Vec<f32> = (0..100).map(|i| (i % 23 - 11) as f32).collect();

    assert_eq!(sum_f32(&values), -60.0);
    assert_eq!(max_f32(&values), 11.0);
    assert_eq!(min_f32(&values), -11.0);
    assert_eq!(argmax_f32(&values), 22);
    assert_eq!(argmin_f32(&values), 0);
}

#[test]
fn reductions_f64_preserve_first_index_on_ties() {
    let values = [5.0_f64, -2.0, 5.0, -2.0];

    assert_eq!(sum_f64(&values), 6.0);
    assert_eq!(max_f64(&values), 5.0);
    assert_eq!(min_f64(&values), -2.0);
    assert_eq!(argmax_f64(&values), 0);
    assert_eq!(argmin_f64(&values), 1);
}

#[test]
fn arg_reductions_ignore_nan_when_numeric_values_are_present() {
    let values = [f32::NAN, 4.0, f32::NAN, -3.0];

    assert_eq!(argmax_f32(&values), 1);
    assert_eq!(argmin_f32(&values), 3);
    assert_eq!(argmax_f32(&[f32::NAN, f32::NAN]), 0);
}

#[test]
fn value_reductions_have_neutral_values_for_empty_slices() {
    assert_eq!(sum_f32(&[]), 0.0);
    assert_eq!(max_f32(&[]), f32::NEG_INFINITY);
    assert_eq!(min_f64(&[]), f64::INFINITY);
}

#[test]
fn stable_l2_norm_handles_overflow_and_underflow() {
    let large_f32 = [f32::MAX / 2.0; 2];
    let small_f32 = [1.0e-30_f32; 2];
    let large_f64 = [f64::MAX / 2.0; 2];
    let small_f64 = [1.0e-200_f64; 2];

    assert!(stable_norm_l2_f32(&large_f32).is_finite());
    assert!((stable_norm_l2_f32(&small_f32) / 1.0e-30 - 2.0_f32.sqrt()).abs() < 1e-6);
    assert!(stable_norm_l2_f64(&large_f64).is_finite());
    assert!((stable_norm_l2_f64(&small_f64) / 1.0e-200 - 2.0_f64.sqrt()).abs() < 1e-14);
    assert_eq!(stable_norm_l2_f64(&[]), 0.0);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_reductions_match_sequential_results() {
    let values: Vec<f64> = (0..100).map(|i| (i % 23 - 11) as f64).collect();

    assert_eq!(par_sum_f64(&values, 17), sum_f64(&values));
    assert_eq!(par_max_f64(&values, 17), max_f64(&values));
    assert_eq!(par_min_f64(&values, 17), min_f64(&values));
    assert_eq!(par_argmax_f64(&values, 17), argmax_f64(&values));
    assert_eq!(par_argmin_f64(&values, 17), argmin_f64(&values));
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_stable_l2_norm_handles_scaling_and_non_finite_values() {
    let values = [f64::MAX / 2.0, 1.0e-200, f64::MAX / 2.0];
    assert_eq!(par_stable_norm_l2_f64(&values, 1), stable_norm_l2_f64(&values));
    assert_eq!(par_stable_norm_l2_f32(&[f32::INFINITY, 1.0], 1), f32::INFINITY);
    assert!(par_stable_norm_l2_f64(&[f64::NAN, f64::INFINITY], 1).is_nan());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_arg_reductions_ignore_nan_consistently() {
    let values = [f64::NAN, 4.0, f64::NAN, -3.0];

    assert_eq!(par_argmax_f64(&values, 2), 1);
    assert_eq!(par_argmin_f64(&values, 2), 3);
    assert_eq!(par_argmax_f64(&[f64::NAN, f64::NAN], 1), 0);
}

#[test]
fn dot_rejects_mismatched_slice_lengths() {
    assert!(std::panic::catch_unwind(|| dot_f32(&[1.0, 2.0], &[3.0])).is_err());
}

#[test]
fn l1_norm_handles_partial_simd_vector_groups() {
    for len in 0..65 {
        let values: Vec<f64> = (0..len).map(|i| (i as f64 - 17.0) * 0.5).collect();
        let expected = values.iter().map(|value| value.abs()).sum::<f64>();
        assert_eq!(norm_l1_f64(&values), expected);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_dot_rejects_mismatched_slice_lengths() {
    assert!(std::panic::catch_unwind(|| par_dot_f64(&[1.0, 2.0], &[3.0], 1)).is_err());
}
