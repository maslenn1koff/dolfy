use dolfy::linalg::reductions;

macro_rules! axis_reduction_tests {
    ($name:ident, $ty:ty, $suffix:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _contiguous_rows>]() {
                let x: [$ty; 12] = [
                    1.0, 5.0, 3.0, 4.0,
                    -2.0, 8.0, 7.0, 0.0,
                    9.0, 6.0, 2.0, 3.0,
                ];
                let shape = [3, 4];
                let strides = [4, 1];
                let (mut sums, mut mins, mut maxs) = ([0.0; 3], [0.0; 3], [0.0; 3]);
                let (mut argmins, mut argmaxs) = ([0; 3], [0; 3]);

                reductions::[<sum_axis_ $suffix>](&x, &shape, &strides, 1, &mut sums);
                reductions::[<min_axis_ $suffix>](&x, &shape, &strides, 1, &mut mins);
                reductions::[<max_axis_ $suffix>](&x, &shape, &strides, 1, &mut maxs);
                reductions::[<argmin_axis_ $suffix>](&x, &shape, &strides, 1, &mut argmins);
                reductions::[<argmax_axis_ $suffix>](&x, &shape, &strides, 1, &mut argmaxs);

                assert_eq!(sums, [13.0, 13.0, 20.0]);
                assert_eq!(mins, [1.0, -2.0, 2.0]);
                assert_eq!(maxs, [5.0, 8.0, 9.0]);
                assert_eq!(argmins, [0, 0, 2]);
                assert_eq!(argmaxs, [1, 1, 0]);
            }

            #[test]
            fn [<$name _strided_and_non_last_axis>]() {
                let x: [$ty; 6] = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
                let shape = [2, 3];
                let strides = [1, 2];
                let (mut sums, mut mins, mut maxs) = ([0.0; 2], [0.0; 2], [0.0; 2]);
                let (mut argmins, mut argmaxs) = ([0; 2], [0; 2]);

                reductions::[<sum_axis_ $suffix>](&x, &shape, &strides, 1, &mut sums);
                reductions::[<min_axis_ $suffix>](&x, &shape, &strides, 1, &mut mins);
                reductions::[<max_axis_ $suffix>](&x, &shape, &strides, 1, &mut maxs);
                reductions::[<argmin_axis_ $suffix>](&x, &shape, &strides, 1, &mut argmins);
                reductions::[<argmax_axis_ $suffix>](&x, &shape, &strides, 1, &mut argmaxs);

                assert_eq!(sums, [6.0, 60.0]);
                assert_eq!(mins, [1.0, 10.0]);
                assert_eq!(maxs, [3.0, 30.0]);
                assert_eq!(argmins, [0, 0]);
                assert_eq!(argmaxs, [2, 2]);

                let strides = [3, 1];
                let mut column_sums = [0.0; 3];
                reductions::[<sum_axis_ $suffix>](
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    &shape,
                    &strides,
                    0,
                    &mut column_sums,
                );
                assert_eq!(column_sums, [5.0, 7.0, 9.0]);
            }

            #[test]
            fn [<$name _nan_ties_and_empty_axis>]() {
                let x = [<$ty>::NAN, 4.0, 4.0, <$ty>::NAN, <$ty>::NAN, <$ty>::NAN];
                let shape = [2, 3];
                let strides = [3, 1];
                let (mut argmaxs, mut argmins) = ([0; 2], [0; 2]);
                reductions::[<argmax_axis_ $suffix>](&x, &shape, &strides, 1, &mut argmaxs);
                reductions::[<argmin_axis_ $suffix>](&x, &shape, &strides, 1, &mut argmins);
                assert_eq!(argmaxs, [1, 0]);
                assert_eq!(argmins, [1, 0]);

                let mut empty_axis = [1.0; 3];
                reductions::[<sum_axis_ $suffix>](&[], &[0, 3], &[3, 1], 0, &mut empty_axis);
                assert_eq!(empty_axis, [0.0; 3]);
                let mut zeros = [1.0; 3];
                reductions::[<sum_axis_ $suffix>](&[], &[3, 0], &[0, 1], 1, &mut zeros);
                assert_eq!(zeros, [0.0; 3]);
            }

            #[test]
            fn [<$name _rejects_invalid_metadata>]() {
                let mut output = [0.0; 2];
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    reductions::[<sum_axis_ $suffix>](&[1.0; 4], &[2, 2], &[2], 1, &mut output);
                })).is_err());
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    reductions::[<sum_axis_ $suffix>](&[1.0; 4], &[2, 2], &[2, 1], 2, &mut output);
                })).is_err());
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    reductions::[<sum_axis_ $suffix>](&[1.0; 3], &[2, 2], &[2, 1], 1, &mut output);
                })).is_err());
            }
        }
    };
}

axis_reduction_tests!(axis_reductions_f32, f32, f32);
axis_reduction_tests!(axis_reductions_f64, f64, f64);
