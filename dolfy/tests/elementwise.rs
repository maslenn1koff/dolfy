use dolfy::linalg::elementwise as ew;

macro_rules! elementwise_tests {
    ($name:ident, $ty:ty, $suffix:ident, $tolerance:expr) => {
        paste::paste! {
            #[test]
            fn $name() {
                let x: Vec<$ty> = (0..73).map(|i| (i + 1) as $ty * 0.125).collect();
                let y: Vec<$ty> = (0..73).map(|i| (i % 9 + 1) as $ty * 0.25).collect();
                let z: Vec<$ty> = (0..73).map(|i| (i as $ty - 36.0) * 0.015625).collect();
                let negative_x: Vec<$ty> = x.iter().map(|&value| -value).collect();
                let mut dst = vec![<$ty>::NAN; x.len()];

                macro_rules! assert_output {
                    ($expected:expr) => {
                        for (actual, expected) in dst.iter().zip($expected) {
                            let tolerance: $ty = $tolerance;
                            assert!(
                                (*actual - expected).abs()
                                    <= tolerance * expected.abs().max(1.0),
                                "actual={actual}, expected={expected}"
                            );
                        }
                    };
                }

                ew::[<add_ $suffix>](&x, &y, &mut dst);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a + b));
                ew::[<sub_ $suffix>](&x, &y, &mut dst);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a - b));
                ew::[<mul_ $suffix>](&x, &y, &mut dst);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a * b));
                ew::[<div_ $suffix>](&x, &y, &mut dst);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a / b));
                ew::[<max_ $suffix>](&x, &z, &mut dst);
                assert_output!(x.iter().zip(&z).map(|(&a, &b)| a.max(b)));
                ew::[<min_ $suffix>](&x, &z, &mut dst);
                assert_output!(x.iter().zip(&z).map(|(&a, &b)| a.min(b)));
                ew::[<mul_add_ $suffix>](&x, &y, &z, &mut dst);
                assert_output!(x.iter().zip(&y).zip(&z).map(|((&a, &b), &c)| a.mul_add(b, c)));

                ew::[<sqrt_ $suffix>](&x, &mut dst);
                assert_output!(x.iter().map(|value| value.sqrt()));
                ew::[<abs_ $suffix>](&negative_x, &mut dst);
                assert_output!(x.iter().copied());
                ew::[<neg_ $suffix>](&x, &mut dst);
                assert_output!(x.iter().map(|value| -*value));
                ew::[<ln_ $suffix>](&x, &mut dst);
                assert_output!(x.iter().map(|value| value.ln()));
                ew::[<exp_ $suffix>](&z, &mut dst);
                assert_output!(z.iter().map(|value| value.exp()));
                ew::[<log_ $suffix>](&x, 10.0, &mut dst);
                assert_output!(x.iter().map(|value| value.log(10.0)));

                let mut empty: [$ty; 0] = [];
                ew::[<add_ $suffix>](&[], &[], &mut empty);
                ew::[<mul_add_ $suffix>](&[], &[], &[], &mut empty);
                ew::[<sqrt_ $suffix>](&[], &mut empty);
                ew::[<ln_ $suffix>](&[], &mut empty);
                ew::[<log_ $suffix>](&[], 10.0, &mut empty);
            }

            #[cfg(feature = "parallel")]
            #[test]
            fn [<$name _parallel>]() {
                let x: Vec<$ty> = (0..73).map(|i| (i + 1) as $ty * 0.125).collect();
                let y: Vec<$ty> = (0..73).map(|i| (i % 9 + 1) as $ty * 0.25).collect();
                let z: Vec<$ty> = (0..73).map(|i| (i as $ty - 36.0) * 0.015625).collect();
                let negative_x: Vec<$ty> = x.iter().map(|&value| -value).collect();
                let mut dst = vec![<$ty>::NAN; x.len()];

                macro_rules! assert_output {
                    ($expected:expr) => {
                        for (actual, expected) in dst.iter().zip($expected) {
                            let tolerance: $ty = $tolerance;
                            assert!(
                                (*actual - expected).abs()
                                    <= tolerance * expected.abs().max(1.0),
                                "actual={actual}, expected={expected}"
                            );
                        }
                    };
                }

                ew::[<par_add_ $suffix>](&x, &y, &mut dst, 7);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a + b));
                ew::[<par_sub_ $suffix>](&x, &y, &mut dst, 7);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a - b));
                ew::[<par_mul_ $suffix>](&x, &y, &mut dst, 7);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a * b));
                ew::[<par_div_ $suffix>](&x, &y, &mut dst, 7);
                assert_output!(x.iter().zip(&y).map(|(&a, &b)| a / b));
                ew::[<par_max_ $suffix>](&x, &z, &mut dst, 7);
                assert_output!(x.iter().zip(&z).map(|(&a, &b)| a.max(b)));
                ew::[<par_min_ $suffix>](&x, &z, &mut dst, 7);
                assert_output!(x.iter().zip(&z).map(|(&a, &b)| a.min(b)));
                ew::[<par_mul_add_ $suffix>](&x, &y, &z, &mut dst, 7);
                assert_output!(x.iter().zip(&y).zip(&z).map(|((&a, &b), &c)| a.mul_add(b, c)));
                ew::[<par_sqrt_ $suffix>](&x, &mut dst, 7);
                assert_output!(x.iter().map(|value| value.sqrt()));
                ew::[<par_abs_ $suffix>](&negative_x, &mut dst, 7);
                assert_output!(x.iter().copied());
                ew::[<par_neg_ $suffix>](&x, &mut dst, 7);
                assert_output!(x.iter().map(|value| -*value));

                let mut empty: [$ty; 0] = [];
                ew::[<par_add_ $suffix>](&[], &[], &mut empty, 3);
                ew::[<par_mul_add_ $suffix>](&[], &[], &[], &mut empty, 3);
                ew::[<par_sqrt_ $suffix>](&[], &mut empty, 3);
            }
        }
    };
}

elementwise_tests!(elementwise_f32_matches_scalar_operations, f32, f32, 1e-5);
elementwise_tests!(elementwise_f64_matches_scalar_operations, f64, f64, 1e-12);

#[test]
fn elementwise_handles_lengths_around_simd_boundaries() {
    for len in [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65] {
        let x: Vec<f64> = (0..len).map(|i| i as f64 + 1.0).collect();
        let y: Vec<f64> = (0..len).map(|i| i as f64 + 2.0).collect();
        let z = vec![1.0; len];
        let mut dst = vec![f64::NAN; len];

        ew::add_f64(&x, &y, &mut dst);
        assert_eq!(dst, x.iter().zip(&y).map(|(&a, &b)| a + b).collect::<Vec<_>>());
        ew::mul_add_f64(&x, &y, &z, &mut dst);
        assert_eq!(dst, x.iter().zip(&y).map(|(&a, &b)| a.mul_add(b, 1.0)).collect::<Vec<_>>());
        ew::sqrt_f64(&x, &mut dst);
        assert_eq!(dst, x.iter().map(|value| value.sqrt()).collect::<Vec<_>>());
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_elementwise_handles_lengths_around_chunk_and_simd_boundaries() {
    for len in [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65] {
        let x: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
        let y: Vec<f32> = (0..len).map(|i| i as f32 + 2.0).collect();
        let z = vec![1.0; len];
        let mut dst = vec![f32::NAN; len];

        ew::par_add_f32(&x, &y, &mut dst, 7);
        assert_eq!(dst, x.iter().zip(&y).map(|(&a, &b)| a + b).collect::<Vec<_>>());
        ew::par_mul_add_f32(&x, &y, &z, &mut dst, 7);
        assert_eq!(dst, x.iter().zip(&y).map(|(&a, &b)| a.mul_add(b, 1.0)).collect::<Vec<_>>());
        ew::par_sqrt_f32(&x, &mut dst, 7);
        assert_eq!(dst, x.iter().map(|value| value.sqrt()).collect::<Vec<_>>());
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_elementwise_rejects_mismatched_slice_lengths() {
    let mut dst_f32 = [0.0; 2];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ew::par_add_f32(&[1.0, 2.0], &[3.0], &mut dst_f32, 1);
        }))
        .is_err()
    );
    let mut dst_f64 = [0.0; 2];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ew::par_mul_add_f64(&[1.0], &[2.0], &[3.0, 4.0], &mut dst_f64, 1);
        }))
        .is_err()
    );
}
