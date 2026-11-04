#![allow(clippy::too_many_arguments)]

mod define;
use define::define_matmul;

pub const DEFAULT_MC: usize = 96;
pub const DEFAULT_NC: usize = 1024;
pub const DEFAULT_KC: usize = 256;
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 2097152;

pub(crate) const MR: usize = 6;
pub(crate) const NV: usize = 2;

#[cfg(feature = "parallel")]
pub(crate) const MIN_WORK_PER_THREAD: usize = 524288;

#[cfg(feature = "parallel")]
const MIN_PARALLEL_ROWS: usize = 4 * MR;

#[cfg(feature = "parallel")]
#[inline]
fn use_parallel(m: usize, n: usize, k: usize, threshold: usize) -> bool {
    let work = m.saturating_mul(n).saturating_mul(k);
    if m < MIN_PARALLEL_ROWS || work < threshold {
        return false;
    }
    let row_blocks = m.div_ceil(MR);
    let threads = (work / MIN_WORK_PER_THREAD).min(rayon::current_num_threads());
    row_blocks >= 2 && threads >= 2
}

define_matmul!(f32, imp_f32);
define_matmul!(f64, imp_f64);

use imp_f32::{
    configure_matmul as im_configure_matmul_f32, configure_matmul_serial as im_configure_matmul_serial_f32, matmul as im_matmul_f32, matmul_serial as im_matmul_serial_f32,
};
#[cfg(feature = "parallel")]
use imp_f32::{configure_matmul_parallel as im_configure_matmul_parallel_f32, matmul_parallel as im_matmul_parallel_f32};

use imp_f64::{
    configure_matmul as im_configure_matmul_f64, configure_matmul_serial as im_configure_matmul_serial_f64, matmul as im_matmul_f64, matmul_serial as im_matmul_serial_f64,
};
#[cfg(feature = "parallel")]
use imp_f64::{configure_matmul_parallel as im_configure_matmul_parallel_f64, matmul_parallel as im_matmul_parallel_f64};

macro_rules! define_fn {
    ($ty:ident) => {
        paste::paste! {
            pub fn [<matmul _ $ty>](
            m: usize,
            n: usize,
            k: usize,
            alpha: $ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: $ty,
            c: &mut [$ty],
            ldc: usize,
        ) {
            [<im_matmul _ $ty>](m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        pub fn [<configure_matmul _ $ty>](
            m: usize,
            n: usize,
            k: usize,
            alpha: $ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: $ty,
            c: &mut [$ty],
            ldc: usize,
            mc: usize,
            nc: usize,
            kc: usize,
            parallel_threshold: usize,
        ) {
            [<im_configure_matmul _ $ty>](
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
                mc,
                nc,
                kc,
                parallel_threshold,
            );
        }

        pub fn [<matmul_serial _ $ty>](
            m: usize,
            n: usize,
            k: usize,
            alpha: $ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: $ty,
            c: &mut [$ty],
            ldc: usize,
        ) {
            [<im_matmul_serial _ $ty>](m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        pub fn [<configure_matmul_serial _ $ty>](
            m: usize,
            n: usize,
            k: usize,
            alpha: $ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: $ty,
            c: &mut [$ty],
            ldc: usize,
            mc: usize,
            nc: usize,
            kc: usize,
        ) {
            [<im_configure_matmul_serial _ $ty>](m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mc, nc, kc);
        }

        #[cfg(feature = "parallel")]
        pub fn [<matmul_parallel _ $ty>](
            m: usize,
            n: usize,
            k: usize,
            alpha: $ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: $ty,
            c: &mut [$ty],
            ldc: usize,
        ) {
            [<im_matmul_parallel _ $ty>](m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        #[cfg(feature = "parallel")]
        pub fn [<configure_matmul_parallel _ $ty>](
            m: usize,
            n: usize,
            k: usize,
            alpha: $ty,
            a: &[$ty],
            lda: usize,
            b: &[$ty],
            ldb: usize,
            beta: $ty,
            c: &mut [$ty],
            ldc: usize,
            mc: usize,
            nc: usize,
            kc: usize,
        ) {
            [<im_configure_matmul_parallel _ $ty>](m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mc, nc, kc);
        }
        }
    };
}

define_fn!(f32);
define_fn!(f64);
