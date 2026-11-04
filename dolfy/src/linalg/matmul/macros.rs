#[macro_export]
macro_rules! define_matmul {
    ($ty:ident, $module:ident) => {
        paste::paste! {
            mod $module {
                use super::{DEFAULT_KC, DEFAULT_MC, DEFAULT_NC, DEFAULT_PARALLEL_THRESHOLD, MR, NV};
                #[cfg(feature = "parallel")]
                use super::{MIN_WORK_PER_THREAD, use_parallel};

                use reviero::{SimdFn, dispatch, traits::Simd};
                use std::cell::RefCell;
                use std::thread::LocalKey;

                #[cfg(feature = "parallel")]
                use rayon::prelude::*;

                const DIRECT_MAX_A_ELEMS: usize = 1 << 17;
                const DIRECT_MAX_ROWS: usize = 192;
                const DIRECT_MAX_TAIL: usize = 1 << 16;
                const DIRECT_SERIAL_B_TRAFFIC: usize = 1 << 20;
                #[cfg(feature = "parallel")]
                const DIRECT_PARALLEL_B_TRAFFIC: usize = 1 << 21;
                const PACK_PB: usize = 64;

                #[inline(always)]
                pub(crate) fn prefer_direct_serial(m: usize, n: usize, k: usize, nr: usize) -> bool {
                    m <= DIRECT_MAX_ROWS && m.saturating_mul(k) <= DIRECT_MAX_A_ELEMS && m.div_ceil(MR).saturating_mul(k).saturating_mul(n) <= DIRECT_SERIAL_B_TRAFFIC && k.saturating_mul(nr) <= DIRECT_MAX_TAIL
                }

                #[cfg(feature = "parallel")]
                #[inline(always)]
                pub(crate) fn prefer_direct_parallel(m: usize, n: usize, k: usize, nr: usize, tasks: usize) -> bool {
                    m <= DIRECT_MAX_ROWS && m.saturating_mul(k) <= DIRECT_MAX_A_ELEMS && tasks.saturating_mul(k).saturating_mul(n) <= DIRECT_PARALLEL_B_TRAFFIC && k.saturating_mul(nr) <= DIRECT_MAX_TAIL
                }

                thread_local! {
                    static A_BUF: RefCell<Vec<$ty>> = const { RefCell::new(Vec::new()) };
                    static B_BUF: RefCell<Vec<$ty>> = const { RefCell::new(Vec::new()) };
                    static T_BUF: RefCell<Vec<$ty>> = const { RefCell::new(Vec::new()) };
                    static P_BUF: RefCell<Vec<$ty>> = const { RefCell::new(Vec::new()) };
                }

                #[inline(always)]
                fn with_buf<R>(key: &'static LocalKey<RefCell<Vec<$ty>>>, len: usize, f: impl FnOnce(&mut [$ty]) -> R) -> R {
                    key.with(|cell| match cell.try_borrow_mut() {
                        Ok(mut v) => {
                            if v.len() < len {
                                v.resize(len, 0.0);
                            } else if v.len() > len.saturating_mul(4).max(1 << 20) {
                                v.truncate(len);
                                v.shrink_to_fit();
                            }
                            f(&mut v[..len])
                        }
                        Err(_) => {
                            let mut tmp = vec![0.0; len];
                            f(tmp.as_mut_slice())
                        }
                    })
                }

                #[inline(always)]
                fn validate(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, c: &[$ty], ldc: usize, mc: usize, nc: usize, kc: usize) {
                    if m == 0 || n == 0 {
                        return;
                    }

                    assert!(ldc >= n);
                    let c_needed = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n)).expect("C shape overflow");
                    assert!(c.len() >= c_needed);

                    if k == 0 || alpha == 0.0 {
                        return;
                    }

                    assert!(mc > 0 && nc > 0 && kc > 0);
                    assert!(lda >= k);
                    assert!(ldb >= n);

                    let a_needed = (m - 1).checked_mul(lda).and_then(|v| v.checked_add(k)).expect("A shape overflow");
                    let b_needed = (k - 1).checked_mul(ldb).and_then(|v| v.checked_add(n)).expect("B shape overflow");
                    assert!(a.len() >= a_needed);
                    assert!(b.len() >= b_needed);
                }

                #[inline(always)]
                fn scale_c(m: usize, n: usize, beta: $ty, c: &mut [$ty], ldc: usize) {
                    if beta == 1.0 {
                        return;
                    }

                    for i in 0..m {
                        let row = &mut c[i * ldc..i * ldc + n];
                        if beta == 0.0 {
                            row.fill(0.0);
                        } else {
                            for v in row {
                                *v *= beta;
                            }
                        }
                    }
                }

                struct Lanes;

                impl SimdFn for Lanes {
                    type Result = usize;

                    #[inline(always)]
                    fn call<S: Simd>(self, _simd: S) -> usize {
                        S::[<$ty s_lanes>]
                    }
                }

                #[inline(always)]
                fn pack_a(kc: usize, mc: usize, a: &[$ty], lda: usize, alpha: $ty, a_pack: &mut [$ty]) {
                    let unit_alpha = alpha == 1.0;

                    for (panel_idx, panel) in a_pack.chunks_exact_mut(MR * kc).take(mc.div_ceil(MR)).enumerate() {
                        let ir = panel_idx * MR;
                        let rows = MR.min(mc - ir);

                        for p0 in (0..kc).step_by(PACK_PB) {
                            let pb = PACK_PB.min(kc - p0);
                            let dst = &mut panel[p0 * MR..(p0 + pb) * MR];

                            for i in 0..rows {
                                let base = (ir + i) * lda + p0;
                                let src = &a[base..base + pb];
                                if unit_alpha {
                                    for (t, &value) in src.iter().enumerate() {
                                        dst[t * MR + i] = value;
                                    }
                                } else {
                                    for (t, &value) in src.iter().enumerate() {
                                        dst[t * MR + i] = alpha * value;
                                    }
                                }
                            }

                            for i in rows..MR {
                                for t in 0..pb {
                                    dst[t * MR + i] = 0.0;
                                }
                            }
                        }
                    }
                }

                #[inline(always)]
                fn pack_b_panel(nr: usize, cols: usize, b: &[$ty], ldb: usize, panel: &mut [$ty]) {
                    for (p, dst) in panel.chunks_exact_mut(nr).enumerate() {
                        let src = &b[p * ldb..p * ldb + cols];
                        if cols == nr {
                            dst.copy_from_slice(src);
                        } else {
                            dst[..cols].copy_from_slice(src);
                            dst[cols..].fill(0.0);
                        }
                    }
                }

                #[inline(always)]
                fn pack_b(nr: usize, kc: usize, nc: usize, b: &[$ty], ldb: usize, b_pack: &mut [$ty]) {
                    for (panel_idx, panel) in b_pack.chunks_exact_mut(nr * kc).take(nc.div_ceil(nr)).enumerate() {
                        let jr = panel_idx * nr;
                        pack_b_panel(nr, nr.min(nc - jr), &b[jr..], ldb, panel);
                    }
                }

                #[inline(always)]
                fn macro_kernel<S: Simd>(simd: S, nr: usize, mc: usize, nc: usize, kc: usize, a_pack: &[$ty], b_pack: &[$ty], c: &mut [$ty], ldc: usize, beta: $ty, c_tile: &mut [$ty]) {
                    let a_panel_size = kc * MR;
                    let b_panel_size = kc * nr;

                    for (jp, jr) in (0..nc).step_by(nr).enumerate() {
                        let nr_eff = nr.min(nc - jr);
                        let b_panel = &b_pack[jp * b_panel_size..(jp + 1) * b_panel_size];

                        for (ip, ir) in (0..mc).step_by(MR).enumerate() {
                            let mr_eff = MR.min(mc - ir);
                            let a_panel = &a_pack[ip * a_panel_size..(ip + 1) * a_panel_size];

                            micro_kernel(simd, nr, kc, a_panel, b_panel, &mut c[ir * ldc + jr..], ldc, mr_eff, nr_eff, beta, c_tile);
                        }
                    }
                }

                #[allow(clippy::needless_range_loop, clippy::chunks_exact_to_as_chunks)]
                #[inline(always)]
                fn store_direct<S: Simd>(simd: S, c: &mut [$ty], ldc: usize, acc: &[[S::[<$ty s>]; NV]; MR], rows: usize, beta: $ty, scale: $ty) {
                    let lanes = S::[<$ty s_lanes>];
                    if scale == 1.0 {
                        if beta == 0.0 {
                            for i in 0..rows {
                                for v in 0..NV {
                                    let off = i * ldc + v * lanes;
                                    simd.[<store_ $ty s>](&mut c[off..off + lanes], acc[i][v]);
                                }
                            }
                        } else if beta == 1.0 {
                            for i in 0..rows {
                                for v in 0..NV {
                                    let off = i * ldc + v * lanes;
                                    let dst = &mut c[off..off + lanes];
                                    let cv = simd.[<load_ $ty s>](dst);
                                    simd.[<store_ $ty s>](dst, simd.[<add_ $ty s>](cv, acc[i][v]));
                                }
                            }
                        } else {
                            let beta_v = simd.[<splat_ $ty s>](beta);
                            for i in 0..rows {
                                for v in 0..NV {
                                    let off = i * ldc + v * lanes;
                                    let dst = &mut c[off..off + lanes];
                                    let cv = simd.[<load_ $ty s>](dst);
                                    simd.[<store_ $ty s>](dst, simd.[<mul_add_ $ty s>](cv, beta_v, acc[i][v]));
                                }
                            }
                        }
                        return;
                    }

                    let scale_v = simd.[<splat_ $ty s>](scale);
                    if beta == 0.0 {
                        for i in 0..rows {
                            for v in 0..NV {
                                let off = i * ldc + v * lanes;
                                simd.[<store_ $ty s>](&mut c[off..off + lanes], simd.[<mul_ $ty s>](scale_v, acc[i][v]));
                            }
                        }
                    } else if beta == 1.0 {
                        for i in 0..rows {
                            for v in 0..NV {
                                let off = i * ldc + v * lanes;
                                let dst = &mut c[off..off + lanes];
                                let cv = simd.[<load_ $ty s>](dst);
                                simd.[<store_ $ty s>](dst, simd.[<mul_add_ $ty s>](scale_v, acc[i][v], cv));
                            }
                        }
                    } else {
                        let beta_v = simd.[<splat_ $ty s>](beta);
                        for i in 0..rows {
                            for v in 0..NV {
                                let off = i * ldc + v * lanes;
                                let dst = &mut c[off..off + lanes];
                                let cv = simd.[<load_ $ty s>](dst);
                                let bv = simd.[<mul_ $ty s>](beta_v, cv);
                                simd.[<store_ $ty s>](dst, simd.[<mul_add_ $ty s>](scale_v, acc[i][v], bv));
                            }
                        }
                    }
                }

                #[allow(clippy::needless_range_loop)]
                #[inline(always)]
                fn store_partial<S: Simd>(simd: S, c: &mut [$ty], ldc: usize, c_tile: &mut [$ty], nr: usize, acc: &[[S::[<$ty s>]; NV]; MR], rows: usize, nr_eff: usize, beta: $ty, scale: $ty) {
                    let lanes = S::[<$ty s_lanes>];

                    if scale == 1.0 {
                        for i in 0..rows {
                            for v in 0..NV {
                                let off = i * nr + v * lanes;
                                simd.[<store_ $ty s>](&mut c_tile[off..off + lanes], acc[i][v]);
                            }
                        }
                    } else {
                        let scale_v = simd.[<splat_ $ty s>](scale);
                        for i in 0..rows {
                            for v in 0..NV {
                                let off = i * nr + v * lanes;
                                simd.[<store_ $ty s>](&mut c_tile[off..off + lanes], simd.[<mul_ $ty s>](scale_v, acc[i][v]));
                            }
                        }
                    }

                    if beta == 0.0 {
                        for i in 0..rows {
                            let src = &c_tile[i * nr..i * nr + nr_eff];
                            c[i * ldc..i * ldc + nr_eff].copy_from_slice(src);
                        }
                    } else if beta == 1.0 {
                        for i in 0..rows {
                            let src = &c_tile[i * nr..i * nr + nr_eff];
                            let dst = &mut c[i * ldc..i * ldc + nr_eff];
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d += *s;
                            }
                        }
                    } else {
                        for i in 0..rows {
                            let src = &c_tile[i * nr..i * nr + nr_eff];
                            let dst = &mut c[i * ldc..i * ldc + nr_eff];
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d = beta * *d + *s;
                            }
                        }
                    }
                }

                #[allow(clippy::needless_range_loop, clippy::chunks_exact_to_as_chunks)]
                #[inline(always)]
                fn micro_kernel<S: Simd>(simd: S, nr: usize, kc: usize, a: &[$ty], b: &[$ty], c: &mut [$ty], ldc: usize, mr_eff: usize, nr_eff: usize, beta: $ty, c_tile: &mut [$ty]) {
                    let lanes = S::[<$ty s_lanes>];

                    if nr_eff == nr {
                        let mut acc = [[simd.[<splat_ $ty s>](0.0); NV]; MR];

                        if mr_eff == MR {
                            for (a_col, b_row) in a.chunks_exact(MR).zip(b.chunks_exact(nr)).take(kc) {
                                let mut bv = [simd.[<splat_ $ty s>](0.0); NV];
                                for v in 0..NV {
                                    bv[v] = simd.[<load_ $ty s>](&b_row[v * lanes..(v + 1) * lanes]);
                                }

                                for i in 0..MR {
                                    let av = simd.[<splat_ $ty s>](a_col[i]);
                                    for v in 0..NV {
                                        acc[i][v] = simd.[<mul_add_ $ty s>](av, bv[v], acc[i][v]);
                                    }
                                }
                            }
                        } else {
                            for (a_col, b_row) in a.chunks_exact(MR).zip(b.chunks_exact(nr)).take(kc) {
                                let mut bv = [simd.[<splat_ $ty s>](0.0); NV];
                                for v in 0..NV {
                                    bv[v] = simd.[<load_ $ty s>](&b_row[v * lanes..(v + 1) * lanes]);
                                }

                                for i in 0..mr_eff {
                                    let av = simd.[<splat_ $ty s>](a_col[i]);
                                    for v in 0..NV {
                                        acc[i][v] = simd.[<mul_add_ $ty s>](av, bv[v], acc[i][v]);
                                    }
                                }
                            }
                        }

                        store_direct(simd, c, ldc, &acc, mr_eff, beta, 1.0);
                        return;
                    }

                    let mut acc = [[simd.[<splat_ $ty s>](0.0); NV]; MR];

                    for (a_col, b_row) in a.chunks_exact(MR).zip(b.chunks_exact(nr)).take(kc) {
                        let mut bv = [simd.[<splat_ $ty s>](0.0); NV];
                        for v in 0..NV {
                            bv[v] = simd.[<load_ $ty s>](&b_row[v * lanes..(v + 1) * lanes]);
                        }

                        for i in 0..MR {
                            let av = simd.[<splat_ $ty s>](a_col[i]);
                            for v in 0..NV {
                                acc[i][v] = simd.[<mul_add_ $ty s>](av, bv[v], acc[i][v]);
                            }
                        }
                    }

                    store_partial(simd, c, ldc, c_tile, nr, &acc, mr_eff, nr_eff, beta, 1.0);
                }

                #[allow(clippy::needless_range_loop)]
                #[inline(always)]
                fn direct_micro_kernel<S: Simd>(
                    simd: S,
                    k: usize,
                    alpha: $ty,
                    a: &[$ty],
                    lda: usize,
                    b: &[$ty],
                    ldb: usize,
                    c: &mut [$ty],
                    ldc: usize,
                    mr_eff: usize,
                    nr_eff: usize,
                    beta: $ty,
                    c_tile: &mut [$ty],
                ) {
                    let lanes = S::[<$ty s_lanes>];
                    let nr = NV * lanes;
                    let mut acc = [[simd.[<splat_ $ty s>](0.0); NV]; MR];

                    if mr_eff == MR {
                        let rows = [
                            &a[0..k],
                            &a[lda..lda + k],
                            &a[2 * lda..2 * lda + k],
                            &a[3 * lda..3 * lda + k],
                            &a[4 * lda..4 * lda + k],
                            &a[5 * lda..5 * lda + k],
                        ];

                        for p in 0..k {
                            let b_row = &b[p * ldb..];
                            let mut bv = [simd.[<splat_ $ty s>](0.0); NV];
                            for v in 0..NV {
                                bv[v] = simd.[<load_ $ty s>](&b_row[v * lanes..(v + 1) * lanes]);
                            }
                            for i in 0..MR {
                                let av = simd.[<splat_ $ty s>](rows[i][p]);
                                for v in 0..NV {
                                    acc[i][v] = simd.[<mul_add_ $ty s>](av, bv[v], acc[i][v]);
                                }
                            }
                        }
                    } else {
                        for p in 0..k {
                            let b_row = &b[p * ldb..];
                            let mut bv = [simd.[<splat_ $ty s>](0.0); NV];
                            for v in 0..NV {
                                bv[v] = simd.[<load_ $ty s>](&b_row[v * lanes..(v + 1) * lanes]);
                            }
                            for i in 0..mr_eff {
                                let av = simd.[<splat_ $ty s>](a[i * lda + p]);
                                for v in 0..NV {
                                    acc[i][v] = simd.[<mul_add_ $ty s>](av, bv[v], acc[i][v]);
                                }
                            }
                        }
                    }

                    if nr_eff == nr {
                        store_direct(simd, c, ldc, &acc, mr_eff, beta, alpha);
                    } else {
                        store_partial(simd, c, ldc, c_tile, nr, &acc, mr_eff, nr_eff, beta, alpha);
                    }
                }

                struct Directmatmul<'a> {
                    m: usize,
                    n: usize,
                    k: usize,
                    alpha: $ty,
                    beta: $ty,
                    a: &'a [$ty],
                    b: &'a [$ty],
                    c: &'a mut [$ty],
                    lda: usize,
                    ldb: usize,
                    ldc: usize,
                    c_tile: &'a mut [$ty],
                    b_tail: &'a mut [$ty],
                }

                impl SimdFn for Directmatmul<'_> {
                    type Result = ();

                    #[inline(always)]
                    fn call<S: Simd>(self, simd: S) {
                        let Self {
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            a,
                            b,
                            c,
                            lda,
                            ldb,
                            ldc,
                            c_tile,
                            b_tail,
                        } = self;
                        let nr = NV * S::[<$ty s_lanes>];
                        let full_panels = n / nr;

                        for panel in 0..full_panels {
                            let jp = panel * nr;

                            for ip in (0..m).step_by(MR) {
                                let mr_eff = MR.min(m - ip);
                                direct_micro_kernel(simd, k, alpha, &a[ip * lda..], lda, &b[jp..], ldb, &mut c[ip * ldc + jp..], ldc, mr_eff, nr, beta, c_tile);
                            }
                        }

                        let rem = n - full_panels * nr;
                        if rem != 0 {
                            let jp = full_panels * nr;
                            pack_b_panel(nr, rem, &b[jp..], ldb, b_tail);

                            for ip in (0..m).step_by(MR) {
                                let mr_eff = MR.min(m - ip);
                                direct_micro_kernel(simd, k, alpha, &a[ip * lda..], lda, b_tail, nr, &mut c[ip * ldc + jp..], ldc, mr_eff, rem, beta, c_tile);
                            }
                        }
                    }
                }

                fn direct_matmul(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, beta: $ty, c: &mut [$ty], ldc: usize) {
                    let nr = NV * dispatch(Lanes);
                    let tail_len = if n.is_multiple_of(nr) { 0 } else { k * nr };

                    with_buf(&T_BUF, MR * nr, |c_tile| {
                        with_buf(&P_BUF, tail_len, |b_tail| {
                            dispatch(Directmatmul {
                                m,
                                n,
                                k,
                                alpha,
                                beta,
                                a,
                                b,
                                c,
                                lda,
                                ldb,
                                ldc,
                                c_tile,
                                b_tail,
                            });
                        });
                    });
                }

                struct Serialmatmul<'a> {
                    m: usize,
                    n: usize,
                    k: usize,
                    alpha: $ty,
                    beta: $ty,
                    a: &'a [$ty],
                    b: &'a [$ty],
                    c: &'a mut [$ty],
                    lda: usize,
                    ldb: usize,
                    ldc: usize,
                    kc: usize,

                    mc_max: usize,
                    nc_max: usize,

                    a_pack: &'a mut [$ty],
                    b_pack: &'a mut [$ty],
                    c_tile: &'a mut [$ty],
                }

                impl<'a> SimdFn for Serialmatmul<'a> {
                    type Result = ();

                    #[inline(always)]
                    fn call<S: Simd>(self, simd: S) {
                        let Self {
                            m,
                            n,
                            k,
                            alpha,
                            beta,
                            a,
                            b,
                            c,
                            lda,
                            ldb,
                            ldc,
                            kc,
                            mc_max,
                            nc_max,
                            a_pack,
                            b_pack,
                            c_tile,
                        } = self;

                        let nr = NV * S::[<$ty s_lanes>];

                        for jc in (0..n).step_by(nc_max) {
                            let nc_cur = nc_max.min(n - jc);

                            for pc in (0..k).step_by(kc) {
                                let kc_cur = kc.min(k - pc);

                                let beta = if pc == 0 { beta } else { 1.0 };

                                pack_b(nr, kc_cur, nc_cur, &b[pc * ldb + jc..], ldb, b_pack);

                                for ic in (0..m).step_by(mc_max) {
                                    let mc_cur = mc_max.min(m - ic);

                                    pack_a(kc_cur, mc_cur, &a[ic * lda + pc..], lda, alpha, a_pack);

                                    macro_kernel(simd, nr, mc_cur, nc_cur, kc_cur, a_pack, b_pack, &mut c[ic * ldc + jc..], ldc, beta, c_tile);
                                }
                            }
                        }
                    }
                }

                pub fn matmul_serial(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, beta: $ty, c: &mut [$ty], ldc: usize) {
                    configure_matmul_serial(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, DEFAULT_MC, DEFAULT_NC, DEFAULT_KC);
                }

                pub fn configure_matmul_serial(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, beta: $ty, c: &mut [$ty], ldc: usize, mc: usize, nc: usize, kc: usize) {
                    validate(m, n, k, alpha, a, lda, b, ldb, c, ldc, mc, nc, kc);

                    if m == 0 || n == 0 {
                        return;
                    }

                    if k == 0 || alpha == 0.0 {
                        scale_c(m, n, beta, c, ldc);
                        return;
                    }

                    let nr = NV * dispatch(Lanes);

                    if prefer_direct_serial(m, n, k, nr) {
                        direct_matmul(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
                        return;
                    }

                    let mc_max = mc.min(m).div_ceil(MR) * MR;
                    let nc_max = nc.min(n).div_ceil(nr) * nr;
                    let kc_max = kc.min(k);

                    with_buf(&A_BUF, mc_max * kc_max, |a_pack| {
                        with_buf(&B_BUF, nc_max * kc_max, |b_pack| {
                            with_buf(&T_BUF, MR * nr, |c_tile| {
                                dispatch(Serialmatmul {
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    beta,
                                    a,
                                    b,
                                    c,
                                    lda,
                                    ldb,
                                    ldc,
                                    kc: kc_max,
                                    mc_max,
                                    nc_max,
                                    a_pack,
                                    b_pack,
                                    c_tile,
                                });
                            })
                        })
                    });
                }

                #[cfg(feature = "parallel")]
                const PAR_PACK_B_MIN: usize = 1 << 15;

                #[cfg(feature = "parallel")]
                struct MacroFn<'a> {
                    mc: usize,
                    nc: usize,
                    kc: usize,
                    a_pack: &'a [$ty],
                    b_pack: &'a [$ty],
                    c: &'a mut [$ty],
                    ldc: usize,
                    beta: $ty,
                    c_tile: &'a mut [$ty],
                }

                #[cfg(feature = "parallel")]
                impl<'a> SimdFn for MacroFn<'a> {
                    type Result = ();

                    #[inline(always)]
                    fn call<S: Simd>(self, simd: S) {
                        macro_kernel(simd, NV * S::[<$ty s_lanes>], self.mc, self.nc, self.kc, self.a_pack, self.b_pack, self.c, self.ldc, self.beta, self.c_tile);
                    }
                }

                #[cfg(feature = "parallel")]
                #[inline(always)]
                fn balanced_mc(m: usize, mc: usize, threads: usize) -> usize {
                    let mc = mc.div_ceil(MR) * MR;
                    let split = if m.div_ceil(mc) <= threads { 4 } else { 2 };
                    let target = m.div_ceil(threads.saturating_mul(split)).div_ceil(MR) * MR;
                    target.min(mc).max(MR)
                }

                #[cfg(feature = "parallel")]
                #[inline(always)]
                fn pack_b_par(nr: usize, kc: usize, nc: usize, b: &[$ty], ldb: usize, b_pack: &mut [$ty]) {
                    b_pack.par_chunks_mut(nr * kc).take(nc.div_ceil(nr)).enumerate().for_each(|(panel_idx, panel)| {
                        let jr = panel_idx * nr;
                        pack_b_panel(nr, nr.min(nc - jr), &b[jr..], ldb, panel);
                    });
                }

                #[cfg(feature = "parallel")]
                pub fn matmul_parallel(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, beta: $ty, c: &mut [$ty], ldc: usize) {
                    configure_matmul_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, DEFAULT_MC, DEFAULT_NC, DEFAULT_KC);
                }

                #[cfg(feature = "parallel")]
                pub fn configure_matmul_parallel(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, beta: $ty, c: &mut [$ty], ldc: usize, mc: usize, nc: usize, kc: usize) {
                    validate(m, n, k, alpha, a, lda, b, ldb, c, ldc, mc, nc, kc);

                    if m == 0 || n == 0 {
                        return;
                    }

                    if k == 0 || alpha == 0.0 {
                        scale_c(m, n, beta, c, ldc);
                        return;
                    }

                    let work = m.saturating_mul(n).saturating_mul(k);
                    let row_blocks = m.div_ceil(MR);
                    let threads = (work / MIN_WORK_PER_THREAD).clamp(1, rayon::current_num_threads().max(1)).min(row_blocks);

                    if threads < 2 {
                        return configure_matmul_serial(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mc, nc, kc);
                    }

                    let nr = NV * dispatch(Lanes);
                    let mc_max = balanced_mc(m, mc, threads);
                    let tasks = m.div_ceil(mc_max);

                    if prefer_direct_parallel(m, n, k, nr, tasks) {
                        let c_len = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n)).expect("C shape overflow");
                        let c = &mut c[..c_len];

                        c.par_chunks_mut(mc_max * ldc).enumerate().for_each(|(block, c_block)| {
                            let ic = block * mc_max;
                            let mc_cur = mc_max.min(m - ic);
                            direct_matmul(mc_cur, n, k, alpha, &a[ic * lda..], lda, b, ldb, beta, c_block, ldc);
                        });
                        return;
                    }

                    let nc_max = nc.min(n).div_ceil(nr) * nr;
                    let kc_max = kc.min(k);

                    let c_len = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n)).expect("C shape overflow");
                    let c = &mut c[..c_len];

                    with_buf(&B_BUF, nc_max * kc_max, |b_pack| {
                        for jc in (0..n).step_by(nc_max) {
                            let nc_cur = nc_max.min(n - jc);

                            for pc in (0..k).step_by(kc) {
                                let kc_cur = kc.min(k - pc);

                                let beta_p = if pc == 0 { beta } else { 1.0 };

                                if nc_cur * kc_cur >= PAR_PACK_B_MIN {
                                    pack_b_par(nr, kc_cur, nc_cur, &b[pc * ldb + jc..], ldb, b_pack);
                                } else {
                                    pack_b(nr, kc_cur, nc_cur, &b[pc * ldb + jc..], ldb, b_pack);
                                }

                                let b_pack_ref: &[$ty] = &*b_pack;

                                c.par_chunks_mut(mc_max * ldc).enumerate().for_each(|(block, c_block)| {
                                    let ic = block * mc_max;
                                    let mc_cur = mc_max.min(m - ic);

                                    with_buf(&A_BUF, mc_max * kc_max, |a_pack| {
                                        with_buf(&T_BUF, MR * nr, |c_tile| {
                                            pack_a(kc_cur, mc_cur, &a[ic * lda + pc..], lda, alpha, a_pack);

                                            dispatch(MacroFn {
                                                mc: mc_cur,
                                                nc: nc_cur,
                                                kc: kc_cur,
                                                a_pack: &*a_pack,
                                                b_pack: b_pack_ref,
                                                c: &mut c_block[jc..],
                                                ldc,
                                                beta: beta_p,
                                                c_tile,
                                            });
                                        })
                                    });
                                });
                            }
                        }
                    });
                }

                pub fn matmul(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, beta: $ty, c: &mut [$ty], ldc: usize) {
                    configure_matmul(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, DEFAULT_MC, DEFAULT_NC, DEFAULT_KC, DEFAULT_PARALLEL_THRESHOLD);
                }

                pub fn configure_matmul(
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
                    #[cfg(feature = "parallel")]
                    if use_parallel(m, n, k, parallel_threshold) {
                        return configure_matmul_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mc, nc, kc);
                    }

                    #[cfg(not(feature = "parallel"))]
                    let _ = parallel_threshold;
                    configure_matmul_serial(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mc, nc, kc)
                }

                #[cfg(test)]
                mod tests {
                    use super::*;

                    fn naive(m: usize, n: usize, k: usize, alpha: $ty, a: &[$ty], lda: usize, b: &[$ty], ldb: usize, beta: $ty, c: &mut [$ty], ldc: usize) {
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0;
                                for p in 0..k {
                                    sum += a[i * lda + p] * b[p * ldb + j];
                                }
                                c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
                            }
                        }
                    }

                    fn check_case(m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize, alpha: $ty, beta: $ty, mc: usize, nc: usize, kc: usize, parallel: bool) {
                        let a: Vec<$ty> = (0..m * lda).map(|i| ((i * 17 % 101) as $ty - 50.0) / 37.0).collect();
                        let b: Vec<$ty> = (0..k * ldb).map(|i| ((i * 29 % 103) as $ty - 51.0) / 41.0).collect();
                        let initial: Vec<$ty> = (0..m * ldc).map(|i| ((i * 13 % 97) as $ty - 48.0) / 53.0).collect();

                        let mut expected = initial.clone();
                        naive(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut expected, ldc);

                        let mut actual = initial.clone();
                        #[cfg(feature = "parallel")]
                        if parallel {
                            configure_matmul_parallel(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut actual, ldc, mc, nc, kc);
                            assert_results(m, n, ldc, &initial, &expected, &actual);
                            return;
                        }
                        #[cfg(not(feature = "parallel"))]
                        let _ = parallel;
                        configure_matmul_serial(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut actual, ldc, mc, nc, kc);
                        assert_results(m, n, ldc, &initial, &expected, &actual);
                    }

                    fn assert_results(m: usize, n: usize, ldc: usize, initial: &[$ty], expected: &[$ty], actual: &[$ty]) {
                        for i in 0..m {
                            for j in 0..n {
                                let index = i * ldc + j;
                                let tolerance = 1e-3 * expected[index].abs().max(1.0);
                                assert!((actual[index] - expected[index]).abs() <= tolerance, "mismatch at ({i}, {j}): {} vs {}", actual[index], expected[index]);
                            }
                            for j in n..ldc {
                                let index = i * ldc + j;
                                assert_eq!(actual[index], initial[index]);
                            }
                        }
                    }

                    #[test]
                    fn degenerate_shapes() {
                        let mut c = vec![1.0; 4];
                        matmul(2, 2, 0, 1.0, &[], 0, &[], 0, 2.0, &mut c, 2);
                        assert_eq!(c, vec![2.0; 4]);

                        let mut c = vec![1.0; 4];
                        matmul(2, 2, 3, 0.0, &[], 0, &[], 0, 3.0, &mut c, 2);
                        assert_eq!(c, vec![3.0; 4]);

                        matmul(0, 5, 7, 1.0, &[], 0, &[], 0, 0.0, &mut [], 0);
                        matmul(5, 0, 7, 1.0, &[], 0, &[], 0, 0.0, &mut [], 0);
                        matmul(0, 0, 0, 1.0, &[], 0, &[], 0, 0.0, &mut [], 0);
                    }

                    #[test]
                    fn matches_naive_with_strides() {
                        let shapes = [(1, 1, 1), (5, 7, 3), (6, 17, 10), (12, 33, 17), (23, 31, 47), (7, 19, 300), (24, 47, 33)];
                        let coeffs = [(1.0, 0.0), (1.0, 1.0), (1.25, 0.5), (-0.75, 0.0), (0.0, 2.0), (1.0, -1.0)];

                        for &(m, n, k) in &shapes {
                            let lda = k + 2;
                            let ldb = n + 3;
                            let ldc = n + 5;
                            for &(alpha, beta) in &coeffs {
                                check_case(m, n, k, lda, ldb, ldc, alpha, beta, DEFAULT_MC, DEFAULT_NC, DEFAULT_KC, false);
                            }
                        }
                    }

                    #[cfg(feature = "parallel")]
                    #[test]
                    fn parallel_matches_serial() {
                        let coeffs = [(1.0, 0.0), (1.0, 1.0), (1.25, -0.5)];

                        for &(alpha, beta) in &coeffs {
                            check_case(24, 257, 257, 259, 260, 262, alpha, beta, 12, 64, 32, true);
                        }
                    }

                    #[cfg(feature = "parallel")]
                    #[test]
                    fn auto_parallel_matches_naive() {
                        let (m, n, k) = (24, 256, 384);
                        let lda = k + 1;
                        let ldb = n + 2;
                        let ldc = n + 3;
                        let a: Vec<$ty> = (0..m * lda).map(|i| ((i * 17 % 101) as $ty - 50.0) / 37.0).collect();
                        let b: Vec<$ty> = (0..k * ldb).map(|i| ((i * 29 % 103) as $ty - 51.0) / 41.0).collect();
                        let initial: Vec<$ty> = (0..m * ldc).map(|i| ((i * 13 % 97) as $ty - 48.0) / 53.0).collect();

                        let mut expected = initial.clone();
                        naive(m, n, k, 1.25, &a, lda, &b, ldb, 0.5, &mut expected, ldc);

                        let mut actual = initial.clone();
                        matmul(m, n, k, 1.25, &a, lda, &b, ldb, 0.5, &mut actual, ldc);

                        assert_results(m, n, ldc, &initial, &expected, &actual);
                    }
                }
            }
        }
    };
}

pub(crate) use define_matmul;
