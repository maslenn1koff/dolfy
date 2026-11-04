macro_rules! define_binary_elementwise_op {
    ($ty:ident, $op:ident) => {
        paste::paste! {
            #[inline(always)]
            pub fn [<$op _ $ty>](x: &[$ty], y: &[$ty], dst: &mut [$ty]) {
                assert_eq!(x.len(), y.len());
                assert_eq!(x.len(), dst.len());

                struct Impl<'a>(&'a [$ty], &'a [$ty], &'a mut [$ty]);

                impl<'a> SimdFn for Impl<'a> {
                    type Result = ();

                    #[inline(always)]
                    fn call<S: reviero::traits::Simd>(self, simd: S) -> Self::Result {
                        let len = self.0.len();

                        let mut i = 0;
                        while i + 4 * S::[<$ty s_lanes>] <= len {
                            let (x0, x1, x2, x3) = (
                                simd.[<load_ $ty s>](&self.0[i..i + S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.0[i + S::[<$ty s_lanes>]..i + 2 * S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.0[i + 2 * S::[<$ty s_lanes>]..i + 3 * S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.0[i + 3 * S::[<$ty s_lanes>]..i + 4 * S::[<$ty s_lanes>]]),
                            );

                            let (y0, y1, y2, y3) = (
                                simd.[<load_ $ty s>](&self.1[i..i + S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.1[i + S::[<$ty s_lanes>]..i + 2 * S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.1[i + 2 * S::[<$ty s_lanes>]..i + 3 * S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.1[i + 3 * S::[<$ty s_lanes>]..i + 4 * S::[<$ty s_lanes>]]),
                            );

                            simd.[<store_ $ty s>](&mut self.2[i..i + S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x0, y0));
                            simd.[<store_ $ty s>](&mut self.2[i + S::[<$ty s_lanes>]..i + 2 * S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x1, y1));
                            simd.[<store_ $ty s>](&mut self.2[i + 2 * S::[<$ty s_lanes>]..i + 3 * S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x2, y2));
                            simd.[<store_ $ty s>](&mut self.2[i + 3 * S::[<$ty s_lanes>]..i + 4 * S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x3, y3));

                            i += 4 * S::[<$ty s_lanes>];
                        }

                        while i + S::[<$ty s_lanes>] <= len {
                            let x0 = simd.[<load_ $ty s>](&self.0[i..i + S::[<$ty s_lanes>]]);
                            let y0 = simd.[<load_ $ty s>](&self.1[i..i + S::[<$ty s_lanes>]]);
                            simd.[<store_ $ty s>](&mut self.2[i..i + S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x0, y0));
                            i += S::[<$ty s_lanes>];
                        }

                        while i < len {
                            self.2[i] = self.0[i].[<$op>](self.1[i]);
                            i+=1;
                        }
                    }
                }

                dispatch(Impl(x, y, dst))
            }

            #[cfg(feature = "parallel")]
            #[inline(always)]
            pub fn [<par_ $op _ $ty>](x: &[$ty], y: &[$ty], dst: &mut [$ty], chunk_size: usize) {
                assert!(chunk_size > 0);
                x.par_chunks(chunk_size).zip(y.par_chunks(chunk_size)).zip(dst.par_chunks_mut(chunk_size))
                    .for_each(|((x, y), dst)| { [< $op _ $ty>](x, y, dst) });
            }
        }
    };
}

macro_rules! define_unary_elementwise_op {
    ($ty:ident, $op:ident) => {
        paste::paste! {
            #[inline(always)]
            pub fn [<$op _ $ty>](x: &[$ty], dst: &mut [$ty]) {
                assert_eq!(x.len(), dst.len());

                struct Impl<'a>(&'a [$ty], &'a mut [$ty]);

                impl<'a> SimdFn for Impl<'a> {
                    type Result = ();

                    #[inline(always)]
                    fn call<S: reviero::traits::Simd>(self, simd: S) -> Self::Result {
                        let len = self.0.len();

                        let mut i = 0;
                        while i + 4 * S::[<$ty s_lanes>] <= len {
                            let (x0, x1, x2, x3) = (
                                simd.[<load_ $ty s>](&self.0[i..i + S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.0[i + S::[<$ty s_lanes>]..i + 2 * S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.0[i + 2 * S::[<$ty s_lanes>]..i + 3 * S::[<$ty s_lanes>]]),
                                simd.[<load_ $ty s>](&self.0[i + 3 * S::[<$ty s_lanes>]..i + 4 * S::[<$ty s_lanes>]]),
                            );

                            simd.[<store_ $ty s>](&mut self.1[i..i + S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x0));
                            simd.[<store_ $ty s>](&mut self.1[i + S::[<$ty s_lanes>]..i + 2 * S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x1));
                            simd.[<store_ $ty s>](&mut self.1[i + 2 * S::[<$ty s_lanes>]..i + 3 * S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x2));
                            simd.[<store_ $ty s>](&mut self.1[i + 3 * S::[<$ty s_lanes>]..i + 4 * S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x3));

                            i += 4 * S::[<$ty s_lanes>];
                        }

                        while i + S::[<$ty s_lanes>] <= len {
                            let x0 = simd.[<load_ $ty s>](&self.0[i..i + S::[<$ty s_lanes>]]);
                            simd.[<store_ $ty s>](&mut self.1[i..i + S::[<$ty s_lanes>]], simd.[<$op _ $ty s>](x0));
                            i += S::[<$ty s_lanes>];
                        }

                        while i < len {
                            self.1[i] = self.0[i].[<$op>]();
                            i+=1;
                        }
                    }
                }

                dispatch(Impl(x, dst))
            }

            #[cfg(feature = "parallel")]
            #[inline(always)]
            pub fn [<par_ $op _ $ty>](x: &[$ty], dst: &mut [$ty], chunk_size: usize) {
                assert!(chunk_size > 0);
                x.par_chunks(chunk_size).zip(dst.par_chunks_mut(chunk_size)).for_each(|(x_chunk, dst_chunk)| {
                    [<$op _ $ty>](x_chunk, dst_chunk);
                });
            }
        }
    };
}

macro_rules! define_scalar_op {
    ($ty:ty, $op:ident) => {
        paste::paste! {
            #[inline(always)]
            pub fn [<$op _ $ty>](x: &[$ty], dst: &mut [$ty]) {
                assert_eq!(x.len(), dst.len());

                for i in 0..x.len() {
                    dst[i] = x[i].[<$op>]();
                }
            }
        }
    };

    ($ty:ty, $op:ident, $base:ident) => {
        paste::paste! {
            #[inline(always)]
            pub fn [<$op _ $ty>](x: &[$ty], base: $base, dst: &mut [$ty]) {
                assert_eq!(x.len(), dst.len());

                for i in 0..x.len() {
                    dst[i] = x[i].[<$op>](base);
                }
            }
        }
    };
}

pub(crate) use define_binary_elementwise_op;
pub(crate) use define_unary_elementwise_op;
pub(crate) use define_scalar_op;
