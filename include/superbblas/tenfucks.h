/// TENsor FUture Contraction K-dimensional Subroutine (TenFuCKS)

#ifndef __SUPERBBLAS_TENFUCKS__
#define __SUPERBBLAS_TENFUCKS__

#include "platform.h"
#ifdef SUPERBBLAS_USE_XSIMD
#    include "xsimd/xsimd.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#if __cplusplus >= 202002L
#    include <experimental/simd>
#endif

namespace superbblas {
    namespace detail_xp {

        using T = double;
        using cT = std::complex<T>;
        using Idx = int;

#ifdef SUPERBBLAS_USE_XSIMD
        using vc4 = xsimd::make_sized_batch<T, 4>::type;
        using vc8 = xsimd::make_sized_batch<T, 8>::type;
        using vi4 = xsimd::batch<uint64_t, vc4::arch_type>;
        using vi8 = xsimd::batch<uint64_t, vc8::arch_type>;

        inline vc8 flip_ri(const vc8 &b) {
            return xsimd::swizzle(b, xsimd::batch_constant<vi8, 1, 0, 3, 2, 5, 4, 4, 4>());
        }

        inline vc8 scalar_mult(cT s, const vc8 &b) {
            //if (s == cT{1}) return b;
            T r = *(T *)&s;
            T i = ((T *)&s)[1];
            return s == cT{1} ? b
                              : xsimd::fma(vc8(r), b, vc8(-i, i, -i, i, -i, i, i, i) * flip_ri(b));
        }

        constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, bool the_real) {
            return i * 2 * ldr + j * 2 * ldc + (the_real ? 0 : 1);
        }

        constexpr bool the_real = true;
        constexpr bool the_imag = false;
        inline std::array<vc8, 2> get_col_intr(const T *a, Idx ldr, Idx ldc, Idx d) {
            auto va = vc8::gather(a,
                                  vi8(                                                  //
                                      get_disp_3x3(0, (d + 0) % 3, ldr, ldc, the_real), //
                                      get_disp_3x3(0, (d + 0) % 3, ldr, ldc, the_imag), //
                                      get_disp_3x3(1, (d + 1) % 3, ldr, ldc, the_real), //
                                      get_disp_3x3(1, (d + 1) % 3, ldr, ldc, the_imag), //
                                      get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_real), //
                                      get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag), //
                                      get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag), //
                                      get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag)));
            return {xsimd::shuffle(va, va, xsimd::batch_constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>()),
                    xsimd::shuffle(
                        xsimd::neg(va), va,
                        xsimd::batch_constant<vi8, 1, 8 + 1, 3, 8 + 3, 5, 8 + 5, 8 + 5, 8 + 5>())};
        }

        inline vi8 get_8_ri(Idx ld) {
            return vi8(ld * 2 * 0, ld * 2 * 0 + 1, ld * 2 * 1, ld * 2 * 1 + 1, ld * 2 * 2,
                       ld * 2 * 2 + 1, ld * 2 * 2 + 1, ld * 2 * 2 + 1);
        }

        inline void gemm_basic_3x3c_intr(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar,
                                         Idx ldac, const cT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                         cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                         cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const T *SB_RESTRICT a = (const T *)(a_);
            const T *SB_RESTRICT b = (const T *)(b_);
            const T *SB_RESTRICT c = (const T *)(c_);
            T *SB_RESTRICT d = (T *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            for (Idx j = 0; j < N; ++j) {
                auto b0 = vc8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 =
                    beta == T{0}
                        ? vc8(0)
                        : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j, vi8(get_8_ri(ldcr))));
                for (int disp = 0; disp < 3; ++disp) {
                    auto a01 = get_col_intr(a, ldar, ldac, disp);
                    if (disp > 0) { b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1()); }
                    c0 = xsimd::fma(std::get<0>(a01), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(std::get<1>(a01), b0, c0);
                }

                if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        inline void gemm_basic_3x3c_intr2(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const cT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const T *SB_RESTRICT a = (const T *)(a_);
            const T *SB_RESTRICT b = (const T *)(b_);
            const T *SB_RESTRICT c = (const T *)(c_);
            T *SB_RESTRICT d = (T *)(d_);
            //using vi8_seq = xsimd::batch_constant<vi8, 0, 2, 4, 6, 8, 10, 10, 10>;
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            Idx j = 0;
            if (N % 2 != 0) {
                gemm_basic_3x3c_intr(N % 2, alpha, a_, ldar, ldac, b_, ldbr, ldbc, beta, c_, ldcr,
                                     ldcc, d_, lddr, lddc);
                j = N % 2;
            }

            for (; j < N; j += 2) {
                int j0 = j, j1 = j + 1;
                auto b0 = vc8::gather(b + ldbc * 2 * j0, get_8_ri(ldbr));
                auto b1 = vc8::gather(b + ldbc * 2 * j1, get_8_ri(ldbr));
                auto c0 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j0, get_8_ri(ldcr)));
                auto c1 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j1, get_8_ri(ldcr)));

                for (int disp = 0; disp < 3; ++disp) {
                    auto a01 = get_col_intr(a, ldar, ldac, disp);
                    if (disp > 0) {
                        b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                        b1 = xsimd::swizzle(b1, vi8_flip_and_plus_1());
                    }
                    c0 = xsimd::fma(std::get<0>(a01), b0, c0);
                    c1 = xsimd::fma(std::get<0>(a01), b1, c1);

                    b0 = flip_ri(b0);
                    b1 = flip_ri(b1);
                    c0 = xsimd::fma(std::get<1>(a01), b0, c0);
                    c1 = xsimd::fma(std::get<1>(a01), b1, c1);
                }

                if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
                if (alpha != T{1}) c1 = scalar_mult(alpha, c1);
                c0.scatter(d + lddc * 2 * j0, get_8_ri(lddr));
                c1.scatter(d + lddc * 2 * j1, get_8_ri(lddr));
            }
        }

        inline std::array<vc8, 2> get_col_intr(const T *a, Idx ldr, Idx ldc, Idx d, bool first_time,
                                               T v[3 * 8]) {
            vc8 va;
            if (first_time) {
                va = vc8::gather(a,
                                 vi8(                                                  //
                                     get_disp_3x3(0, (d + 0) % 3, ldr, ldc, the_real), //
                                     get_disp_3x3(0, (d + 0) % 3, ldr, ldc, the_imag), //
                                     get_disp_3x3(1, (d + 1) % 3, ldr, ldc, the_real), //
                                     get_disp_3x3(1, (d + 1) % 3, ldr, ldc, the_imag), //
                                     get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_real), //
                                     get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag), //
                                     get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag), //
                                     get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag)));
                va.store_aligned(&v[8 * d]);
            } else {
                va = vc8::load_aligned(&v[8 * d]);
            }
            return {xsimd::shuffle(va, va, xsimd::batch_constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>()),
                    xsimd::shuffle(
                        xsimd::neg(va), va,
                        xsimd::batch_constant<vi8, 1, 8 + 1, 3, 8 + 3, 5, 8 + 5, 8 + 5, 8 + 5>())};
        }

        inline void gemm_basic_3x3c_intr3(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const cT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const T *SB_RESTRICT a = (const T *)(a_);
            const T *SB_RESTRICT b = (const T *)(b_);
            const T *SB_RESTRICT c = (const T *)(c_);
            T *SB_RESTRICT d = (T *)(d_);
            //using vi8_seq = xsimd::batch_constant<vi8, 0, 2, 4, 6, 8, 10, 10, 10>;
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            //    if (N%2 != 0) {
            //gemm_basic_3x3c_intr(N%2, alpha, a_, ldar, ldac, b_, ldbr, ldbc, beta, c_, ldcr, ldcc, d_, lddr, lddc);
            //j =N%2;
            //    }

            alignas(vc8::arch_type::alignment()) T a_aux[3 * 8];

            Idx j = 0;
            for (; j + 2 <= N; j += 2) {
                int j0 = j, j1 = j + 1;
                auto b0 = vc8::gather(b + ldbc * 2 * j0, get_8_ri(ldbr));
                auto b1 = vc8::gather(b + ldbc * 2 * j1, get_8_ri(ldbr));
                auto c0 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j0, get_8_ri(ldcr)));
                auto c1 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j1, get_8_ri(ldcr)));

                for (int disp = 0; disp < 3; ++disp) {
                    auto a01 = get_col_intr(a, ldar, ldac, disp, j == 0, a_aux);
                    if (disp > 0) {
                        b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                        b1 = xsimd::swizzle(b1, vi8_flip_and_plus_1());
                    }
                    c0 = xsimd::fma(std::get<0>(a01), b0, c0);
                    c1 = xsimd::fma(std::get<0>(a01), b1, c1);

                    b0 = flip_ri(b0);
                    b1 = flip_ri(b1);
                    c0 = xsimd::fma(std::get<1>(a01), b0, c0);
                    c1 = xsimd::fma(std::get<1>(a01), b1, c1);
                }

                if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
                if (alpha != T{1}) c1 = scalar_mult(alpha, c1);
                c0.scatter(d + lddc * 2 * j0, get_8_ri(lddr));
                c1.scatter(d + lddc * 2 * j1, get_8_ri(lddr));
            }
            if (j < N) {
                auto b0 = vc8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j, get_8_ri(ldcr)));

                for (int disp = 0; disp < 3; ++disp) {
                    auto a01 = get_col_intr(a, ldar, ldac, disp, j == 0, a_aux);
                    if (disp > 0) { b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1()); }
                    c0 = xsimd::fma(std::get<0>(a01), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(std::get<1>(a01), b0, c0);
                }

                if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        inline vc8 get_cols_aux(const T *a, Idx ldr, Idx ldc, Idx d) {
            return vc8::gather(a,
                               vi8(                                                  //
                                   get_disp_3x3(0, (d + 0) % 3, ldr, ldc, the_real), //
                                   get_disp_3x3(0, (d + 0) % 3, ldr, ldc, the_imag), //
                                   get_disp_3x3(1, (d + 1) % 3, ldr, ldc, the_real), //
                                   get_disp_3x3(1, (d + 1) % 3, ldr, ldc, the_imag), //
                                   get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_real), //
                                   get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag), //
                                   get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag), //
                                   get_disp_3x3(2, (d + 2) % 3, ldr, ldc, the_imag)));
        }

        inline std::array<vc8, 3> get_cols(const T *a, Idx ldr, Idx ldc) {
            return {get_cols_aux(a, ldr, ldc, 0), //
                    get_cols_aux(a, ldr, ldc, 1), //
                    get_cols_aux(a, ldr, ldc, 2)};
        }

        template <bool the_real> inline vc8 get_col(vc8 va) {
            return the_real ? xsimd::shuffle(va, va,
                                             xsimd::batch_constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>())
                            : xsimd::shuffle(xsimd::neg(va), va,
                                             xsimd::batch_constant<vi8, 1, 8 + 1, 3, 8 + 3, 5,
                                                                   8 + 5, 8 + 5, 8 + 5>());
        }

        inline void gemm_basic_3x3c_intr4(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const cT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const T *SB_RESTRICT a = (const T *)(a_);
            const T *SB_RESTRICT b = (const T *)(b_);
            const T *SB_RESTRICT c = (const T *)(c_);
            T *SB_RESTRICT d = (T *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            auto a012 = get_cols(a, ldar, ldac);
            for (Idx j = 0; j < N; ++j) {
                auto b0 = vc8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j, get_8_ri(ldcr)));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                    c0 = xsimd::fma(get_col<the_real>(a012[disp]), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(get_col<the_imag>(a012[disp]), b0, c0);
                }

                if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        inline void gemm_basic_3x3c_intr5(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const cT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const T *SB_RESTRICT a = (const T *)(a_);
            const T *SB_RESTRICT b = (const T *)(b_);
            const T *SB_RESTRICT c = (const T *)(c_);
            T *SB_RESTRICT d = (T *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            auto a012 = get_cols(a, ldar, ldac);
            Idx j = 0;
            if (j % 2 != 0) {
                auto b0 = vc8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j, get_8_ri(ldcr)));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                    c0 = xsimd::fma(get_col<the_real>(a012[disp]), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(get_col<the_imag>(a012[disp]), b0, c0);
                }

                if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
                j++;
            }
            for (; j + 2 <= N; j += 2) {
                Idx j0 = j, j1 = j + 1;
                auto b0 = vc8::gather(b + ldbc * 2 * j0, get_8_ri(ldbr));
                auto b1 = vc8::gather(b + ldbc * 2 * j1, get_8_ri(ldbr));
                auto c0 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j0, get_8_ri(ldcr)));
                auto c1 = beta == T{0}
                              ? vc8(0)
                              : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j1, get_8_ri(ldcr)));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                    if (disp > 0) b1 = xsimd::swizzle(b1, vi8_flip_and_plus_1());
                    c0 = xsimd::fma(get_col<the_real>(a012[disp]), b0, c0);
                    c1 = xsimd::fma(get_col<the_real>(a012[disp]), b1, c1);

                    b0 = flip_ri(b0);
                    b1 = flip_ri(b1);
                    c0 = xsimd::fma(get_col<the_imag>(a012[disp]), b0, c0);
                    c1 = xsimd::fma(get_col<the_imag>(a012[disp]), b1, c1);
                }

                if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
                if (alpha != T{1}) c1 = scalar_mult(alpha, c1);
                c0.scatter(d + lddc * 2 * j0, get_8_ri(lddr));
                c1.scatter(d + lddc * 2 * j1, get_8_ri(lddr));
            }
        }

#elif __cpp_lib_experimental_parallel_simd >= 201803

        /// Implementation based on experimental simd C++ interface

        namespace stdx = std::experimental;
        using vc8 = stdx::fixed_size_simd<double, 8>;

        constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, bool the_real) {
            return i * 2 * ldr + j * 2 * ldc + (the_real ? 0 : 1);
        }

        inline vc8 get_A_cols_aux(const T *SB_RESTRICT a, Idx ldr, Idx ldc, Idx d) {
            return vc8([=](auto i) {
                return a[i < 6 ? get_disp_3x3(i / 2, (d + i / 2) % 3, ldr, ldc,
                                              i % 2 == 0 /* is real? */)
                               : get_disp_3x3(5 / 2, (d + 5 / 2) % 3, ldr, ldc, 5 % 2 == 0)];
            });
        }

        inline std::array<vc8, 3> get_A_cols(const T *SB_RESTRICT a, Idx ldr, Idx ldc) {
            return {get_A_cols_aux(a, ldr, ldc, 0), //
                    get_A_cols_aux(a, ldr, ldc, 1), //
                    get_A_cols_aux(a, ldr, ldc, 2)};
        }

        constexpr bool the_real = true;
        constexpr bool the_imag = false;

        template <bool is_real> inline vc8 get_A_col(vc8 va) {
            return is_real ? vc8([=](auto i) { return va[i / 2 * 2]; }) : vc8([=](auto i) {
                return i < 6 ? (i % 2 == 0 ? -va[i / 2 * 2 + 1] : va[i / 2 * 2 + 1])
                             : va[5 / 2 * 2 + 1];
            });
        }

        constexpr Idx get_8_ri(Idx i, Idx ld) {
            return i < 6 ? ld * 2 * (i / 2) + i % 2 : ld * 2 * (5 / 2) + 5 % 2;
        }

        inline vc8 get_B_col(const T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc) {
            return vc8([=](auto i) { return b[ldc * 2 * j + get_8_ri(i, ldr)]; });
        }

        inline void set_B_col(vc8 x, T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc) {
            for (std::size_t i = 0; i < 6; ++i) b[ldc * 2 * j + get_8_ri(i, ldr)] = x[i];
        }

        inline vc8 flip_ri(vc8 x) {
            return vc8([=](auto i) { return x[i / 2 * 2 + (i + 1) % 2]; });
        }

        inline vc8 scalar_mult(cT s, vc8 b) {
            T r = *(T *)&s;
            T i = ((T *)&s)[1];
            return s == cT{1} ? b : stdx::fma(vc8(r), b, vc8([=](auto j) {
                                                             return j < 6 ? (j % 2 == 0 ? -i : i)
                                                                          : i;
                                                         }) * flip_ri(b));
        }
        /// It should return: x[{3, 2, 5, 4, 1, 0, 0, 0}]
        inline vc8 flip_ri_plus_1(vc8 x) {
            return vc8([=](auto i) { return x[i < 6 ? ((i / 2 + 1) % 3) * 2 + (i + 1) % 2 : 0]; });
        }

        inline void gemm_basic_3x3c_intr4(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const cT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const T *SB_RESTRICT a = (const T *)(a_);
            const T *SB_RESTRICT b = (const T *)(b_);
            const T *SB_RESTRICT c = (const T *)(c_);
            T *SB_RESTRICT d = (T *)(d_);

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            auto a012 = get_A_cols(a, ldar, ldac);
            for (Idx j = 0; j < N; ++j) {
                vc8 b0 = get_B_col(b, j, ldbr, ldbc);
                auto c0 = beta == T{0} ? vc8(0) : scalar_mult(beta, get_B_col(c, j, ldcr, ldcc));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = flip_ri_plus_1(b0);
                    c0 = stdx::fma(get_A_col<the_real>(a012[disp]), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = stdx::fma(get_A_col<the_imag>(a012[disp]), b0, c0);
                }
                set_B_col(scalar_mult(alpha, c0), d, j, lddr, lddc);
            }
        }
#endif // SUPERBBLAS_USE_XSIMD
    }
}
#endif // __SUPERBBLAS_TENFUCKS__
