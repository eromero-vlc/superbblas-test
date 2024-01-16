/// TENsor FUture Contraction K-dimensional Subroutine (TenFuCKS)

/// Proposed strategy for matrix-matrix multiplication:
/// - Only operation: element wise sums and products.
/// - Read columns of the two input matrices and accumulate the results on the output matrix.
///
/// Example 2x2:
/// [ a_00 a_01 ] x [ b_0 ] -> [ c_0 ]
/// [ a_10 a_11 ]   [ b_1 ]    [ c_1 ]
/// 0) 0 -> c
/// 1) [ a_00 ] x [ b_0 ] -> [ c_0 ]
///    [ a_11 ]   [ b_1 ]    [ c_1 ]
/// 2) [ a_01 ] x [ b_0 ] -> [ c_0 ]
///    [ a_10 ]   [ b_1 ] +  [ c_1 ]

#ifndef __SUPERBBLAS_TENFUCKS__
#define __SUPERBBLAS_TENFUCKS__

#include "platform.h"
#ifdef SUPERBBLAS_USE_XSIMD
#    include "xsimd/xsimd.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <complex>
#include <execution>
#include <numeric>
#if __cplusplus >= 202002L
#    include <experimental/simd>
#endif

namespace superbblas {
    namespace detail_xp {

        using zT = std::complex<double>;
        using cT = std::complex<float>;
        using Idx = unsigned int;

#ifdef SUPERBBLAS_USE_XSIMD
        using vc16 = xsimd::make_sized_batch<float, 16>::type;
        using vi16 = xsimd::batch<unsigned int, vc16::arch_type>;
        using vz8 = xsimd::make_sized_batch<double, 8>::type;
        using vi8 = xsimd::batch<uint64_t, vz8::arch_type>;

        inline vz8 flip_ri(const vz8 &b) {
            return xsimd::swizzle(b, xsimd::batch_constant<vi8, 1, 0, 3, 2, 5, 4, 4, 4>());
        }

        inline vz8 scalar_mult(zT s, const vz8 &b) {
            //if (s == zT{1}) return b;
            double r = *(double *)&s;
            double i = ((double *)&s)[1];
            return s == zT{1} ? b
                              : xsimd::fma(vz8(r), b, vz8(-i, i, -i, i, -i, i, i, i) * flip_ri(b));
        }

        constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, bool the_real) {
            return i * 2 * ldr + j * 2 * ldc + (the_real ? 0 : 1);
        }

        constexpr bool the_real = true;
        constexpr bool the_imag = false;
        inline std::array<vz8, 2> get_col_intr(const double *a, Idx ldr, Idx ldc, Idx d) {
            auto va = vz8::gather(a,
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

        inline void gemm_basic_3x3c_intr(Idx N, zT alpha, const zT *SB_RESTRICT a_, Idx ldar,
                                         Idx ldac, const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                         zT beta, const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                         zT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const double *SB_RESTRICT a = (const double *)(a_);
            const double *SB_RESTRICT b = (const double *)(b_);
            const double *SB_RESTRICT c = (const double *)(c_);
            double *SB_RESTRICT d = (double *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            for (Idx j = 0; j < N; ++j) {
                auto b0 = vz8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 =
                    beta == zT{0}
                        ? vz8(0)
                        : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j, vi8(get_8_ri(ldcr))));
                for (int disp = 0; disp < 3; ++disp) {
                    auto a01 = get_col_intr(a, ldar, ldac, disp);
                    if (disp > 0) { b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1()); }
                    c0 = xsimd::fma(std::get<0>(a01), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(std::get<1>(a01), b0, c0);
                }

                c0 = scalar_mult(alpha, c0);
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        inline void gemm_basic_3x3c_intr2(Idx N, zT alpha, const zT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          zT beta, const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          zT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const double *SB_RESTRICT a = (const double *)(a_);
            const double *SB_RESTRICT b = (const double *)(b_);
            const double *SB_RESTRICT c = (const double *)(c_);
            double *SB_RESTRICT d = (double *)(d_);
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
                auto b0 = vz8::gather(b + ldbc * 2 * j0, get_8_ri(ldbr));
                auto b1 = vz8::gather(b + ldbc * 2 * j1, get_8_ri(ldbr));
                auto c0 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j0, get_8_ri(ldcr)));
                auto c1 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j1, get_8_ri(ldcr)));

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

                c0 = scalar_mult(alpha, c0);
                c1 = scalar_mult(alpha, c1);
                c0.scatter(d + lddc * 2 * j0, get_8_ri(lddr));
                c1.scatter(d + lddc * 2 * j1, get_8_ri(lddr));
            }
        }

        inline std::array<vz8, 2> get_col_intr(const double *a, Idx ldr, Idx ldc, Idx d,
                                               bool first_time, double v[3 * 8]) {
            vz8 va;
            if (first_time) {
                va = vz8::gather(a,
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
                va = vz8::load_aligned(&v[8 * d]);
            }
            return {xsimd::shuffle(va, va, xsimd::batch_constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>()),
                    xsimd::shuffle(
                        xsimd::neg(va), va,
                        xsimd::batch_constant<vi8, 1, 8 + 1, 3, 8 + 3, 5, 8 + 5, 8 + 5, 8 + 5>())};
        }

        inline void gemm_basic_3x3c_intr3(Idx N, zT alpha, const zT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          zT beta, const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          zT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const double *SB_RESTRICT a = (const double *)(a_);
            const double *SB_RESTRICT b = (const double *)(b_);
            const double *SB_RESTRICT c = (const double *)(c_);
            double *SB_RESTRICT d = (double *)(d_);
            //using vi8_seq = xsimd::batch_constant<vi8, 0, 2, 4, 6, 8, 10, 10, 10>;
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            //    if (N%2 != 0) {
            //gemm_basic_3x3c_intr(N%2, alpha, a_, ldar, ldac, b_, ldbr, ldbc, beta, c_, ldcr, ldcc, d_, lddr, lddc);
            //j =N%2;
            //    }

            alignas(vz8::arch_type::alignment()) double a_aux[3 * 8];

            Idx j = 0;
            for (; j + 2 <= N; j += 2) {
                int j0 = j, j1 = j + 1;
                auto b0 = vz8::gather(b + ldbc * 2 * j0, get_8_ri(ldbr));
                auto b1 = vz8::gather(b + ldbc * 2 * j1, get_8_ri(ldbr));
                auto c0 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j0, get_8_ri(ldcr)));
                auto c1 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j1, get_8_ri(ldcr)));

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

                c0 = scalar_mult(alpha, c0);
                c1 = scalar_mult(alpha, c1);
                c0.scatter(d + lddc * 2 * j0, get_8_ri(lddr));
                c1.scatter(d + lddc * 2 * j1, get_8_ri(lddr));
            }
            if (j < N) {
                auto b0 = vz8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j, get_8_ri(ldcr)));

                for (int disp = 0; disp < 3; ++disp) {
                    auto a01 = get_col_intr(a, ldar, ldac, disp, j == 0, a_aux);
                    if (disp > 0) { b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1()); }
                    c0 = xsimd::fma(std::get<0>(a01), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(std::get<1>(a01), b0, c0);
                }

                c0 = scalar_mult(alpha, c0);
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        inline vz8 get_cols_aux(const double *a, Idx ldr, Idx ldc, Idx d) {
            return vz8::gather(a,
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

        inline std::array<vz8, 3> get_cols(const double *a, Idx ldr, Idx ldc) {
            return {get_cols_aux(a, ldr, ldc, 0), //
                    get_cols_aux(a, ldr, ldc, 1), //
                    get_cols_aux(a, ldr, ldc, 2)};
        }

        template <bool the_real> inline vz8 get_col(vz8 va) {
            return the_real ? xsimd::shuffle(va, va,
                                             xsimd::batch_constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>())
                            : xsimd::shuffle(xsimd::neg(va), va,
                                             xsimd::batch_constant<vi8, 1, 8 + 1, 3, 8 + 3, 5,
                                                                   8 + 5, 8 + 5, 8 + 5>());
        }

        /// Matrix-matrix multiplication, D = alpha*A*B + beta*C, where
        /// A is a 3x3 matrix, B is a 3xN matrix, and C and D are 3xN matrices.
        ///
        /// \param N: number of columns of B
        /// \param alpha: scale on the matrix multiplication
        /// \param a_: pointer to the first element of matrix A
        /// \param ldar: row leading dimension for matrix A
        /// \param ldac: column leading dimension for matrix A
        /// \param b_: pointer to the first leading dimension for the matrix B
        /// \param ldbr: row leading dimension for matrix B
        /// \param ldbc: column leading dimension for matrix B
        /// \param beta: scale of the addition on matrix C
        /// \param ldbr: row leading dimension for matrix B
        /// \param ldbc: column leading dimension for matrix B
        /// \param c_: pointer to the first element of matrix C
        /// \param ldcr: row leading dimension for matrix C
        /// \param ldcc: column leading dimension for matrix C
        /// \param d_: pointer to the first element of matrix C
        /// \param lddr: row leading dimension for matrix D
        /// \param lddc: column leading dimension for matrix D

        __attribute__((always_inline)) inline void
        gemm_basic_3x3c_intr4(Idx N, zT alpha, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                              const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc, zT beta,
                              const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc, zT *SB_RESTRICT d_,
                              Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const double *SB_RESTRICT a = (const double *)(a_);
            const double *SB_RESTRICT b = (const double *)(b_);
            const double *SB_RESTRICT c = (const double *)(c_);
            double *SB_RESTRICT d = (double *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            auto a012 = get_cols(a, ldar, ldac);
            for (Idx j = 0; j < N; ++j) {
                auto b0 = vz8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                vz8 c0{0};
                auto c1 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j, get_8_ri(ldcr)));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                    c0 = xsimd::fma(get_col<the_real>(a012[disp]), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(get_col<the_imag>(a012[disp]), b0, c0);
                }

                c0 = scalar_mult(alpha, c0) + c1;
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        /// Matrix-matrix multiplication, D = A*B + C, where
        /// A is a 3x3 matrix, B is a 3xN matrix, and C and D are 3xN matrices.
        ///
        /// \param N: number of columns of B
        /// \param a_: pointer to the first element of matrix A
        /// \param ldar: row leading dimension for matrix A
        /// \param ldac: column leading dimension for matrix A
        /// \param b_: pointer to the first leading dimension for the matrix B
        /// \param ldbr: row leading dimension for matrix B
        /// \param ldbc: column leading dimension for matrix B
        /// \param ldbr: row leading dimension for matrix B
        /// \param ldbc: column leading dimension for matrix B
        /// \param c_: pointer to the first element of matrix C
        /// \param ldcr: row leading dimension for matrix C
        /// \param ldcc: column leading dimension for matrix C
        /// \param d_: pointer to the first element of matrix C
        /// \param lddr: row leading dimension for matrix D
        /// \param lddc: column leading dimension for matrix D

        __attribute__((always_inline)) inline void
        gemm_basic_3x3c_intr4_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                                           const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                           const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                           zT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const double *SB_RESTRICT a = (const double *)(a_);
            const double *SB_RESTRICT b = (const double *)(b_);
            const double *SB_RESTRICT c = (const double *)(c_);
            double *SB_RESTRICT d = (double *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            auto a012 = get_cols(a, ldar, ldac);
            for (Idx j = 0; j < N; ++j) {
                auto b0 = vz8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 = vz8::gather(c + ldcc * 2 * j, get_8_ri(ldcr));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                    c0 = xsimd::fma(get_col<the_real>(a012[disp]), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(get_col<the_imag>(a012[disp]), b0, c0);
                }
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        __attribute__((always_inline)) inline void gemm_basic_3x3c_intr4_alpha1_beta1_perm(
            Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac, const zT *SB_RESTRICT b_, Idx ldbr,
            Idx ldbc, const Idx *SB_RESTRICT b_cols_perm, Idx b_cols_modulus,
            const zT *SB_RESTRICT alphas, const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
            zT *SB_RESTRICT d_, Idx lddr, Idx lddc) {

            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const double *SB_RESTRICT a = (const double *)(a_);
            const double *SB_RESTRICT b = (const double *)(b_);
            const double *SB_RESTRICT c = (const double *)(c_);
            double *SB_RESTRICT d = (double *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            auto a012 = get_cols(a, ldar, ldac);
            for (Idx j = 0; j < N; ++j) {
                auto b0 = vz8::gather(
                    b + ldbc * 2 *
                            (j / b_cols_modulus * b_cols_modulus + b_cols_perm[j % b_cols_modulus]),
                    get_8_ri(ldbr));
                vz8 c0{0};
                auto c1 = vz8::gather(c + ldcc * 2 * j, get_8_ri(ldcr));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                    c0 = xsimd::fma(get_col<the_real>(a012[disp]), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(get_col<the_imag>(a012[disp]), b0, c0);
                }

                c0 = scalar_mult(alphas[j % b_cols_modulus], c0) + c1;
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
            }
        }

        inline void gemm_basic_3x3c_intr5(Idx N, zT alpha, const zT *SB_RESTRICT a_, Idx ldar,
                                          Idx ldac, const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                          zT beta, const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
                                          zT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
            //constexpr Idx M = 3;
            //constexpr Idx K = 3;
            const double *SB_RESTRICT a = (const double *)(a_);
            const double *SB_RESTRICT b = (const double *)(b_);
            const double *SB_RESTRICT c = (const double *)(c_);
            double *SB_RESTRICT d = (double *)(d_);
            using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;

            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            auto a012 = get_cols(a, ldar, ldac);
            Idx j = 0;
            if (j % 2 != 0) {
                auto b0 = vz8::gather(b + ldbc * 2 * j, get_8_ri(ldbr));
                auto c0 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j, get_8_ri(ldcr)));
                for (int disp = 0; disp < 3; ++disp) {
                    if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                    c0 = xsimd::fma(get_col<the_real>(a012[disp]), b0, c0);

                    b0 = flip_ri(b0);
                    c0 = xsimd::fma(get_col<the_imag>(a012[disp]), b0, c0);
                }

                c0 = scalar_mult(alpha, c0);
                c0.scatter(d + lddc * 2 * j, get_8_ri(lddr));
                j++;
            }
            for (; j + 2 <= N; j += 2) {
                Idx j0 = j, j1 = j + 1;
                auto b0 = vz8::gather(b + ldbc * 2 * j0, get_8_ri(ldbr));
                auto b1 = vz8::gather(b + ldbc * 2 * j1, get_8_ri(ldbr));
                auto c0 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j0, get_8_ri(ldcr)));
                auto c1 = beta == zT{0}
                              ? vz8(0)
                              : scalar_mult(beta, vz8::gather(c + ldcc * 2 * j1, get_8_ri(ldcr)));
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

                c0 = scalar_mult(alpha, c0);
                c1 = scalar_mult(alpha, c1);
                c0.scatter(d + lddc * 2 * j0, get_8_ri(lddr));
                c1.scatter(d + lddc * 2 * j1, get_8_ri(lddr));
            }
        }

#elif __cpp_lib_experimental_parallel_simd >= 201803

        /// Implementation based on experimental simd C++ interface

        namespace stdx = std::experimental;

        /// Implementation for
        ///  - complex double on SIMD 512 bits (avx512)
        ///  - complex float  on SIMD 256 bits (avx)
        ///  - complex half   on SIMD 128 bits (sse2?, probably not useful, although it may be possible to take advantage of the intel instruction to convert 4 half precision into float)

        template <typename T, bool default_leading_dimensions = false> struct gemm_3x3_8parts {
            using zT = std::complex<T>;
            using vc8 = stdx::fixed_size_simd<T, 8>;

            static constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, Idx reality) {
                return i * 2 * ldr + j * 2 * ldc + reality;
            }

            static inline vc8 get_A_cols_aux(const T *SB_RESTRICT a, Idx ldr, Idx ldc, Idx d) {
                return vc8([=](auto i) {
                    return a[i < 6 ? get_disp_3x3(i / 2, (d + i / 2) % 3, ldr, ldc, i % 2)
                                   : (i == 6
                                          ? get_disp_3x3(4 / 2, (d + 4 / 2) % 3, ldr, ldc, 4 % 2)
                                          : get_disp_3x3(5 / 2, (d + 5 / 2) % 3, ldr, ldc, 5 % 2))];
                });
            }

            static inline std::array<vc8, 3> get_A_cols(const T *SB_RESTRICT a, Idx ldr, Idx ldc) {
                return {get_A_cols_aux(a, ldr, ldc, 0), //
                        get_A_cols_aux(a, ldr, ldc, 1), //
                        get_A_cols_aux(a, ldr, ldc, 2)};
            }

            static constexpr bool the_real = true;
            static constexpr bool the_imag = false;

            template <bool is_real> static inline vc8 get_A_col(vc8 va) {
                return is_real ? vc8([=](auto i) { return va[i / 2 * 2]; })
                               : vc8([=](auto i) { return i % 2 == 0 ? T{-1} : T{1}; }) *
                                     vc8([=](auto i) { return va[i / 2 * 2 + 1]; });
            }

            static constexpr Idx get_8_ri(Idx i, Idx ld) {
                return i < 6 ? ld * 2 * (i / 2) + i % 2 : ld * 2 * (5 / 2) + 5 % 2;
            }

            static inline vc8 get_B_col(const T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc) {
                const T *SB_RESTRICT bj = std::assume_aligned<sizeof(T) * 2>(b + ldc * 2 * j);
                return vc8([=](auto i) { return bj[get_8_ri(i, ldr)]; });
            }

            static inline void set_B_col(vc8 x, T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc) {
                T *SB_RESTRICT bj = std::assume_aligned<sizeof(T) * 2>(b + ldc * 2 * j);
                vc8([=](auto i) {
                    bj[get_8_ri(i, ldr)] = x[i];
                    return T{0};
                });
            }

            static inline vc8 flip_ri(vc8 x) {
                return vc8([=](auto i) {
                    return x[i < 6 ? i / 2 * 2 + (i + 1) % 2 : 5 / 2 * 2 + (5 + 1) % 2];
                });
            }

            /// It should return: x[{3, 2, 5, 4, 1, 0, 0, 0}]
            static inline vc8 flip_ri_plus_1(vc8 x) {
                return vc8([=](auto i) {
                    return x[i < 6 ? ((i / 2 + 1) % 3) * 2 + (i + 1) % 2
                                   : ((5 / 2 + 1) % 3) * 2 + (5 + 1) % 2];
                });
            }

            static inline void gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_,
                                                            Idx ldar, Idx ldac,
                                                            const zT *SB_RESTRICT b_, Idx ldbr,
                                                            Idx ldbc, zT *SB_RESTRICT c_, Idx ldcr,
                                                            Idx ldcc) {
                if (default_leading_dimensions) {
                    ldar = ldbr = ldcr = 1;
                    ldac = ldbc = ldcc = 3;
                }
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b = (const T *)(b_);
                T *SB_RESTRICT c = (T *)(c_);

                // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
                auto a012 = get_A_cols(a, ldar, ldac);
                for (Idx j = 0; j < N; ++j) {
                    vc8 b0 = get_B_col(b, j, ldbr, ldbc);
                    vc8 c0{0};
                    auto c1 = get_B_col(c, j, ldcr, ldcc);
                    for (int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = flip_ri_plus_1(b0);
                        c0 = stdx::fma(get_A_col<the_real>(a012[disp]), b0, c0);

                        b0 = flip_ri(b0);
                        c0 = stdx::fma(get_A_col<the_imag>(a012[disp]), b0, c0);
                    }
                    set_B_col(c0 + c1, c, j, ldcr, ldcc);
                }
            }
        };

        /// Implementation for
        ///  - complex float  on SIMD 512 bits (avx512)
        ///  - complex half   on SIMD 256 bits (avx, probably not useful)

        template <typename T, bool default_leading_dimensions = false> struct gemm_3x3_16parts {
            using zT = std::complex<T>;
            using vc8 = stdx::fixed_size_simd<T, 8>;
            using vi8 = stdx::fixed_size_simd<Idx, 8>;
            using vi16 = stdx::native_simd<Idx>;
            using vc16 = stdx::native_simd<T>;
            using vc32 = stdx::fixed_size_simd<T, 32>;

            static inline vc16 reshuffle(vc16 x, vi16 p) {
                return vc16([=](auto i) { return x[p[i]]; });
            }

            static constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, Idx reality) {
                return i * 2 * ldr + j * 2 * ldc + reality;
            }

            static inline vi8 get_A_cols_aux(Idx ldr, Idx ldc, Idx d) {
                return vi8([=](auto i) {
                    return (
                        Idx)(i < 6 ? get_disp_3x3(i / 2, (d + i / 2) % 3, ldr, ldc, i % 2)
                                   : (i == 6
                                          ? get_disp_3x3(4 / 2, (d + 4 / 2) % 3, ldr, ldc, 4 % 2)
                                          : get_disp_3x3(5 / 2, (d + 5 / 2) % 3, ldr, ldc, 5 % 2)));
                });
            }

            static inline vc32 get_A_cols(const T *SB_RESTRICT a, Idx ldr, Idx ldc) {
                auto p = concat(get_A_cols_aux(ldr, ldc, 0), //
                                get_A_cols_aux(ldr, ldc, 1), //
                                get_A_cols_aux(ldr, ldc, 2), //
                                vi8(Idx{0}));
                return vc32([=](auto i) { return a[p[i]]; });
            }

            static constexpr bool the_real = true;
            static constexpr bool the_imag = false;

            template <bool is_real> static inline vc16 get_A_col(vc32 va, Idx col) {
                auto r = is_real ? vc8([=](auto i) { return va[col * 8 + i / 2 * 2]; })
                                 : vc8([=](auto i) { return (i % 2 == 0 ? T{-1} : T{1}); }) *
                                       vc8([=](auto i) { return va[col * 8 + i / 2 * 2 + 1]; });
                return concat(r, r);
            }

            static constexpr Idx get_8_ri(Idx i, Idx ld) {
                return i < 6 ? ld * 2 * (i / 2) + i % 2 : ld * 2 * (5 / 2) + 5 % 2;
            }

            static inline vc16 get_B_col(const T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc) {
                const T *SB_RESTRICT bj = std::assume_aligned<sizeof(T) * 2>(b + ldc * 2 * j);
                return concat(vc8([=](auto i) { return bj[get_8_ri(i, ldr)]; }), vc8(T{0}));
            }

            static inline vi16 get_two_cols_perm(Idx ldr, Idx ldc) {
                return vi16(
                    [=](auto i) { return (Idx)(ldc * 2 * (i / 8) + get_8_ri(i % 8, ldr)); });
            }

            static inline vc16 get_B_two_cols(const T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc,
                                              vi16 p) {
                (void)ldr;
                const T *SB_RESTRICT bj = std::assume_aligned<sizeof(T) * 2>(b + ldc * 2 * j);
                return vc16([=](auto i) { return bj[p[i]]; });
            }

            static inline void set_B_col(vc16 x, T *SB_RESTRICT b, Idx ldr) {
                (void)ldr;
                T *SB_RESTRICT bj = std::assume_aligned<sizeof(T) * 2>(b);
                vi8([=](auto i) {
                    bj[get_8_ri(i, ldr)] = x[i];
                    return Idx{0};
                });
            }

            static inline void set_B_two_cols(vc16 x, T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc,
                                              vi16 p) {
                (void)ldr;
                T *SB_RESTRICT bj = std::assume_aligned<sizeof(T) * 2>(b + ldc * 2 * j);
                vi16([=](auto i) {
                    bj[p[i]] = x[i];
                    return Idx{0};
                });
            }

            /// It should return: x[{3, 2, 5, 4, 1, 0, 0, 0}]
            static inline vi16 flip_ri_plus_1() {
                return vi16([=](auto i) {
                    return (Idx)((i % 8) < 6 ? ((i / 2 + 1) % 3) * 2 + (i + 1) % 2
                                             : (((i / 8 * 8 + 5) / 2 + 1) % 3) * 2 +
                                                   ((i / 8 * 8 + 5) + 1) % 2);
                });
            }

            static inline vi16 flip_ri() {
                return vi16([=](auto i) {
                    return (Idx)((i % 8) < 6 ? i / 2 * 2 + (i + 1) % 2
                                             : (i / 8 * 8 + 5) / 2 * 2 + ((i / 8 * 8 + 5) + 1) % 2);
                });
            }

            static inline void gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_,
                                                            Idx ldar, Idx ldac,
                                                            const zT *SB_RESTRICT b_, Idx ldbr,
                                                            Idx ldbc, zT *SB_RESTRICT c_, Idx ldcr,
                                                            Idx ldcc) {
                if (default_leading_dimensions) {
                    ldar = ldbr = ldcr = 1;
                    ldac = ldbc = ldcc = 3;
                }
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b = (const T *)(b_);
                T *SB_RESTRICT c = (T *)(c_);

                // d{0,1}[i,j] = c{0,1}[i,j] + sum_0^k a{0,1}[i,k] * b{0,1}[k,j]
                auto a012 = get_A_cols(a, ldar, ldac);
                auto p_flip_ri = flip_ri();
                auto p_flip_ri_plus_1 = flip_ri_plus_1();
                if (N % 2 != 0) {
                    vc16 b0 = get_B_col(b, 0, ldbr, ldbc);
                    vc16 d0{0};
                    vc16 c0 = get_B_col(c, 0, ldcr, ldcc);
                    for (unsigned int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = reshuffle(b0, p_flip_ri_plus_1);
                        d0 = stdx::fma(get_A_col<the_real>(a012, disp), b0, d0);

                        b0 = reshuffle(b0, p_flip_ri);
                        d0 = stdx::fma(get_A_col<the_imag>(a012, disp), b0, d0);
                    }
                    set_B_col(c0 + d0, c, ldcr);
                }
                vi16 p_b = get_two_cols_perm(ldbr, ldbc);
                vi16 p_c = get_two_cols_perm(ldcr, ldcc);
                for (Idx j = N % 2; j < N; j += 2) {
                    vc16 b0 = get_B_two_cols(b, j, ldbr, ldbc, p_b);
                    vc16 d0{0};
                    vc16 c0 = get_B_two_cols(c, j, ldcr, ldcc, p_c);
                    for (unsigned int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = reshuffle(b0, p_flip_ri_plus_1);
                        d0 = stdx::fma(get_A_col<the_real>(a012, disp), b0, d0);

                        b0 = reshuffle(b0, p_flip_ri);
                        d0 = stdx::fma(get_A_col<the_imag>(a012, disp), b0, d0);
                    }
                    set_B_two_cols(c0 + d0, c, j, ldcr, ldcc, p_c);
                }
            }
        };

        /// Implementation for
        ///  - complex half   on SIMD 512 bits

        template <typename T, bool default_leading_dimensions = false> struct gemm_3x3_32parts {
            using zT = std::complex<T>;
            using vc8 = stdx::fixed_size_simd<T, 8>;
            using vc16 = stdx::fixed_size_simd<T, 16>;
            using vc32 = stdx::fixed_size_simd<T, 32>;

            static constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, Idx reality) {
                return i * 2 * ldr + j * 2 * ldc + reality;
            }

            static inline vc32 get_A_cols_aux(const T *SB_RESTRICT a[5], Idx ldr, Idx ldc, Idx d) {
                return vc32([=](auto i0) {
                    auto i = i0 % 6;
                    return i0 > 29
                               ? T{0}
                               : a[i0 / 6][get_disp_3x3(i / 2, (d + i / 2) % 3, ldr, ldc, i % 2)];
                });
            }

            static inline std::array<vc32, 3> get_A_cols(const T *SB_RESTRICT a[5], Idx ldr,
                                                         Idx ldc) {
                return {get_A_cols_aux(a, ldr, ldc, 0), //
                        get_A_cols_aux(a, ldr, ldc, 1), //
                        get_A_cols_aux(a, ldr, ldc, 2)};
            }

            static constexpr bool the_real = true;
            static constexpr bool the_imag = false;

            template <bool is_real> static inline vc32 get_A_col(vc32 va) {
                return is_real ? vc32([=](auto i) { return va[i / 2 * 2]; }) : vc32([=](auto i) {
                    return (i % 2 == 0 ? -va[i / 2 * 2 + 1] : va[i / 2 * 2 + 1]);
                });
            }

            static constexpr Idx get_8_ri(Idx i, Idx ld) {
                return i < 6 ? ld * 2 * (i / 2) + i % 2 : ld * 2 * (5 / 2) + 5 % 2;
            }

            static inline vc8 get_B_col(const T *SB_RESTRICT b, Idx j, Idx ldr, Idx ldc) {
                return vc8([=](auto i) { return b[ldc * 2 * j + get_8_ri(i, ldr)]; });
            }

            static inline vc32 get_B_col(const T *SB_RESTRICT b[5], Idx j, Idx ldr, Idx ldc) {
                return vc32([=](auto i) {
                    return i <= 29 ? b[i / 6][ldc * 2 * j + get_8_ri(i % 6, ldr)] : T{0};
                });
            }

            static inline void set_B_col(vc8 x0, vc32 x, T *SB_RESTRICT b, Idx j, Idx ldr,
                                         Idx ldc) {
                for (std::size_t i = 0; i < 6; ++i)
                    b[ldc * 2 * j + get_8_ri(i, ldr)] =
                        x0[i] + x[i] + x[i + 6] + x[i + 6 * 2] + x[i + 6 * 3] + x[i + 6 * 4];
            }

            static inline vc32 flip_ri(vc32 x) {
                return vc32([=](auto i) { return x[i / 2 * 2 + (i + 1) % 2]; });
            }

            /// It should return: x[{3, 2, 5, 4, 1, ...}]
            static inline vc32 flip_ri_plus_1(vc32 x) {
                return vc32([=](auto i) {
                    return x[i <= 29 ? i / 6 * 6 + (((i % 6) / 2 + 1) % 3) * 2 + (i + 1) % 2 : i];
                });
            }

            static inline void gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_[5],
                                                            Idx ldar, Idx ldac,
                                                            const zT *SB_RESTRICT b_[5], Idx ldbr,
                                                            Idx ldbc, const zT *SB_RESTRICT c_,
                                                            Idx ldcr, Idx ldcc) {
                if (default_leading_dimensions) {
                    ldar = ldbr = ldcr = 1;
                    ldac = ldbc = ldcc = 3;
                }
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a[5];
                const T *SB_RESTRICT b[5];
                for (std::size_t i = 0; i < 5; ++i) a[i] = (const T *SB_RESTRICT)a_[i];
                for (std::size_t i = 0; i < 5; ++i) b[i] = (const T *SB_RESTRICT)b_[i];
                T *SB_RESTRICT c = (T *)(c_);

                // c{0,1}[i,j] += sum_0^k a{0,1}[i,k] * b{0,1}[k,j]
                auto a012 = get_A_cols(a, ldar, ldac);
                for (Idx j = 0; j < N; ++j) {
                    vc32 b0 = get_B_col(b, j, ldbr, ldbc);
                    vc32 d{0};
                    auto c0 = get_B_col(c, j, ldcr, ldcc);
                    for (int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = flip_ri_plus_1(b0);
                        d = stdx::fma(get_A_col<the_real>(a012[disp]), b0, d);

                        b0 = flip_ri(b0);
                        d = stdx::fma(get_A_col<the_imag>(a012[disp]), b0, d);
                    }
                    set_B_col(c0, d, c, j, ldcr, ldcc);
                }
            }
        };

        /// Matrix-matrix multiplication, D = \sum_i A[i]*B[i] + C, where
        /// A[i] is a 3x3 matrix, B[i] is a 3xN matrix, and C and D are 3xN matrices.
        ///
        /// \param N: number of columns of B
        /// \param a: pointer to the first element of matrix A
        /// \param ldar: row leading dimension for matrix A
        /// \param ldac: column leading dimension for matrix A
        /// \param b: pointer to the first leading dimension for the matrix B
        /// \param ldbr: row leading dimension for matrix B
        /// \param ldbc: column leading dimension for matrix B
        /// \param ldbr: row leading dimension for matrix B
        /// \param ldbc: column leading dimension for matrix B
        /// \param c: pointer to the first element of matrix C
        /// \param ldcr: row leading dimension for matrix C
        /// \param ldcc: column leading dimension for matrix C
        /// \param d: pointer to the first element of matrix C
        /// \param lddr: row leading dimension for matrix D
        /// \param lddc: column leading dimension for matrix D

        template <typename T>
        inline void gemm_basic_3x3c_alpha1_beta1(Idx N, const T *SB_RESTRICT a, Idx ldar, Idx ldac,
                                                 const T *SB_RESTRICT b, Idx ldbr, Idx ldbc,
                                                 T *SB_RESTRICT c, Idx ldcr, Idx ldcc) {
            constexpr std::size_t native_size = stdx::native_simd<typename T::value_type>::size();
            bool default_leading_dimensions =
                (ldar == 1 && ldbr == 1 && ldcr == 1 && ldac == 3 && ldbc == 3 && ldcc == 3);
            if constexpr (native_size == 8) {
                if (default_leading_dimensions)
                    gemm_3x3_8parts<typename T::value_type, true>::gemm_basic_3x3c_alpha1_beta1(
                        N, a, ldar, ldac, b, ldbr, ldbc, c, ldcr, ldcc);
                else
                    gemm_3x3_8parts<typename T::value_type, false>::gemm_basic_3x3c_alpha1_beta1(
                        N, a, ldar, ldac, b, ldbr, ldbc, c, ldcr, ldcc);
            } else if constexpr (native_size == 16) {
                if (default_leading_dimensions)
                    gemm_3x3_16parts<typename T::value_type, true>::gemm_basic_3x3c_alpha1_beta1(
                        N, a, ldar, ldac, b, ldbr, ldbc, c, ldcr, ldcc);
                else
                    gemm_3x3_16parts<typename T::value_type, false>::gemm_basic_3x3c_alpha1_beta1(
                        N, a, ldar, ldac, b, ldbr, ldbc, c, ldcr, ldcc);
            }
        }
#endif // SUPERBBLAS_USE_XSIMD

#ifdef SUPERBBLAS_USE_FLOAT16
        template <bool Conj, typename T> inline T cond_conj(const T &t);

        template <> _Float16 cond_conj<false, _Float16>(const _Float16 &t) { return t; }
        template <>
        std::complex<_Float16>
        cond_conj<false, std::complex<_Float16>>(const std::complex<_Float16> &t) {
            return t;
        }
        template <>
        std::complex<_Float16>
        cond_conj<true, std::complex<_Float16>>(const std::complex<_Float16> &t) {
            return std::conj(t);
        }

        template <typename Idx, bool ConjA, bool ConjB, typename T>
        void gemm_basic(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar, Idx ldac,
                        const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta, T *SB_RESTRICT c,
                        Idx ldcr, Idx ldcc) {
            // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            bool beta_is_one = (beta == T{1});
            bool beta_is_zero = (beta == T{0});
            for (Idx i = 0; i < M; ++i) {
                for (Idx j = 0; j < N; ++j) {
                    T r{0};
                    for (Idx k = 0; k < K; ++k)
                        r += cond_conj<ConjA, T>(a[ldar * i + ldac * k]) *
                             cond_conj<ConjB, T>(b[ldbr * k + ldbc * j]);
                    if (beta_is_one)
                        c[ldcr * i + ldcc * j] += alpha * r;
                    else
                        c[ldcr * i + ldcc * j] =
                            (!beta_is_zero ? beta * c[ldcr * i + ldcc * j] : T{0}) + alpha * r;
                }
            }
        }

        template <unsigned int MM, unsigned int NN, unsigned int KK, typename Idx, bool ConjA,
                  bool ConjB, typename T>
        void gemm_blk_ikj_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar,
                                   Idx ldac, const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta,
                                   T *SB_RESTRICT c, Idx ldcr, Idx ldcc) {
            // c[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
            if (beta != T{1})
                for (Idx i = 0; i < i; ++i)
                    for (Idx j = 0; j < j; ++j)
                        c[ldcr * i + ldcc * j] =
                            (beta != T{0} ? beta * c[ldcr * i + ldcc * j] : T{0});
            for (Idx i = 0, ii = std::min(M, MM); i < M; i += ii, ii = std::min(M - i, MM)) {
                for (Idx k = 0, kk = std::min(K, KK); k < K; k += kk, kk = std::min(K - k, KK)) {
                    for (Idx j = 0, jj = std::min(N, NN); j < N;
                         j += jj, jj = std::min(N - j, NN)) {
                        if (ii == MM && jj == NN && kk == KK)
                            gemm_basic<Idx, ConjA, ConjB>(MM, NN, KK, alpha,
                                                          a + ldar * i + ldac * k, ldar, ldac,
                                                          b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                                          c + ldcr * i + ldcc * j, ldcr, ldcc);
                        else if (ii == MM && kk == KK)
                            gemm_basic<Idx, ConjA, ConjB>(MM, jj, KK, alpha,
                                                          a + ldar * i + ldac * k, ldar, ldac,
                                                          b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                                          c + ldcr * i + ldcc * j, ldcr, ldcc);
                        else
                            gemm_basic<Idx, ConjA, ConjB>(ii, jj, kk, alpha,
                                                          a + ldar * i + ldac * k, ldar, ldac,
                                                          b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                                          c + ldcr * i + ldcc * j, ldcr, ldcc);
                    }
                }
            }
        }

        template <unsigned int MM, unsigned int NN, typename Idx, bool ConjA, bool ConjB,
                  typename T>
        void gemm_blk_ikj_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar,
                                   Idx ldac, const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta,
                                   T *SB_RESTRICT c, Idx ldcr, Idx ldcc) {

            if (K <= 1 || (detail::is_complex<T>::value && MM > 1 && NN > 1))
                gemm_blk_ikj_nobuffer<MM, NN, 1, Idx, ConjA, ConjB, T>(
                    M, N, K, alpha, a, ldar, ldac, b, ldbr, ldbc, beta, c, ldcr, ldcc);
            else if (K <= 2)
                gemm_blk_ikj_nobuffer<MM, NN, 2, Idx, ConjA, ConjB, T>(
                    M, N, K, alpha, a, ldar, ldac, b, ldbr, ldbc, beta, c, ldcr, ldcc);
            else if (K <= 3)
                gemm_blk_ikj_nobuffer<MM, NN, 3, Idx, ConjA, ConjB, T>(
                    M, N, K, alpha, a, ldar, ldac, b, ldbr, ldbc, beta, c, ldcr, ldcc);
            else
                gemm_blk_ikj_nobuffer<MM, NN, 4, Idx, ConjA, ConjB, T>(
                    M, N, K, alpha, a, ldar, ldac, b, ldbr, ldbc, beta, c, ldcr, ldcc);
        }

        template <unsigned int MM, typename Idx, bool ConjA, bool ConjB, typename T>
        void gemm_blk_ikj_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar,
                                   Idx ldac, const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta,
                                   T *SB_RESTRICT c, Idx ldcr, Idx ldcc) {

            if (N <= 1)
                gemm_blk_ikj_nobuffer<MM, 1, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                                   ldbr, ldbc, beta, c, ldcr, ldcc);
            else if (N <= 2)
                gemm_blk_ikj_nobuffer<MM, 2, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                                   ldbr, ldbc, beta, c, ldcr, ldcc);
            else if (N <= 3)
                gemm_blk_ikj_nobuffer<MM, 3, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                                   ldbr, ldbc, beta, c, ldcr, ldcc);
            else
                gemm_blk_ikj_nobuffer<MM, 4, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                                   ldbr, ldbc, beta, c, ldcr, ldcc);
        }

        template <typename Idx, bool ConjA, bool ConjB, typename T>
        void gemm_blk_ikj_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar,
                                   Idx ldac, const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta,
                                   T *SB_RESTRICT c, Idx ldcr, Idx ldcc) {

            if (M <= 1)
                gemm_blk_ikj_nobuffer<1, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                               ldbr, ldbc, beta, c, ldcr, ldcc);
            else if (M <= 2)
                gemm_blk_ikj_nobuffer<2, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                               ldbr, ldbc, beta, c, ldcr, ldcc);
            else if (M <= 3)
                gemm_blk_ikj_nobuffer<3, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                               ldbr, ldbc, beta, c, ldcr, ldcc);
            else
                gemm_blk_ikj_nobuffer<4, Idx, ConjA, ConjB, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                               ldbr, ldbc, beta, c, ldcr, ldcc);
        }

        template <typename Idx, typename T>
        void gemm_blk_ikj_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, bool conja,
                                   Idx ldar, Idx ldac, const T *SB_RESTRICT b, bool conjb, Idx ldbr,
                                   Idx ldbc, T beta, T *SB_RESTRICT c, Idx ldcr, Idx ldcc) {
            if constexpr (detail::is_complex<T>::value) {
                if (!conja && !conjb)
                    gemm_blk_ikj_nobuffer<Idx, false, false, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                                ldbr, ldbc, beta, c, ldcr, ldcc);
                else if (conja && !conjb)
                    gemm_blk_ikj_nobuffer<Idx, true, false, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                               ldbr, ldbc, beta, c, ldcr, ldcc);
                else if (!conja && conjb)
                    gemm_blk_ikj_nobuffer<Idx, false, true, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                               ldbr, ldbc, beta, c, ldcr, ldcc);
                else
                    gemm_blk_ikj_nobuffer<Idx, true, true, T>(M, N, K, alpha, a, ldar, ldac, b,
                                                              ldbr, ldbc, beta, c, ldcr, ldcc);
            } else {
                gemm_blk_ikj_nobuffer<Idx, false, false, T>(M, N, K, alpha, a, ldar, ldac, b, ldbr,
                                                            ldbc, beta, c, ldcr, ldcc);
            }
        }
    }

    namespace detail {
        template <typename T,
                  typename std::enable_if<std::is_same<_Float16, T>::value ||
                                              std::is_same<std::complex<_Float16>, T>::value,
                                          bool>::type = true>
        inline void xgemm(char transa, char transb, int m, int n, int k, const T &alpha, const T *a,
                          int lda, const T *b, int ldb, const T &beta, T *c, int ldc, Cpu) {
            if (m == 0 || n == 0) return;
            bool ta = (transa != 'n' && transa != 'N');
            bool ca = (transa == 'c' || transa == 'C');
            bool tb = (transb != 'n' && transb != 'N');
            bool cb = (transb == 'c' || transb == 'C');

            detail_xp::gemm_blk_ikj_nobuffer<unsigned int>(m, n, k, alpha, a, ca, !ta ? 1 : lda,
                                                           !ta ? lda : 1, b, cb, !tb ? 1 : ldb,
                                                           !tb ? ldb : 1, beta, c, 1, ldc);
        }

        template <typename T,
                  typename std::enable_if<std::is_same<_Float16, T>::value ||
                                              std::is_same<std::complex<_Float16>, T>::value,
                                          bool>::type = true>
        inline void xgemv(char transa, int m, int n, T alpha, const T *a, int lda, const T *x,
                          int incx, T beta, T *y, int incy, Cpu) {
            if (m == 0) return;
            bool ta = (transa != 'n' && transa != 'N');
            bool ca = (transa == 'c' || transa == 'C');
            detail_xp::gemm_blk_ikj_nobuffer<unsigned int>(m, 1, n, alpha, a, ca, !ta ? 1 : lda,
                                                           !ta ? lda : 1, x, false, incx, m, beta,
                                                           y, incy, m);
        }
#endif // SUPERBBLAS_USE_FLOAT16
    }
}
#endif // __SUPERBBLAS_TENFUCKS__
