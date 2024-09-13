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

#ifndef __SUPERBBLAS_TENFUCKS_CPU__
#define __SUPERBBLAS_TENFUCKS_CPU__

#include "platform.h"
#ifndef SUPERBBLAS_LIB
#    ifdef SUPERBBLAS_USE_XSIMD
#        include "xsimd/xsimd.hpp"
#    elif __cplusplus >= 202002L
#        include <experimental/simd>
#    endif
#endif
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

#ifdef SUPERBBLAS_CREATING_LIB
/// Generate template instantiations for xgemm_alt_alpha1_beta1 function with template parameter T

#    define DECL_XGEMM_ALT_ALPHA1_BETA1_T(...)                                                     \
        EMIT REPLACE1(xgemm_alt_alpha1_beta1, superbblas::detail::xgemm_alt_alpha1_beta1<T>)       \
            REPLACE(T, SUPERBBLAS_COMPLEX_TYPES) template __VA_ARGS__;

/// Generate template instantiations for xgemm_alt_alpha1_beta1_perm function with template parameter T

#    define DECL_XGEMM_ALT_ALPHA1_BETA0_PERM_T(...)                                                \
        EMIT REPLACE1(xgemm_alt_alpha1_beta0_perm,                                                 \
                      superbblas::detail::xgemm_alt_alpha1_beta0_perm<T>)                          \
            REPLACE(T, SUPERBBLAS_COMPLEX_TYPES) template __VA_ARGS__;

/// Generate template instantiations for xgemm_alt_alpha1_beta1_perm function with template parameter T

#    define DECL_AVAILABLE_BSR_KRON_3x3_PERM_T(...)                                                \
        EMIT REPLACE1(available_bsr_kron_3x3_perm,                                                 \
                      superbblas::detail::available_bsr_kron_3x3_perm<T>)                          \
            REPLACE(T, SUPERBBLAS_COMPLEX_TYPES) template __VA_ARGS__;
#else
#    define DECL_XGEMM_ALT_ALPHA1_BETA1_T(...) __VA_ARGS__
#    define DECL_XGEMM_ALT_ALPHA1_BETA0_PERM_T(...) __VA_ARGS__
#    define DECL_AVAILABLE_BSR_KRON_3x3_PERM_T(...) __VA_ARGS__
#endif

namespace superbblas {
#ifndef SUPERBBLAS_LIB
    namespace detail_xp {

        using Idx = unsigned int;

        template <typename T> struct get_native_size {
            static constexpr std::size_t size = 0;
        };

        template <std::size_t Parts, typename T> struct gemm_3x3_in_parts {
            constexpr static bool supported = false;
        };

#    ifdef SUPERBBLAS_USE_XSIMD
#        define SUPERBBLAS_USE_SHORTCUTS_FOR_GEMM_3x3

        template <typename T> struct equivalent_int;
        template <> struct equivalent_int<float> {
            using type = uint32_t;
        };
        template <> struct equivalent_int<double> {
            using type = uint64_t;
        };

        template <typename B, typename B::value_type... Values>
        using constant = typename xsimd::batch_constant<typename B::value_type,
                                                        typename B::arch_type, Values...>;

        /// Implementation for
        ///  - complex double on SIMD 256 bits (avx)
        ///  - complex float  on SIMD 128 bits (sse2)

        template <typename T> struct gemm_3x3_in_parts<4, T> {
            constexpr static bool supported = true;

            using vc4 = typename xsimd::make_sized_batch<T, 4>::type;
            using vi4 = xsimd::batch<typename equivalent_int<T>::type, typename vc4::arch_type>;
            using zT = std::complex<T>;

            static constexpr bool the_real = true;
            static constexpr bool the_imag = false;

            static constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, bool reality) {
                return i * 2 * ldr + j * 2 * ldc + (reality ? 0 : 1);
            }

            static inline vc4 get_A_cols_aux(const T *SB_RESTRICT a, Idx ldr, Idx ldc, Idx d,
                                             bool reality) {
                return vc4::gather(a,
                                   vi4(                                                 //
                                       get_disp_3x3(0, (d + 0) % 3, ldr, ldc, reality), //
                                       get_disp_3x3(1, (d + 1) % 3, ldr, ldc, reality), //
                                       get_disp_3x3(2, (d + 2) % 3, ldr, ldc, reality), //
                                       get_disp_3x3(2, (d + 2) % 3, ldr, ldc, reality)));
            }

            static inline std::array<vc4, 6> get_A_cols(const T *SB_RESTRICT a, Idx ldr, Idx ldc) {
                return {get_A_cols_aux(a, ldr, ldc, 0, the_real), //
                        get_A_cols_aux(a, ldr, ldc, 0, the_imag), //
                        get_A_cols_aux(a, ldr, ldc, 1, the_real), //
                        get_A_cols_aux(a, ldr, ldc, 1, the_imag), //
                        get_A_cols_aux(a, ldr, ldc, 2, the_real), //
                        get_A_cols_aux(a, ldr, ldc, 2, the_imag)};
            }

            template <bool the_real> static inline vi4 get_4_ri(Idx ld) {
                constexpr Idx r{(the_real ? 0 : 1)};
                return vi4(ld * 2 * 0 + r, ld * 2 * 1 + r, ld * 2 * 2 + r, ld * 2 * 2 + r);
            }

            template <bool default_leading_dimensions = false>
            static inline void
            gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                                         const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                         zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
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
                using vi4_plus_1 = constant<vi4, 1, 2, 0, 0>;
                auto a012 = get_A_cols(a, ldar, ldac);
                auto vi4_r_b = get_4_ri<the_real>(ldbr);
                auto vi4_i_b = get_4_ri<the_imag>(ldbr);
                auto vi4_r_c = get_4_ri<the_real>(ldcr);
                auto vi4_i_c = get_4_ri<the_imag>(ldcr);
                if (default_leading_dimensions) vi4_r_c = vi4_r_b, vi4_i_c = vi4_i_b;
                for (Idx j = 0; j < N; ++j) {
                    auto b0r = vc4::gather(b + ldbc * 2 * j, vi4_r_b);
                    auto b0i = vc4::gather(b + ldbc * 2 * j, vi4_i_b);
                    auto c1r = vc4::gather(c + ldcc * 2 * j, vi4_r_c);
                    auto c1i = vc4::gather(c + ldcc * 2 * j, vi4_i_c);
                    vc4 c0r(T{0}), c0i(T{0});
                    for (int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) {
                            b0r = xsimd::swizzle(b0r, vi4_plus_1());
                            b0i = xsimd::swizzle(b0i, vi4_plus_1());
                        }
                        // c0r += ar * br
                        c0r = xsimd::fma(a012[disp * 2], b0r, c0r);
                        // c0i += ar * bi
                        c0i = xsimd::fma(a012[disp * 2], b0i, c0i);
                        // c0r -= ai * bi
                        c0r = xsimd::fnma(a012[disp * 2 + 1], b0i, c0r);
                        // c0i += ai * br
                        c0i = xsimd::fma(a012[disp * 2 + 1], b0r, c0i);
                    }
                    (c0r + c1r).scatter(c + ldcc * 2 * j, vi4_r_c);
                    (c0i + c1i).scatter(c + ldcc * 2 * j, vi4_i_c);
                }
            }

            static inline void gemm_basic_3x3c_alpha1_beta0_perm(
                Idx Nmats, Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                const zT *SB_RESTRICT b0_, const zT *SB_RESTRICT b1_, int bj01, int *SB_RESTRICT bj,
                Idx bjprod, Idx ldbr, Idx ldbc, int perm_size, int *SB_RESTRICT perm,
                int *SB_RESTRICT perm_sign, zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b_0 = (const T *)(b0_);
                const T *SB_RESTRICT b_1 = (const T *)(b1_);
                T *SB_RESTRICT c = (T *)(c_);
                bjprod *= 2;

                // c[i,j] = sum_{m=0:Nmats-1} sum_{k=0:2} a_m[i,k] * b_m[k,j/perm_size*perm_size+perm_m[j%perm_size]] * perm_sign_m[j%perm_size]
                using vi4_plus_1 = constant<vi4, 1, 2, 0, 0>;
                auto vi4_r_b = get_4_ri<the_real>(ldbr);
                auto vi4_i_b = get_4_ri<the_imag>(ldbr);
                auto vi4_r_c = get_4_ri<the_real>(ldcr);
                auto vi4_i_c = get_4_ri<the_imag>(ldcr);
                for (Idx j = 0; j < N; j += 4) {
                    vc4 c0r(T{0}), c0i(T{0});
                    vc4 c1r(T{0}), c1i(T{0});
                    vc4 c2r(T{0}), c2i(T{0});
                    vc4 c3r(T{0}), c3i(T{0});
                    for (Idx mat = 0; mat < Nmats; ++mat) {
                        auto a012 = get_A_cols(a + 2 * 3 * 3 * mat, ldar, ldac);
                        auto baseb = (bj[mat] < bj01 ? b_0 : b_1) +
                                     (bj[mat] - (bj[mat] < bj01 ? 0 : bj01)) * bjprod;
                        auto b0p = baseb + ldbc * 2 *
                                               (j / perm_size * perm_size +
                                                perm[mat * perm_size + j % perm_size]);
                        auto b0r = vc4::gather(b0p, vi4_r_b);
                        b0r = (perm_sign[mat * perm_size + j % perm_size] == 1 ? b0r : -b0r);
                        auto b0i = vc4::gather(b0p, vi4_i_b);
                        b0i = (perm_sign[mat * perm_size + j % perm_size] == 1 ? b0i : -b0i);

                        auto b1p = baseb + ldbc * 2 *
                                               ((j + 1) / perm_size * perm_size +
                                                perm[mat * perm_size + (j + 1) % perm_size]);
                        auto b1r = vc4::gather(b1p, vi4_r_b);
                        b1r = (perm_sign[mat * perm_size + (j + 1) % perm_size] == 1 ? b1r : -b1r);
                        auto b1i = vc4::gather(b1p, vi4_i_b);
                        b1i = (perm_sign[mat * perm_size + (j + 1) % perm_size] == 1 ? b1i : -b1i);

                        auto b2p = baseb + ldbc * 2 *
                                               ((j + 2) / perm_size * perm_size +
                                                perm[mat * perm_size + (j + 2) % perm_size]);
                        auto b2r = vc4::gather(b2p, vi4_r_b);
                        b2r = (perm_sign[mat * perm_size + (j + 2) % perm_size] == 1 ? b2r : -b2r);
                        auto b2i = vc4::gather(b2p, vi4_i_b);
                        b2i = (perm_sign[mat * perm_size + (j + 2) % perm_size] == 1 ? b2i : -b2i);

                        auto b3p = baseb + ldbc * 2 *
                                               ((j + 3) / perm_size * perm_size +
                                                perm[mat * perm_size + (j + 3) % perm_size]);
                        auto b3r = vc4::gather(b3p, vi4_r_b);
                        b3r = (perm_sign[mat * perm_size + (j + 3) % perm_size] == 1 ? b3r : -b3r);
                        auto b3i = vc4::gather(b3p, vi4_i_b);
                        b3i = (perm_sign[mat * perm_size + (j + 3) % perm_size] == 1 ? b3i : -b3i);

                        for (int disp = 0; disp < 3; ++disp) {
                            if (disp > 0) {
                                b0r = xsimd::swizzle(b0r, vi4_plus_1());
                                b0i = xsimd::swizzle(b0i, vi4_plus_1());
                                b1r = xsimd::swizzle(b1r, vi4_plus_1());
                                b1i = xsimd::swizzle(b1i, vi4_plus_1());
                                b2r = xsimd::swizzle(b2r, vi4_plus_1());
                                b2i = xsimd::swizzle(b2i, vi4_plus_1());
                                b3r = xsimd::swizzle(b3r, vi4_plus_1());
                                b3i = xsimd::swizzle(b3i, vi4_plus_1());
                            }
                            auto ar = a012[disp * 2];
                            c0r = xsimd::fma(ar, b0r, c0r);
                            c0i = xsimd::fma(ar, b0i, c0i);
                            c1r = xsimd::fma(ar, b1r, c1r);
                            c1i = xsimd::fma(ar, b1i, c1i);
                            c2r = xsimd::fma(ar, b2r, c2r);
                            c2i = xsimd::fma(ar, b2i, c2i);
                            c3r = xsimd::fma(ar, b3r, c3r);
                            c3i = xsimd::fma(ar, b3i, c3i);

                            auto ai = a012[disp * 2 + 1];
                            c0r = xsimd::fnma(ai, b0i, c0r);
                            c0i = xsimd::fma(ai, b0r, c0i);
                            c1r = xsimd::fnma(ai, b1i, c1r);
                            c1i = xsimd::fma(ai, b1r, c1i);
                            c2r = xsimd::fnma(ai, b2i, c2r);
                            c2i = xsimd::fma(ai, b2r, c2i);
                            c3r = xsimd::fnma(ai, b3i, c3r);
                            c3i = xsimd::fma(ai, b3r, c3i);
                        }
                    }
                    c0r.scatter(c + ldcc * 2 * j, vi4_r_c);
                    c0i.scatter(c + ldcc * 2 * j, vi4_i_c);
                    c1r.scatter(c + ldcc * 2 * (j + 1), vi4_r_c);
                    c1i.scatter(c + ldcc * 2 * (j + 1), vi4_i_c);
                    c2r.scatter(c + ldcc * 2 * (j + 2), vi4_r_c);
                    c2i.scatter(c + ldcc * 2 * (j + 2), vi4_i_c);
                    c3r.scatter(c + ldcc * 2 * (j + 3), vi4_r_c);
                    c3i.scatter(c + ldcc * 2 * (j + 3), vi4_i_c);
                }
            }
        };

        /// Implementation for
        ///  - complex double on SIMD 512 bits (avx512)
        ///  - complex float  on SIMD 256 bits (avx)
        ///  - complex half   on SIMD 128 bits (sse2?, probably not useful, although it may be possible to take advantage of the intel instruction to convert 4 half precision into float)

        template <typename T> struct gemm_3x3_in_parts<8, T> {
            constexpr static bool supported = true;

            using vc8 = typename xsimd::make_sized_batch<T, 8>::type;
            using vi8 = xsimd::batch<typename equivalent_int<T>::type, typename vc8::arch_type>;
            using zT = std::complex<T>;

            static constexpr bool the_real = true;
            static constexpr bool the_imag = false;

            static constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, bool reality) {
                return i * 2 * ldr + j * 2 * ldc + (reality ? 0 : 1);
            }

            static inline vc8 get_A_cols_aux(const T *SB_RESTRICT a, Idx ldr, Idx ldc, Idx d) {
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

            static inline std::array<vc8, 3> get_A_cols(const T *SB_RESTRICT a, Idx ldr, Idx ldc) {
                return {get_A_cols_aux(a, ldr, ldc, 0), //
                        get_A_cols_aux(a, ldr, ldc, 1), //
                        get_A_cols_aux(a, ldr, ldc, 2)};
            }

            template <bool the_real> static inline vc8 get_A_col(vc8 va) {
                return the_real ? xsimd::shuffle(va, va, constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>())
                                : xsimd::shuffle(
                                      xsimd::neg(va), va,
                                      constant<vi8, 1, 8 + 1, 3, 8 + 3, 5, 8 + 5, 8 + 5, 8 + 5>());
            }

            static inline vi8 get_8_ri(Idx ld) {
                return vi8(ld * 2 * 0, ld * 2 * 0 + 1, ld * 2 * 1, ld * 2 * 1 + 1, ld * 2 * 2,
                           ld * 2 * 2 + 1, ld * 2 * 2 + 1, ld * 2 * 2 + 1);
            }

            template <bool default_leading_dimensions = false>
            static inline void
            gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                                         const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                         zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
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
                using vi8_flip_ri = constant<vi8, 1, 0, 3, 2, 5, 4, 4, 4>;
                using vi8_flip_and_plus_1 = constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;
                auto a012 = get_A_cols(a, ldar, ldac);
                auto vi8_ri_b = get_8_ri(ldbr);
                auto vi8_ri_c = get_8_ri(ldcr);
                if (default_leading_dimensions) vi8_ri_c = vi8_ri_b;
                for (Idx j = 0; j < N; ++j) {
                    auto b0 = vc8::gather(b + ldbc * 2 * j, vi8_ri_b);
                    auto c1 = vc8::gather(c + ldcc * 2 * j, vi8_ri_c);
                    vc8 c0(T{0});
                    for (int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                        c0 = xsimd::fma(get_A_col<the_real>(a012[disp]), b0, c0);

                        b0 = xsimd::swizzle(b0, vi8_flip_ri());
                        c0 = xsimd::fma(get_A_col<the_imag>(a012[disp]), b0, c0);
                    }
                    (c0 + c1).scatter(c + ldcc * 2 * j, vi8_ri_c);
                }
            }

            static inline void gemm_basic_3x3c_alpha1_beta0_perm(
                Idx Nmats, Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                const zT *SB_RESTRICT b0_, const zT *SB_RESTRICT b1_, int bj01, int *SB_RESTRICT bj,
                Idx bjprod, Idx ldbr, Idx ldbc, int perm_size, int *SB_RESTRICT perm,
                int *SB_RESTRICT perm_sign, zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b_0 = (const T *)(b0_);
                const T *SB_RESTRICT b_1 = (const T *)(b1_);
                T *SB_RESTRICT c = (T *)(c_);
                bjprod *= 2;

                // c[i,j] = sum_{m=0:Nmats-1} sum_{k=0:2} a_m[i,k] * b_m[k,j/perm_size*perm_size+perm_m[j%perm_size]] * perm_sign_m[j%perm_size]
                using vi8_flip_ri = constant<vi8, 1, 0, 3, 2, 5, 4, 4, 4>;
                using vi8_flip_and_plus_1 = constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;
                auto vi8_ri_b = get_8_ri(ldbr);
                auto vi8_ri_c = get_8_ri(ldcr);
                for (Idx j = 0; j < N; j += 4) {
                    vc8 c0(T{0});
                    vc8 c1(T{0});
                    vc8 c2(T{0});
                    vc8 c3(T{0});
                    for (Idx mat = 0; mat < Nmats; ++mat) {
                        auto a012 = get_A_cols(a + 2 * 3 * 3 * mat, ldar, ldac);
                        auto baseb = (bj[mat] < bj01 ? b_0 : b_1) +
                                     (bj[mat] - (bj[mat] < bj01 ? 0 : bj01)) * bjprod;
                        auto b0 = vc8::gather(baseb + ldbc * 2 *
                                                          (j / perm_size * perm_size +
                                                           perm[mat * perm_size + j % perm_size]),
                                              vi8_ri_b);
                        b0 = (perm_sign[mat * perm_size + j % perm_size] == 1 ? b0 : -b0);
                        auto b1 =
                            vc8::gather(baseb + ldbc * 2 *
                                                    ((j + 1) / perm_size * perm_size +
                                                     perm[mat * perm_size + (j + 1) % perm_size]),
                                        vi8_ri_b);
                        b1 = (perm_sign[mat * perm_size + (j + 1) % perm_size] == 1 ? b1 : -b1);
                        auto b2 =
                            vc8::gather(baseb + ldbc * 2 *
                                                    ((j + 2) / perm_size * perm_size +
                                                     perm[mat * perm_size + (j + 2) % perm_size]),
                                        vi8_ri_b);
                        b2 = (perm_sign[mat * perm_size + (j + 2) % perm_size] == 1 ? b2 : -b2);
                        auto b3 =
                            vc8::gather(baseb + ldbc * 2 *
                                                    ((j + 3) / perm_size * perm_size +
                                                     perm[mat * perm_size + (j + 3) % perm_size]),
                                        vi8_ri_b);
                        b3 = (perm_sign[mat * perm_size + (j + 3) % perm_size] == 1 ? b3 : -b3);
                        for (int disp = 0; disp < 3; ++disp) {
                            if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                            if (disp > 0) b1 = xsimd::swizzle(b1, vi8_flip_and_plus_1());
                            if (disp > 0) b2 = xsimd::swizzle(b2, vi8_flip_and_plus_1());
                            if (disp > 0) b3 = xsimd::swizzle(b3, vi8_flip_and_plus_1());
                            auto ar = get_A_col<the_real>(a012[disp]);
                            c0 = xsimd::fma(ar, b0, c0);
                            c1 = xsimd::fma(ar, b1, c1);
                            c2 = xsimd::fma(ar, b2, c2);
                            c3 = xsimd::fma(ar, b3, c3);

                            b0 = xsimd::swizzle(b0, vi8_flip_ri());
                            b1 = xsimd::swizzle(b1, vi8_flip_ri());
                            b2 = xsimd::swizzle(b2, vi8_flip_ri());
                            b3 = xsimd::swizzle(b3, vi8_flip_ri());
                            auto ai = get_A_col<the_imag>(a012[disp]);
                            c0 = xsimd::fma(ai, b0, c0);
                            c1 = xsimd::fma(ai, b1, c1);
                            c2 = xsimd::fma(ai, b2, c2);
                            c3 = xsimd::fma(ai, b3, c3);
                        }
                    }
                    c0.scatter(c + ldcc * 2 * j, vi8_ri_c);
                    c1.scatter(c + ldcc * 2 * (j + 1), vi8_ri_c);
                    c2.scatter(c + ldcc * 2 * (j + 2), vi8_ri_c);
                    c3.scatter(c + ldcc * 2 * (j + 3), vi8_ri_c);
                }
            }
        };

        /// Implementation for
        ///  - complex float  on SIMD 512 bits (avx512)
        ///  - complex half   on SIMD 256 bits (avx, probably not useful)

        template <typename T> struct gemm_3x3_in_parts<16, T> {
            constexpr static bool supported = true;

            using vc8 = typename xsimd::make_sized_batch<T, 8>::type;
            using vi8 = xsimd::batch<typename equivalent_int<T>::type, typename vc8::arch_type>;
            using vc16 = typename xsimd::make_sized_batch<T, 16>::type;
            using vi16 = xsimd::batch<typename equivalent_int<T>::type, typename vc16::arch_type>;
            using zT = std::complex<T>;

            static constexpr bool the_real = true;
            static constexpr bool the_imag = false;

            static constexpr Idx get_disp_3x3(Idx i, Idx j, Idx ldr, Idx ldc, bool reality) {
                return i * 2 * ldr + j * 2 * ldc + (reality ? 0 : 1);
            }

            static inline vc8 get_A_cols_aux(const T *SB_RESTRICT a, Idx ldr, Idx ldc, Idx d) {
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

            static inline std::array<vc8, 3> get_A_cols(const T *SB_RESTRICT a, Idx ldr, Idx ldc) {
                return {get_A_cols_aux(a, ldr, ldc, 0), //
                        get_A_cols_aux(a, ldr, ldc, 1), //
                        get_A_cols_aux(a, ldr, ldc, 2)};
            }

            template <bool the_real> static inline vc8 get_A_col(vc8 va) {
                return the_real ? xsimd::shuffle(va, va, constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>())
                                : xsimd::shuffle(
                                      xsimd::neg(va), va,
                                      constant<vi8, 1, 8 + 1, 3, 8 + 3, 5, 8 + 5, 8 + 5, 8 + 5>());
            }

            template <bool the_real> static inline vc16 get_A_col_double(vc8 va) {
                auto x = get_A_col<the_real>(va);
                alignas(vc16::arch_type::alignment()) T buffer[16];
                x.store_aligned(&buffer[0]);
                x.store_aligned(&buffer[8]);
                return vc16::load_aligned(&buffer[0]);
            }

            static inline vi8 get_8_ri(Idx ld) {
                return vi8(ld * 2 * 0, ld * 2 * 0 + 1, ld * 2 * 1, ld * 2 * 1 + 1, ld * 2 * 2,
                           ld * 2 * 2 + 1, ld * 2 * 2 + 1, ld * 2 * 2 + 1);
            }

            static inline vi16 get_16_ri(Idx ldr, Idx ldc) {
                return vi16(ldr * 2 * 0,               //
                            ldr * 2 * 0 + 1,           //
                            ldr * 2 * 1,               //
                            ldr * 2 * 1 + 1,           //
                            ldr * 2 * 2,               //
                            ldr * 2 * 2 + 1,           //
                            ldr * 2 * 2 + 1,           //
                            ldr * 2 * 2 + 1,           //
                            ldc * 2 + ldr * 2 * 0,     //
                            ldc * 2 + ldr * 2 * 0 + 1, //
                            ldc * 2 + ldr * 2 * 1,     //
                            ldc * 2 + ldr * 2 * 1 + 1, //
                            ldc * 2 + ldr * 2 * 2,     //
                            ldc * 2 + ldr * 2 * 2 + 1, //
                            ldc * 2 + ldr * 2 * 2 + 1, //
                            ldc * 2 + ldr * 2 * 2 + 1);
            }

            template <typename VI8>
            static inline vc16 get_B_col_double(const T *SB_RESTRICT b0, int sign0,
                                                const T *SB_RESTRICT b1, int sign1, VI8 ri) {
                alignas(vc16::arch_type::alignment()) T buffer[16];
                auto x0 = vc8::gather(b0, ri);
                auto x1 = vc8::gather(b1, ri);
                (sign0 == 1 ? x0 : -x0).store_aligned(&buffer[0]);
                (sign1 == 1 ? x1 : -x1).store_aligned(&buffer[8]);
                return vc16::load_aligned(&buffer[0]);
            }

            template <bool default_leading_dimensions = false>
            static inline void
            gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                                         const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                         zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
                if (default_leading_dimensions) {
                    ldar = ldbr = ldcr = 1;
                    ldac = ldbc = ldcc = 3;
                }
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b = (const T *)(b_);
                T *SB_RESTRICT c = (T *)(c_);

                using vi8_flip_ri = constant<vi8, 1, 0, 3, 2, 5, 4, 4, 4>;
                using vi8_flip_and_plus_1 = constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;
                auto a012 = get_A_cols(a, ldar, ldac);
                if (N % 2 != 0) {
                    auto vi8_ri_b = get_8_ri(ldbr);
                    auto vi8_ri_c = get_8_ri(ldcr);
                    if (default_leading_dimensions) vi8_ri_c = vi8_ri_b;
                    auto b0 = vc8::gather(b, vi8_ri_b);
                    auto c1 = vc8::gather(c, vi8_ri_c);
                    vc8 c0(T{0});
                    for (int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
                        c0 = xsimd::fma(get_A_col<the_real>(a012[disp]), b0, c0);

                        b0 = xsimd::swizzle(b0, vi8_flip_ri());
                        c0 = xsimd::fma(get_A_col<the_imag>(a012[disp]), b0, c0);
                    }
                    (c0 + c1).scatter(c, vi8_ri_c);
                }
                using vi16_flip_ri =
                    constant<vi16, 1, 0, 3, 2, 5, 4, 4, 4, //
                             8 + 1, 8 + 0, 8 + 3, 8 + 2, 8 + 5, 8 + 4, 8 + 4, 8 + 4>;
                using vi16_flip_and_plus_1 =
                    constant<vi16, 3, 2, 5, 4, 1, 0, 0, 0, //
                             8 + 3, 8 + 2, 8 + 5, 8 + 4, 8 + 1, 8 + 0, 8 + 0, 8 + 0>;
                auto vi16_ri_b = get_16_ri(ldbr, ldbc);
                auto vi16_ri_c = get_16_ri(ldcr, ldcc);
                if (default_leading_dimensions) vi16_ri_c = vi16_ri_b;
                for (Idx j = N % 2; j < N; j += 2) {
                    auto b0 = vc16::gather(b + ldbc * 2 * j, vi16_ri_b);
                    auto c1 = vc16::gather(c + ldcc * 2 * j, vi16_ri_c);
                    vc16 c0(T{0});
                    for (int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = xsimd::swizzle(b0, vi16_flip_and_plus_1());
                        c0 = xsimd::fma(get_A_col_double<the_real>(a012[disp]), b0, c0);

                        b0 = xsimd::swizzle(b0, vi16_flip_ri());
                        c0 = xsimd::fma(get_A_col_double<the_imag>(a012[disp]), b0, c0);
                    }
                    (c0 + c1).scatter(c + ldcc * 2 * j, vi16_ri_c);
                }
            }

            static inline void gemm_basic_3x3c_alpha1_beta0_perm(
                Idx Nmats, Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                const zT *SB_RESTRICT b0_, const zT *SB_RESTRICT b1_, int bj01, int *SB_RESTRICT bj,
                Idx bjprod, Idx ldbr, Idx ldbc, int perm_size, int *SB_RESTRICT perm,
                int *SB_RESTRICT perm_sign, zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b_0 = (const T *)(b0_);
                const T *SB_RESTRICT b_1 = (const T *)(b1_);
                T *SB_RESTRICT c = (T *)(c_);
                bjprod *= 2;

                // TEMP!!!
                //ldar = ldbr = ldcr = 1;
                //ldac = ldbc = ldcc = 3;
                //perm_size = 4;

                auto vi8_ri_b = get_8_ri(ldbr);
                using vi16_flip_ri =
                    constant<vi16, 1, 0, 3, 2, 5, 4, 4, 4, //
                             8 + 1, 8 + 0, 8 + 3, 8 + 2, 8 + 5, 8 + 4, 8 + 4, 8 + 4>;
                using vi16_flip_and_plus_1 =
                    constant<vi16, 3, 2, 5, 4, 1, 0, 0, 0, //
                             8 + 3, 8 + 2, 8 + 5, 8 + 4, 8 + 1, 8 + 0, 8 + 0, 8 + 0>;
                auto vi16_ri_c = get_16_ri(ldcr, ldcc);
                for (Idx j = 0; j < N; j += 16) {
                    vc16 c0(T{0});
                    vc16 c2(T{0});
                    vc16 c4(T{0});
                    vc16 c6(T{0});
                    vc16 c8(T{0});
                    vc16 c10(T{0});
                    vc16 c12(T{0});
                    vc16 c14(T{0});
                    for (Idx mat = 0; mat < Nmats; ++mat) {
                        auto a012 = get_A_cols(a + 2 * 3 * 3 * mat, ldar, ldac);
                        auto baseb = (bj[mat] < bj01 ? b_0 : b_1) +
                                     (bj[mat] - (bj[mat] < bj01 ? 0 : bj01)) * bjprod;
                        auto b0 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        (j / perm_size * perm_size +
                                         perm[mat * perm_size + j % perm_size]),
                            perm_sign[mat * perm_size + j % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 1) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 1) % perm_size]),
                            perm_sign[mat * perm_size + (j + 1) % perm_size], vi8_ri_b);
                        auto b2 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        ((j + 2) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 2) % perm_size]),
                            perm_sign[mat * perm_size + (j + 2) % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 3) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 3) % perm_size]),
                            perm_sign[mat * perm_size + (j + 3) % perm_size], vi8_ri_b);
                        auto b4 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        ((j + 4) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 4) % perm_size]),
                            perm_sign[mat * perm_size + (j + 4) % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 5) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 5) % perm_size]),
                            perm_sign[mat * perm_size + (j + 5) % perm_size], vi8_ri_b);
                        auto b6 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        ((j + 6) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 6) % perm_size]),
                            perm_sign[mat * perm_size + (j + 6) % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 7) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 7) % perm_size]),
                            perm_sign[mat * perm_size + (j + 7) % perm_size], vi8_ri_b);
                        auto b8 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        ((j + 8) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 8) % perm_size]),
                            perm_sign[mat * perm_size + (j + 8) % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 9) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 9) % perm_size]),
                            perm_sign[mat * perm_size + (j + 9) % perm_size], vi8_ri_b);
                        auto b10 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        ((j + 10) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 10) % perm_size]),
                            perm_sign[mat * perm_size + (j + 10) % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 11) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 11) % perm_size]),
                            perm_sign[mat * perm_size + (j + 11) % perm_size], vi8_ri_b);
                        auto b12 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        ((j + 12) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 12) % perm_size]),
                            perm_sign[mat * perm_size + (j + 12) % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 13) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 13) % perm_size]),
                            perm_sign[mat * perm_size + (j + 13) % perm_size], vi8_ri_b);
                        auto b14 = get_B_col_double(
                            baseb + ldbc * 2 *
                                        ((j + 14) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 14) % perm_size]),
                            perm_sign[mat * perm_size + (j + 14) % perm_size],
                            baseb + ldbc * 2 *
                                        ((j + 15) / perm_size * perm_size +
                                         perm[mat * perm_size + (j + 15) % perm_size]),
                            perm_sign[mat * perm_size + (j + 15) % perm_size], vi8_ri_b);

                        for (int disp = 0; disp < 3; ++disp) {
                            if (disp > 0) b0 = xsimd::swizzle(b0, vi16_flip_and_plus_1());
                            if (disp > 0) b2 = xsimd::swizzle(b2, vi16_flip_and_plus_1());
                            if (disp > 0) b4 = xsimd::swizzle(b4, vi16_flip_and_plus_1());
                            if (disp > 0) b6 = xsimd::swizzle(b6, vi16_flip_and_plus_1());
                            if (disp > 0) b8 = xsimd::swizzle(b8, vi16_flip_and_plus_1());
                            if (disp > 0) b10 = xsimd::swizzle(b10, vi16_flip_and_plus_1());
                            if (disp > 0) b12 = xsimd::swizzle(b12, vi16_flip_and_plus_1());
                            if (disp > 0) b14 = xsimd::swizzle(b14, vi16_flip_and_plus_1());
                            auto ar = get_A_col_double<the_real>(a012[disp]);
                            c0 = xsimd::fma(ar, b0, c0);
                            c2 = xsimd::fma(ar, b2, c2);
                            c4 = xsimd::fma(ar, b4, c4);
                            c6 = xsimd::fma(ar, b6, c6);
                            c8 = xsimd::fma(ar, b8, c8);
                            c10 = xsimd::fma(ar, b10, c10);
                            c12 = xsimd::fma(ar, b12, c12);
                            c14 = xsimd::fma(ar, b14, c14);

                            b0 = xsimd::swizzle(b0, vi16_flip_ri());
                            b2 = xsimd::swizzle(b2, vi16_flip_ri());
                            b4 = xsimd::swizzle(b4, vi16_flip_ri());
                            b6 = xsimd::swizzle(b6, vi16_flip_ri());
                            b8 = xsimd::swizzle(b8, vi16_flip_ri());
                            b10 = xsimd::swizzle(b10, vi16_flip_ri());
                            b12 = xsimd::swizzle(b12, vi16_flip_ri());
                            b14 = xsimd::swizzle(b14, vi16_flip_ri());
                            auto ai = get_A_col_double<the_imag>(a012[disp]);
                            c0 = xsimd::fma(ai, b0, c0);
                            c2 = xsimd::fma(ai, b2, c2);
                            c4 = xsimd::fma(ai, b4, c4);
                            c6 = xsimd::fma(ai, b6, c6);
                            c8 = xsimd::fma(ai, b8, c8);
                            c10 = xsimd::fma(ai, b10, c10);
                            c12 = xsimd::fma(ai, b12, c12);
                            c14 = xsimd::fma(ai, b14, c14);
                        }
                    }
                    c0.scatter(c + ldcc * 2 * j, vi16_ri_c);
                    c2.scatter(c + ldcc * 2 * (j + 2), vi16_ri_c);
                    c4.scatter(c + ldcc * 2 * (j + 4), vi16_ri_c);
                    c6.scatter(c + ldcc * 2 * (j + 6), vi16_ri_c);
                    c8.scatter(c + ldcc * 2 * (j + 8), vi16_ri_c);
                    c10.scatter(c + ldcc * 2 * (j + 10), vi16_ri_c);
                    c12.scatter(c + ldcc * 2 * (j + 12), vi16_ri_c);
                    c14.scatter(c + ldcc * 2 * (j + 14), vi16_ri_c);
                }
            }
        };

        template <typename T> struct get_native_size<std::complex<T>> {
            static constexpr std::size_t size = xsimd::batch<T>::size;
        };

#    elif __cpp_lib_experimental_parallel_simd >= 201803
#        define SUPERBBLAS_USE_SHORTCUTS_FOR_GEMM_3x3

        /// Implementation based on experimental simd C++ interface

        namespace stdx = std::experimental;

        /// Implementation for
        ///  - complex double on SIMD 512 bits (avx512)
        ///  - complex float  on SIMD 256 bits (avx)
        ///  - complex half   on SIMD 128 bits (sse2?, probably not useful, although it may be possible to take advantage of the intel instruction to convert 4 half precision into float)

        template <typename T> struct gemm_3x3_in_parts<8, T> {
            constexpr static bool supported = true;

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

            template <bool default_leading_dimensions = false>
            static inline void
            gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                                         const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                         zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
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

            static inline void gemm_basic_3x3c_alpha1_beta0_perm(
                Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac, const zT *SB_RESTRICT b_,
                Idx ldbr, Idx ldbc, int perm_size, int *SB_RESTRICT perm,
                int *SB_RESTRICT perm_sign, zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b = (const T *)(b_);
                T *SB_RESTRICT c = (T *)(c_);

                // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j/perm_size*perm_size+perm[j%perm_size]] * perm_sign[j%perm_size]
                auto a012 = get_A_cols(a, ldar, ldac);
                for (Idx j = 0; j < N; ++j) {
                    vc8 b0 =
                        get_B_col(b, j / perm_size * perm_size + perm[j % perm_size], ldbr, ldbc);
                    vc8 c0{0};
                    auto c1 = get_B_col(c, j, ldcr, ldcc);
                    for (int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = flip_ri_plus_1(b0);
                        c0 = stdx::fma(get_A_col<the_real>(a012[disp]), b0, c0);

                        b0 = flip_ri(b0);
                        c0 = stdx::fma(get_A_col<the_imag>(a012[disp]), b0, c0);
                    }
                    set_B_col(perm_sign[j % perm_size] == 1 ? c1 + c0 : c1 - c0, c, j, ldcr, ldcc);
                }
            }
        };

        /// Implementation for
        ///  - complex float  on SIMD 512 bits (avx512)
        ///  - complex half   on SIMD 256 bits (avx, probably not useful)

        template <typename T> struct gemm_3x3_in_parts<16, T> {
            constexpr static bool supported = true;

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

            static inline vc16 get_B_two_cols(const T *SB_RESTRICT b0, int sign0,
                                              const T *SB_RESTRICT b1, int sign1, Idx ldr, vi8 p) {
                auto v0 = vc8([=](auto i) { return b0[p[i]]; });
                auto v1 = vc8([=](auto i) { return b1[p[i]]; });
                return concat(sign0 == 0 ? v0 : -v0, sign1 == 0 ? v1 : -v1);
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

            template <bool default_leading_dimensions = false>
            static inline void
            gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                                         const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                         zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
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

            static inline void
            gemm_basic_3x3c_alpha1_beta0_perm(Idx N, const zT *SB_RESTRICT a_, Idx ldar, Idx ldac,
                                              const zT *SB_RESTRICT b_, Idx ldbr, Idx ldbc,
                                              int perm_size, int *SB_RESTRICT perm,
                                              int *SB_RESTRICT perm_sign,

                                              zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
                //constexpr Idx M = 3;
                //constexpr Idx K = 3;
                const T *SB_RESTRICT a = (const T *)(a_);
                const T *SB_RESTRICT b = (const T *)(b_);
                T *SB_RESTRICT c = (T *)(c_);

                // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j/perm_size*perm_size+perm[j%perm_size]] * perm_sign[j%perm_size]
                auto a012 = get_A_cols(a, ldar, ldac);
                auto p_flip_ri = flip_ri();
                auto p_flip_ri_plus_1 = flip_ri_plus_1();
                if (N % 2 != 0) {
                    vc16 b0 = get_B_col(b, perm[0], ldbr, ldbc);
                    vc16 d0{0};
                    vc16 c0 = get_B_col(c, 0, ldcr, ldcc);
                    for (unsigned int disp = 0; disp < 3; ++disp) {
                        if (disp > 0) b0 = reshuffle(b0, p_flip_ri_plus_1);
                        d0 = stdx::fma(get_A_col<the_real>(a012, disp), b0, d0);

                        b0 = reshuffle(b0, p_flip_ri);
                        d0 = stdx::fma(get_A_col<the_imag>(a012, disp), b0, d0);
                    }
                    set_B_col(perm_sign[0] == 1 ? c0 + d0 : c0 - d0, c, ldcr);
                }
                vi8 p_b = get_B_col_perm(ldbr);
                vi16 p_c = get_two_cols_perm(ldcr, ldcc);
                for (Idx j = N % 2; j < N; j += 2) {
                    vc16 b0 = get_B_two_cols(
                        b + ldbc * (j / perm_size * perm_size + perm[j % perm_size]),
                        perm_sign[j % perm_size],
                        b + ldbc * ((j + 1) / perm_size * perm_size + perm[(j + 1) % perm_size]),
                        perm_sign[(j + 1) perm_size], p_b);
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

        template <typename T> struct gemm_3x3_in_parts<32, T> {
            constexpr static bool supported = true;

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

            template <bool default_leading_dimensions = false>
            static inline void
            gemm_basic_3x3c_alpha1_beta1(Idx N, const zT *SB_RESTRICT a_[5], Idx ldar, Idx ldac,
                                         const zT *SB_RESTRICT b_[5], Idx ldbr, Idx ldbc,
                                         const zT *SB_RESTRICT c_, Idx ldcr, Idx ldcc) {
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

        template <typename T> struct get_native_size<std::complex<T>> {
            static constexpr std::size_t size = stdx::native_simd<T>::size();
        };
#    endif // SUPERBBLAS_USE_XSIMD

        template <typename T, bool supported = (get_native_size<T>::size >= 4)>
        struct gemm_basic_3x3c_alpha1_beta1_wrapper;

#    ifdef SUPERBBLAS_USE_SHORTCUTS_FOR_GEMM_3x3
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
            constexpr std::size_t native_size = get_native_size<T>::size;
            bool default_leading_dimensions =
                (ldar == 1 && ldbr == 1 && ldcr == 1 && ldac == 3 && ldbc == 3 && ldcc == 3);
            if (default_leading_dimensions)
                gemm_3x3_in_parts<native_size, typename T::value_type>::
                    template gemm_basic_3x3c_alpha1_beta1<true>(N, a, ldar, ldac, b, ldbr, ldbc, c,
                                                                ldcr, ldcc);
            else
                gemm_3x3_in_parts<native_size, typename T::value_type>::
                    template gemm_basic_3x3c_alpha1_beta1<false>(N, a, ldar, ldac, b, ldbr, ldbc, c,
                                                                 ldcr, ldcc);
        }

        template <typename T> struct gemm_basic_3x3c_alpha1_beta1_wrapper<T, true> {
            static constexpr bool available() { return true; }

            static inline void func(char transa, char transb, int m, int n, int k, const T *a,
                                    int lda, const T *b, int ldb, T *c, int ldc) {
                if (m == 0 || n == 0) return;

                bool ta = (transa != 'n' && transa != 'N');
                bool tb = (transb != 'n' && transb != 'N');
                if (k == 3) {
                    if (m == 3) {
                        gemm_basic_3x3c_alpha1_beta1(n, a, !ta ? 1 : lda, !ta ? lda : 1, b,
                                                     !tb ? 1 : ldb, !tb ? ldb : 1, c, 1, ldc);
                        return;
                    } else if (n == 3) {
                        gemm_basic_3x3c_alpha1_beta1(m, b, tb ? 1 : ldb, tb ? ldb : 1, a,
                                                     ta ? 1 : lda, ta ? lda : 1, c, ldc, 1);
                        return;
                    }
                }
                xgemm(transa, transb, m, n, k, T{1}, a, lda, b, ldb, T{1}, c, ldc, detail::Cpu{});
            }

            static inline void func_perm(int nmats, int m, int n, int k, const T *a, int ldar,
                                         int ldac, const T *b0, const T *b1, int bj01, int *bj,
                                         int bjprod, int ldbr, int ldbc, int perm_size, int *perm,
                                         int *perm_sign, T *c, int ldcr, int ldcc) {
                (void)k;
                if (m != 3 || k != 3) abort();
                if (m == 0 || n == 0) return;

                constexpr std::size_t native_size = get_native_size<T>::size;
                gemm_3x3_in_parts<native_size, typename T::value_type>::
                    gemm_basic_3x3c_alpha1_beta0_perm(nmats, n, a, ldar, ldac, b0, b1, bj01, bj,
                                                      bjprod, ldbr, ldbc, perm_size, perm,
                                                      perm_sign, c, ldcr, ldcc);
            }
        };
#    endif // SUPERBBLAS_USE_SHORTCUTS_FOR_GEMM_3x3

        template <typename T> struct gemm_basic_3x3c_alpha1_beta1_wrapper<T, false> {
            static constexpr bool available() { return false; }
            static inline void func(char transa, char transb, int m, int n, int k, const T *a,
                                    int lda, const T *b, int ldb, T *c, int ldc) {
                xgemm(transa, transb, m, n, k, T{1}, a, lda, b, ldb, T{1}, c, ldc, detail::Cpu{});
            }
            static inline void func_perm(int, int, int, int, const T *, int, int, const T *,
                                         const T *, int, int *, int, int, int, int, int *, int *,
                                         T *, int, int) {
                abort();
            }
        };
    }
#endif // SUPERBBLAS_LIB

    namespace detail {
        /// Perform the matrix multiplication C += A * B with shortcuts for some matrix sizes
        /// \param transa: whether A is 'n': non-transposed, 't': transposed, 'c': conjugated transposed
        /// \param transb: whether B is 'n': non-transposed, 't': transposed, 'c': conjugated transposed
        /// \param m: rows of C
        /// \param n: columns of C
        /// \param k: columns of A and rows of B
        /// \param a: pointer to the first element of A
        /// \param lda: number of elements to skip a column in A
        /// \param b: pointer to the first element of B
        /// \param ldb: number of elements to skip a column in B
        /// \param c: pointer to the first element of C
        /// \param ldc: number of elements to skip a column in C

        template <typename T>
        DECL_XGEMM_ALT_ALPHA1_BETA1_T(void xgemm_alt_alpha1_beta1(char transa, char transb, int m,
                                                                  int n, int k, const T *a, int lda,
                                                                  const T *b, int ldb, T *c,
                                                                  int ldc, Cpu))
        IMPL({
            superbblas::detail_xp::gemm_basic_3x3c_alpha1_beta1_wrapper<T>::func(
                transa, transb, m, n, k, a, lda, b, ldb, c, ldc);
        })

        /// Perform the matrix multiplication C = \sum_i A_i * B_i * P_i * S_i with shortcuts for some matrix sizes,
        /// where P_i is a periodic permutation matrix, and S_i is a periodic diagonal matrix with 1 or -1 elements.
        /// For example: for nmat=2, n=8, perm_size=4, perm={0,1,2,3, 3,2,1,0}, perm_sign={1,-1,1,-1, 1,1,1,1}, then
        /// P_0 = [1 0 0 0        ]  S_0=diag(1,-1,1,-1, 1,-1,1,-1)
        ///       [0 1 0 0        ]
        ///       [0 0 1 0        ]
        ///       [0 0 0 1        ]
        ///       [        1 0 0 0]
        ///       [        0 1 0 0]
        ///       [        0 0 1 0]
        ///       [        0 0 0 1]
        /// P_1 = [0 0 0 1        ]  S_1=diag(1,1,1,1,  1,1,1,1)
        ///       [0 0 1 0        ]
        ///       [0 1 0 0        ]
        ///       [1 0 0 0        ]
        ///       [        0 0 0 1]
        ///       [        0 0 1 0]
        ///       [        0 1 0 0]
        ///       [        1 0 0 0]
        ///
        /// \param nmats: number of matrix products, sum_{i=0}^{i=nmats-1} A_i * B_i * P_i * S_i
        /// \param m: rows of C
        /// \param n: columns of C
        /// \param k: columns of all A_i and rows of all B_i
        /// \param a: pointer to the first element of A_0; all A_i are consecutive in memory
        /// \param ldar: number of elements to skip a row in A_i
        /// \param ldac: number of elements to skip a column in A_i
        /// \param b0: pointer base to all B_i matrices, B_i starts at b0[bj[i]*bjprod]
        ///            if bj[i] < bj01
        /// \param b1: pointer base to all B_i matrices, B_i starts at b1[bj[i]*bjprod]
        ///            if bj[i] >= bj01
        /// \param bj01: discriminator to use b0 or b1 depending on bj[i]
        /// \param bjprod: product on b0[i] or b1[i]
        /// \param ldbr: number of elements to skip a row in B_i
        /// \param ldbc: number of elements to skip a column in B_i
        /// \param perm_size: period of the permutation and the scalar matrix
        /// \param perm: pointer to the first column index for representing P_0
        /// \param perm_sign: pointer to the first value of S_0
        /// \param c: pointer to the first element of C
        /// \param ldcr: number of elements to skip a row in C
        /// \param ldcc: number of elements to skip a column in C

        template <typename T>
        DECL_XGEMM_ALT_ALPHA1_BETA0_PERM_T(void xgemm_alt_alpha1_beta0_perm(
            int nmats, int m, int n, int k, const T *a, int ldar, int ldac, const T *b0,
            const T *b1, int bj01, int *bj, int bjprod, int ldbr, int ldbc, int perm_size,
            int *perm, int *perm_sign, T *c, int ldcr, int ldcc, Cpu))
        IMPL({
            superbblas::detail_xp::gemm_basic_3x3c_alpha1_beta1_wrapper<T>::func_perm(
                nmats, m, n, k, a, ldar, ldac, b0, b1, bj01, bj, bjprod, ldbr, ldbc, perm_size,
                perm, perm_sign, c, ldcr, ldcc);
        })

        /// Return whether a crafted kernel is available
        /// \tparam T: matrix type

        template <typename T>
        DECL_AVAILABLE_BSR_KRON_3x3_PERM_T(bool available_bsr_kron_3x3_perm()) IMPL({
            return superbblas::detail_xp::gemm_basic_3x3c_alpha1_beta1_wrapper<T>::available();
        })
    }

    namespace detail_xp {

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
#endif // __SUPERBBLAS_TENFUCKS_CPU__
