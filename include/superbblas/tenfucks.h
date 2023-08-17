/// TENsor FUture Contraction K-dimensional Subroutine (TenFuCKS)

#ifndef __SUPERBBLAS_TENFUCKS__
#define __SUPERBBLAS_TENFUCKS__

#include "platform.h"
#include "xsimd/xsimd.hpp"
#include <algorithm>
#include <numeric>
#include <execution>

namespace superbblas {
namespace detail_xp {

using T = double;
using cT = std::complex<T>;
using vc4 = xsimd::make_sized_batch<T, 4>::type;
using vc8 = xsimd::make_sized_batch<T, 8>::type;
using vi4 = xsimd::batch<uint64_t, vc4::arch_type>;
using vi8 = xsimd::batch<uint64_t, vc8::arch_type>;
using Idx = int;

inline vc8 flip_ri(const vc8 &b) {
    return xsimd::swizzle(b, xsimd::batch_constant<vi8, 1, 0, 3, 2, 5, 4, 4, 4>());
}

inline vc8 scalar_mult(cT s, const vc8 &b) {
	//if (s == cT{1}) return b;
    T r = *(T*)&s;
    T i = ((T*)&s)[1];
    return s == cT{1} ? b : xsimd::fma(vc8(r), b, vc8(-i, i, -i, i, -i, i, i, i) * flip_ri(b));
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
    return {
        xsimd::shuffle(va, va, xsimd::batch_constant<vi8, 0, 0, 2, 2, 4, 4, 4, 4>()),
        xsimd::shuffle(xsimd::neg(va), va,
                       xsimd::batch_constant<vi8, 1, 8 + 1, 3, 8 + 3, 5, 8 + 5, 8 + 5, 8 + 5>())};
}

inline void gemm_basic_3x3c_intr(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar, Idx ldac, const cT *SB_RESTRICT b_,
                          Idx ldbr, Idx ldbc, cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
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
    for (Idx j = 0; j < N; ++j) {
        auto b0 = vc8::gather(b + ldbc * 2 * j, vi8(ldbr*2*0, ldbr*2*0+1, ldbr*2*1, ldbr*2*1+1, ldbr*2*2, ldbr*2*2+1, ldbr*2*2+1, ldbr*2*2+1));
        auto c0 = beta == T{0}
                      ? vc8(0)
                      : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j, vi8(ldcr*2*0, ldcr*2*0+1, ldcr*2*1, ldcr*2*1+1, ldcr*2*2, ldcr*2*2+1, ldcr*2*2+1, ldcr*2*2+1)));
	for (int disp=0; disp<3; ++disp) {
        	auto a01 = get_col_intr(a, ldar, ldac, disp);
		if (disp > 0) {
        		b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
		}
        	c0 = xsimd::fma(std::get<0>(a01), b0, c0);

        	b0 = flip_ri(b0);
        	c0 = xsimd::fma(std::get<1>(a01), b0, c0);
	}

        if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
        c0.scatter(d + lddc * 2 * j, vi8(lddr*2*0, lddr*2*0+1, lddr*2*1, lddr*2*1+1, lddr*2*2, lddr*2*2+1, lddr*2*2+1, lddr*2*2+1));
    }
}

inline vi8 get_8_ri(Idx ld) {
	return vi8(ld*2*0, ld*2*0+1, ld*2*1, ld*2*1+1, ld*2*2, ld*2*2+1, ld*2*2+1, ld*2*2+1);
}

inline void gemm_basic_3x3c_intr2(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar, Idx ldac, const cT *SB_RESTRICT b_,
                          Idx ldbr, Idx ldbc, cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
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
    if (N%2 != 0) {
gemm_basic_3x3c_intr(N%2, alpha, a_, ldar, ldac, b_, ldbr, ldbc, beta, c_, ldcr, ldcc, d_, lddr, lddc);
j =N%2;
    }

    for (; j < N; j+=2) {
	int j0=j, j1=j+1;
        auto b0 = vc8::gather(b + ldbc * 2 * j0, get_8_ri(ldbr));
        auto b1 = vc8::gather(b + ldbc * 2 * j1, get_8_ri(ldbr));
        auto c0 = beta == T{0}
                      ? vc8(0)
                      : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j0, get_8_ri(ldcr)));
        auto c1 = beta == T{0}
                      ? vc8(0)
                      : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j1, get_8_ri(ldcr)));

	for (int disp=0; disp<3; ++disp) {
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


//
////inline void gemm_basic_3x3c_intr_pf(Idx N, cT alpha, const cT *SB_RESTRICT a_, Idx ldar, Idx ldac, const cT *SB_RESTRICT b_,
////                          Idx ldbr, Idx ldbc, cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
////                          cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
////    //constexpr Idx M = 3;
////    //constexpr Idx K = 3;
////    const T *SB_RESTRICT a = (const T *)(a_);
////    const T *SB_RESTRICT b = (const T *)(b_);
////    const T *SB_RESTRICT c = (const T *)(c_);
////    T *SB_RESTRICT d = (T *)(d_);
////    using vi8_seq = xsimd::batch_constant<vi8, 0, 2, 4, 6, 8, 10, 10, 10>;
////    using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;
////
////    // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
////    std::for_each(std::execution::unseq, (char*)0, (char*)0+N, [&](const char& j_) {
////	int j = &j_ - (const char*)0;
////        auto b0 = vc8::gather(b + ldbc * 2 * j, vi8(ldbr) * vi8_seq());
////        auto c0 = beta == T{0}
////                      ? vc8(0)
////                      : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j, vi8(ldcr) * vi8_seq()));
////        auto a01 = get_col_intr(a, ldar, ldac, 0);
////        c0 = xsimd::fma(std::get<0>(a01), b0, c0);
////
////        b0 = flip_ri(b0);
////        c0 = xsimd::fma(std::get<1>(a01), b0, c0);
////
////        a01 = get_col_intr(a, ldar, ldac, 1);
////        b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
////        c0 = xsimd::fma(std::get<0>(a01), b0, c0);
////
////        b0 = flip_ri(b0);
////        c0 = xsimd::fma(std::get<1>(a01), b0, c0);
////
////        a01 = get_col_intr(a, ldar, ldac, 2);
////        b0 = xsimd::swizzle(b0, vi8_flip_and_plus_1());
////        c0 = xsimd::fma(std::get<0>(a01), b0, c0);
////
////        b0 = flip_ri(b0);
////        c0 = xsimd::fma(std::get<1>(a01), b0, c0);
////
////        if (alpha != T{1}) c0 = scalar_mult(alpha, c0);
////        c0.scatter(d + lddc * 2 * j, vi8(lddr) * vi8_seq());
////    });
////}
//
//template<Idx N>
//inline void gemm_basic_3x3c_intr2(cT alpha, const cT *SB_RESTRICT a_, Idx ldar, Idx ldac, const cT *SB_RESTRICT b_,
//                          Idx ldbr, Idx ldbc, cT beta, const cT *SB_RESTRICT c_, Idx ldcr, Idx ldcc,
//                          cT *SB_RESTRICT d_, Idx lddr, Idx lddc) {
//    //constexpr Idx M = 3;
//    //constexpr Idx K = 3;
//    const T *SB_RESTRICT a = (const T *)(a_);
//    const T *SB_RESTRICT b = (const T *)(b_);
//    const T *SB_RESTRICT c = (const T *)(c_);
//    T *SB_RESTRICT d = (T *)(d_);
//    using vi8_seq = xsimd::batch_constant<vi8, 0, 2, 4, 6, 8, 10, 10, 10>;
//    using vi8_flip_and_plus_1 = xsimd::batch_constant<vi8, 3, 2, 5, 4, 1, 0, 0, 0>;
//
//    // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
//        std::array<vc8, N> b0, c0;
//	for (Idx j = 0; j < N; ++j) b0[j] = vc8::gather(b + ldbc * 2 * j, vi8(ldbr) * vi8_seq());
//        for (Idx j = 0; j < N; ++j) c0[j] = beta == T{0}
//                      ? vc8(0)
//                      : scalar_mult(beta, vc8::gather(c + ldcc * 2 * j, vi8(ldcr) * vi8_seq()));
//        auto a01 = get_col_intr(a, ldar, ldac, 0);
//        for (Idx j = 0; j < N; ++j) c0[j] = xsimd::fma(std::get<0>(a01), b0[j], c0[j]);
//
//        for (Idx j = 0; j < N; ++j) b0[j] = flip_ri(b0[j]);
//        for (Idx j = 0; j < N; ++j) c0[j] = xsimd::fma(std::get<1>(a01), b0[j], c0[j]);
//
//        a01 = get_col_intr(a, ldar, ldac, 1);
//	auto flip_and_plus_1 = vi8_flip_and_plus_1();
//        for (Idx j = 0; j < N; ++j) b0[j] = xsimd::swizzle(b0[j], flip_and_plus_1);
//        for (Idx j = 0; j < N; ++j) c0[j] = xsimd::fma(std::get<0>(a01), b0[j], c0[j]);
//
//        for (Idx j = 0; j < N; ++j) b0[j] = flip_ri(b0[j]);
//        for (Idx j = 0; j < N; ++j) c0[j] = xsimd::fma(std::get<1>(a01), b0[j], c0[j]);
//
//        for (Idx j = 0; j < N; ++j) a01 = get_col_intr(a, ldar, ldac, 2);
//        for (Idx j = 0; j < N; ++j) b0[j] = xsimd::swizzle(b0[j], flip_and_plus_1);
//        for (Idx j = 0; j < N; ++j) c0[j] = xsimd::fma(std::get<0>(a01), b0[j], c0[j]);
//
//        for (Idx j = 0; j < N; ++j) b0[j] = flip_ri(b0[j]);
//        for (Idx j = 0; j < N; ++j) c0[j] = xsimd::fma(std::get<1>(a01), b0[j], c0[j]);
//
//        if (alpha != T{1}) for (Idx j = 0; j < N; ++j) c0[j] = scalar_mult(alpha, c0[j]);
//        for (Idx j = 0; j < N; ++j) c0[j].scatter(d + lddc * 2 * j, vi8(lddr) * vi8_seq());
//}

}
}
#endif // __SUPERBBLAS_TENFUCKS__
