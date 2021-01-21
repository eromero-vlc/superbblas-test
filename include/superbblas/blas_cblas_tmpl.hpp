
#ifndef THIS_FILE
#    define THIS_FILE "blas_cblas_tmpl.hpp"
#endif

#include "cblas.h"
#include "platform.h"
#include "template.h"
#include <complex>

namespace superbblas {

    namespace detail {

        using BLASINT = int;

#define REAL ARTIH(, , float, float, double, double, , )
#define SCALAR ARTIH(, , float, std::complex<float>, double, std::complex<double>, , )

// Pass constant values by value for non-complex types, and by reference otherwise
#define PASS_SCALAR(X) ARTIH(X, &(X), X, &(X), X, &(X), X, &(X))

#define CBLAS_FUNCTION(X) CONCAT(cblas_, X)

// clang-format off
#define XCOPY     CBLAS_FUNCTION(ARITH(hcopy , kcopy , scopy , ccopy , dcopy , zcopy , , ))
#define XSWAP     CBLAS_FUNCTION(ARITH(hswap , kswap , sswap , cswap , dswap , zswap , , ))
#define XGEMM     CBLAS_FUNCTION(ARITH(hgemm , kgemm , sgemm , cgemm , dgemm , zgemm , , ))
#define XTRMM     CBLAS_FUNCTION(ARITH(htrmm , ktrmm , strmm , ctrmm , dtrmm , ztrmm , , ))
#define XTRSM     CBLAS_FUNCTION(ARITH(htrsm , ktrsm , strsm , ctrsm , dtrsm , ztrsm , , ))
#define XHEMM     CBLAS_FUNCTION(ARITH(hsymm , khemm , ssymm , chemm , dsymm , zhemm , , ))
#define XHEMV     CBLAS_FUNCTION(ARITH(hsymv , khemv , ssymv , chemv , dsymv , zhemv , , ))
#define XAXPY     CBLAS_FUNCTION(ARITH(haxpy , kaxpy , saxpy , caxpy , daxpy , zaxpy , , ))
#define XGEMV     CBLAS_FUNCTION(ARITH(hgemv , kgemv , sgemv , cgemv , dgemv , zgemv , , ))
#define XSCAL     CBLAS_FUNCTION(ARITH(hscal , kscal , sscal , cscal , dscal , zscal , , ))
#define XDOT      CBLAS_FUNCTION(ARITH(hdot  , kdotc_sub, sdot, cdotc_sub, ddot, zdotc_sub, , ))
        // clang-format on

        inline CBLAS_TRANSPOSE toCblasTrans(char trans) {
            switch (trans) {
            case 'n':
            case 'N': return CblasNoTrans;
            case 't':
            case 'T': return CblasTrans;
            case 'c':
            case 'C': return CblasConjTrans;
            default: throw std::runtime_error("Not valid value of trans");
            }
        }

#if !defined(__SUPERBBLAS_USE_HALF) && !defined(__SUPERBBLAS_USE_HALFCOMPLEX)
        inline void xcopy(BLASINT n, const SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy,
                          Cpu cpu) {
            (void)cpu;
            XCOPY(n, x, incx, y, incy);
        }

        inline void xswap(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu cpu) {
            (void)cpu;
            XSWAP(n, x, incx, y, incy);
        }

        inline void xgemm(char transa, char transb, BLASINT m, BLASINT n, BLASINT k, SCALAR alpha,
                          const SCALAR *a, BLASINT lda, const SCALAR *b, BLASINT ldb, SCALAR beta,
                          SCALAR *c, BLASINT ldc, Cpu cpu) {
            (void)cpu;
            XGEMM(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m, n, k,
                  PASS_SCALAR(alpha), a, lda, b, ldb, PASS_SCALAR(beta), c, ldc);
        }

        inline void xgemv(char transa, BLASINT m, BLASINT n, SCALAR alpha, SCALAR *a, BLASINT lda,
                          const SCALAR *x, BLASINT incx, SCALAR beta, SCALAR *y, BLASINT incy,
                          Cpu cpu) {
            (void)cpu;
            XGEMV(CblasColMajor, toCblasTrans(transa), m, n, PASS_SCALAR(alpha), a, lda, x, incx,
                  PASS_SCALAR(beta), y, incy);
        }

        inline void xtrmm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu cpu) {
            (void)cpu;
            XTRMM(CblasColMajor, toCblasTrans(side), toCblasTrans(uplo), toCblasTrans(transa),
                  toCblasTrans(diag), m, n, PASS_SCALAR(alpha), a, lda, b, ldb);
        }

        inline void xtrsm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu cpu) {
            (void)cpu;
            XTRSM(CblasColMajor, toCblasTrans(side), toCblasTrans(uplo), toCblasTrans(transa),
                  toCblasTrans(diag), m, n, PASS_SCALAR(alpha), a, lda, b, ldb);
        }

        inline void xhemm(char side, char uplo, BLASINT m, BLASINT n, SCALAR alpha, SCALAR *a,
                          BLASINT lda, SCALAR *b, BLASINT ldb, SCALAR beta, SCALAR *c, BLASINT ldc,
                          Cpu cpu) {
            (void)cpu;
            XHEMM(CblasColMajor, toCblasTrans(side), toCblasTrans(uplo), m, n, PASS_SCALAR(alpha),
                  a, lda, b, ldb, PASS_SCALAR(beta), c, ldc);
        }

        inline void xhemv(char uplo, BLASINT n, SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *x,
                          BLASINT incx, SCALAR beta, SCALAR *y, BLASINT incy, Cpu cpu) {
            (void)cpu;
            XHEMV(CblasColMajor, toCblasTrans(uplo), n, PASS_SCALAR(alpha), a, lda, x, incx,
                  PASS_SCALAR(beta), y, incy);
        }

        inline void xaxpy(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy,
                          Cpu cpu) {
            (void)cpu;
            XAXPY(n, PASS_SCALAR(alpha), x, incx, y, incy);
        }

        inline SCALAR xdot(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu cpu) {
            (void)cpu;
#    ifndef __SUPERBBLAS_USE_COMPLEX
            return XDOT(n, x, incx, y, incy);
#    else
            SCALAR r = (SCALAR)0.0;
            XDOT(n, x, incx, y, incy, &r);
            return r;
#    endif // __SUPERBBLAS_USE_COMPLEX
        }

        inline void xscal(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, Cpu cpu) {
            (void)cpu;
            XSCAL(n, PASS_SCALAR(alpha), x, incx);
        }

#endif // !defined(__SUPERBBLAS_USE_HALF) && !defined(__SUPERBBLAS_USE_HALFCOMPLEX)

#undef XCOPY
#undef XSWAP
#undef XGEMM
#undef XTRMM
#undef XTRSM
#undef XHEMM
#undef XHEMV
#undef XAXPY
#undef XGEMV
#undef XDOT
#undef XSCAL

#undef REAL
#undef SCALAR
    }
}
