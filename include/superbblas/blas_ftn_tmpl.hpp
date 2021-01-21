
#ifndef THIS_FILE
#    define THIS_FILE "blas_ftn_tmpl.hpp"
#endif

#include "platform.h"
#include "template.h"
#include <complex>

namespace superbblas {

    namespace detail {

        using BLASINT = int;

#define REAL ARTIH(, , float, float, double, double, , )
#define SCALAR ARITH(, , float, std::complex<float>, double, std::complex<double>, , )

// clang-format off
#define XCOPY     FORTRAN_FUNCTION(ARITH(hcopy , kcopy , scopy , ccopy , dcopy , zcopy , , ))
#define XSWAP     FORTRAN_FUNCTION(ARITH(hswap , kswap , sswap , cswap , dswap , zswap , , ))
#define XGEMM     FORTRAN_FUNCTION(ARITH(hgemm , kgemm , sgemm , cgemm , dgemm , zgemm , , ))
#define XTRMM     FORTRAN_FUNCTION(ARITH(htrmm , ktrmm , strmm , ctrmm , dtrmm , ztrmm , , ))
#define XTRSM     FORTRAN_FUNCTION(ARITH(htrsm , ktrsm , strsm , ctrsm , dtrsm , ztrsm , , ))
#define XHEMM     FORTRAN_FUNCTION(ARITH(hsymm , khemm , ssymm , chemm , dsymm , zhemm , , ))
#define XHEMV     FORTRAN_FUNCTION(ARITH(hsymv , khemv , ssymv , chemv , dsymv , zhemv , , ))
#define XAXPY     FORTRAN_FUNCTION(ARITH(haxpy , kaxpy , saxpy , caxpy , daxpy , zaxpy , , ))
#define XGEMV     FORTRAN_FUNCTION(ARITH(hgemv , kgemv , sgemv , cgemv , dgemv , zgemv , , ))
#define XDOT      FORTRAN_FUNCTION(ARITH(hdot  ,       , sdot  ,       , ddot  ,       , , ))
#define XSCAL     FORTRAN_FUNCTION(ARITH(hscal , kscal , sscal , cscal , dscal , zscal , , ))
        // clang-format on

        using BLASSTRING = const char *;

#if !defined(__SUPERBBLAS_USE_HALF) && !defined(__SUPERBBLAS_USE_HALFCOMPLEX)

        extern "C" {

        void XCOPY(BLASINT *n, const SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
        void XSWAP(BLASINT *n, SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
        void XGEMM(BLASSTRING transa, BLASSTRING transb, BLASINT *m, BLASINT *n, BLASINT *k,
                   SCALAR *alpha, const SCALAR *a, BLASINT *lda, const SCALAR *b, BLASINT *ldb,
                   SCALAR *beta, SCALAR *c, BLASINT *ldc);
        void XGEMV(BLASSTRING transa, BLASINT *m, BLASINT *n, SCALAR *alpha, const SCALAR *a,
                   BLASINT *lda, const SCALAR *x, BLASINT *incx, SCALAR *beta, SCALAR *y,
                   BLASINT *incy);
        void XTRMM(BLASSTRING side, BLASSTRING uplo, BLASSTRING transa, BLASSTRING diag, BLASINT *m,
                   BLASINT *n, SCALAR *alpha, SCALAR *a, BLASINT *lda, SCALAR *b, BLASINT *ldb);
        void XTRSM(BLASSTRING side, BLASSTRING uplo, BLASSTRING transa, BLASSTRING diag, BLASINT *m,
                   BLASINT *n, SCALAR *alpha, SCALAR *a, BLASINT *lda, SCALAR *b, BLASINT *ldb);
        void XHEMM(BLASSTRING side, BLASSTRING uplo, BLASINT *m, BLASINT *n, SCALAR *alpha,
                   SCALAR *a, BLASINT *lda, SCALAR *b, BLASINT *ldb, SCALAR *beta, SCALAR *c,
                   BLASINT *ldc);
        void XHEMV(BLASSTRING uplo, BLASINT *n, SCALAR *alpha, SCALAR *a, BLASINT *lda, SCALAR *x,
                   BLASINT *lncx, SCALAR *beta, SCALAR *y, BLASINT *lncy);
        void XAXPY(BLASINT *n, SCALAR *alpha, SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
// NOTE: avoid calling Fortran functions that return complex values
#    ifndef __SUPERBBLAS_USE_COMPLEX
        SCALAR XDOT(BLASINT *n, SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
#    endif // __SUPERBBLAS_USE_COMPLEX
        void XSCAL(BLASINT *n, SCALAR *alpha, SCALAR *x, BLASINT *incx);
        }

        inline void xcopy(BLASINT n, const SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy,
                          Cpu cpu) {
            (void)cpu;
            XCOPY(&n, x, &incx, y, &incy);
        }

        inline void xswap(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu cpu) {
            (void)cpu;
            XSWAP(&n, x, &incx, y, &incy);
        }

        inline void xgemm(char transa, char transb, BLASINT m, BLASINT n, BLASINT k, SCALAR alpha,
                          const SCALAR *a, BLASINT lda, const SCALAR *b, BLASINT ldb, SCALAR beta,
                          SCALAR *c, BLASINT ldc, Cpu cpu) {
            (void)cpu;
            XGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        }

        inline void xgemv(char transa, BLASINT m, BLASINT n, SCALAR alpha, const SCALAR *a,
                          BLASINT lda, const SCALAR *x, BLASINT incx, SCALAR beta, SCALAR *y,
                          BLASINT incy, Cpu cpu) {
            (void)cpu;
            XGEMV(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        }

        inline void xtrmm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu cpu) {
            (void)cpu;
            XTRMM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
        }

        inline void xtrsm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu cpu) {
            (void)cpu;
            XTRSM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
        }

        inline void xhemm(char side, char uplo, BLASINT m, BLASINT n, SCALAR alpha, SCALAR *a,
                          BLASINT lda, SCALAR *b, BLASINT ldb, SCALAR beta, SCALAR *c, BLASINT ldc,
                          Cpu cpu) {
            (void)cpu;
            XHEMM(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        }

        inline void xhemv(char uplo, BLASINT n, SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *x,
                          BLASINT incx, SCALAR beta, SCALAR *y, BLASINT incy, Cpu cpu) {
            (void)cpu;
            XHEMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        }

        inline void xaxpy(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy,
                          Cpu cpu) {
            (void)cpu;
            XAXPY(&n, &alpha, x, &incx, y, &incy);
        }

        inline SCALAR xdot(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu cpu) {
            (void)cpu;
#    ifndef __SUPERBBLAS_USE_COMPLEX
            return XDOT(&n, x, &incx, y, &incy);
#    else
            SCALAR r = (SCALAR)0.0;
#        ifdef _OPENMP
#            pragma omp for
#        endif
            for (int i = 0; i < n; i++) r += std::conj(x[i * incx]) * y[i * incy];
            return r;
#    endif // __SUPERBBLAS_USE_COMPLEX
        }

        inline void xscal(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, Cpu cpu) {
            (void)cpu;
            XSCAL(&n, &alpha, x, &incx);
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
