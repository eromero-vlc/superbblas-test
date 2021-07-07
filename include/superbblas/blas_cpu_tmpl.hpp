
#ifndef THIS_FILE
#    define THIS_FILE "blas_cpu_tmpl.hpp"
#endif

#include "platform.h"
#include "template.h"
#ifdef SUPERBBLAS_USE_MKL
#    include "mkl.h"
#    include "mkl_spblas.h"
#    ifndef SUPERBBLAS_USE_CBLAS
#        define SUPERBBLAS_USE_CBLAS
#    endif
#    define MKL_SCALAR ARTIH(, , float, MKL_Complex8, double, MKL_Complex16, , )
#elif defined(SUPERBBLAS_USE_CBLAS)
#    include "cblas.h"
#endif // SUPERBBLAS_USE_MKL

#include <complex>

#if !defined(__SUPERBBLAS_USE_HALF) && !defined(__SUPERBBLAS_USE_HALFCOMPLEX)

namespace superbblas {

    namespace detail {

#    ifdef SUPERBBLAS_USE_MKL
#        define BLASINT MKL_INT
#    else
#        define BLASINT int
#    endif

#    define REAL ARTIH(, , float, float, double, double, , )
#    define SCALAR ARITH(, , float, std::complex<float>, double, std::complex<double>, , )

        //
        // Basic BLAS
        //

#    ifndef SUPERBBLAS_USE_CBLAS

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
#        ifndef __SUPERBBLAS_USE_COMPLEX
        SCALAR XDOT(BLASINT *n, SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
#        endif // __SUPERBBLAS_USE_COMPLEX
        void XSCAL(BLASINT *n, SCALAR *alpha, SCALAR *x, BLASINT *incx);
        }

#    else //  SUPERBBLAS_USE_CBLAS

// Pass constant values by value for non-complex types, and by reference otherwise
#        define PASS_SCALAR(X) ARTIH(X, &(X), X, &(X), X, &(X), X, &(X))

#        define CBLAS_FUNCTION(X) CONCAT(cblas_, X)

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

#    endif //  SUPERBBLAS_USE_CBLAS

        inline void xcopy(BLASINT n, const SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XCOPY(&n, x, &incx, y, &incy);
#    else
            XCOPY(n, x, incx, y, incy);
#    endif
        }

        inline void xswap(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XSWAP(&n, x, &incx, y, &incy);
#    else
            XSWAP(n, x, incx, y, incy);
#    endif
        }

        inline void xgemm(char transa, char transb, BLASINT m, BLASINT n, BLASINT k, SCALAR alpha,
                          const SCALAR *a, BLASINT lda, const SCALAR *b, BLASINT ldb, SCALAR beta,
                          SCALAR *c, BLASINT ldc, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#    else
            XGEMM(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m, n, k,
                  PASS_SCALAR(alpha), a, lda, b, ldb, PASS_SCALAR(beta), c, ldc);
#    endif
        }

        inline void xgemv(char transa, BLASINT m, BLASINT n, SCALAR alpha, const SCALAR *a,
                          BLASINT lda, const SCALAR *x, BLASINT incx, SCALAR beta, SCALAR *y,
                          BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XGEMV(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#    else
            XGEMV(CblasColMajor, toCblasTrans(transa), m, n, PASS_SCALAR(alpha), a, lda, x, incx,
                  PASS_SCALAR(beta), y, incy);
#    endif
        }

        inline void xtrmm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XTRMM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
#    else
            XTRMM(CblasColMajor, toCblasTrans(side), toCblasTrans(uplo), toCblasTrans(transa),
                  toCblasTrans(diag), m, n, PASS_SCALAR(alpha), a, lda, b, ldb);
#    endif
        }

        inline void xtrsm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XTRSM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
#    else
            XTRSM(CblasColMajor, toCblasTrans(side), toCblasTrans(uplo), toCblasTrans(transa),
                  toCblasTrans(diag), m, n, PASS_SCALAR(alpha), a, lda, b, ldb);
#    endif
        }

        inline void xhemm(char side, char uplo, BLASINT m, BLASINT n, SCALAR alpha, SCALAR *a,
                          BLASINT lda, SCALAR *b, BLASINT ldb, SCALAR beta, SCALAR *c, BLASINT ldc,
                          Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XHEMM(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#    else
            XHEMM(CblasColMajor, toCblasTrans(side), toCblasTrans(uplo), m, n, PASS_SCALAR(alpha),
                  a, lda, b, ldb, PASS_SCALAR(beta), c, ldc);
#    endif
        }

        inline void xhemv(char uplo, BLASINT n, SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *x,
                          BLASINT incx, SCALAR beta, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XHEMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#    else
            XHEMV(CblasColMajor, toCblasTrans(uplo), n, PASS_SCALAR(alpha), a, lda, x, incx,
                  PASS_SCALAR(beta), y, incy);
#    endif
        }

        inline void xaxpy(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy,
                          Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XAXPY(&n, &alpha, x, &incx, y, &incy);
#    else
            XAXPY(n, PASS_SCALAR(alpha), x, incx, y, incy);
#    endif
        }

        inline SCALAR xdot(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef __SUPERBBLAS_USE_COMPLEX
#        ifndef SUPERBBLAS_USE_CBLAS
            return XDOT(&n, x, &incx, y, &incy);
#        else
            SCALAR r = (SCALAR)0.0;
            XDOT(n, x, incx, y, incy, &r);
            return r;
#        endif
#    else
            SCALAR r = (SCALAR)0.0;
#        ifdef _OPENMP
#            pragma omp for
#        endif
            for (int i = 0; i < n; i++) r += std::conj(x[i * incx]) * y[i * incy];
            return r;
#    endif // __SUPERBBLAS_USE_COMPLEX
        }

        inline void xscal(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, Cpu) {
            if (std::fabs(alpha) == SCALAR{0.0}) {
#    ifdef _OPENMP
#        pragma omp for
#    endif
                for (BLASINT i = 0; i < n; ++i) x[i * incx] = SCALAR{0};
                return;
            }
            if (alpha == SCALAR{1.0}) return;
#    ifndef SUPERBBLAS_USE_CBLAS
            XSCAL(&n, &alpha, x, &incx);
#    else
            XSCAL(n, PASS_SCALAR(alpha), x, incx);
#    endif
        }

#    undef XCOPY
#    undef XSWAP
#    undef XGEMM
#    undef XTRMM
#    undef XTRSM
#    undef XHEMM
#    undef XHEMV
#    undef XAXPY
#    undef XGEMV
#    undef XDOT
#    undef XSCAL
#    ifdef SUPERBBLAS_USE_CBLAS
#        undef PASS_SCALAR
#        undef CBLAS_FUNCTION
#    endif

        //
        // Batched GEMM
        //

#    ifdef SUPERBBLAS_USE_MKL

        inline void xgemm_batch_strided(char transa, char transb, int m, int n, int k, float alpha,
                                        const SCALAR *a, int lda, int stridea, const SCALAR *b,
                                        int ldb, int strideb, SCALAR beta, SCALAR *c, int ldc,
                                        int stridec, int batch_size, Cpu) {

            CONCAT(cblas_, CONCAT(ARITH(, , s, c, d, z, , ), gemm_batch_strided))
            (CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m, n, k, alpha, a, lda,
             stridea, b, ldb, strideb, beta, c, ldc, stridec, batch_size);
        }

#    else // SUPERBBLAS_USE_MKL

        inline void xgemm_batch_strided(char transa, char transb, int m, int n, int k, SCALAR alpha,
                                        const SCALAR *a, int lda, int stridea, const SCALAR *b,
                                        int ldb, int strideb, SCALAR beta, SCALAR *c, int ldc,
                                        int stridec, int batch_size, Cpu cpu) {

#        ifdef _OPENMP
#            pragma omp for
#        endif
            for (int i = 0; i < batch_size; ++i) {
                xgemm(transa, transb, m, n, k, alpha, a + stridea * i, lda, b + strideb * i, ldb,
                      beta, c + stridec * i, ldc, cpu);
            }
        }

#    endif // SUPERBBLAS_USE_MKL

        //
        // MKL Sparse
        //

#    ifdef SUPERBBLAS_USE_MKL
// Pass constant values by value for non-complex types, and by reference otherwise
#        define PASS_SCALAR(X)                                                                     \
            ARTIH(X, &(X), X, *(MKL_Complex8 *)&(X), X, *(MKL_Complex16 *)&(X), X, &(X))

#        define MKL_SP_FUNCTION(X) CONCAT(mkl_sparse_, X)

// clang-format off
#define XSPCREATEBSR    MKL_SP_FUNCTION(ARITH( , , s_create_bsr , c_create_bsr , d_create_bsr , z_create_bsr , , ))
#define XSPMM           MKL_SP_FUNCTION(ARITH( , , s_mm , c_mm , d_mm , z_mm , , ))
        // clang-format on

        inline sparse_matrix_t mkl_sparse_create_bsr(sparse_matrix_t *A,
                                                     sparse_index_base_t indexing,
                                                     sparse_layout_t block_layout, MKL_INT rows,
                                                     MKL_INT cols, MKL_INT block_size,
                                                     MKL_INT *rows_start, MKL_INT *rows_end,
                                                     MKL_INT *col_indx, SCALAR *values) {
            return XSPCREATEBSR(A, indexing, block_layout, rows, cols, block_size, rows_start,
                                rows_end, col_indx, (MKL_SCALAR *)values);
        }

        inline sparse_status_t mkl_sparse_mm(const sparse_operation_t operation, SCALAR alpha,
                                             sparse_matrix_t A, struct matrix_descr descr,
                                             sparse_layout_t layout, const SCALAR *B,
                                             MKL_INT columns, MKL_INT ldb, SCALAR beta, SCALAR *C,
                                             MKL_INT ldc) {
            return XSPMM(operation, PASS_SCALAR(alpha), (MKL_SCALAR *)A, descr, layout,
                         (MKL_SCALAR *)B, columns, ldb, PASS_SCALAR(beta), (MKL_SCALAR *)C, ldc);
        }

#        undef PASS_SCALAR
#        undef MKL_SP_FUNCTION

#    endif // SUPERBBLAS_USE_MKL

#    undef BLASINT
#    undef REAL
#    undef SCALAR
    }
}
#endif // !defined(__SUPERBBLAS_USE_HALF) && !defined(__SUPERBBLAS_USE_HALFCOMPLEX)
