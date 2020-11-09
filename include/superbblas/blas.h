#ifndef __SUPERBBLAS_BLAS__
#define __SUPERBBLAS_BLAS__

#include "platform.h"

#ifdef SUPERBBLAS_USE_MKL
#    include "mkl.h"
#    ifndef SUPERBBLAS_USE_CBLAS
#        define SUPERBBLAS_USE_CBLAS
#    endif
#endif // SUPERBBLAS_USE_MKL

#ifndef SUPERBBLAS_USE_CBLAS
#    include "blas_ftn_tmpl.hpp"
#else
#    include "blas_cblas_tmpl.hpp"
#endif

namespace superbblas {

    namespace detail {

        /// Template multiple GEMM

        template <typename T>
        void xgemm_batch_strided(char transa, char transb, int m, int n, int k, T alpha, const T *a,
                                 int lda, int stridea, const T *b, int ldb, int strideb, T beta,
                                 T *c, int ldc, int stridec, int batch_size, Cpu cpu);

#ifdef SUPERBBLAS_USE_MKL
        template <>
        void xgemm_batch_strided<float>(char transa, char transb, int m, int n, int k, float alpha,
                                        const float *a, int lda, int stridea, const float *b,
                                        int ldb, int strideb, float beta, float *c, int ldc,
                                        int stridec, int batch_size, Cpu cpu) {

            (void)cpu;
            cblas_sgemm_batch_strided(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m,
                                      n, k, alpha, a, lda, stridea, b, ldb, strideb, beta, c, ldc,
                                      stridec, batch_size);
        }

        template <>
        void xgemm_batch_strided<std::complex<float>>(
            char transa, char transb, int m, int n, int k, std::complex<float> alpha,
            const std::complex<float> *a, int lda, int stridea, const std::complex<float> *b,
            int ldb, int strideb, std::complex<float> beta, std::complex<float> *c, int ldc,
            int stridec, int batch_size, Cpu cpu) {

            (void)cpu;
            cblas_cgemm_batch_strided(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m,
                                      n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc,
                                      stridec, batch_size);
        }

        template <>
        void xgemm_batch_strided<double>(char transa, char transb, int m, int n, int k,
                                         double alpha, const double *a, int lda, int stridea,
                                         const double *b, int ldb, int strideb, double beta,
                                         double *c, int ldc, int stridec, int batch_size, Cpu cpu) {

            (void)cpu;
            cblas_dgemm_batch_strided(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m,
                                      n, k, alpha, a, lda, stridea, b, ldb, strideb, beta, c, ldc,
                                      stridec, batch_size);
        }

        template <>
        void xgemm_batch_strided<std::complex<double>>(
            char transa, char transb, int m, int n, int k, std::complex<double> alpha,
            const std::complex<double> *a, int lda, int stridea, const std::complex<double> *b,
            int ldb, int strideb, std::complex<double> beta, std::complex<double> *c, int ldc,
            int stridec, int batch_size, Cpu cpu) {

            (void)cpu;
            cblas_zgemm_batch_strided(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m,
                                      n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc,
                                      stridec, batch_size);
        }

#else // SUPERBBLAS_USE_MKL

        template <typename T>
        void xgemm_batch_strided(char transa, char transb, int m, int n, int k, T alpha, const T *a,
                                 int lda, int stridea, const T *b, int ldb, int strideb, T beta,
                                 T *c, int ldc, int stridec, int batch_size, Cpu cpu) {

#    ifdef _OPENMP
#        pragma omp for
#    endif
            for (int i = 0; i < batch_size; ++i) {
                xgemm(transa, transb, m, n, k, alpha, a + stridea * i, lda, b + strideb * i, ldb,
                      beta, c + stridec * i, ldc, cpu);
            }
        }

#endif // SUPERBBLAS_USE_MKL
    }
}


#endif // __SUPERBBLAS_BLAS__
