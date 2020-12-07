#ifndef __SUPERBBLAS_BLAS__
#define __SUPERBBLAS_BLAS__

#include "platform.h"
#include <type_traits>
#include <vector>

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

#ifdef SUPERBBLAS_USE_CUDA
#    include <cublas_v2.h>
#    include <thrust/device_ptr.h>
#    include <thrust/device_vector.h>
#    include <thrust/fill.h>
#    include <thrust/iterator/permutation_iterator.h>
#endif

namespace superbblas {

    namespace detail {

        /// Vector type
        /// \param T: type of the vector's elements
        /// \param XPU: device type, one of Cpu, Cuda, Gpuamd

        template <typename T, typename XPU>
        using vector = typename std::conditional<
            std::is_same<XPU, Cpu>::value, std::vector<T>,
#ifdef SUPERBBLAS_USE_CUDA
            typename std::conditional<std::is_same<XPU, Cuda>::value, thrust::device_vector<T>,
                                      void>::type
#else
            void
#endif // SUPERBBLAS_USE_CUDA
            >::type;

        /// Constant iterator vector type
        /// \param T: type of the vector's elements
        /// \param XPU: device type, one of Cpu, Cuda, Gpuamd

        template <typename T, typename XPU>
        using vector_const_iterator = typename vector<T, XPU>::const_iterator;

        /// Pointer to data type
        /// \param T: type of the vector's elements
        /// \param XPU: device type, one of Cpu, Cuda, Gpuamd

        template <typename T, typename XPU>
        using data =
            typename std::conditional<std::is_same<XPU, Cpu>::value, T *,
#ifdef SUPERBBLAS_USE_CUDA
                                      typename std::conditional<std::is_same<XPU, Cuda>::value,
                                                                thrust::device_ptr<T>, void>::type
#else
                                      void
#endif // SUPERBBLAS_USE_CUDA
                                      >::type;

#ifdef SUPERBBLAS_USE_CUDA
        /// Return a device pointer suitable for making iterators

        template <typename T> thrust::device_ptr<T> encapsulate_pointer(T *ptr) {
            return thrust::device_pointer_cast(ptr);
        }
#endif

        namespace EWOp {
            /// Copy the values of the origin vector into the destination vector
            struct Copy {};

            /// Add the values from the origin vector to the destination vector
            struct Add {};
        }

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indices, Cpu cpuv,
                    std::size_t n, data<T, Cpu> w, Cpu cpuw, EWOp::Copy) {
            (void)cpuv;
            (void)cpuw;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[i] = v[indices[i]];
        }

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu cpuv, std::size_t n, data<T, Cpu> w,
                    vector_const_iterator<IndexType, Cpu> indices, Cpu cpuw, EWOp::Copy) {
            (void)cpuv;
            (void)cpuw;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indices[i]] = v[i];
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu cpuv,
                    std::size_t n, data<T, Cpu> w, vector_const_iterator<IndexType, Cpu> indicesw,
                    Cpu cpuw, EWOp::Copy) {
            (void)cpuv;
            (void)cpuw;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = v[indicesv[i]];
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu cpuv,
                    std::size_t n, data<T, Cpu> w, vector_const_iterator<IndexType, Cpu> indicesw,
                    Cpu cpuw, EWOp::Add) {
            (void)cpuv;
            (void)cpuw;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] += v[indicesv[i]];
        }

        /// Copy and reduce n values, w[indicesw[i]] += sum(v[perm[perm_distinct[i]:perm_distinct[i+1]]])
        template <typename IndexType, typename T>
        void copy_reduce_n(data<const T, Cpu> v, Cpu, vector_const_iterator<IndexType, Cpu> perm,
                           vector_const_iterator<IndexType, Cpu> perm_distinct,
                           std::size_t ndistinct, Cpu cpuv, data<T, Cpu> w,
                           vector_const_iterator<IndexType, Cpu> indicesw, Cpu cpuw, EWOp::Copy) {
            (void)cpuv;
            (void)cpuw;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < ndistinct - 1; ++i)
                for (std::size_t j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                    w[indicesw[i]] += v[perm[j]];
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indices,
                    Cuda cudav, std::size_t n, data<T, Cpu> w, Cpu cpuw, EWOp::Copy) {
            (void)cudav;
            (void)cpuw;
            thrust::copy_n(thrust::make_permutation_iterator(v, indices), n, w);
        }

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu cpuv, std::size_t n, data<T, Cuda> w,
                    vector_const_iterator<IndexType, Cuda> indices, Cuda cudaw, EWOp::Copy) {
            (void)cpuv;
            (void)cudaw;
            thrust::copy_n(v, n, thrust::make_permutation_iterator(w, indices));
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu xpuv,
                    std::size_t n, data<T, Cuda> w, vector_const_iterator<IndexType, Cuda> indicesw,
                    Cuda xpuw, EWOp::Copy) {
            (void)xpuv;
            (void)xpuw;
            thrust::copy_n(thrust::make_permutation_iterator(v, indicesv), n,
                           thrust::make_permutation_iterator(w, indicesw));
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu xpuv,
                    std::size_t n, data<T, Cuda> w, vector_const_iterator<IndexType, Cuda> indicesw,
                    Cuda xpuw, EWOp::Add) {
            (void)xpuv;
            (void)xpuw;
            auto vit = thrust::make_permutation_iterator(v, indicesv);
            thrust::transform(vit, vit + n, thrust::make_permutation_iterator(w, indicesw),
                              thrust::make_permutation_iterator(w, indicesw), thrust::plus<T>());
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indicesv,
                    Cuda xpuv, std::size_t n, data<T, Cpu> w,
                    vector_const_iterator<IndexType, Cpu> indicesw, Cpu xpuw, EWOp::Copy) {
            (void)xpuv;
            (void)xpuw;
            thrust::copy_n(thrust::make_permutation_iterator(v, indicesv), n,
                           thrust::make_permutation_iterator(w, indicesw));
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indicesv,
                    Cuda xpuv, std::size_t n, data<T, Cpu> w,
                    vector_const_iterator<IndexType, Cpu> indicesw, Cpu xpuw, EWOp::Add) {
            (void)xpuv;
            (void)xpuw;
            auto vit = thrust::make_permutation_iterator(v, indicesv);
            thrust::transform(vit, vit + n, thrust::make_permutation_iterator(w, indicesw),
                              thrust::make_permutation_iterator(w, indicesw), thrust::plus<T>());
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indicesv,
                    Cuda xpuv, std::size_t n, data<T, Cuda> w,
                    vector_const_iterator<IndexType, Cuda> indicesw, Cuda xpuw, EWOp::Copy) {
            (void)xpuv;
            (void)xpuw;
            thrust::copy_n(thrust::make_permutation_iterator(v, indicesv), n,
                           thrust::make_permutation_iterator(w, indicesw));
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indicesv,
                    Cuda xpuv, std::size_t n, data<T, Cuda> w,
                    vector_const_iterator<IndexType, Cuda> indicesw, Cuda xpuw, EWOp::Add) {
            (void)xpuv;
            (void)xpuw;
            auto vit = thrust::make_permutation_iterator(v, indicesv);
            thrust::transform(vit, vit + n, thrust::make_permutation_iterator(w, indicesw),
                              thrust::make_permutation_iterator(w, indicesw), thrust::plus<T>());
        }

        /// Copy and reduce n values, w[indicesw[i]] += sum(v[perm[perm_distinct[i]:perm_distinct[i+1]]])
        template <typename IndexType, typename T>
        void copy_reduce_n(data<const T, Cpu> v, Cpu, vector_const_iterator<IndexType, Cpu> perm,
                           vector_const_iterator<IndexType, Cpu> perm_distinct,
                           std::size_t ndistinct, Cpu cpuv, data<T, Cuda> w,
                           vector_const_iterator<IndexType, Cuda> indicesw, Cuda cudaw) {
            (void)cpuv;
            (void)cudaw;
            std::vector<T> w_host(ndistinct - 1);
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < ndistinct - 1; ++i)
                for (std::size_t j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                    w_host[i] += v[perm[j]];
            vector<T, Cuda> w_device = w_host;

            thrust::transform(w_device.begin(), w_device.end(),
                              thrust::make_permutation_iterator(w, indicesw),
                              thrust::make_permutation_iterator(w, indicesw), thrust::plus<T>());
        }

#endif // SUPERBBLAS_USE_CUDA

        /// Set the first `n` elements with a value
        /// \param it: first element to set
        /// \param n: number of elements to set
        /// \param v: value to set
        /// \param cpu: device context

        template <typename Iterator>
        void fill_n(Iterator it, std::size_t n,
                    typename std::iterator_traits<Iterator>::value_type v, Cpu cpu) {
            (void)cpu;
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) it[i] = v;
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Set the first `n` elements with a value
        /// \param it: first element to set
        /// \param n: number of elements to set
        /// \param v: value to set
        /// \param cpu: device context

        template <typename Iterator>
        void fill_n(Iterator it, std::size_t n,
                    typename std::iterator_traits<Iterator>::value_type v, Cuda cuda) {
            (void)cuda;
            thrust::fill(it, it + n, v);
        }
#endif

        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename Iterator>
        typename std::iterator_traits<Iterator>::value_type *raw_pointer(Iterator it) {
            return &*it;
        }

        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename Iterator>
        const typename std::iterator_traits<Iterator>::value_type *const_raw_pointer(Iterator it) {
            return &*it;
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename T> T *raw_pointer(thrust::device_ptr<T> it) { return it.get(); }

        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename T> const T *const_raw_pointer(thrust::device_ptr<T> it) {
            return it.get();
        }
#endif

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

#ifdef SUPERBBLAS_USE_CUDA
        template <typename T> inline cudaDataType_t toCudaDataType(void);

        template <> inline cudaDataType_t toCudaDataType<float>(void) { return CUDA_R_32F; }
        template <> inline cudaDataType_t toCudaDataType<std::complex<float>>(void) {
            return CUDA_C_32F;
        }
        template <> inline cudaDataType_t toCudaDataType<double>(void) { return CUDA_R_64F; }
        template <> inline cudaDataType_t toCudaDataType<std::complex<double>>(void) {
            return CUDA_C_64F;
        }

        template <typename T> inline cublasComputeType_t toCudaComputeType(void);

        template <> inline cublasComputeType_t toCudaComputeType<float>(void) {
            return CUBLAS_COMPUTE_32F;
        }
        template <> inline cublasComputeType_t toCudaComputeType<std::complex<float>>(void) {
            return CUBLAS_COMPUTE_32F;
        }
        template <> inline cublasComputeType_t toCudaComputeType<double>(void) {
            return CUBLAS_COMPUTE_64F;
        }
        template <> inline cublasComputeType_t toCudaComputeType<std::complex<double>>(void) {
            return CUBLAS_COMPUTE_64F;
        }

        inline cublasOperation_t toCublasTrans(char trans) {
            switch (trans) {
            case 'n':
            case 'N': return CUBLAS_OP_N;
            case 't':
            case 'T': return CUBLAS_OP_T;
            case 'c':
            case 'C': return CUBLAS_OP_C;
            default: throw std::runtime_error("Not valid value of trans");
            }
        }


        template <typename T>
        void xgemm_batch_strided(char transa, char transb, int m, int n, int k, T alpha, const T *a,
                                 int lda, int stridea, const T *b, int ldb, int strideb, T beta,
                                 T *c, int ldc, int stridec, int batch_size, Cuda cuda) {

            cudaDataType_t cT = toCudaDataType<T>();
            cublasCheck(cublasGemmStridedBatchedEx(
                cuda.cublasHandle, toCublasTrans(transa), toCublasTrans(transb), m, n, k, &alpha,
                a, cT, lda, stridea, b, cT, ldb, strideb, &beta, c, cT, ldc, stridec, batch_size,
                toCudaComputeType<T>(), CUBLAS_GEMM_DEFAULT));
        }
#endif // SUPERBBLAS_USE_CUDA
    }
}


#endif // __SUPERBBLAS_BLAS__
