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
#    include <thrust/complex.h>
#    include <thrust/device_ptr.h>
#    include <thrust/device_vector.h>
#    include <thrust/fill.h>
#    include <thrust/iterator/permutation_iterator.h>
#endif

namespace superbblas {

    namespace detail {

#ifdef SUPERBBLAS_USE_CUDA
        /// Replace std::complex by thrust complex
        /// \tparam T: one of float, double, std::complex<T>, std::array<T,N>
        /// \return cuda_complex<T>::type has the new type

        template <typename T> struct cuda_complex { using type = T; };
        template <typename T> struct cuda_complex<std::complex<T>> {
            using type = thrust::complex<T>;
        };
        template <typename T> struct cuda_complex<const T> {
            using type = const typename cuda_complex<T>::type;
        };
        template <typename T, std::size_t N> struct cuda_complex<std::array<T, N>> {
            using type = std::array<typename cuda_complex<T>::type, N>;
        };

        /// Replace const T by const thrust::device_ptr<T> and T by thrust::device_ptr<T>
        /// \tparam T: one of float, double, std::complex<T>, std::array<T,N>
        /// \return cuda_ptr<T>::type has the new type

        template <typename T> struct cuda_ptr { using type = thrust::device_ptr<T>; };
        template <typename T> struct cuda_ptr<const T> {
            using type = const typename cuda_ptr<T>::type;
        };
#endif // SUPERBBLAS_USE_CUDA

        /// Vector type
        /// \param T: type of the vector's elements
        /// \param XPU: device type, one of Cpu, Cuda, Gpuamd

        template <typename T, typename XPU>
        using vector = typename std::conditional<
            std::is_same<XPU, Cpu>::value, std::vector<T>,
#ifdef SUPERBBLAS_USE_CUDA
            typename std::conditional<std::is_same<XPU, Cuda>::value,
                                      thrust::device_vector<typename cuda_complex<T>::type>,
                                      void>::type
#else
            void
#endif // SUPERBBLAS_USE_CUDA
            >::type;

        /// Return a std::vector like for the given data type and platform
        /// \tparam T: one of float, double, std::complex<T>, std::array<T,N>
        /// \param size: length of the new vector
        /// xpu: device context

        template <typename T> vector<T, Cpu> new_vector(std::size_t size, Cpu) {
            return vector<T, Cpu>(size);
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Return a std::vector like for the given data type and platform
        /// \tparam T: one of float, double, std::complex<T>, std::array<T,N>
        /// \param size: length of the new vector
        /// xpu: device context

        template <typename T> vector<T, Cuda> new_vector(std::size_t size, Cuda cuda) {
            static_assert(std::is_same<typename std::remove_const<T>::type, T>::value,
                          "No const type supported!");
            cudaCheck(cudaSetDevice(deviceId(cuda)));
            vector<T, Cuda> t(size);
            return t;
        }
#endif // SUPERBBLAS_USE_CUDA

        /// Constant iterator vector type
        /// \param T: type of the vector's elements
        /// \param XPU: device type, one of Cpu, Cuda, Gpuamd

        template <typename T, typename XPU>
        using vector_const_iterator = typename vector<T, XPU>::const_iterator;

        /// Pointer to data type
        /// \param T: type of the vector's elements
        /// \param XPU: device type, one of Cpu, Cuda, Gpuamd

        template <typename T, typename XPU>
        using data = typename std::conditional<
            std::is_same<XPU, Cpu>::value, T *,
#ifdef SUPERBBLAS_USE_CUDA
            typename std::conditional<std::is_same<XPU, Cuda>::value,
                                      typename cuda_ptr<typename cuda_complex<T>::type>::type,
                                      void>::type
#else
                                      void
#endif // SUPERBBLAS_USE_CUDA
            >::type;

#ifdef SUPERBBLAS_USE_CUDA
        /// Return a device pointer suitable for making iterators

        template <typename T>
        thrust::device_ptr<typename cuda_complex<T>::type> encapsulate_pointer(T *ptr) {
            return thrust::device_pointer_cast(
                reinterpret_cast<typename cuda_complex<T>::type *>(ptr));
        }
#endif

        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename T> T *raw_pointer(data<T, Cpu> v) { return &*v; }

        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename T> const T *const_raw_pointer(data<const T, Cpu> v) { return &*v; }

#ifdef SUPERBBLAS_USE_CUDA
        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename T> T *raw_pointer(data<T, Cuda> v) {
            return reinterpret_cast<T *>(v.get());
        }

        /// Return the pointer associated to an iterator
        /// \param it: iterator

        template <typename T> const T *const_raw_pointer(data<const T, Cuda> v) {
            return reinterpret_cast<const T *>(v.get());
        }
#endif

        inline void sync(Cpu) {}

#ifdef SUPERBBLAS_USE_CUDA
        inline void sync(Cuda cuda) {
            cudaCheck(cudaSetDevice(deviceId(cuda)));
            cudaDeviceSynchronize();
        }
#endif

        namespace EWOp {
            /// Copy the values of the origin vector into the destination vector
            struct Copy {};

            /// Add the values from the origin vector to the destination vector
            struct Add {};
        }

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu, std::size_t n, data<T, Cpu> w, Cpu, EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[i] = v[i];
        }

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu, std::size_t n, data<T, Cpu> w, Cpu, EWOp::Add) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[i] += v[i];
        }

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indices, Cpu,
                    std::size_t n, data<T, Cpu> w, Cpu, EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[i] = v[indices[i]];
        }

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu, std::size_t n, data<T, Cpu> w,
                    vector_const_iterator<IndexType, Cpu> indices, Cpu, EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indices[i]] = v[i];
        }

        /// Copy n values, w[indices[i]] += v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu, std::size_t n, data<T, Cpu> w,
                    vector_const_iterator<IndexType, Cpu> indices, Cpu, EWOp::Add) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indices[i]] += v[i];
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu,
                    std::size_t n, data<T, Cpu> w, vector_const_iterator<IndexType, Cpu> indicesw,
                    Cpu, EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = v[indicesv[i]];
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu,
                    std::size_t n, data<T, Cpu> w, vector_const_iterator<IndexType, Cpu> indicesw,
                    Cpu, EWOp::Add) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] += v[indicesv[i]];
        }

        /// Copy and reduce n values, w[indicesw[i]] += sum(v[perm[perm_distinct[i]:perm_distinct[i+1]]])
        template <typename IndexType, typename T>
        void copy_reduce_n(data<const T, Cpu> v, Cpu, vector_const_iterator<IndexType, Cpu> perm,
                           vector_const_iterator<IndexType, Cpu> perm_distinct,
                           std::size_t ndistinct, Cpu, data<T, Cpu> w,
                           vector_const_iterator<IndexType, Cpu> indicesw, Cpu) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < ndistinct - 1; ++i)
                for (IndexType j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                    w[indicesw[i]] += v[perm[j]];
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, Cuda cudav, std::size_t n, data<T, Cpu> w, Cpu,
                    EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            cudaCheck(cudaMemcpy(raw_pointer<T>(w), const_raw_pointer<T>(v), sizeof(T) * n,
                                 cudaMemcpyDeviceToHost));
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, Cuda cudav, std::size_t n, data<T, Cpu> w, Cpu,
                    EWOp::Add) {
            vector<T, Cpu> t(n);
            copy_n<IndexType, T>(v, cudav, n, t.data(), Cpu{}, EWOp::Copy{});
            copy_n<IndexType, T>(t.data(), Cpu{}, n, w, Cpu{}, EWOp::Add{});
        }

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu , std::size_t n, data<T, Cuda> w, Cuda cudaw,
                    EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            cudaCheck(cudaMemcpy(raw_pointer<T>(w), const_raw_pointer<T>(v), sizeof(T) * n,
                                 cudaMemcpyHostToDevice));
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu , std::size_t n, data<T, Cuda> w, Cuda cudaw,
                    EWOp::Add) {
            vector<T, Cuda> t = new_vector<T>(n, cudaw);
            copy_n<IndexType, T>(v, Cpu{}, n, t.data(), cudaw, EWOp::Copy{});
            copy_n<IndexType, T>(t.data(), cudaw, n, w, cudaw, EWOp::Add{});
        }

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, Cuda cudav, std::size_t n, data<T, Cuda> w, Cuda cudaw,
                    EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            if (deviceId(cudav) == deviceId(cudaw)) {
                cudaCheck(cudaMemcpy(raw_pointer<T>(w), const_raw_pointer<T>(v), sizeof(T) * n,
                                     cudaMemcpyDeviceToDevice));
            } else {
                cudaCheck(cudaMemcpyPeer(raw_pointer<T>(w), deviceId(cudaw),
                                         const_raw_pointer<T>(v), deviceId(cudav), sizeof(T) * n));
            }
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, Cuda cudav, std::size_t n, data<T, Cuda> w, Cuda cudaw,
                    EWOp::Add) {
            if (deviceId(cudav) == deviceId(cudaw)) {
                cudaCheck(cudaSetDevice(deviceId(cudaw)));
                thrust::transform(v, v + n, w, w, thrust::plus<typename cuda_complex<T>::type>());
            } else {
                vector<T, Cuda> t = new_vector<T>(n, cudaw);
                copy_n<IndexType, T>(v, cudav, n, t.data(), cudaw, EWOp::Copy{});
                copy_n<IndexType, T>(t.data(), cudaw, n, w, cudaw, EWOp::Add{});
            }
        }

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indices,
                    Cuda cudav, std::size_t n, data<T, Cpu> w, Cpu, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            thrust::copy_n(thrust::make_permutation_iterator(v, indices), n,
                           (typename cuda_complex<T>::type *)w);
        }

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indices, Cpu,
                    std::size_t n, data<T, Cuda> w, Cuda cudaw, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            thrust::copy_n(
                thrust::make_permutation_iterator((typename cuda_complex<T>::type *)v, indices), n,
                w);
        }

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, Cpu, std::size_t n, data<T, Cuda> w,
                    vector_const_iterator<IndexType, Cuda> indices, Cuda cudaw, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            thrust::copy_n((typename cuda_complex<T>::type *)v, n,
                           thrust::make_permutation_iterator(w, indices));
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu,
                    std::size_t n, data<T, Cuda> w, vector_const_iterator<IndexType, Cuda> indicesw,
                    Cuda cudaw, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            thrust::copy_n(
                thrust::make_permutation_iterator((typename cuda_complex<T>::type *)v, indicesv), n,
                thrust::make_permutation_iterator(w, indicesw));
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cpu> v, vector_const_iterator<IndexType, Cpu> indicesv, Cpu,
                    std::size_t n, data<T, Cuda> w, vector_const_iterator<IndexType, Cuda> indicesw,
                    Cuda cudaw, EWOp::Add) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            std::vector<T> v_gather(n);
            copy_n<IndexType, T>(v, indicesv, Cpu{}, n, v_gather.data(), Cpu{}, EWOp::Copy{});
            vector<T, Cuda> v_dev = v_gather;
            thrust::transform(v_dev.begin(), v_dev.end(),
                              thrust::make_permutation_iterator(w, indicesw),
                              thrust::make_permutation_iterator(w, indicesw),
                              thrust::plus<typename cuda_complex<T>::type>());
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indicesv,
                    Cuda cudav, std::size_t n, data<T, Cpu> w,
                    vector_const_iterator<IndexType, Cpu> indicesw, Cpu, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            thrust::copy_n(
                thrust::make_permutation_iterator(v, indicesv), n,
                thrust::make_permutation_iterator((typename cuda_complex<T>::type *)w, indicesw));
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T>
        void copy_n(data<const T, Cuda> v, vector_const_iterator<IndexType, Cuda> indicesv,
                    Cuda cudav, std::size_t n, data<T, Cpu> w,
                    vector_const_iterator<IndexType, Cpu> indicesw, Cpu cpuw, EWOp::Add) {
            vector<T, Cpu> v_gather(n);
            copy_n<IndexType, T>(v, indicesv, cudav, n, v_gather.data(), cpuw, EWOp::Copy{});
            copy_n<IndexType, T>(v_gather.data(), cpuw, n, w, indicesw, cpuw, EWOp::Add{});
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
                              thrust::make_permutation_iterator(w, indicesw),
                              thrust::plus<typename cuda_complex<T>::type>());
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
                for (IndexType j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                    w_host[i] += v[perm[j]];
            vector<T, Cuda> w_device = w_host;

            thrust::transform(w_device.begin(), w_device.end(),
                              thrust::make_permutation_iterator(w, indicesw),
                              thrust::make_permutation_iterator(w, indicesw),
                              thrust::plus<typename cuda_complex<T>::type>());
        }

#endif // SUPERBBLAS_USE_CUDA

        /// Set the first `n` elements with a value
        /// \param it: first element to set
        /// \param n: number of elements to set
        /// \param v: value to set
        /// \param cpu: device context

        template <typename T>
        void zero_n(data<T, Cpu> v, std::size_t n, Cpu) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) v[i] = T{0};
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Set the first `n` elements with a zero value
        /// \param it: first element to set
        /// \param n: number of elements to set
        /// \param v: value to set
        /// \param cuda: device context

        template <typename T> void zero_n(data<T, Cuda> v, std::size_t n, Cuda cuda) {
            cudaCheck(cudaSetDevice(deviceId(cuda)));
            cudaCheck(cudaMemset(raw_pointer<T>(v), 0, sizeof(T)*n));
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
            // Quick exits
            if (m == 0 || n == 0) return;

            // Replace some invalid arguments when k is zero
            if (k == 0) {
                a = b = c;
                lda = ldb = 1;
            }

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
