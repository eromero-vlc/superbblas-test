#ifndef __SUPERBBLAS_BLAS__
#define __SUPERBBLAS_BLAS__

#include "platform.h"
#include <stdexcept>
#include <type_traits>
#include <utility>
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

//////////////////////
// NOTE:
// Functions in this file that uses `thrust` should be instrumented to remove the dependency from
// `thrust` when the superbblas library is used not as header-only. Use the macro `IMPL` to hide
// the definition of functions using `thrust` and use DECL_... macros to generate template
// instantiations to be included in the library.

#ifdef SUPERBBLAS_USE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#    ifndef SUPERBBLAS_LIB
#        include <thrust/complex.h>
#        include <thrust/device_ptr.h>
#        include <thrust/device_vector.h>
#        include <thrust/fill.h>
#        include <thrust/iterator/permutation_iterator.h>
#        define SUPERBBLAS_USE_THRUST
#    endif
#endif

#ifdef SUPERBBLAS_CREATING_FLAGS
#    ifdef SUPERBBLAS_USE_CBLAS
EMIT_define(SUPERBBLAS_USE_CBLAS)
#    endif
    ;
#endif

#ifdef SUPERBBLAS_CREATING_LIB
#    define SUPERBBLAS_REAL_TYPES float, double
#    define SUPERBBLAS_COMPLEX_TYPES std::complex<float>, std::complex<double>
#    define SUPERBBLAS_TYPES SUPERBBLAS_REAL_TYPES, SUPERBBLAS_COMPLEX_TYPES
// #    define SUPERBBLAS_ARRAY_RANGES 3, 4, 12

#if SUPERBBLAS_ARRAY_RANGES
// clang-format off
#    define META_ARRAY_TYPES_T                                                                     \
        REPLACE $(T $ T $ std::array<T,NNN>) REPLACE(NNN, SUPERBBLAS_ARRAY_RANGES)
// clang-format on
#else
#    define META_ARRAY_TYPES_T
#endif

/// Generate template instantiations for copy_n functions with template parameters IndexType and T

#    define DECL_COPY_T(...)                                                                       \
        EMIT REPLACE1(copy_n, superbblas::detail::copy_n<IndexType, T>)                            \
            META_ARRAY_TYPES_T REPLACE(T, SUPERBBLAS_TYPES) template __VA_ARGS__;

// When generating template instantiations for copy_n functions with different input and output
// types, avoid copying from complex types to non-complex types (note the missing TCOMPLEX QREAL
// from the definition of macro META_TYPES)

#    define META_TYPES TREAL QREAL, TREAL QCOMPLEX, TCOMPLEX QCOMPLEX
#    define REPLACE_META_TYPES                                                                     \
        REPLACE(TREAL, SUPERBBLAS_REAL_TYPES)                                                      \
        REPLACE(QREAL, SUPERBBLAS_REAL_TYPES) REPLACE(TCOMPLEX, SUPERBBLAS_COMPLEX_TYPES)          \
            REPLACE(QCOMPLEX, SUPERBBLAS_COMPLEX_TYPES)

#if SUPERBBLAS_ARRAY_RANGES
// clang-format off
#    define META_ARRAY_TYPES_T_Q                                                                   \
        REPLACE$(T Q $ T Q $ std::array<T,NNN> std::array<Q,NNN>)                                  \
        REPLACE(NNN, SUPERBBLAS_ARRAY_RANGES)
// clang-format on
#else
#        define META_ARRAY_TYPES_T_Q REPLACE(T Q, IndexType IndexType, T Q)
#endif

/// Generate template instantiations for copy_n functions with template parameters IndexType, T and Q

#    define DECL_COPY_T_Q(...)                                                                     \
        EMIT REPLACE1(copy_n, superbblas::detail::copy_n<IndexType, T, Q>) META_ARRAY_TYPES_T_Q    \
        REPLACE(T Q, META_TYPES) REPLACE_META_TYPES template __VA_ARGS__;

/// Generate template instantiations for copy_reduce_n functions with template parameters IndexType and T

#    define DECL_COPY_REDUCE(...)                                                                  \
        EMIT REPLACE1(copy_reduce_n, superbblas::detail::copy_reduce_n<IndexType, T>)              \
            REPLACE(T, SUPERBBLAS_TYPES) template __VA_ARGS__;
#else
#    define DECL_COPY_T(...) __VA_ARGS__
#    define DECL_COPY_T_Q(...) __VA_ARGS__
#    define DECL_COPY_REDUCE(...) __VA_ARGS__
#endif

namespace superbblas {

    namespace detail {

        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param cpu: context

        template <typename T> T *allocate(std::size_t n, Cpu) {
            T *r = new T[n];
            if (n > 0 && r == nullptr) std::runtime_error("Memory allocation failed!");
            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param cpu: context

        template <typename T> void deallocate(T *ptr, Cpu) {
            delete[] ptr;
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param cuda: context

        template <typename T> T *allocate(std::size_t n, Cuda cuda) {
            cudaCheck(cudaSetDevice(deviceId(cuda)));
            T *r = nullptr;
            cudaCheck(cudaMalloc(&r, sizeof(T) * n));
            if (n > 0 && r == nullptr) std::runtime_error("Memory allocation failed!");
            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param cuda: context

        template <typename T> void deallocate(T *ptr, Cuda cuda) {
            detail::cudaCheck(cudaSetDevice(deviceId(cuda)));
            detail::cudaCheck(cudaFree((void *)ptr));
        }
#endif

#ifdef SUPERBBLAS_USE_THRUST
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
#endif // SUPERBBLAS_USE_THRUST

        /// Vector type a la python, that is, operator= does a reference not a copy
        /// \param T: type of the vector's elements
        /// \param XPU: device type, one of Cpu, Cuda, Gpuamd

        template <typename T, typename XPU> struct vector {
            /// Type `T` without const
            using T_no_const = typename std::remove_const<T>::type;

            /// Type returned by `begin()` and `end()`
            using iterator = T *;

            /// Default constructor: create an empty vector
            vector() : vector(0, XPU{}) {}

            /// `Cpu` vectors can construct a `vector` without providing a device context
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            vector(std::size_t n = 0) : vector(n, Cpu{}) {}

            /// Construct a vector with `n` elements a with context device `xpu_`
            vector(std::size_t n, XPU xpu_)
                : n(n),
                  ptr(allocate<T_no_const>(n, xpu_),
                      [=](const T_no_const *ptr) { deallocate(ptr, xpu_); }),
                  xpu(xpu_) {}

            /// Construct a vector from a given pointer `ptr` with `n` elements and with context
            /// device `xpu`. `ptr` is not deallocated after the destruction of the `vector`.
            vector(std::size_t n, T *ptr, XPU xpu)
                : n(n), ptr((T_no_const *)ptr, [&](const T_no_const *) {}), xpu(xpu) {}

            /// Low-level constructor
            vector(std::size_t n, std::shared_ptr<T_no_const> ptr, XPU xpu)
                : n(n), ptr(ptr), xpu(xpu) {}

            /// Return the number of allocated elements
            std::size_t size() const { return n; }

            /// Return a pointer to the allocated space
            T *data() const { return ptr.get(); }

            /// Return a pointer to the first element allocated
            T *begin() const { return ptr.get(); }

            /// Return a pointer to the first element non-allocated after an allocated element
            T *end() const { return ptr.get() + n; }

            /// Return the device context
            XPU ctx() const { return xpu; }

            /// Return a reference to i-th allocated element, for Cpu `vector`
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            T &operator[](std::size_t i) const {
                return ptr.get()[i];
            }

            /// Conversion from `vector<T, XPU>` to `vector<const T, XPU>`
            template <typename U = T, typename std::enable_if<std::is_same<U, T_no_const>::value,
                                                              bool>::type = true>
            operator vector<const T, XPU>() const {
                return {n, ptr, xpu};
            }

        private:
            std::size_t n;                     ///< Number of allocated `T` elements
            std::shared_ptr<T_no_const> ptr;   ///< Pointer to the allocated memory
            XPU xpu;                           ///< Context
        };

        /// Construct a `vector<T, Cpu>` with the given pointer and context

        template <typename T> vector<T, Cpu> to_vector(T *ptr, Cpu cpu) {
            return vector<T, Cpu>(0, ptr, cpu);
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Construct a `vector<T, Cuda>` with the given pointer and context

        template <typename T> vector<T, Cuda> to_vector(T *ptr, Cuda cuda) {
            return vector<T, Cuda>(0, ptr, cuda);
        }
#endif

#ifdef SUPERBBLAS_USE_THRUST
        /// Return a device pointer suitable for making iterators

        template <typename T>
        thrust::device_ptr<typename cuda_complex<T>::type> encapsulate_pointer(T *ptr) {
            return thrust::device_pointer_cast(
                reinterpret_cast<typename cuda_complex<T>::type *>(ptr));
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
        void copy_n(const T *v, Cpu, std::size_t n, T *w, Cpu, EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[i] = v[i];
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T>
        void copy_n(const T *v, Cpu, std::size_t n, T *w, Cpu, EWOp::Add) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[i] += v[i];
        }

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T, typename Q>
        void copy_n(const T *v, const IndexType *indices, Cpu, std::size_t n, Q *w, Cpu,
                    EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[i] = v[indices[i]];
        }

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(const T *v, Cpu, std::size_t n, Q *w, const IndexType *indices, Cpu,
                    EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indices[i]] = v[i];
        }

        /// Copy n values, w[indices[i]] += v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(const T *v, Cpu, std::size_t n, Q *w, const IndexType *indices, Cpu,
                    EWOp::Add) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indices[i]] += v[i];
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n(const T *v, const IndexType *indicesv, Cpu, std::size_t n, Q *w,
                    const IndexType *indicesw, Cpu, EWOp::Copy) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = v[indicesv[i]];
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n(const T *v, const IndexType *indicesv, Cpu, std::size_t n, Q *w,
                    const IndexType *indicesw, Cpu, EWOp::Add) {
#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] += v[indicesv[i]];
        }

        /// Copy and reduce n values, w[indicesw[i]] += sum(v[perm[perm_distinct[i]:perm_distinct[i+1]]])
        template <typename IndexType, typename T>
        void copy_reduce_n(const T *v, Cpu, const IndexType *perm, const IndexType *perm_distinct,
                           std::size_t ndistinct, Cpu, T *w, const IndexType *indicesw, Cpu) {
            if (indicesw) {
#ifdef _OPENMP
#    pragma omp for
#endif
                for (std::size_t i = 0; i < ndistinct - 1; ++i)
                    for (IndexType j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                        w[indicesw[i]] += v[perm[j]];
            } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                for (std::size_t i = 0; i < ndistinct - 1; ++i)
                    for (IndexType j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                        w[i] += v[perm[j]];
            }
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(const T *v, Cuda cudav, std::size_t n, T *w, Cpu, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            cudaCheck(cudaMemcpy(w, v, sizeof(T) * n, cudaMemcpyDeviceToHost));
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T>
        void copy_n(const T *v, Cuda cudav, std::size_t n, T *w, Cpu, EWOp::Add) {
            vector<T, Cpu> t(n, Cpu{});
            copy_n<IndexType, T>(v, cudav, n, t.data(), Cpu{}, EWOp::Copy{});
            copy_n<IndexType, T>(t.data(), Cpu{}, n, w, Cpu{}, EWOp::Add{});
        }

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(const T *v, Cpu, std::size_t n, T *w, Cuda cudaw, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            cudaCheck(cudaMemcpy(w, v, sizeof(T) * n, cudaMemcpyHostToDevice));
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T>
        void copy_n(const T *v, Cpu, std::size_t n, T *w, Cuda cudaw, EWOp::Add) {
            vector<T, Cuda> t(n, cudaw);
            copy_n<IndexType, T>(v, Cpu{}, n, t.data(), cudaw, EWOp::Copy{});
            copy_n<IndexType, T>(t.data(), cudaw, n, w, cudaw, EWOp::Add{});
        }

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T>
        void copy_n(const T *v, Cuda cudav, std::size_t n, T *w, Cuda cudaw, EWOp::Copy) {
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            if (deviceId(cudav) == deviceId(cudaw)) {
                cudaCheck(cudaMemcpy(w, v, sizeof(T) * n, cudaMemcpyDeviceToDevice));
            } else {
                cudaCheck(cudaMemcpyPeer(w, deviceId(cudaw), v, deviceId(cudav), sizeof(T) * n));
            }
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T>
        DECL_COPY_T(void copy_n(const T *v, Cuda cudav, std::size_t n, T *w, Cuda cudaw, EWOp::Add))
        IMPL({
            if (deviceId(cudav) == deviceId(cudaw)) {
                cudaCheck(cudaSetDevice(deviceId(cudaw)));
                thrust::transform(encapsulate_pointer(v), encapsulate_pointer(v) + n,
                                  encapsulate_pointer(w), encapsulate_pointer(w),
                                  thrust::plus<typename cuda_complex<T>::type>());
            } else {
                vector<T, Cuda> t(n, cudaw);
                copy_n<IndexType, T>(v, cudav, n, t.data(), cudaw, EWOp::Copy{});
                copy_n<IndexType, T>(t.data(), cudaw, n, w, cudaw, EWOp::Add{});
            }
        });

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, const IndexType *indices, Cuda cudav, std::size_t n,
                                  Q *w, Cpu, EWOp::Copy))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            thrust::copy_n(thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                             encapsulate_pointer(indices)),
                           n, (typename cuda_complex<Q>::type *)w);
        });

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, const IndexType *indices, Cpu, std::size_t n, Q *w,
                                  Cuda cudaw, EWOp::Copy))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            thrust::copy_n(
                thrust::make_permutation_iterator((typename cuda_complex<T>::type *)v, indices), n,
                encapsulate_pointer(w));
        });

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, Cpu, std::size_t n, Q *w, const IndexType *indices,
                                  Cuda cudaw, EWOp::Copy))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            thrust::copy_n((typename cuda_complex<T>::type *)v, n,
                           thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                             encapsulate_pointer(indices)));
        });

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, Cuda cudav, std::size_t n, Q *w,
                                  const IndexType *indices, EWOp::Copy))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            thrust::copy_n(encapsulate_pointer(v), n,
                           thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                             encapsulate_pointer(indices)));
        });

#    ifdef SUPERBBLAS_USE_THRUST
        /// Addition of two values with different types
        template <typename T, typename Q> struct plus {
            typedef T first_argument_type;

            typedef Q second_argument_type;

            typedef Q result_type;

            __thrust_exec_check_disable__ __host__ __device__ result_type
            operator()(const T &lhs, const Q &rhs) const {
                return lhs + rhs;
            }
        };
#    endif // SUPERBBLAS_USE_THRUST

        /// Copy n values, w[indices[i]] += v[i]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, Cuda cudav, std::size_t n, Q *w,
                                  const IndexType *indices, EWOp::Add))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            auto itw = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                         encapsulate_pointer(indices));
            thrust::transform(
                encapsulate_pointer(v), encapsulate_pointer(v) + n, itw, itw,
                plus<typename cuda_complex<T>::type, typename cuda_complex<Q>::type>());
        });

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, const IndexType *indicesv, Cpu, std::size_t n, Q *w,
                                  const IndexType *indicesw, Cuda cudaw, EWOp::Copy))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            thrust::copy_n(
                thrust::make_permutation_iterator((typename cuda_complex<T>::type *)v, indicesv), n,
                thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                  encapsulate_pointer(indicesw)));
        });

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, const IndexType *indicesv, Cpu, std::size_t n, Q *w,
                                  const IndexType *indicesw, Cuda cudaw, EWOp::Add))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudaw)));
            std::vector<Q> v_gather(n);
            copy_n<IndexType, T, Q>(v, indicesv, Cpu{}, n, v_gather.data(), Cpu{}, EWOp::Copy{});
            vector<Q, Cuda> v_dev(n, cudaw);
            copy_n<IndexType, Q>(v_gather.data(), Cpu{}, n, v_dev.data(), cudaw, EWOp::Copy{});
            auto itw = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                         encapsulate_pointer(indicesw));
            thrust::transform(encapsulate_pointer(v_dev.begin()), encapsulate_pointer(v_dev.end()),
                              itw, itw, thrust::plus<typename cuda_complex<Q>::type>());
        });

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, const IndexType *indicesv, Cuda cudav, std::size_t n,
                                  Q *w, const IndexType *indicesw, Cpu, EWOp::Copy))
        IMPL({
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            thrust::copy_n(
                thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                  encapsulate_pointer(indicesv)),
                n,
                thrust::make_permutation_iterator((typename cuda_complex<Q>::type *)w, indicesw));
        });

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T, typename Q>
        void copy_n(const T *v, const IndexType *indicesv, Cuda cudav, std::size_t n, Q *w,
                    const IndexType *indicesw, Cpu cpuw, EWOp::Add) {
            vector<T, Cpu> v_gather(n, Cpu{});
            copy_n<IndexType, T, T>(v, indicesv, cudav, n, v_gather.data(), cpuw, EWOp::Copy{});
            copy_n<IndexType, T, Q>(v_gather.data(), cpuw, n, w, indicesw, cpuw, EWOp::Add{});
        }

#    ifdef SUPERBBLAS_USE_THRUST

        /// Copy n values, w[indicesw[i]] = (or +=) v[indicesv[i]] cudav and cudaw being on
        /// different devices

        template <typename IndexType, typename T, typename Q, typename EWOP>
        void copy_n_gen(const T *v, const IndexType *indicesv, Cuda cudav, std::size_t n, Q *w,
                        const IndexType *indicesw, Cuda cudaw, EWOP) {
            vector<T, Cuda> v_gather(n, cudav);
            cudaCheck(cudaSetDevice(deviceId(cudav)));
            thrust::copy_n(thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                             encapsulate_pointer(indicesv)),
                           n, encapsulate_pointer(v_gather.data()));
            vector<T, Cuda> w_gather(n, cudaw);
            copy_n<IndexType, T>(v_gather.data(), cudav, n, w_gather.data(), cudaw, EWOp::Copy{});
            copy_n<IndexType, T, Q>(w_gather.data(), cudaw, n, w, indicesw, EWOP{});
        }

        /// Assign array to another array of the same type

        template <typename IndexType, typename T, typename Q, std::size_t N> struct assign_array {

            const T *const v;
            const IndexType *const indicesv;
            Q *const w;
            const IndexType *const indicesw;

            assign_array(const T *v, const IndexType *indicesv, Q *w, const IndexType *indicesw)
                : v(v), indicesv(indicesv), w(w), indicesw(indicesw) {}

            typedef IndexType first_argument_type;

            typedef void result_type;

            __thrust_exec_check_disable__ __host__ __device__ result_type
            operator()(const IndexType &i) const {
                w[indicesw[i / N] * N + i % N] = v[indicesv[i / N] * N + i % N];
            }
        };

#    endif // SUPERBBLAS_USE_THRUST

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]

        template <typename T> struct is_std_array : std::false_type {};
        template <typename T, std::size_t N>
        struct is_std_array<std::array<T, N>> : std::true_type {};

        template <typename IndexType, typename T, typename Q,
                  typename std::enable_if<!is_std_array<T>::value, bool>::type = true>
        DECL_COPY_T_Q(void copy_n(const T *v, const IndexType *indicesv, Cuda cudav, std::size_t n,
                                  Q *w, const IndexType *indicesw, Cuda cudaw, EWOp::Copy))
        IMPL({
            if (deviceId(cudav) == deviceId(cudaw)) {
                thrust::copy_n(thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                                 encapsulate_pointer(indicesv)),
                               n,
                               thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                                 encapsulate_pointer(indicesw)));
            } else {
                copy_n_gen<IndexType, T, Q>(v, indicesv, cudav, n, w, indicesw, cudaw,
                                            EWOp::Copy{});
            }
        });

        template <typename IndexType, typename T, typename Q,
                  std::size_t N = std::tuple_size<T>::value>
        void copy_n(const T *v, const IndexType *indicesv, Cuda cudav, std::size_t n, Q *w,
                    const IndexType *indicesw, Cuda cudaw, EWOp::Copy) IMPL({
            if (deviceId(cudav) == deviceId(cudaw)) {
                // thrust::copy_n(thrust::make_permutation_iterator(v, indicesv), n,
                //                thrust::make_permutation_iterator(w, indicesw));
                thrust::for_each_n(
                    thrust::counting_iterator<IndexType>(0), n * N,
                    assign_array<IndexType, typename T::value_type, typename Q::value_type, N>(
                        (typename T::const_pointer)v, indicesv, (typename Q::pointer)w, indicesw));
            } else {
                copy_n_gen<IndexType, T, Q>(v, indicesv, cudav, n, w, indicesw, cudaw,
                                            EWOp::Copy{});
            }
        });

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]

        template <typename IndexType, typename T, typename Q>
        DECL_COPY_T_Q(void copy_n(const T *v, const IndexType *indicesv, Cuda cudav, std::size_t n,
                                  Q *w, const IndexType *indicesw, Cuda cudaw, EWOp::Add))
        IMPL({
            if (deviceId(cudav) == deviceId(cudaw)) {
                auto vit = thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                             encapsulate_pointer(indicesv));
                auto wit = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                             encapsulate_pointer(indicesw));
                thrust::transform(
                    vit, vit + n, wit, wit,
                    plus<typename cuda_complex<T>::type, typename cuda_complex<Q>::type>());
            } else {
                copy_n_gen<IndexType, T, Q, EWOp::Add>(v, indicesv, cudav, n, w, indicesw, cudaw,
                                                       EWOp::Add{});
            }
        });

        /// Copy and reduce n values, w[indicesw[i]] += sum(v[perm[perm_distinct[i]:perm_distinct[i+1]]])
        template <typename IndexType, typename T>
        DECL_COPY_REDUCE(void copy_reduce_n(const T *v, Cpu, const IndexType *perm,
                                            const IndexType *perm_distinct, std::size_t ndistinct,
                                            Cpu cpuv, T *w, const IndexType *indicesw, Cuda cudaw))
        IMPL({
            std::vector<T> w_host(ndistinct - 1);
            copy_reduce_n<IndexType, T>(v, Cpu{}, perm, perm_distinct, ndistinct, cpuv,
                                        w_host.data(), nullptr, Cpu{});
            vector<T, Cuda> w_device(ndistinct - 1, cudaw);
            copy_n<IndexType, T>(w_host.data(), cpuv, ndistinct - 1, w_device.data(), cudaw,
                                 EWOp::Copy{});

            auto itw = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                         encapsulate_pointer(indicesw));
            thrust::transform(encapsulate_pointer(w_device.begin()),
                              encapsulate_pointer(w_device.end()), itw, itw,
                              thrust::plus<typename cuda_complex<T>::type>());
        });

#endif // SUPERBBLAS_USE_CUDA

        /// Set the first `n` elements with a value
        /// \param it: first element to set
        /// \param n: number of elements to set
        /// \param v: value to set
        /// \param cpu: device context

        template <typename T> void zero_n(T *v, std::size_t n, Cpu) {
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

        template <typename T> void zero_n(T *v, std::size_t n, Cuda cuda) {
            cudaCheck(cudaSetDevice(deviceId(cuda)));
            cudaCheck(cudaMemset(v, 0, sizeof(T) * n));
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

    /// Allocate memory on a device
    /// \param n: number of element of type `T` to allocate
    /// \param ctx: context
 
    template <typename T> T *allocate(std::size_t n, Context ctx) {
        switch (ctx.plat) {
        case CPU: return detail::allocate<T>(n, ctx.toCpu());
#ifdef SUPERBBLAS_USE_CUDA
        case CUDA: return detail::allocate<T>(n, ctx.toCuda());
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
        return nullptr;
    }

    /// Deallocate memory on a device
    /// \param ptr: pointer to the memory to deallocate
    /// \param ctx: context

    template <typename T> void deallocate(T *ptr, Context ctx) {
        switch (ctx.plat) {
        case CPU: detail::deallocate(ptr, ctx.toCpu()); break;
#ifdef SUPERBBLAS_USE_CUDA
        case CUDA: detail::deallocate(ptr, ctx.toCuda()); break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }
}

#endif // __SUPERBBLAS_BLAS__
