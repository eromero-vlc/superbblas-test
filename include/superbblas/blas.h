#ifndef __SUPERBBLAS_BLAS__
#define __SUPERBBLAS_BLAS__

#include "performance.h"
#include "platform.h"
#include <algorithm>
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
#    ifndef SUPERBBLAS_LIB
#        include <thrust/complex.h>
#        include <thrust/device_ptr.h>
#        include <thrust/device_vector.h>
#        include <thrust/fill.h>
#        include <thrust/iterator/permutation_iterator.h>
#        include <thrust/iterator/transform_iterator.h>
#        define SUPERBBLAS_USE_THRUST
#    endif
#endif

#ifdef SUPERBBLAS_CREATING_FLAGS
#    ifdef SUPERBBLAS_USE_CBLAS
EMIT_define(SUPERBBLAS_USE_CBLAS)
#    endif
#endif

#ifdef SUPERBBLAS_CREATING_LIB
#    define SUPERBBLAS_REAL_TYPES float, double
#    define SUPERBBLAS_COMPLEX_TYPES std::complex<float>, std::complex<double>
#    define SUPERBBLAS_TYPES SUPERBBLAS_REAL_TYPES, SUPERBBLAS_COMPLEX_TYPES

// When generating template instantiations for copy_n functions with different input and output
// types, avoid copying from complex types to non-complex types (note the missing TCOMPLEX QREAL
// from the definition of macro META_TYPES)

#    define META_TYPES TREAL QREAL, TREAL QCOMPLEX, TCOMPLEX QCOMPLEX
#    define REPLACE_META_TYPES                                                                     \
        REPLACE(TREAL, SUPERBBLAS_REAL_TYPES)                                                      \
        REPLACE(QREAL, SUPERBBLAS_REAL_TYPES)                                                      \
        REPLACE(TCOMPLEX, SUPERBBLAS_COMPLEX_TYPES) REPLACE(QCOMPLEX, SUPERBBLAS_COMPLEX_TYPES)
#    define REPLACE_T_Q                                                                            \
        REPLACE(T Q, IndexType IndexType, T Q) REPLACE(T Q, META_TYPES) REPLACE_META_TYPES

#    define REPLACE_EWOP REPLACE(EWOP, EWOp::Copy, EWOp::Add)

/// Generate template instantiations for copy_n functions with template parameters IndexType, T and Q

#    define DECL_COPY_T_Q_EWOP(...)                                                                \
        EMIT REPLACE1(copy_n, superbblas::detail::copy_n<IndexType, T, Q, EWOP>)                   \
            REPLACE_T_Q REPLACE_EWOP template __VA_ARGS__;

/// Generate template instantiations for copy_reduce_n functions with template parameters IndexType and T

#    define DECL_COPY_REDUCE(...)                                                                  \
        EMIT REPLACE1(copy_reduce_n, superbblas::detail::copy_reduce_n<IndexType, T>)              \
            REPLACE(T, superbblas::IndexType, SUPERBBLAS_TYPES) template __VA_ARGS__;

#else
#    define DECL_COPY_T_Q_EWOP(...) __VA_ARGS__
#    define DECL_COPY_REDUCE(...) __VA_ARGS__
#endif

namespace superbblas {

    /// elem<T>::type is T::value_type if T is an array; otherwise it is T

    template <typename T> struct elem { using type = T; };
    template <typename T, std::size_t N> struct elem<std::array<T, N>> { using type = T; };

    namespace detail {

        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param cpu: context

        template <typename T> T *allocate(std::size_t n, Cpu) {
            // Shortcut for zero allocations
            if (n == 0) return nullptr;

            // Do the allocation
            T *r = new T[n];
            if (r == nullptr) std::runtime_error("Memory allocation failed!");

            // Annotate allocation
            if (getTrackingMemory()) {
                getAllocations()[(void *)r] = sizeof(T) * n;
                getCpuMemUsed() += double(sizeof(T) * n);
            }

            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param cpu: context

        template <typename T> void deallocate(T *ptr, Cpu) {
            // Shortcut for zero allocations
            if (!ptr) return;

            // Deallocate the pointer
            delete[] ptr;

            // Remove annotation
            if (getTrackingMemory()) {
                const auto &it = getAllocations().find((void *)ptr);
                if (it == getAllocations().end())
                    throw std::runtime_error("Unexpected pointer to deallocate");
                getCpuMemUsed() -= double(it->second);
                getAllocations().erase(it);
            }
        }

#ifdef SUPERBBLAS_USE_CUDA

        /// Set the current device as the one passed
        /// \param cuda: context

        inline void setDevice(Cuda cuda) {
            int currentDevice;
            cudaCheck(cudaGetDevice(&currentDevice));
            if (currentDevice != deviceId(cuda)) cudaCheck(cudaSetDevice(deviceId(cuda)));
        }

        inline void setDevice(Cpu) {}

        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param cuda: context

        template <typename T> T *allocate(std::size_t n, Cuda cuda) {
            // Shortcut for zero allocations
            if (n == 0) return nullptr;

            // Do the allocation
            setDevice(cuda);
            T *r = nullptr;
            if (cuda.alloc) {
                r = (T *)cuda.alloc(sizeof(T) * n, CUDA);
            } else {
                cudaCheck(cudaMalloc(&r, sizeof(T) * n));
            }
            if (r == nullptr) std::runtime_error("Memory allocation failed!");

            // Annotate allocation
            if (getTrackingMemory()) {
                getAllocations()[(void *)r] = sizeof(T) * n;
                getGpuMemUsed() += double(sizeof(T) * n);
            }

            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param cuda: context

        template <typename T> void deallocate(T *ptr, Cuda cuda) {
            // Shortcut for zero allocations
            if (!ptr) return;

            // Deallocate the pointer
            setDevice(cuda);
            if (cuda.dealloc)
                cuda.dealloc((void *)ptr, CUDA);
            else
                detail::cudaCheck(cudaFree((void *)ptr));

            // Remove annotation
            if (getTrackingMemory()) {
                const auto &it = getAllocations().find((void *)ptr);
                if (it == getAllocations().end())
                    throw std::runtime_error("Unexpected pointer to deallocate");
                getGpuMemUsed() -= double(it->second);
                getAllocations().erase(it);
            }
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

            /// Operator == compares size and content
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            bool operator==(const vector<T, U> &v) const {
                if (n != v.size()) return false;
                for (std::size_t i = 0; i < n; ++i)
                    if ((*this)[i] != v[i]) return false;
                return true;
            }

            /// Clone content
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            vector<T, Cpu> clone() const {
                vector<T_no_const, Cpu> r(n);
                std::copy_n(data(), n, r.data());
                return r;
            }

        private:
            std::size_t n;                   ///< Number of allocated `T` elements
            std::shared_ptr<T_no_const> ptr; ///< Pointer to the allocated memory
            XPU xpu;                         ///< Context
        };

        /// Construct a `vector<T, Cpu>` with the given pointer and context

        template <typename T> vector<T, Cpu> to_vector(T *ptr, std::size_t n = 0) {
            return vector<T, Cpu>(n, ptr, Cpu{});
        }

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
            setDevice(cuda);
            cudaCheck(cudaDeviceSynchronize());
        }
#endif

        template <typename T, std::size_t N>
        std::array<T, N> operator+(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = a[i] + b[i];
            return r;
        }

        template <typename T, std::size_t N>
        std::array<T, N> &operator+=(std::array<T, N> &a, const std::array<T, N> &b) {
            for (std::size_t i = 0; i < N; i++) a[i] += b[i];
            return a;
        }

        template <typename T, std::size_t N>
        std::array<T, N> operator*(T a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = a * b[i];
            return r;
        }

        namespace EWOp {
            /// Copy the values of the origin vector into the destination vector
            struct Copy {};

            /// Add the values from the origin vector to the destination vector
            struct Add {};
        }

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w, Cpu,
                    EWOp::Copy) {
            assert((n == 0 || (void *)v != (void *)w || std::is_same<T, Q>::value));
            if (alpha == typename elem<T>::type{1}) {
                if (std::is_same<T, Q>::value && (void *)v == (void *)w) return;
#ifdef _OPENMP
#    pragma omp for
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] = v[i];
            } else {
                if (std::is_same<T, Q>::value && (void *)v == (void *)w) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] *= alpha;
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] = alpha * v[i];
                }
            }
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w, Cpu,
                    EWOp::Add) {
            assert((n == 0 || (void *)v != (void *)w || std::is_same<T, Q>::value));
            if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp for
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] += v[i];
            } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] += alpha * v[i];
            }
        }

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indices, Cpu,
                    std::size_t n, Q *w, Cpu, EWOp::Copy) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Copy{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] = v[indices[i]];
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] = alpha * v[indices[i]];
                }
            }
        }

        /// Copy n values, w[i] += v[indices[i]]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indices, Cpu,
                    std::size_t n, Q *w, Cpu, EWOp::Add) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Add{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] += v[indices[i]];
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] += alpha * v[indices[i]];
                }
            }
        }

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w,
                    const IndexType *indices, Cpu, EWOp::Copy) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Copy{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] = v[i];
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] = alpha * v[i];
                }
            }
        }

        /// Copy n values, w[indices[i]] += v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w,
                    const IndexType *indices, Cpu, EWOp::Add) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Add{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] += v[i];
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] += alpha * v[i];
                }
            }
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indicesv, Cpu,
                    std::size_t n, Q *w, const IndexType *indicesw, Cpu, EWOp::Copy) {
            if (indicesv == nullptr) {
                copy_n(alpha, v, Cpu{}, n, w, indicesw, Cpu{}, EWOp::Copy{});
            } else if (indicesw == nullptr) {
                copy_n(alpha, v, indicesv, Cpu{}, n, w, Cpu{}, EWOp::Copy{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = v[indicesv[i]];
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = alpha * v[indicesv[i]];
                }
            }
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indicesv, Cpu,
                    std::size_t n, Q *w, const IndexType *indicesw, Cpu, EWOp::Add) {
            if (indicesv == nullptr) {
                copy_n(alpha, v, Cpu{}, n, w, indicesw, Cpu{}, EWOp::Add{});
            } else if (indicesw == nullptr) {
                copy_n(alpha, v, indicesv, Cpu{}, n, w, Cpu{}, EWOp::Add{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] += v[indicesv[i]];
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] += alpha * v[indicesv[i]];
                }
            }
        }

        /// Copy and reduce n values, w[indicesw[i]] += sum(v[perm[perm_distinct[i]:perm_distinct[i+1]]])
        template <typename IndexType, typename T>
        void copy_reduce_n(typename elem<T>::type alpha, const T *v, Cpu, const IndexType *perm,
                           const IndexType *perm_distinct, std::size_t ndistinct, Cpu, T *w,
                           const IndexType *indicesw, Cpu) {
            assert(ndistinct == 0 || (void *)v != (void *)w);
            if (alpha == typename elem<T>::type{1}) {
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
            } else {
                if (indicesw) {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < ndistinct - 1; ++i)
                        for (IndexType j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                            w[indicesw[i]] += alpha * v[perm[j]];
                } else {
#ifdef _OPENMP
#    pragma omp for
#endif
                    for (std::size_t i = 0; i < ndistinct - 1; ++i)
                        for (IndexType j = perm_distinct[i]; j < perm_distinct[i + 1]; ++j)
                            w[i] += alpha * v[perm[j]];
                }
            }
        }

#ifdef SUPERBBLAS_USE_CUDA
#    ifdef SUPERBBLAS_USE_THRUST
        /// Addition of two values with different types
        template <typename T, typename Q> struct plus {
            typedef T first_argument_type;

            typedef Q second_argument_type;

            typedef Q result_type;

            __host__ __device__ result_type operator()(const T &lhs, const Q &rhs) const {
                return lhs + rhs;
            }
        };

        // Scala of a number
        template <typename T>
        struct scale : public thrust::unary_function<typename cuda_complex<T>::type,
                                                     typename cuda_complex<T>::type> {
            using cuda_T = typename cuda_complex<T>::type;
            using scalar_type = typename elem<cuda_T>::type;
            const scalar_type a;
            scale(scalar_type a) : a(a) {}
            __host__ __device__ cuda_T operator()(const cuda_T &i) const { return a * i; }
        };

        template <typename T, typename Q, typename IteratorV, typename IteratorW>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, const IteratorW &itw,
                                    EWOp::Copy) {
            thrust::copy_n(itv, n, itw);
        }

        template <typename T, typename Q, typename IteratorV, typename IteratorW>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, const IteratorW &itw,
                                    EWOp::Add) {
            thrust::transform(
                itv, itv + n, itw, itw,
                plus<typename cuda_complex<T>::type, typename cuda_complex<Q>::type>());
        }

        template <typename IndexType, typename T, typename Q, typename IteratorV, typename EWOP>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, Q *w,
                                    const IndexType *indicesw, EWOP) {
            if (indicesw == nullptr) {
                copy_n_same_dev_thrust<T, Q>(itv, n, encapsulate_pointer(w), EWOP{});
            } else {
                auto itw = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                             encapsulate_pointer(indicesw));
                copy_n_same_dev_thrust<T, Q>(itv, n, itw, EWOP{});
            }
        }

        template <typename IndexType, typename T, typename Q, typename EWOP>
        void copy_n_same_dev_thrust(typename elem<T>::type alpha, const T *v,
                                    const IndexType *indicesv, std::size_t n, Q *w,
                                    const IndexType *indicesw, Cuda cuda, EWOP) {
            setDevice(cuda);
            if (indicesv == nullptr) {
                auto itv = encapsulate_pointer(v);
                if (alpha == typename elem<T>::type{1}) {
                    copy_n_same_dev_thrust<IndexType, T, Q>(itv, n, w, indicesw, EWOP{});
                } else {
                    copy_n_same_dev_thrust<IndexType, T, Q>(
                        thrust::make_transform_iterator(itv, scale<T>(alpha)), n, w, indicesw,
                        EWOP{});
                }
            } else {
                auto itv = thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                             encapsulate_pointer(indicesv));
                if (alpha == typename elem<T>::type{1}) {
                    copy_n_same_dev_thrust<IndexType, T, Q>(itv, n, w, indicesw, EWOP{});
                } else {
                    copy_n_same_dev_thrust<IndexType, T, Q>(
                        thrust::make_transform_iterator(itv, scale<T>(alpha)), n, w, indicesw,
                        EWOP{});
                }
            }
        }

#    endif // SUPERBBLAS_USE_THRUST

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] when v and w are on device

        template <typename IndexType, typename T, typename Q, typename EWOP>
        DECL_COPY_T_Q_EWOP(void copy_n(typename elem<T>::type alpha, const T *v,
                                       const IndexType *indicesv, Cuda cudav, std::size_t n, Q *w,
                                       const IndexType *indicesw, Cuda cudaw, EWOP))
        IMPL({
            assert((n == 0 || (void *)v != (void *)w || std::is_same<T, Q>::value));
            if (n == 0) return;

            // Actions when the v and w are on the same device
            if (deviceId(cudav) == deviceId(cudaw)) {
                if (alpha == typename elem<T>::type{1} && std::is_same<T, Q>::value &&
                    std::is_same<EWOP, EWOp::Copy>::value) {
                    setDevice(cudaw);
                    cudaCheck(cudaMemcpy(w, v, sizeof(T) * n, cudaMemcpyDeviceToDevice));
                } else {
                    copy_n_same_dev_thrust(alpha, v, indicesv, n, w, indicesw, cudaw, EWOP{});
                }
            }

            // Simple case when the v and w are NOT on the same device and no permutation is involved
            else if (indicesv == nullptr && indicesw == nullptr &&
                     alpha == typename elem<T>::type{1} && std::is_same<T, Q>::value &&
                     std::is_same<EWOP, EWOp::Copy>::value && deviceId(cudav) != deviceId(cudaw)) {
                setDevice(cudaw);
                cudaCheck(cudaMemcpyPeer(w, deviceId(cudaw), v, deviceId(cudav), sizeof(T) * n));
            }

            // If v is permuted, copy v[indices[i]] in a contiguous chunk, and then copy
            else if (indicesv != nullptr) {
                vector<Q, Cuda> v0(n, cudav);
                copy_n<IndexType>(alpha, v, indicesv, cudav, n, v0.data(), nullptr, cudav,
                                  EWOp::Copy{});
                copy_n<IndexType>(Q{1}, v0.data(), nullptr, cudav, n, w, indicesw, cudaw, EWOP{});
            }

            // Otherwise copy v to cudaw, and then copy it to the w[indices[i]]
            else {
                vector<T, Cuda> v1(n, cudaw);
                copy_n<IndexType>(T{1}, v, indicesv, cudav, n, v1.data(), nullptr, cudaw,
                                  EWOp::Copy{});
                copy_n<IndexType>(T{alpha}, v1.data(), nullptr, cudaw, n, w, indicesw, cudaw,
                                  EWOP{});
            }
        })

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] from device to host and vice versa

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPU0, XPU1>::value, bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indicesv, XPU0 xpu0,
                    std::size_t n, Q *w, const IndexType *indicesw, XPU1 xpu1, EWOP) {
            if (n == 0) return;

            // Base case
            if (std::is_same<T, Q>::value && std::is_same<EWOP, EWOp::Copy>::value &&
                indicesv == nullptr && indicesw == nullptr) {
                setDevice(xpu0);
                setDevice(xpu1);
                cudaCheck(cudaMemcpy(w, v, sizeof(T) * n,
                                     std::is_same<XPU0, Cuda>::value ? cudaMemcpyDeviceToHost
                                                                     : cudaMemcpyHostToDevice));
                // Scale by alpha
                copy_n<IndexType>(alpha, w, nullptr, xpu1, n, w, nullptr, xpu1, EWOp::Copy{});
            }

            // If v is permuted, copy v[indices[i]] in a contiguous chunk, and then copy
            else if (indicesv != nullptr) {
                vector<Q, XPU0> v0(n, xpu0);
                copy_n<IndexType>(alpha, v, indicesv, xpu0, n, v0.data(), nullptr, xpu0,
                                  EWOp::Copy{});
                copy_n<IndexType>(Q{1}, v0.data(), nullptr, xpu0, n, w, indicesw, xpu1, EWOP{});
            }

            // Otherwise copy v to xpu1, and then copy it to the w[indices[i]]
            else {
                vector<T, XPU1> v1(n, xpu1);
                copy_n<IndexType>(T{1}, v, indicesv, xpu0, n, v1.data(), nullptr, xpu1,
                                  EWOp::Copy{});
                copy_n<IndexType>(T{alpha}, v1.data(), nullptr, xpu1, n, w, indicesw, xpu1, EWOP{});
            }
        }

        /// Copy n values, w[i] (+)= v[i] when v or w is on device

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<std::is_same<XPU0, Cuda>::value |
                                              std::is_same<XPU1, Cuda>::value,
                                          bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, XPU0 xpu0, std::size_t n, Q *w,
                    XPU1 xpu1, EWOP) {
            copy_n<IndexType>(alpha, v, nullptr, xpu0, n, w, nullptr, xpu1, EWOP{});
        }

        /// Copy n values, w[i] (+)= v[indices[i]] when v or w is on device

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<std::is_same<XPU0, Cuda>::value |
                                              std::is_same<XPU1, Cuda>::value,
                                          bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indices, XPU0 xpu0,
                    std::size_t n, Q *w, XPU1 xpu1, EWOP) {
            copy_n<IndexType>(alpha, v, indices, xpu0, n, w, nullptr, xpu1, EWOP{});
        }

        /// Copy n values, w[indices[i]] (+)= v[i] when v or w is on device

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<std::is_same<XPU0, Cuda>::value |
                                              std::is_same<XPU1, Cuda>::value,
                                          bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, XPU0 xpu0, std::size_t n, Q *w,
                    const IndexType *indices, XPU1 xpu1, EWOP) {
            copy_n<IndexType>(alpha, v, nullptr, xpu0, n, w, indices, xpu1, EWOP{});
        }

        /// Copy and reduce n values, w[indicesw[i]] += sum(v[perm[perm_distinct[i]:perm_distinct[i+1]]])
        template <typename IndexType, typename T>
        DECL_COPY_REDUCE(void copy_reduce_n(typename elem<T>::type alpha, const T *v, Cpu,
                                            const IndexType *perm, const IndexType *perm_distinct,
                                            std::size_t ndistinct, Cpu cpuv, T *w,
                                            const IndexType *indicesw, Cuda cudaw))
        IMPL({
            std::vector<T> w_host(ndistinct - 1);
            copy_reduce_n<IndexType, T>(1, v, Cpu{}, perm, perm_distinct, ndistinct, cpuv,
                                        w_host.data(), nullptr, Cpu{});
            vector<T, Cuda> w_device(ndistinct - 1, cudaw);
            copy_n<IndexType, T>(T{1}, w_host.data(), cpuv, ndistinct - 1, w_device.data(), cudaw,
                                 EWOp::Copy{});
            auto itw_dev = encapsulate_pointer(w_device.begin());
            auto itw = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                         encapsulate_pointer(indicesw));
            if (alpha == typename elem<T>::type{1}) {
                thrust::transform(itw_dev, itw_dev + ndistinct - 1, itw, itw,
                                  thrust::plus<typename cuda_complex<T>::type>());
            } else {
                auto itw_dev_scale = thrust::make_transform_iterator(itw_dev, scale<T>(alpha));
                thrust::transform(itw_dev_scale, itw_dev_scale + ndistinct - 1, itw, itw,
                                  thrust::plus<typename cuda_complex<T>::type>());
            }
        })

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
            if (n == 0) return;
            setDevice(cuda);
            cudaCheck(cudaMemset(v, 0, sizeof(T) * n));
        }

        template <typename T> inline cudaDataType_t toCudaDataType(void);

        template <> inline cudaDataType_t toCudaDataType<float>(void) { return CUDA_R_32F; }
        template <> inline cudaDataType_t toCudaDataType<std::complex<float>>(void) {
            return CUDA_C_32F;
        }
        template <> inline cudaDataType_t toCudaDataType<double>(void) { return CUDA_R_64F; }
        template <> inline cudaDataType_t toCudaDataType<std::complex<double>>(void) {
            return CUDA_C_64F;
        }

        /// Template scal for GPUs

        template <typename T,
                  typename std::enable_if<!std::is_same<int, T>::value, bool>::type = true>
        inline void xscal(int n, T alpha, T *x, int incx, Cuda cuda) {
            if (std::fabs(alpha) == 0.0) {
                setDevice(cuda);
                cudaMemset2D(x, sizeof(T) * incx, 0, sizeof(T), n);
                return;
            }
            if (alpha == typename elem<T>::type{1}) return;
            cudaDataType_t cT = toCudaDataType<T>();
            cublasCheck(cublasScalEx(cuda.cublasHandle, n, &alpha, cT, x, cT, incx, cT));
        }
#endif

        /// Template scal for integers
        template <typename XPU> inline void xscal(int n, int alpha, int *x, int incx, XPU xpu) {
            if (alpha == 1) return;
            if (incx != 1) throw std::runtime_error("Unsupported xscal variant");
            if (std::abs(alpha) == 0) {
                zero_n(x, n, xpu);
            } else {
                copy_n<int>(alpha, x, xpu, n, x, xpu, EWOp::Copy{});
            }
        }

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

#    if CUDART_VERSION >= 11000
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
#    else
        template <typename T> inline cudaDataType_t toCudaComputeType(void) {
            return toCudaDataType<T>();
        }
#    endif

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
                cuda.cublasHandle, toCublasTrans(transa), toCublasTrans(transb), m, n, k, &alpha, a,
                cT, lda, stridea, b, cT, ldb, strideb, &beta, c, cT, ldc, stridec, batch_size,
                toCudaComputeType<T>(), CUBLAS_GEMM_DEFAULT));
        }
#endif // SUPERBBLAS_USE_CUDA

        template <typename IndexType, typename T> vector<T, Cpu> toCpu(const vector<T, Cpu> &v) {
            return v;
        }

        template <typename IndexType, typename T, typename XPU,
                  typename std::enable_if<!std::is_same<Cpu, XPU>::value, bool>::type = true>
        vector<T, Cpu> toCpu(const vector<T, XPU> &v) {
            vector<T, Cpu> r(v.size());
            copy_n<IndexType, T>(T{1}, v.data(), v.ctx(), v.size(), r.data(), r.ctx(),
                                 EWOp::Copy{});
            return r;
        }
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
