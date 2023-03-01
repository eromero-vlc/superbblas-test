#ifndef __SUPERBBLAS_BLAS__
#define __SUPERBBLAS_BLAS__

#include "alloc.h"
#include "blas_cpu_tmpl.hpp"
#include "performance.h"
#include "platform.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

//////////////////////
// NOTE:
// Functions in this file that uses `thrust` should be instrumented to remove the dependency from
// `thrust` when the superbblas library is used not as header-only. Use the macro `IMPL` to hide
// the definition of functions using `thrust` and use DECL_... macros to generate template
// instantiations to be included in the library.

#if (defined(SUPERBBLAS_USE_CUDA) || defined(SUPERBBLAS_USE_HIP)) &&                               \
    !defined(SUPERBBLAS_CREATING_FLAGS) && !defined(SUPERBBLAS_CREATING_LIB) &&                    \
    !defined(SUPERBBLAS_LIB)
#    define SUPERBBLAS_USE_THRUST
#endif
#ifdef SUPERBBLAS_USE_THRUST
#    ifndef SUPERBBLAS_LIB
#        include <thrust/complex.h>
#        include <thrust/copy.h>
#        include <thrust/device_ptr.h>
#        include <thrust/device_vector.h>
#        include <thrust/execution_policy.h>
#        include <thrust/iterator/permutation_iterator.h>
#        include <thrust/iterator/transform_iterator.h>
#        include <thrust/transform.h>
#    endif
#endif

#ifdef SUPERBBLAS_CREATING_FLAGS
#    ifdef SUPERBBLAS_USE_CBLAS
EMIT_define(SUPERBBLAS_USE_CBLAS)
#    endif
#endif

#ifdef SUPERBBLAS_CREATING_LIB
#    define SUPERBBLAS_INDEX_TYPES superbblas::IndexType, std::size_t
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
#    define REPLACE_T REPLACE(T, superbblas::IndexType, std::size_t, SUPERBBLAS_TYPES)
#    define REPLACE_T_Q                                                                            \
        REPLACE(T Q, superbblas::IndexType superbblas::IndexType, std::size_t std::size_t, T Q)    \
        REPLACE(T Q, META_TYPES) REPLACE_META_TYPES
#    define REPLACE_IndexType REPLACE(IndexType, superbblas::IndexType, std::size_t)

#    define REPLACE_EWOP REPLACE(EWOP, EWOp::Copy, EWOp::Add)

#    define REPLACE_XPU REPLACE(XPU, XPU_GPU)

#    if defined(SUPERBBLAS_USE_CUDA)
#        define XPU_GPU Cuda
#    elif defined(SUPERBBLAS_USE_HIP)
#        define XPU_GPU Hip
#    else
#        define XPU_GPU Cpu
#    endif

/// Generate template instantiations for sum functions with template parameter T

#    define DECL_SUM_T(...)                                                                        \
        EMIT REPLACE1(sum, superbblas::detail::sum<T>) REPLACE_T template __VA_ARGS__;

/// Generate template instantiations for sum functions with template parameter T

#    define DECL_SELECT_T(...)                                                                     \
        EMIT REPLACE1(select, superbblas::detail::select<IndexType, T>) REPLACE_IndexType REPLACE( \
            T, superbblas::IndexType, std::size_t, SUPERBBLAS_REAL_TYPES) template __VA_ARGS__;

#else
#    define DECL_SUM_T(...) __VA_ARGS__
#    define DECL_SELECT_T(...) __VA_ARGS__
#endif

#define SB_RESTRICT __restrict__

namespace superbblas {

    /// elem<T>::type is T::value_type if T is an array; otherwise it is T

    template <typename T> struct elem { using type = T; };
    template <typename T, std::size_t N> struct elem<std::array<T, N>> {
        using type = typename elem<T>::type;
    };
    template <typename T, std::size_t N> struct elem<const std::array<T, N>> {
        using type = typename elem<T>::type;
    };

    namespace detail {

        /// is_array<T>::value is true if T is std::array
        /// \tparam T: type to inspect

        template <typename T> struct is_array { static const bool value = false; };
        template <typename T, std::size_t N> struct is_array<std::array<T, N>> {
            static const bool value = true;
        };
        template <typename T> struct is_array<const T> {
            static const bool value = is_array<T>::value;
        };

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

        /// Copy n values from v to w
        /// \param v: first element to read
        /// \param xpu0: context of v
        /// \param n: number of elements to copy
        /// \param w: first element to write
        /// \param xpu1: context of w

        template <typename T, typename XPU0, typename XPU1>
        void copy_n(const T *SB_RESTRICT v, XPU0 xpu0, std::size_t n, T *SB_RESTRICT w, XPU1 xpu1) {
            if (n == 0 || v == w) return;

            const bool v_is_on_cpu = deviceId(xpu0) == CPU_DEVICE_ID;
            const bool w_is_on_cpu = deviceId(xpu1) == CPU_DEVICE_ID;

            if (v_is_on_cpu && w_is_on_cpu &&
                (std::is_same<XPU0, Cpu>::value || std::is_same<XPU1, Cpu>::value)) {
                // Both pointers are on cpu

                // Synchronize the contexts just in case there is disguised cpu context on a gpu context
                sync(xpu0);
                sync(xpu1);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] = v[i];

            }
#ifdef SUPERBBLAS_USE_GPU
            else if (v_is_on_cpu && w_is_on_cpu) {
                // Both pointers are on cpu but disguised as gpu contexts
                causalConnectTo(xpu1, xpu0);
                launchHostKernel([=] { std::memcpy((void *)w, (void *)v, sizeof(T) * n); }, xpu0);
                causalConnectTo(xpu0, xpu1);
            } else if (v_is_on_cpu != w_is_on_cpu) {
                // One pointer is on device and the other on host

                // Perform the operation on the first context stream if it is a gpu (disguised cpu or not)
                constexpr bool op_on_first = !std::is_same<XPU0, Cpu>::value;
                GpuStream stream;
                if (op_on_first) {
                    causalConnectTo(xpu1, xpu0);
                    setDevice(xpu0);
                    stream = getStream(xpu0);
                } else {
                    causalConnectTo(xpu0, xpu1);
                    setDevice(xpu1);
                    stream = getStream(xpu1);
                }
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(MemcpyAsync)(
                    w, v, sizeof(T) * n,
                    !v_is_on_cpu ? SUPERBBLAS_GPU_SYMBOL(MemcpyDeviceToHost)
                                 : SUPERBBLAS_GPU_SYMBOL(MemcpyHostToDevice),
                    stream));
                if (op_on_first) {
                    causalConnectTo(xpu0, xpu1);
                } else {
                    causalConnectTo(xpu1, xpu0);
                }
            } else {
                // Both pointers are on device
                causalConnectTo(xpu1, xpu0);
                setDevice(xpu0);
                if (deviceId(xpu0) == deviceId(xpu1)) {
                    gpuCheck(SUPERBBLAS_GPU_SYMBOL(MemcpyAsync)(
                        w, v, sizeof(T) * n, SUPERBBLAS_GPU_SYMBOL(MemcpyDeviceToDevice),
                        getStream(xpu0)));
                } else {
                    gpuCheck(SUPERBBLAS_GPU_SYMBOL(MemcpyPeerAsync)(
                        w, deviceId(xpu1), v, deviceId(xpu0), sizeof(T) * n, getStream(xpu0)));
                }
                causalConnectTo(xpu0, xpu1);
            }
#endif // SUPERBBLAS_USE_GPU
        }

        /// Whether to cache allocation
        enum CacheAlloc { dontCacheAlloc, doCacheAlloc };

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

            /// Construct a vector with `n` elements a with context device `xpu_`
            vector(std::size_t n, XPU xpu_, CacheAlloc cacheAlloc = dontCacheAlloc,
                   std::size_t alignment = 0)
                : n(n), xpu(xpu_) {
                auto alloc = cacheAlloc == doCacheAlloc
                                 ? allocateBufferResouce<T_no_const>(n, xpu, alignment)
                                 : allocateResouce<T_no_const>(n, xpu, alignment);
                ptr_aligned = alloc.first;
                ptr = alloc.second;
            }

            /// Construct a vector with `n` elements a with context device `xpu_`
            vector(std::size_t n, XPU xpu_, std::size_t alignment)
                : vector(n, xpu_, dontCacheAlloc, alignment) {}

            /// Construct a vector from a given pointer `ptr` with `n` elements and with context
            /// device `xpu`. `ptr` is not deallocated after the destruction of the `vector`.
            vector(std::size_t n, T *ptr, XPU xpu)
                : n(n), ptr_aligned(ptr), ptr((char *)ptr, [&](const char *) {}), xpu(xpu) {}

            /// Low-level constructor
            vector(std::size_t n, T *ptr_aligned, std::shared_ptr<char> ptr, XPU xpu)
                : n(n), ptr_aligned(ptr_aligned), ptr(ptr), xpu(xpu) {}

            /// Conversion from `vector<T, XPU>` to `vector<const T, XPU>`
            template <typename U = T_no_const,
                      typename std::enable_if<!std::is_const<U>::value && std::is_const<T>::value &&
                                                  std::is_same<const U, T>::value,
                                              bool>::type = true>
            vector(const vector<U, XPU> &v) : vector{v.n, (T *)v.ptr_aligned, v.ptr, v.xpu} {}

            /// Release all elements in the vector
            void clear() {
                n = 0;
                ptr.reset();
                ptr_aligned = nullptr;
            }

            /// Return the number of allocated elements
            std::size_t size() const { return n; }

            /// Return a pointer to the allocated space
            T *data() const { return ptr_aligned; }

            /// Return a pointer to the first element allocated
            T *begin() const { return ptr_aligned; }

            /// Return a pointer to the first element non-allocated after an allocated element
            T *end() const { return begin() + n; }

            /// Return the device context
            XPU ctx() const { return xpu; }

            /// Resize to a smaller size vector
            void resize(std::size_t new_n) {
                if (new_n > n) throw std::runtime_error("Unsupported operation");
                n = new_n;
            }

            /// Return a reference to i-th allocated element, for Cpu `vector`
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            const T &operator[](std::size_t i) const {
                return ptr_aligned[i];
            }

            /// Return a reference to i-th allocated element, for Cpu `vector`
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            T &operator[](std::size_t i) {
                return ptr_aligned[i];
            }

            /// Return a reference to the last element, for Cpu `vector`
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            const T &back() const {
                return ptr_aligned[n - 1];
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

            /// Operator == compares size and content
            template <typename U = XPU,
                      typename std::enable_if<std::is_same<U, Cpu>::value, bool>::type = true>
            bool operator!=(const vector<T, U> &v) const {
                return !operator==(v);
            }

            /// Return an alias of the vector with another context
            /// \param new_xpu: new context
            vector withNewContext(const XPU &new_xpu) const {
                return vector{n, ptr_aligned, ptr, new_xpu};
            }

            std::size_t n;             ///< Number of allocated `T` elements
            T *ptr_aligned;            ///< Pointer aligned
            std::shared_ptr<char> ptr; ///< Pointer to the allocated memory
            XPU xpu;                   ///< Context
        };

        /// Construct a `vector<T, Cpu>` with the given pointer and context

        template <typename T> vector<T, Cpu> to_vector(T *ptr, std::size_t n, Cpu cpu) {
            check_ptr_align<T>(ptr);
            return vector<T, Cpu>(ptr ? n : 0, ptr, cpu);
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Construct a `vector<T, Gpu>` with the given pointer and context

        template <typename T> vector<T, Gpu> to_vector(T *ptr, std::size_t n, Gpu cuda) {
            check_ptr_align<T>(ptr);
            return vector<T, Gpu>(ptr ? n : 0, ptr, cuda);
        }
#endif

#ifdef SUPERBBLAS_USE_THRUST
        /// Return a device pointer suitable for making iterators

        template <typename T>
        thrust::device_ptr<typename cuda_complex<T>::type> encapsulate_pointer(T *ptr) {
            return thrust::device_pointer_cast(
                reinterpret_cast<typename cuda_complex<T>::type *>(ptr));
        }

        /// Return the stream encapsulated for thrust
        /// \param xpu: context

        inline auto thrust_par_on(const Gpu &xpu) {
            return thrust::
#    ifdef SUPERBBLAS_USE_CUDA
                cuda::
#    elif defined(SUPERBBLAS_USE_HIP)
                hip::
#    endif
#    if THRUST_VERSION >= 101600
                    par_nosync.on(getStream(xpu));
#    else
                    par.on(getStream(xpu));
#    endif
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

        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param n: number of elements to set
        /// \param cpu: device context

        template <typename T> void zero_n(T *v, std::size_t n, Cpu) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
            for (std::size_t i = 0; i < n; ++i) v[i] = T{0};
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Launch a host kernel on the given stream
        /// \param f: function to queue
        /// \param xpu: context where the get the stream

        inline void launchHostKernel(const std::function<void()> &f, const Gpu &xpu) {
            if (deviceId(xpu) != CPU_DEVICE_ID)
                throw std::runtime_error("launchHostKernel: the context should be on cpu");

#    if defined(SUPERBBLAS_USE_CUDA) ||                                                            \
        (defined(SUPERBBLAS_USE_HIP) &&                                                            \
         ((HIP_VERSION_MAJOR > 5) || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 4)))
            struct F {
                static void SUPERBBLAS_GPU_SELECT(, CUDART_CB, ) callback(void *data) {
                    auto f = (std::function<void()> *)data;
                    (*f)();
                    delete f;
                }
            };
            auto fp = new std::function<void()>(f);
            setDevice(xpu);
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(LaunchHostFunc)(
                getStream(xpu), (SUPERBBLAS_GPU_SYMBOL(HostFn_t))F::callback, (void *)fp));
#    else
            sync(xpu);
            f();
#    endif
        }
#endif // SUPERBBLAS_USE_GPU

        inline void launchHostKernel(const std::function<void()> &f, const Cpu &) { f(); }

#ifdef SUPERBBLAS_USE_GPU
        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param n: number of elements to set
        /// \param xpu: device context

        template <typename T> void zero_n(T *v, std::size_t n, const Gpu &xpu) {
            if (n == 0) return;
            if (deviceId(xpu) == CPU_DEVICE_ID) {
                launchHostKernel([=] { std::memset((void *)v, 0, sizeof(T) * n); }, xpu);
            } else {
                setDevice(xpu);
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(MemsetAsync)(v, 0, sizeof(T) * n, getStream(xpu)));
            }
        }
#endif // SUPERBBLAS_USE_GPU

        /// Return a copy of a vector

        template <typename T, typename XPU,
                  typename std::enable_if<!is_array<T>::value, bool>::type = true>
        vector<T, XPU> clone(const vector<T, XPU> &v) {
            using T_no_const = typename std::remove_const<T>::type;
            vector<T_no_const, XPU> r(v.size(), v.ctx());
            copy_n(typename elem<T>::type{1}, v.data(), v.ctx(), v.size(), r.data(), r.ctx(),
                   EWOp::Copy{});
            return r;
        }

        template <typename T, typename std::enable_if<is_array<T>::value, bool>::type = true>
        vector<T, Cpu> clone(const vector<T, Cpu> &v) {
            using T_no_const = typename std::remove_const<T>::type;
            vector<T_no_const, Cpu> r(v.size(), v.ctx());
            std::copy_n(v.data(), v.size(), r.data());
            return r;
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        SUPERBBLAS_GPU_SELECT(xxx, cudaDataType_t, hipblasDatatype_t)
        toCudaDataType(void) {
            if (std::is_same<T, float>::value)
                return SUPERBBLAS_GPU_SELECT(xxx, CUDA_R_32F, HIPBLAS_R_32F);
            if (std::is_same<T, double>::value)
                return SUPERBBLAS_GPU_SELECT(xxx, CUDA_R_64F, HIPBLAS_R_64F);
            if (std::is_same<T, std::complex<float>>::value)
                return SUPERBBLAS_GPU_SELECT(xxx, CUDA_C_32F, HIPBLAS_C_32F);
            if (std::is_same<T, std::complex<double>>::value)
                return SUPERBBLAS_GPU_SELECT(xxx, CUDA_C_64F, HIPBLAS_C_64F);
            throw std::runtime_error("toCudaDataType: unsupported type");
        }

        /// Template scal for GPUs

        template <typename T,
                  typename std::enable_if<!std::is_same<int, T>::value, bool>::type = true>
        void xscal(int n, T alpha, T *x, int incx, Gpu xpu) {
            if (std::norm(alpha) == 0.0) {
                setDevice(xpu);
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(Memset2DAsync)(x, sizeof(T) * incx, 0, sizeof(T), n,
                                                              getStream(xpu)));
                return;
            }
            if (alpha == typename elem<T>::type{1}) return;
            auto cT = toCudaDataType<T>();
            gpuBlasCheck(SUPERBBLAS_GPUBLAS_SYMBOL(ScalEx)(getGpuBlasHandle(xpu), n, &alpha, cT, x,
                                                           cT, incx, cT));
        }
#endif // SUPERBBLAS_USE_GPU

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

#ifdef SUPERBBLAS_USE_GPU
#    ifdef SUPERBBLAS_USE_CUDA
#        if CUDART_VERSION >= 11000
        template <typename T> cublasComputeType_t toCudaComputeType() {
            if (std::is_same<T, float>::value) return CUBLAS_COMPUTE_32F;
            if (std::is_same<T, double>::value) return CUBLAS_COMPUTE_64F;
            if (std::is_same<T, std::complex<float>>::value) return CUBLAS_COMPUTE_32F;
            if (std::is_same<T, std::complex<double>>::value) return CUBLAS_COMPUTE_64F;
            throw std::runtime_error("toCudaDataType: unsupported type");
        }
#        else
        template <typename T> cudaDataType_t toCudaComputeType() { return toCudaDataType<T>(); }
#        endif
#    elif defined(SUPERBBLAS_USE_HIP)
        template <typename T> hipblasDatatype_t toCudaComputeType() { return toCudaDataType<T>(); }
#    endif

        inline SUPERBBLAS_GPUBLAS_SYMBOL(Operation_t) toCublasTrans(char trans) {
            switch (trans) {
            case 'n':
            case 'N': return SUPERBBLAS_GPU_SELECT(xxx, CUBLAS_OP_N, HIPBLAS_OP_N);
            case 't':
            case 'T': return SUPERBBLAS_GPU_SELECT(xxx, CUBLAS_OP_T, HIPBLAS_OP_T);
            case 'c':
            case 'C': return SUPERBBLAS_GPU_SELECT(xxx, CUBLAS_OP_C, HIPBLAS_OP_C);
            default: throw std::runtime_error("Not valid value of trans");
            }
        }

        template <typename T>
        void xgemm_batch(char transa, char transb, int m, int n, int k, T alpha, const T *a[],
                         int lda, const T *b[], int ldb, T beta, T *c[], int ldc, int batch_size,
                         Gpu xpu) {
            // Quick exits
            if (m == 0 || n == 0) return;

            // Replace some invalid arguments when k is zero
            if (k == 0) {
                a = b = (const T **)c;
                lda = ldb = 1;
            }

            auto cT = toCudaDataType<T>();
            gpuBlasCheck(SUPERBBLAS_GPUBLAS_SYMBOL(GemmBatchedEx)(
                getGpuBlasHandle(xpu), toCublasTrans(transa), toCublasTrans(transb), m, n, k,
                &alpha, (const void **)a, cT, lda, (const void **)b, cT, ldb, &beta, (void **)c, cT,
                ldc, batch_size, toCudaComputeType<T>(),
                SUPERBBLAS_GPU_SELECT(xxx, CUBLAS_GEMM_DEFAULT, HIPBLAS_GEMM_DEFAULT)));
        }

        template <typename T>
        void xgemm_batch_strided(char transa, char transb, int m, int n, int k, T alpha, const T *a,
                                 int lda, int stridea, const T *b, int ldb, int strideb, T beta,
                                 T *c, int ldc, int stridec, int batch_size, Gpu xpu) {
            // Quick exits
            if (m == 0 || n == 0 || batch_size == 0) return;

            // Replace some invalid arguments when k is zero
            if (k == 0) {
                a = b = c;
                lda = ldb = 1;
            }

            auto cT = toCudaDataType<T>();
            if (batch_size == 1) {
                gpuBlasCheck(SUPERBBLAS_GPUBLAS_SYMBOL(GemmEx)(
                    getGpuBlasHandle(xpu), toCublasTrans(transa), toCublasTrans(transb), m, n, k,
                    &alpha, a, cT, lda, b, cT, ldb, &beta, c, cT, ldc, toCudaComputeType<T>(),
                    SUPERBBLAS_GPU_SELECT(xxx, CUBLAS_GEMM_DEFAULT, HIPBLAS_GEMM_DEFAULT)));
            } else {
                gpuBlasCheck(SUPERBBLAS_GPUBLAS_SYMBOL(GemmStridedBatchedEx)(
                    getGpuBlasHandle(xpu), toCublasTrans(transa), toCublasTrans(transb), m, n, k,
                    &alpha, a, cT, lda, stridea, b, cT, ldb, strideb, &beta, c, cT, ldc, stridec,
                    batch_size, toCudaComputeType<T>(),
                    SUPERBBLAS_GPU_SELECT(xxx, CUBLAS_GEMM_DEFAULT, HIPBLAS_GEMM_DEFAULT)));
            }
        }
#endif // SUPERBBLAS_USE_GPU

        /// Return a copy of the vector in the given context, or the same vector if its context coincides
        /// \param v: vector to return or to clone with xpu context
        /// \param xpu: target context
        ///
        /// NOTE: implementation when the vector context and the given context are of the same type

        template <typename T, typename XPU>
        vector<T, XPU> makeSure(const vector<T, XPU> &v, XPU xpu,
                                CacheAlloc cacheAlloc = dontCacheAlloc) {
            if (deviceId(v.ctx()) == deviceId(xpu)) {
                causalConnectTo(v.ctx(), xpu);
                return v;
            }
            vector<T, XPU> r(v.size(), xpu, cacheAlloc);
            copy_n(v.data(), v.ctx(), v.size(), r.data(), r.ctx());
            return r;
        }

        /// Return a copy of the vector in the given context
        /// \param v: vector to clone with xpu context
        /// \param xpu: target context
        ///
        /// NOTE: implementation when the vector context and the given context are not of the same type

        template <typename T, typename XPU1, typename XPU0,
                  typename std::enable_if<!std::is_same<XPU0, XPU1>::value, bool>::type = true>
        vector<T, XPU1> makeSure(const vector<T, XPU0> &v, XPU1 xpu1,
                                 CacheAlloc cacheAlloc = dontCacheAlloc) {
            vector<T, XPU1> r(v.size(), xpu1, cacheAlloc);
            copy_n(v.data(), v.ctx(), v.size(), r.data(), r.ctx());
            return r;
        }

        /// Return the sum of all elements in a vector
        /// \param v: vector
        /// \return: the sum of all the elements of v

        template <typename T> T sum(const vector<T, Cpu> &v) {
            T s{0};
            const T *p = v.data();
            for (std::size_t i = 0, n = v.size(); i < n; ++i) s += p[i];
            return s;
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Return the sum of all elements in a vector
        /// \param v: vector
        /// \return: the sum of all the elements of v

        template <typename T>
        DECL_SUM_T(T sum(const vector<T, Gpu> &v))
        IMPL({
            setDevice(v.ctx());
            auto it = encapsulate_pointer(v.begin());
            return thrust::reduce(thrust_par_on(v.ctx()), it, it + v.size());
        })
#endif

        /// Return a new array with only the elements w[i] that mask[v[i]] != 0
        /// \param v: vector of indices used by the mask
        /// \param mask: vector of size v[v.size()-1]
        /// \param w: vector of indices to return
        /// \return: a new vector

        template <typename IndexType, typename T>
        vector<IndexType, Cpu> select(const vector<IndexType, Cpu> &v, const T *m,
                                      const vector<IndexType, Cpu> &w) {
            vector<IndexType, Cpu> r{w.size(), Cpu{}};
            const IndexType *pv = v.data();
            const IndexType *pw = w.data();
            IndexType *pr = r.data();
            std::size_t n = w.size(), nr = 0;
            for (std::size_t i = 0; i < n; ++i)
                if (m[pv[i]] != 0) pr[nr++] = pw[i];
            r.resize(nr);
            return r;
        }

#ifdef SUPERBBLAS_USE_GPU

#    ifdef SUPERBBLAS_USE_THRUST
        // Return whether the element isn't zero
        template <typename T> struct not_zero : public thrust::unary_function<T, bool> {
            __host__ __device__ bool operator()(const T &i) const { return i != T{0}; }
        };
#    endif

        /// Return a new array with only the elements w[i] that mask[v[i]] != 0
        /// \param v: vector of indices used by the mask
        /// \param mask: vector of size v[v.size()-1]
        /// \param w: vector of indices to return
        /// \return: a new vector

        template <typename IndexType, typename T>
        DECL_SELECT_T(vector<IndexType, Gpu> select(const vector<IndexType, Gpu> &v, T *m,
                                                    const vector<IndexType, Gpu> &w))
        IMPL({
            causalConnectTo(w.ctx(), v.ctx());
            setDevice(v.ctx());
            vector<IndexType, Gpu> r{w.size(), v.ctx()};
            auto itv = encapsulate_pointer(v.begin());
            auto itm = encapsulate_pointer(m);
            auto itw = encapsulate_pointer(w.begin());
            auto itr = encapsulate_pointer(r.begin());
            auto itmv = thrust::make_permutation_iterator(itm, itv);
            auto itr_end = thrust::copy_if(thrust_par_on(v.ctx()), itw, itw + w.size(), itmv, itr,
                                           not_zero<T>{});
            r.resize(itr_end - itr);
            causalConnectTo(v.ctx(), w.ctx());
            return r;
        })
#endif // SUPERBBLAS_USE_GPU

#ifdef SUPERBBLAS_USE_GPU
        /// Generate a new stream that branching from the given one that will merge back with `anabranch_end`
        /// \param xpu: context to branch

        inline Gpu anabranch_begin(const Gpu &xpu) {
            // Create a new stream, connect it causally from the given context
            GpuStream new_stream = createStream(xpu);
            causalConnectTo(getStream(xpu), new_stream);
            return xpu.withNewStream(new_stream);
        }
#endif // SUPERBBLAS_USE_GPU

        inline Cpu anabranch_begin(const Cpu &xpu) {
            // Do nothing when context is on cpu
            return xpu;
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Join the context back the given context in `anabranch_begin`
        /// \param xpu: context to merge

        inline void anabranch_end(const Gpu &xpu) {
            // Connect the new stream to the original stream
            setDevice(xpu);
            causalConnectTo(getStream(xpu), getAllocStream(xpu));

            // Destroy the new stream
            destroyStream(xpu, getStream(xpu));
        }
#endif // SUPERBBLAS_USE_GPU

        inline void anabranch_end(const Cpu &) {
            // Do nothing when context is on cpu
        }
    }

    /// Allocate memory on a device
    /// \param n: number of element of type `T` to allocate
    /// \param ctx: context

    template <typename T> T *allocate(std::size_t n, Context ctx) {
        switch (ctx.plat) {
        case CPU: return detail::allocate<T>(n, ctx.toCpu(0));
#ifdef SUPERBBLAS_USE_GPU
        case CUDA: // Do the same as with HIP
        case HIP: return detail::allocate<T>(n, ctx.toGpu(0));
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }

    /// Deallocate memory on a device
    /// \param ptr: pointer to the memory to deallocate
    /// \param ctx: context

    template <typename T> void deallocate(T *ptr, Context ctx) {
        switch (ctx.plat) {
        case CPU: detail::deallocate(ptr, ctx.toCpu(0)); break;
#ifdef SUPERBBLAS_USE_GPU
        case CUDA: // Do the same as with HIP
        case HIP: detail::deallocate(ptr, ctx.toGpu(0)); break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }

    /// Force a synchronization on the device for superbblas stream
    /// \param ctx: context

    inline void sync(Context ctx) {
        switch (ctx.plat) {
        case CPU: detail::sync(ctx.toCpu(0)); break;
#ifdef SUPERBBLAS_USE_GPU
        case CUDA: // Do the same as with HIP
        case HIP: detail::sync(ctx.toGpu(0)); break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }

    /// Force a synchronization on the device for the legacy/default stream
    /// \param ctx: context

    inline void syncLegacyStream(Context ctx) {
        switch (ctx.plat) {
        case CPU: /* do nothing */ break;
#ifdef SUPERBBLAS_USE_GPU
        case CUDA: // Do the same as with HIP
        case HIP: detail::syncLegacyStream(ctx.toGpu(0)); break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }
}

#endif // __SUPERBBLAS_BLAS__
