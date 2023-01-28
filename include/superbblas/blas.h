#ifndef __SUPERBBLAS_BLAS__
#define __SUPERBBLAS_BLAS__

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

        /// Return a pointer aligned or nullptr if it isn't possible
        /// \param alignment: desired alignment of the returned pointer
        /// \param size: desired allocated size
        /// \param ptr: given pointer to align
        /// \param space: storage of the given pointer

        template <typename T>
        T *align(std::size_t alignment, std::size_t size, T *ptr, std::size_t space) {
            if (alignment == 0) return ptr;

                // std::align isn't is old versions of gcc
#if !defined(__GNUC__) || __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9)
            void *ptr0 = (void *)ptr;
            return (T *)std::align(alignment, size, ptr0, space);
#else
            uintptr_t new_ptr = ((uintptr_t(ptr) + (alignment - 1)) & ~(alignment - 1));
            if (new_ptr + size - uintptr_t(ptr) > space) return nullptr;
            return (T *)new_ptr;
#endif
        }

        /// Set default alignment, which is alignof(T) excepting when supporting GPUs that complex
        /// types need special alignment

        template <typename T> struct default_alignment {
            constexpr static std::size_t alignment = 0;
        };

        /// NOTE: thrust::complex requires sizeof(complex<T>) alignment
#ifdef SUPERBBLAS_USE_GPU
        template <typename T> struct default_alignment<std::complex<T>> {
            constexpr static std::size_t alignment = sizeof(std::complex<T>);
        };
#endif

        /// Check the given pointer has proper alignment
        /// \param v: ptr to check

        template <typename T> void check_ptr_align(const void *ptr) {
            if (ptr != nullptr &&
                align(default_alignment<T>::alignment, sizeof(T), ptr, sizeof(T)) == nullptr)
                throw std::runtime_error("Ups! Unaligned pointer");
        }

        /// is_complex<T>::value is true if T is std::complex
        /// \tparam T: type to inspect

        template <typename T> struct is_complex { static const bool value = false; };
        template <typename T> struct is_complex<std::complex<T>> {
            static const bool value = true;
        };
        template <typename T> struct is_complex<const T> {
            static const bool value = is_complex<T>::value;
        };

        /// is_array<T>::value is true if T is std::array
        /// \tparam T: type to inspect

        template <typename T> struct is_array { static const bool value = false; };
        template <typename T, std::size_t N> struct is_array<std::array<T, N>> {
            static const bool value = true;
        };
        template <typename T> struct is_array<const T> {
            static const bool value = is_array<T>::value;
        };

#ifdef SUPERBBLAS_USE_GPU
        /// Wait until everything finishes in the given context
        /// \param xpu: context

        inline void sync(const Gpu &xpu) {
#    ifdef SUPERBBLAS_USE_CUDA
            cudaCheck(cudaStreamSynchronize(xpu.stream));
#    else
            hipCheck(hipStreamSynchronize(xpu.stream));
#    endif
        }

        /// Wait until everything finishes in the device of the given context
        /// \param xpu: context

        inline void syncLegacyStream(const Gpu &xpu) {
            setDevice(xpu);
#    ifdef SUPERBBLAS_USE_CUDA
            cudaCheck(cudaDeviceSynchronize());
#    else
            hipCheck(hipDeviceSynchronize());
#    endif
        }
#endif

        /// Wait until everything finishes in the given context
        /// \param ctx: context
        ///
        /// NOTE: the Cpu implementation does nothing

        inline void sync(Cpu) {}

        /// Wait until everything finishes in the device of the given context
        /// \param ctx: context
        ///
        /// NOTE: the Cpu implementation does nothing

        inline void syncLegacyStream(Cpu) {}

        /// Force the second stream to wait until everything finishes until now from
        /// the first stream.
        /// \param s0: first stream
        /// \param s1: second stream

        void causalConnectTo(GpuStream s0, GpuStream s1) {
            // Trivial case: do nothing when both are the same stream
            if (s0 == s1) return;

                // Otherwise, record an event on s0 and wait on s1
#ifdef SUPERBBLAS_USE_CUDA
            cudaEvent_t ev;
            cudaCheck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            cudaCheck(cudaEventRecord(ev, s0));
            cudaCheck(cudaStreamWaitEvent(s1, ev));
            cudaCheck(cudaEventDestroy(ev));
#elif defined(SUPERBBLAS_USE_HIP)
            hipEvent_t ev;
            hipCheck(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
            hipCheck(hipEventRecord(ev, s0));
            hipCheck(hipStreamWaitEvent(s1, ev));
            hipCheck(hipEventDestroy(ev));
#else
            throw std::runtime_error("superbblas compiled with GPU support!");
#endif
        }

        /// Force the second context to wait until everything finishes until now from
        /// the first context.
        /// \param xpu0: first context
        /// \param xpu1: second context

        template <typename XPU1> void causalConnectTo(const Cpu &, const XPU1 &) {
            // Trivial case: do noting when the first context is on cpu
        }

        template <typename XPU0,
                  typename std::enable_if<!std::is_same<XPU0, Cpu>::value, bool>::type = true>
        void causalConnectTo(const XPU0 &xpu0, const Cpu &) {
            // Trivial case: sync the first context when the second is on cpu
            sync(xpu0);
        }

#ifdef SUPERBBLAS_USE_GPU
        void causalConnectTo(const Gpu &xpu0, const Gpu &xpu1) {
            setDevice(xpu0);
            causalConnectTo(getStream(xpu0), getStream(xpu1));
        }
#endif // SUPERBBLAS_USE_GPU

        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param cpu: context

        template <typename T, typename std::enable_if<!is_complex<T>::value, bool>::type = true>
        T *allocate(std::size_t n, Cpu cpu) {
            // Shortcut for zero allocations
            if (n == 0) return nullptr;

            tracker<Cpu> _t("allocating CPU", cpu);

            // Do the allocation
            T *r = new T[n];
            if (r == nullptr) std::runtime_error("Memory allocation failed!");

            // Annotate allocation
            if (getTrackingMemory()) {
                if (getAllocations(cpu.session).count((void *)r) > 0)
                    throw std::runtime_error("Ups! Allocator returned a pointer already in use");
                getAllocations(cpu.session)[(void *)r] = sizeof(T) * n;
                getCpuMemUsed(cpu.session) += double(sizeof(T) * n);
            }

            check_ptr_align<T>(r);
            return r;
        }

        template <typename T, typename std::enable_if<is_complex<T>::value, bool>::type = true>
        T *allocate(std::size_t n, Cpu cpu) {
            return (T *)allocate<typename T::value_type>(n * 2, cpu);
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param cpu: context

        template <typename T, typename std::enable_if<!is_complex<T>::value, bool>::type = true>
        void deallocate(T *ptr, Cpu cpu) {
            // Shortcut for zero allocations
            if (!ptr) return;

            tracker<Cpu> _t("deallocating CPU", cpu);

            // Remove annotation
            if (getTrackingMemory() && getAllocations(cpu.session).count((void *)ptr) > 0) {
                const auto &it = getAllocations(cpu.session).find((void *)ptr);
                getCpuMemUsed(cpu.session) -= double(it->second);
                getAllocations(cpu.session).erase(it);
            }

            // Deallocate the pointer
            delete[] ptr;
        }

        template <typename T, typename std::enable_if<is_complex<T>::value, bool>::type = true>
        void deallocate(T *ptr, Cpu cpu) {
            deallocate<typename T::value_type>((typename T::value_type *)ptr, cpu);
        }

#ifdef SUPERBBLAS_USE_CUDA

        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param cuda: context

        template <typename T> T *allocate(std::size_t n, Cuda cuda) {
            // Shortcut for zero allocations
            if (n == 0) return nullptr;

            tracker<Cuda> _t("allocating CUDA", cuda);

            // Do the allocation
            setDevice(cuda);
            T *r = nullptr;
            if (cuda.alloc) {
                r = (T *)cuda.alloc(sizeof(T) * n, CUDA);
            } else if (deviceId(cuda) >= 0) {
#    if CUDART_VERSION >= 11020
                cudaCheck(cudaMallocAsync(&r, sizeof(T) * n, getAllocStream(cuda)));
#    else
                cudaCheck(cudaMalloc(&r, sizeof(T) * n));
#    endif
            } else {
                cudaCheck(cudaHostAlloc(&r, sizeof(T) * n, cudaHostAllocPortable));
            }
            if (r == nullptr) std::runtime_error("Memory allocation failed!");

            // Annotate allocation
            if (getTrackingMemory()) {
                if (getAllocations(cuda.session).count((void *)r) > 0)
                    throw std::runtime_error("Ups! Allocator returned a pointer already in use");
                getAllocations(cuda.session)[(void *)r] = sizeof(T) * n;
                if (deviceId(cuda) >= 0)
                    getGpuMemUsed(cuda.session) += double(sizeof(T) * n);
                else
                    getCpuMemUsed(cuda.session) += double(sizeof(T) * n);
            }

            check_ptr_align<T>(r);
            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param cuda: context

        template <typename T> void deallocate(T *ptr, Cuda cuda) {
            // Shortcut for zero allocations
            if (!ptr) return;

            tracker<Cuda> _t("deallocating CUDA", cuda);

            // Remove annotation
            if (getTrackingMemory() && getAllocations(cuda.session).count((void *)ptr) > 0) {
                const auto &it = getAllocations(cuda.session).find((void *)ptr);
                if (deviceId(cuda) >= 0)
                    getGpuMemUsed(cuda.session) -= double(it->second);
                else
                    getCpuMemUsed(cuda.session) -= double(it->second);
                getAllocations(cuda.session).erase(it);
            }

            // Deallocate the pointer
            setDevice(cuda);
            if (cuda.dealloc) {
                cuda.dealloc((void *)ptr, CUDA);
            } else if (deviceId(cuda) >= 0) {
#    if CUDART_VERSION >= 11020
                causalConnectTo(getStream(cuda), getAllocStream(cuda));
                cudaCheck(cudaFreeAsync((void *)ptr, getAllocStream(cuda)));
#    else
                sync(cuda);
                cudaCheck(cudaFree((void *)ptr));
#    endif
            } else {
                sync(cuda);
                cudaCheck(cudaFreeHost((void *)ptr));
            }
        }

#elif defined(SUPERBBLAS_USE_HIP)

        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param hip: context

        template <typename T> T *allocate(std::size_t n, Hip hip) {
            // Shortcut for zero allocations
            if (n == 0) return nullptr;

            tracker<Hip> _t("allocating HIP", hip);

            // Do the allocation
            setDevice(hip);
            T *r = nullptr;
            if (hip.alloc) {
                r = (T *)hip.alloc(sizeof(T) * n, HIP);
            } else if (deviceId(hip) >= 0) {
#    if (HIP_VERSION_MAJOR > 5) || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 3)
                hipCheck(hipMallocAsync(&r, sizeof(T) * n, getAllocStream(hip)));
#    else
                hipCheck(hipMalloc(&r, sizeof(T) * n));
#    endif
            } else {
                hipCheck(hipHostAlloc(&r, sizeof(T) * n, hipHostAllocPortable));
            }
            if (r == nullptr) std::runtime_error("Memory allocation failed!");

            // Annotate allocation
            if (getTrackingMemory()) {
                if (getAllocations(hip.session).count((void *)r) > 0)
                    throw std::runtime_error("Ups! Allocator returned a pointer already in use");
                getAllocations(hip.session)[(void *)r] = sizeof(T) * n;
                if (deviceId(hip) >= 0)
                    getGpuMemUsed(hip.session) += double(sizeof(T) * n);
                else
                    getCpuMemUsed(hip.session) += double(sizeof(T) * n);
            }

            check_ptr_align<T>(r);
            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param hip: context

        template <typename T> void deallocate(T *ptr, Hip hip) {
            // Shortcut for zero allocations
            if (!ptr) return;

            tracker<Hip> _t("deallocating HIP", hip);

            // Remove annotation
            if (getTrackingMemory() && getAllocations(hip.session).count((void *)ptr) > 0) {
                const auto &it = getAllocations(hip.session).find((void *)ptr);
                if (deviceId(hip) >= 0)
                    getGpuMemUsed(hip.session) -= double(it->second);
                else
                    getCpuMemUsed(hip.session) -= double(it->second);
                getAllocations(hip.session).erase(it);
            }

            // Deallocate the pointer
            setDevice(hip);
            if (hip.dealloc) {
                hip.dealloc((void *)ptr, HIP);
            } else if (deviceId(hip) >= 0) {
#    if (HIP_VERSION_MAJOR > 5) || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 3)
                causalConnectTo(getStream(hip), getAllocStream(hip));
                hipCheck(hipFreeAsync((void *)ptr, getAllocStream(hip)));
#    else
                sync(hip);
                hipCheck(hipFree((void *)ptr));
#    endif
            } else {
                sync(hip);
                hipCheck(hipFreeHost((void *)ptr));
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

        /// Copy n values from v to w
        /// \param v: first element to read
        /// \param xpu0: context of v
        /// \param n: number of elements to copy
        /// \param w: first element to write
        /// \param xpu1: context of w

        template <typename T, typename XPU0, typename XPU1>
        void copy_n(const T *v, XPU0 xpu0, std::size_t n, T *w, XPU1 xpu1) {
            if (n == 0 || v == w) return;

            const bool v_is_on_cpu = deviceId(xpu0) == CPU_DEVICE_ID;
            const bool w_is_on_cpu = deviceId(xpu1) == CPU_DEVICE_ID;

            if (v_is_on_cpu && w_is_on_cpu) {
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
            else if (v_is_on_cpu != w_is_on_cpu) {
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
#    ifdef SUPERBBLAS_USE_CUDA
                cudaCheck(cudaMemcpyAsync(
                    w, v, sizeof(T) * n,
                    !v_is_on_cpu ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, stream));
#    elif defined(SUPERBBLAS_USE_HIP)
                hipCheck(hipMemcpyAsync(
                    w, v, sizeof(T) * n,
                    !v_is_on_cpu ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice, stream));
#    endif
                if (op_on_first) {
                    causalConnectTo(xpu0, xpu1);
                } else {
                    causalConnectTo(xpu1, xpu0);
                }
            } else {
                // Both pointers are on device
                causalConnectTo(xpu1, xpu0);
                setDevice(xpu0);
#    ifdef SUPERBBLAS_USE_CUDA
                if (deviceId(xpu0) == deviceId(xpu1)) {
                    cudaCheck(cudaMemcpyAsync(w, v, sizeof(T) * n, cudaMemcpyDeviceToDevice,
                                              getStream(xpu0)));
                } else {
                    cudaCheck(cudaMemcpyPeerAsync(w, deviceId(xpu1), v, deviceId(xpu0),
                                                  sizeof(T) * n, getStream(xpu0)));
                }
#    elif defined(SUPERBBLAS_USE_HIP)
                if (deviceId(xpu0) == deviceId(xpu1)) {
                    hipCheck(hipMemcpyAsync(w, v, sizeof(T) * n, hipMemcpyDeviceToDevice,
                                            getStream(xpu0)));
                } else {
                    hipCheck(hipMemcpyPeerAsync(w, deviceId(xpu1), v, deviceId(xpu0), sizeof(T) * n,
                                                getStream(xpu0)));
                }
#    else
                throw std::runtime_error("superbblas compiled with GPU support!");
#    endif
                causalConnectTo(xpu0, xpu1);
            }
#endif // SUPERBBLAS_USE_GPU
        }

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
            vector(std::size_t n, XPU xpu_,
                   std::size_t alignment = default_alignment<T_no_const>::alignment)
                : n(n),
                  ptr(allocate<T_no_const>(n + (alignment + sizeof(T) - 1) / sizeof(T), xpu_),
                      [=](const T_no_const *ptr) { deallocate(ptr, xpu_); }),
                  xpu(xpu_) {
                std::size_t size = (n + (alignment + sizeof(T) - 1) / sizeof(T)) * sizeof(T);
                ptr_aligned = (T *)align(alignment, sizeof(T) * n, ptr.get(), size);
            }

            /// Construct a vector from a given pointer `ptr` with `n` elements and with context
            /// device `xpu`. `ptr` is not deallocated after the destruction of the `vector`.
            vector(std::size_t n, T *ptr, XPU xpu)
                : n(n),
                  ptr_aligned(ptr),
                  ptr((T_no_const *)ptr, [&](const T_no_const *) {}),
                  xpu(xpu) {}

            /// Low-level constructor
            vector(std::size_t n, T *ptr_aligned, std::shared_ptr<T_no_const> ptr, XPU xpu)
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

            std::size_t n;                   ///< Number of allocated `T` elements
            T *ptr_aligned;                  ///< Pointer aligned
            std::shared_ptr<T_no_const> ptr; ///< Pointer to the allocated memory
            XPU xpu;                         ///< Context
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

#ifdef SUPERBBLAS_USE_CUDA
        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param n: number of elements to set
        /// \param cuda: device context

        template <typename T> void zero_n(T *v, std::size_t n, Cuda cuda) {
            if (n == 0) return;
            setDevice(cuda);
            cudaCheck(cudaMemsetAsync(v, 0, sizeof(T) * n, cuda.stream));
        }

#elif defined(SUPERBBLAS_USE_HIP)
        /// Set the first `n` elements with a zero value
        /// \param v: first element to set
        /// \param n: number of elements to set
        /// \param hip: device context

        template <typename T> void zero_n(T *v, std::size_t n, Hip hip) {
            if (n == 0) return;
            setDevice(hip);
            hipCheck(hipMemsetAsync(v, 0, sizeof(T) * n, hip.stream));
        }

#endif // SUPERBBLAS_USE_CUDA

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

        /// Template scal for GPUs

        template <typename T,
                  typename std::enable_if<!std::is_same<int, T>::value, bool>::type = true>
        inline void xscal(int n, T alpha, T *x, int incx, Cuda cuda) {
            if (std::fabs(alpha) == 0.0) {
                setDevice(cuda);
                cudaMemset2DAsync(x, sizeof(T) * incx, 0, sizeof(T), n, cuda.stream);
                return;
            }
            if (alpha == typename elem<T>::type{1}) return;
            cudaDataType_t cT = toCudaDataType<T>();
            cublasCheck(cublasScalEx(cuda.cublasHandle, n, &alpha, cT, x, cT, incx, cT));
        }

#elif defined(SUPERBBLAS_USE_HIP)
        template <typename T> inline hipblasDatatype_t toHipDataType(void);

        template <> inline hipblasDatatype_t toHipDataType<float>(void) { return HIPBLAS_R_32F; }
        template <> inline hipblasDatatype_t toHipDataType<std::complex<float>>(void) {
            return HIPBLAS_C_32F;
        }
        template <> inline hipblasDatatype_t toHipDataType<double>(void) { return HIPBLAS_R_64F; }
        template <> inline hipblasDatatype_t toHipDataType<std::complex<double>>(void) {
            return HIPBLAS_C_64F;
        }

        /// Template scal for GPUs

        template <typename T,
                  typename std::enable_if<!std::is_same<int, T>::value, bool>::type = true>
        inline void xscal(int n, T alpha, T *x, int incx, Hip hip) {
            if (std::fabs(alpha) == 0.0) {
                setDevice(hip);
                hipMemset2DAsync(x, sizeof(T) * incx, 0, sizeof(T), n, hip.stream);
                return;
            }
            if (alpha == typename elem<T>::type{1}) return;
            hipblasDatatype_t cT = toHipDataType<T>();
            hipCheck(hipblasScalEx(hip.hipblasHandle, n, &alpha, cT, x, cT, incx, cT));
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
        void xgemm_batch(char transa, char transb, int m, int n, int k, T alpha, const T *a[],
                         int lda, const T *b[], int ldb, T beta, T *c[], int ldc, int batch_size,
                         Cuda cuda) {
            // Quick exits
            if (m == 0 || n == 0) return;

            // Replace some invalid arguments when k is zero
            if (k == 0) {
                a = b = (const T **)c;
                lda = ldb = 1;
            }

            cudaDataType_t cT = toCudaDataType<T>();
            cublasCheck(cublasGemmBatchedEx(
                cuda.cublasHandle, toCublasTrans(transa), toCublasTrans(transb), m, n, k, &alpha,
                (const void **)a, cT, lda, (const void **)b, cT, ldb, &beta, (void **)c, cT, ldc,
                batch_size, toCudaComputeType<T>(), CUBLAS_GEMM_DEFAULT));
        }

        template <typename T>
        void xgemm_batch_strided(char transa, char transb, int m, int n, int k, T alpha, const T *a,
                                 int lda, int stridea, const T *b, int ldb, int strideb, T beta,
                                 T *c, int ldc, int stridec, int batch_size, Cuda cuda) {
            // Quick exits
            if (m == 0 || n == 0 || batch_size == 0) return;

            // Replace some invalid arguments when k is zero
            if (k == 0) {
                a = b = c;
                lda = ldb = 1;
            }

            cudaDataType_t cT = toCudaDataType<T>();
            if (batch_size == 1) {
                cublasCheck(cublasGemmEx(cuda.cublasHandle, toCublasTrans(transa),
                                         toCublasTrans(transb), m, n, k, &alpha, a, cT, lda, b, cT,
                                         ldb, &beta, c, cT, ldc, toCudaComputeType<T>(),
                                         CUBLAS_GEMM_DEFAULT));
            } else {

                cublasCheck(cublasGemmStridedBatchedEx(
                    cuda.cublasHandle, toCublasTrans(transa), toCublasTrans(transb), m, n, k,
                    &alpha, a, cT, lda, stridea, b, cT, ldb, strideb, &beta, c, cT, ldc, stridec,
                    batch_size, toCudaComputeType<T>(), CUBLAS_GEMM_DEFAULT));
            }
        }

#elif defined(SUPERBBLAS_USE_HIP)
        template <typename T> inline hipblasDatatype_t toHipComputeType(void) {
            return toHipDataType<T>();
        }

        inline hipblasOperation_t toHipblasTrans(char trans) {
            switch (trans) {
            case 'n':
            case 'N': return HIPBLAS_OP_N;
            case 't':
            case 'T': return HIPBLAS_OP_T;
            case 'c':
            case 'C': return HIPBLAS_OP_C;
            default: throw std::runtime_error("Not valid value of trans");
            }
        }

        template <typename T>
        void xgemm_batch(char transa, char transb, int m, int n, int k, T alpha, const T *a[],
                         int lda, const T *b[], int ldb, T beta, T *c[], int ldc, int batch_size,
                         Hip hip) {
            // Quick exits
            if (m == 0 || n == 0) return;

            // Replace some invalid arguments when k is zero
            if (k == 0) {
                a = b = (const T **)c;
                lda = ldb = 1;
            }

            hipblasDatatype_t cT = toHipDataType<T>();
            hipblasCheck(hipblasGemmBatchedEx(
                hip.hipblasHandle, toHipblasTrans(transa), toHipblasTrans(transb), m, n, k, &alpha,
                (const void **)a, cT, lda, (const void **)b, cT, ldb, &beta, (void **)c, cT, ldc,
                batch_size, toHipComputeType<T>(), HIPBLAS_GEMM_DEFAULT));
        }

        template <typename T>
        void xgemm_batch_strided(char transa, char transb, int m, int n, int k, T alpha, const T *a,
                                 int lda, int stridea, const T *b, int ldb, int strideb, T beta,
                                 T *c, int ldc, int stridec, int batch_size, Hip hip) {
            // Quick exits
            if (m == 0 || n == 0) return;

            // Replace some invalid arguments when k is zero
            if (k == 0) {
                a = b = c;
                lda = ldb = 1;
            }

            hipblasDatatype_t cT = toHipDataType<T>();
            hipblasCheck(hipblasGemmStridedBatchedEx(
                hip.hipblasHandle, toHipblasTrans(transa), toHipblasTrans(transb), m, n, k, &alpha,
                a, cT, lda, stridea, b, cT, ldb, strideb, &beta, c, cT, ldc, stridec, batch_size,
                toHipComputeType<T>(), HIPBLAS_GEMM_DEFAULT));
        }
#endif // SUPERBBLAS_USE_CUDA

        /// Return a copy of the vector in the given context, or the same vector if its context coincides
        /// \param v: vector to return or to clone with xpu context
        /// \param xpu: target context
        ///
        /// NOTE: implementation when the vector context and the given context are of the same type

        template <typename T, typename XPU>
        vector<T, XPU> makeSure(const vector<T, XPU> &v, XPU xpu) {
            if (deviceId(v.ctx()) == deviceId(xpu)) return v;
            vector<T, XPU> r(v.size(), xpu);
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
        vector<T, XPU1> makeSure(const vector<T, XPU0> &v, XPU1 xpu1) {
            vector<T, XPU1> r(v.size(), xpu1);
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

        Gpu anabranch_begin(const Gpu &xpu) {
            // Create a new stream, connect it causally from the given context
            setDevice(xpu);
            GpuStream new_stream = createStream();
            causalConnectTo(getStream(xpu), new_stream);
            return xpu.withNewStream(new_stream);
        }
#endif // SUPERBBLAS_USE_GPU

        Cpu anabranch_begin(const Cpu &xpu) {
            // Do nothing when context is on cpu
            return xpu;
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Join the context back the given context in `anabranch_begin`
        /// \param xpu: context to merge

        void anabranch_end(const Gpu &xpu) {
            // Connect the new stream to the original stream
            setDevice(xpu);
            causalConnectTo(getStream(xpu), getAllocStream(xpu));

            // Destroy the new stream
            destroyStream(getStream(xpu));
        }
#endif // SUPERBBLAS_USE_GPU

        void anabranch_end(const Cpu &) {
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
