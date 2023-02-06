#ifndef __SUPERBBLAS_ALLOC__
#define __SUPERBBLAS_ALLOC__

#include "cache.h"
#include "performance.h"
#include "platform.h"
#include <unordered_set>

namespace superbblas {
    namespace detail {

        /// is_complex<T>::value is true if T is std::complex
        /// \tparam T: type to inspect

        template <typename T> struct is_complex { static const bool value = false; };
        template <typename T> struct is_complex<std::complex<T>> {
            static const bool value = true;
        };
        template <typename T> struct is_complex<const T> {
            static const bool value = is_complex<T>::value;
        };

        /// Return a pointer aligned or nullptr if it isn't possible
        /// \param alignment: desired alignment of the returned pointer
        /// \param size: desired allocated size
        /// \param ptr: given pointer to align
        /// \param space: storage of the given pointer

        template <typename T>
        T *align(std::size_t alignment, std::size_t size, T *ptr, std::size_t space) {
            if (alignment == 0) return ptr;
            if (ptr == nullptr) return nullptr;

            T *r = nullptr;
            // std::align isn't in old versions of gcc
#if !defined(__GNUC__) || __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9)
            void *ptr0 = (void *)ptr;
            r = (T *)std::align(alignment, size, ptr0, space);
#else
            uintptr_t new_ptr = ((uintptr_t(ptr) + (alignment - 1)) & ~(alignment - 1));
            if (new_ptr + size - uintptr_t(ptr) > space)
                r = nullptr;
            else
                r = (T *)new_ptr;
#endif

            if (r == nullptr) throw std::runtime_error("align: fail to align pointer");
            return r;
        }

        /// Set default alignment, which is alignof(T) excepting when supporting GPUs that complex
        /// types need special alignment

        template <typename T> struct default_alignment {
            constexpr static std::size_t alignment = alignof(T);
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
            align<T>(default_alignment<T>::alignment, sizeof(T), (T *)ptr, sizeof(T));
        }

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

        /// Return a memory allocation with at least n elements of type T
        /// \param n: number of elements of the allocation
        /// \param xpu: context
        /// \param alignment: pointer alignment

        template <typename T, typename XPU>
        std::pair<T *, std::shared_ptr<char>> allocateResouce(std::size_t n, XPU xpu,
                                                              std::size_t alignment = 0) {
            // Shortcut for zero allocations
            if (n == 0) return {nullptr, std::shared_ptr<char>()};

            using T_no_const = typename std::remove_const<T>::type;
            if (alignment == 0) alignment = default_alignment<T_no_const>::alignment;
            T *ptr = allocate<T_no_const>(n + (alignment + sizeof(T) - 1) / sizeof(T), xpu);
            std::size_t size = (n + (alignment + sizeof(T) - 1) / sizeof(T)) * sizeof(T);
            T *ptr_aligned = align<T>(alignment, sizeof(T) * n, ptr, size);
            return {ptr_aligned, std::shared_ptr<char>((char *)ptr, [=](char *ptr) {
                        deallocate<T_no_const>((T_no_const *)ptr, xpu);
                    })};
        }

#ifdef SUPERBBLAS_USE_MPI
        /// Return a memory allocation with at least n elements of type T
        /// \param n: number of elements of the allocation
        /// \param xpu: context
        /// \param alignment: pointer alignment

        template <typename T>
        std::pair<T *, std::shared_ptr<char>> allocateResouce_mpi(std::size_t n, Cpu,
                                                                  std::size_t alignment = 0) {
            // Shortcut for zero allocations
            if (n == 0) return {nullptr, std::shared_ptr<char>()};

            if (alignment == 0) alignment = default_alignment<T>::alignment;
            std::size_t size = (n + (alignment + sizeof(T) - 1) / sizeof(T)) * sizeof(T);
            T *ptr = nullptr;
            MPI_check(MPI_Alloc_mem(size, MPI_INFO_NULL, &ptr));
            T *ptr_aligned = align<T>(alignment, sizeof(T) * n, ptr, size);
            return {ptr_aligned, std::shared_ptr<char>((char *)ptr, [=](char *ptr) {
                        MPI_check(MPI_Free_mem(ptr));
                    })};
        }

#    ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        std::pair<T *, std::shared_ptr<char>> allocateResouce_mpi(std::size_t n, const Gpu &xpu,
                                                                  std::size_t alignment = 0) {
            return allocateResouce<T>(n, xpu, alignment);
        }
#    endif // SUPERBBLAS_USE_GPU
#else
        template <typename T, typename XPU>
        std::pair<T *, std::shared_ptr<char>> allocateResouce_mpi(std::size_t n, const XPU &xpu,
                                                                  std::size_t alignment = 0) {
            return allocateResouce<T>(n, xpu, alignment);
        }
#endif // SUPERBBLAS_USE_MPI

        inline std::unordered_set<char *> &getAllocatedBuffers(const Cpu &) {
            static std::unordered_set<char *> allocs(16);
            return allocs;
        }

#ifdef SUPERBBLAS_USE_GPU
        inline std::unordered_set<char *> &getAllocatedBuffers(const Gpu &xpu) {
            static std::vector<std::unordered_set<char *>> allocs(getGpuDevicesCount() + 1,
                                                                  std::unordered_set<char *>(16));
            return allocs[deviceId(xpu) + 1];
        }
#endif

        using AllocationEntry = std::pair<std::size_t, std::shared_ptr<char>>;

        /// Tag class for all `allocateBufferResouce`
        struct allocate_buffer_t {};

        /// Return a memory allocation with at least n elements of type T
        /// \param n: number of elements of the allocation
        /// \param xpu: context
        /// \param alignment: pointer alignment

        template <typename T, typename XPU>
        std::pair<T *, std::shared_ptr<char>> allocateBufferResouce(std::size_t n, XPU xpu,
                                                                    std::size_t alignment = 0) {

            // Shortcut for zero allocations
            if (n == 0) return {nullptr, std::shared_ptr<char>()};

            tracker<Cpu> _t(std::string("allocate buffer ") + platformToStr(xpu), Cpu{});

            // Get alignment and the worst case size to adjust for alignment
            if (alignment == 0) alignment = default_alignment<T>::alignment;
            std::size_t size = (n + (alignment + sizeof(T) - 1) / sizeof(T)) * sizeof(T);

            // Look for the smallest free allocation that can hold the requested size.
            // Also, update `getAllocatedBuffers` by removing the buffers not longer in cache
            auto cache =
                getCache<char *, AllocationEntry, std::hash<char *>, allocate_buffer_t>(xpu);
            auto &all_buffers = getAllocatedBuffers(xpu);
            std::vector<char *> buffers_to_remove;
            std::size_t selected_buffer_size = std::numeric_limits<std::size_t>::max();
            std::shared_ptr<char> selected_buffer;
            for (char *buffer_ptr : all_buffers) {
                auto it = cache.find(buffer_ptr);
                if (it == cache.end()) {
                    buffers_to_remove.push_back(buffer_ptr);
                } else if (it->second.value.second.use_count() == 1 &&
                           it->second.value.first >= size &&
                           it->second.value.first < selected_buffer_size) {
                    selected_buffer_size = it->second.value.first;
                    selected_buffer = it->second.value.second;
                }
            }
            for (char *buffer_ptr : buffers_to_remove) all_buffers.erase(buffer_ptr);

            // If no suitable buffer was found, create a new one and cache it
            if (!selected_buffer) {
                selected_buffer = allocateResouce_mpi<T>(n, xpu, alignment).second;
                selected_buffer_size = size;
                all_buffers.insert(selected_buffer.get());
                cache.insert(selected_buffer.get(), AllocationEntry{size, selected_buffer}, size);
            }

            // Connect the allocation stream with the current stream and make sure to connect back as soon as
            // the caller finishes using the buffer
            GpuStream stream = getStream(xpu), allocStream = getAllocStream(xpu);
            causalConnectTo(allocStream, stream);
            auto return_buffer = std::shared_ptr<char>(
                selected_buffer.get(), [stream, allocStream, selected_buffer](char *) {
                    causalConnectTo(stream, allocStream);
                });

            // Align and return the buffer
            T *ptr_aligned = align<T>(alignment, sizeof(T) * n, (T *)selected_buffer.get(),
                                      selected_buffer_size);
            return {ptr_aligned, return_buffer};
        }
    }
}

#endif // __SUPERBBLAS_ALLOC__
