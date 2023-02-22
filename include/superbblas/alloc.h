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
        /// \param cuda: context

        template <typename T, typename XPU> T *allocate(std::size_t n, const XPU &xpu) {
            // Shortcut for zero allocations
            if (n == 0) return nullptr;

            tracker<XPU> _t(std::string("allocating ") + platformToStr(xpu), xpu);

            // Do the allocation
            setDevice(xpu);
            T *r = nullptr;
            try {
                if (getCustomAllocator()) {
                    r = (T *)getCustomAllocator()(sizeof(T) * n,
                                                  deviceId(xpu) == CPU_DEVICE_ID ? CPU : GPU);
                } else if (std::is_same<Cpu, XPU>::value) {
                    // Allocate the array without calling constructors, specially useful for std::complex
                    r = (T *)::operator new(sizeof(T) * n);
                }
#ifdef SUPERBBLAS_USE_GPU
                else if (deviceId(xpu) == CPU_DEVICE_ID) {
                    gpuCheck(SUPERBBLAS_GPU_SYMBOL(HostAlloc)(
                        &r, sizeof(T) * n, SUPERBBLAS_GPU_SYMBOL(HostAllocPortable)));
                } else {
#    ifdef SUPERBBLAS_USE_CUDA
#        if CUDART_VERSION >= 11020
                    gpuCheck(cudaMallocAsync(&r, sizeof(T) * n, getAllocStream(xpu)));
#        else
                    gpuCheck(cudaMalloc(&r, sizeof(T) * n));
#        endif
#    elif defined(SUPERBBLAS_USE_HIP)
#        if (HIP_VERSION_MAJOR > 5) || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 3)
                    gpuCheck(hipMallocAsync(&r, sizeof(T) * n, getAllocStream(xpu)));
#        else
                    gpuCheck(hipMalloc(&r, sizeof(T) * n));
#        endif
#    endif
                }
#endif // SUPERBBLAS_USE_GPU
                causalConnectTo(getAllocStream(xpu), getStream(xpu));
                if (r == nullptr) std::runtime_error("Memory allocation failed!");
            } catch (...) {
                if (getLogLevel() > 0) {
                    std::cerr << "superbblas::detail::allocate: error allocating " << sizeof(T) * n
                              << " bytes";
                    if (getTrackingMemory()) {
                        std::cerr << "; superbblas mem usage: cpu "
                                  << getCpuMemUsed(0) / 1024 / 1024 << " MiB  gpu "
                                  << getGpuMemUsed(0) / 1024 / 1024 << " MiB";
                    }
                    std::cerr << std::endl;
                }
                throw;
            }

            // Annotate allocation
            if (getTrackingMemory()) {
                if (getAllocations(xpu.session).count((void *)r) > 0)
                    throw std::runtime_error("Ups! Allocator returned a pointer already in use");
                getAllocations(xpu.session)[(void *)r] = sizeof(T) * n;
                if (deviceId(xpu) >= 0)
                    getGpuMemUsed(xpu.session) += double(sizeof(T) * n);
                else
                    getCpuMemUsed(xpu.session) += double(sizeof(T) * n);
            }

            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param cuda: context

        template <typename T, typename XPU> void deallocate(T *ptr, XPU xpu) {
            // Shortcut for zero allocations
            if (!ptr) return;

            tracker<XPU> _t(std::string("deallocating ") + platformToStr(xpu), xpu);

            // Remove annotation
            if (getTrackingMemory() && getAllocations(xpu.session).count((void *)ptr) > 0) {
                const auto &it = getAllocations(xpu.session).find((void *)ptr);
                if (deviceId(xpu) >= 0)
                    getGpuMemUsed(xpu.session) -= double(it->second);
                else
                    getCpuMemUsed(xpu.session) -= double(it->second);
                getAllocations(xpu.session).erase(it);
            }

            // Deallocate the pointer
            setDevice(xpu);
            causalConnectTo(getStream(xpu), getAllocStream(xpu));
            if (getCustomDeallocator()) {
                getCustomDeallocator()((void *)ptr, deviceId(xpu) == CPU_DEVICE_ID ? CPU : GPU);
            } else if (std::is_same<Cpu, XPU>::value) {
                ::operator delete(ptr);
#ifdef SUPERBBLAS_USE_GPU
            } else if (deviceId(xpu) == CPU_DEVICE_ID) {
                sync(getAllocStream(xpu));
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(FreeHost)((void *)ptr));
            } else {
#    ifdef SUPERBBLAS_USE_CUDA
#        if CUDART_VERSION >= 11020
                gpuCheck(cudaFreeAsync((void *)ptr, getAllocStream(xpu)));
#        else
                sync(getAllocStream(xpu));
                gpuCheck(cudaFree((void *)ptr));
#        endif
#    elif defined(SUPERBBLAS_USE_HIP)
#        if (HIP_VERSION_MAJOR > 5) || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 3)
                gpuCheck(hipFreeAsync((void *)ptr, getAllocStream(xpu)));
#        else
                sync(getAllocStream(xpu));
                gpuCheck(hipFree((void *)ptr));
#        endif
#    endif
#endif // SUPERBBLAS_USE_GPU
            }
        }

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
            return allocs.at(deviceId(xpu) + 1);
        }
#endif

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
            // Also, update `getAllocatedBuffers` by removing the buffers not longer in cache.
            // We take extra care for the fake gpu allocations (the ones with device == CPU_DEVICE_ID):
            // we avoid sharing allocations for different backup devices. It should work without this hack,
            // but it avoids correlation between different devices.
            struct AllocationEntry {
                std::size_t size;          // allocation size
                std::shared_ptr<char> res; // allocation resource
                int device;                // allocStream device
            };
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
                } else if (it->second.value.device == backupDeviceId(xpu) &&
                           it->second.value.res.use_count() == 1 && it->second.value.size >= size &&
                           it->second.value.size < selected_buffer_size) {
                    selected_buffer_size = it->second.value.size;
                    selected_buffer = it->second.value.res;
                }
            }
            for (char *buffer_ptr : buffers_to_remove) all_buffers.erase(buffer_ptr);

            // If no suitable buffer was found, create a new one and cache it
            if (!selected_buffer) {
                selected_buffer = allocateResouce_mpi<T>(n, xpu, alignment).second;
                selected_buffer_size = size;
                all_buffers.insert(selected_buffer.get());
                cache.insert(selected_buffer.get(),
                             AllocationEntry{size, selected_buffer, backupDeviceId(xpu)}, size);
            }

            // Connect the allocation stream with the current stream and make sure to connect back as soon as
            // the caller finishes using the buffer
            GpuStream stream = getStream(xpu), allocStream = getAllocStream(xpu);
            int device = backupDeviceId(xpu);
            setDevice(xpu);
            causalConnectTo(allocStream, stream);
            auto return_buffer = std::shared_ptr<char>(
                selected_buffer.get(), [stream, allocStream, selected_buffer, device](char *) {
                    if (stream != allocStream) {
                        setDevice(device);
                        causalConnectTo(stream, allocStream);
                    }
                });

            // Align and return the buffer
            T *ptr_aligned = align<T>(alignment, sizeof(T) * n, (T *)selected_buffer.get(),
                                      selected_buffer_size);
            return {ptr_aligned, return_buffer};
        }
    }
}

#endif // __SUPERBBLAS_ALLOC__
