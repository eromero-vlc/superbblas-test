#ifndef __SUPERBBLAS_PLATFORM__
#define __SUPERBBLAS_PLATFORM__

#include "superbblas_lib.h"
#include <complex>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#ifdef __CUDACC__
#    define __HOST__ __host__
#    define __DEVICE__ __device__
#    ifndef SUPERBBLAS_USE_CUDA
#        define SUPERBBLAS_USE_CUDA
#    endif
#else
#    define __HOST__
#    define __DEVICE__
#endif // __CUDA_ARCH__

#if !defined(SUPERBBLAS_USE_CPU) && defined(SUPERBBLAS_USE_MKL)
#    undef SUPERBBLAS_USE_MKL
#endif

#ifdef SUPERBBLAS_USE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#endif

#ifdef SUPERBBLAS_CREATING_FLAGS
#    ifdef SUPERBBLAS_USE_CUDA
EMIT_define(SUPERBBLAS_USE_CUDA)
#    endif
#    ifdef SUPERBBLAS_USE_MKL
EMIT_define(SUPERBBLAS_USE_MKL)
#    endif
#endif

namespace superbblas {

    /// Where the data is

    enum platform {
        CPU,   ///< tradicional CPUs
        CUDA,  ///< NVIDIA CUDA
        GPUAMD ///< AMD GPU
    };

    /// Default value in `Context`

    constexpr int CPU_DEVICE_ID = -1;

    /// Function to allocate memory
    using Allocator = std::function<void *(std::size_t, enum platform)>;

    /// Function to deallocate memory
    using Deallocator = std::function<void(void *, enum platform)>;

    /// Cache session
    using Session = unsigned int;

    /// Platform and device information of data

    namespace detail {

        struct Cpu {
            /// Cache session
            Session session;

            /// Return a CPU context with the same session
            Cpu toCpu() const { return *this; }
        };

        /// Return a device identification
        inline int deviceId(Cpu) { return CPU_DEVICE_ID; }

#ifdef SUPERBBLAS_USE_CUDA

        /// Throw exception if the given error isn't success
        /// \param err: cuda error code

        inline void cudaCheck(cudaError_t err) {
            if (err != cudaSuccess) {
                std::stringstream s;
                s << "CUDA error: " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err);
                throw std::runtime_error(s.str());
            }
        }

        /// Return the device in which the pointer was allocated

        inline int getPtrDevice(const void *x) {
            struct cudaPointerAttributes ptr_attr;
            if (cudaPointerGetAttributes(&ptr_attr, x) != cudaSuccess) return CPU_DEVICE_ID;

#    if CUDART_VERSION >= 10000
            if (ptr_attr.type == cudaMemoryTypeUnregistered || ptr_attr.type == cudaMemoryTypeHost)
                return CPU_DEVICE_ID;
#    else
            if (!ptr_attr.isManaged && ptr_attr.memoryType == cudaMemoryTypeHost)
                return CPU_DEVICE_ID;
#    endif
            return ptr_attr.device;
        }

        inline const char *cublasStatusToStr(cublasStatus_t status) {
            // clang-format off
            if (status == CUBLAS_STATUS_SUCCESS         ) return "CUBLAS_STATUS_SUCCESS";
            if (status == CUBLAS_STATUS_NOT_INITIALIZED ) return "CUBLAS_STATUS_NOT_INITIALIZED";
            if (status == CUBLAS_STATUS_ALLOC_FAILED    ) return "CUBLAS_STATUS_ALLOC_FAILED";
            if (status == CUBLAS_STATUS_INVALID_VALUE   ) return "CUBLAS_STATUS_INVALID_VALUE";
            if (status == CUBLAS_STATUS_ARCH_MISMATCH   ) return "CUBLAS_STATUS_ARCH_MISMATCH";
            if (status == CUBLAS_STATUS_MAPPING_ERROR   ) return "CUBLAS_STATUS_MAPPING_ERROR";
            if (status == CUBLAS_STATUS_EXECUTION_FAILED) return "CUBLAS_STATUS_EXECUTION_FAILED";
            if (status == CUBLAS_STATUS_INTERNAL_ERROR  ) return "CUBLAS_STATUS_INTERNAL_ERROR";
            if (status == CUBLAS_STATUS_NOT_SUPPORTED   ) return "CUBLAS_STATUS_NOT_SUPPORTED";
            if (status == CUBLAS_STATUS_LICENSE_ERROR   ) return "CUBLAS_STATUS_LICENSE_ERROR";
            // clang-format on
            return "(unknown error code)";
        }

        inline void cublasCheck(cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::stringstream s;
                s << "CUBLAS error: " << cublasStatusToStr(status);
                throw std::runtime_error(s.str());
            }
        }

        struct Cuda {
            int device;
            cublasHandle_t cublasHandle;
            /// Optional function for allocating memory on devices
            Allocator alloc;
            /// Optional function for deallocating memory on devices
            Deallocator dealloc;
            /// Cache session
            Session session;

            /// Return a CPU context with the same session
            Cpu toCpu() const { return Cpu{session}; }
        };

        /// Return a device identification
        inline int deviceId(Cuda cuda) { return cuda.device; }

        /// Set the current device as the one passed
        /// \param cuda: context

        inline void setDevice(Cuda cuda) {
            int currentDevice;
            cudaCheck(cudaGetDevice(&currentDevice));
            if (currentDevice != deviceId(cuda)) cudaCheck(cudaSetDevice(deviceId(cuda)));
        }

#else

        /// Return the device in which the pointer was allocated

        inline int getPtrDevice(const void *) { return CPU_DEVICE_ID; }
#endif

        // struct Gpuamd {int device; };

        /// Return if `T` is a supported type
        template <typename T> struct supported_type { static constexpr bool value = false; };
        template <> struct supported_type<float> { static constexpr bool value = true; };
        template <> struct supported_type<double> { static constexpr bool value = true; };
        template <> struct supported_type<std::complex<float>> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<std::complex<double>> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<_Complex float> { static constexpr bool value = true; };
        template <> struct supported_type<_Complex double> { static constexpr bool value = true; };
        template <typename T> struct supported_type<const T> {
            static constexpr bool value = supported_type<T>::value;
        };
    }

    class Context {
    public:
        enum platform plat; ///< platform where the data is

        /// If `plat` is `CPU`, then `DEFAULT_DEVICE` means to use all the threads on an OpenMP
        /// fashion. If `plat` is `CUDA` and `GPUAMD`, the value is the device identification.
        int device;

    private:
        /// Optional function for allocating memory on devices
        const Allocator alloc;

        /// Optional function for deallocating memory on devices
        const Deallocator dealloc;

#ifdef SUPERBBLAS_USE_CUDA
        std::shared_ptr<cublasHandle_t> cublasHandle;
#endif

    public:
        Context(enum platform plat, int device, Allocator alloc = Allocator(),
                Deallocator dealloc = Deallocator())
            : plat(plat), device(device), alloc(alloc), dealloc(dealloc) {

#ifdef SUPERBBLAS_USE_CUDA
            if (plat == CUDA) {
                int currentDevice = -1;
                detail::cudaCheck(cudaGetDevice(&currentDevice));
                if (currentDevice != device) detail::cudaCheck(cudaSetDevice(device));
                cublasHandle =
                    std::shared_ptr<cublasHandle_t>(new cublasHandle_t, [](cublasHandle_t *p) {
                        detail::cublasCheck(cublasDestroy(*p));
                        delete p;
                    });
                detail::cublasCheck(cublasCreate(cublasHandle.get()));
            }
#endif
        }

        detail::Cpu toCpu(Session session) const { return detail::Cpu{session}; }

#ifdef SUPERBBLAS_USE_CUDA
        detail::Cuda toCuda(Session session) const {
            return detail::Cuda{device, *cublasHandle, alloc, dealloc, session};
        }
#else
        void toCuda(Session) const { throw std::runtime_error("Cuda: unsupported platform"); }
#endif
        void toGpuamd(Session) const { throw std::runtime_error("Gpuamd: unsupported platform"); }
    };

    /// Return a CPU context
    inline Context createCpuContext() { return Context{CPU, CPU_DEVICE_ID}; }

    /// Return a CUDA context
    /// \param device: device ID
    inline Context createCudaContext(int device = 0, Allocator alloc = Allocator(),
                                     Deallocator dealloc = Deallocator()) {
        return Context{CUDA, device, alloc, dealloc};
    }

    /// Return a GPUAMD context
    /// \param device: device ID
    inline Context createGpuamdContext(int device = 0) { return Context{GPUAMD, device}; }

    /// Return if `T` is a supported type
    template <typename T> struct supported_type {
        static constexpr bool value = detail::supported_type<T>::value;
    };
}

#endif // __SUPERBBLAS_PLATFORM__
