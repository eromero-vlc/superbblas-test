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
#elif defined(__HIPCC__)
#    define __HOST__ __host__
#    define __DEVICE__ __device__
#    ifndef SUPERBBLAS_USE_HIP
#        define SUPERBBLAS_USE_HIP
#    endif
#else
#    define __HOST__
#    define __DEVICE__
#endif

#if !defined(SUPERBBLAS_USE_CPU) && defined(SUPERBBLAS_USE_MKL)
#    undef SUPERBBLAS_USE_MKL
#endif

#ifdef SUPERBBLAS_USE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#endif

#ifdef SUPERBBLAS_USE_HIP
#    include <hipblas.h>
#    include <hip/hip_runtime_api.h>
#endif

#ifdef SUPERBBLAS_CREATING_FLAGS
#    ifdef SUPERBBLAS_USE_CUDA
EMIT_define(SUPERBBLAS_USE_CUDA)
#    endif
#    ifdef SUPERBBLAS_USE_HIP
EMIT_define(SUPERBBLAS_USE_HIP)
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

    /// Platform and device information of data

    namespace detail {
        struct Cpu {};

        /// Return a device identification
        inline int deviceId(Cpu) { return CPU_DEVICE_ID; }

#ifdef SUPERBBLAS_USE_CUDA
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
        };

        /// Return a device identification
        inline int deviceId(Cuda cuda) { return cuda.device; }

#elif defined(SUPERBBLAS_USE_HIP)
        inline void hipCheck(hipError_t err) {
            if (err != hipSuccess) {
                std::stringstream s;
                s << "HIP error: " << hipGetErrorName(err) << ": " << hipGetErrorString(err);
                throw std::runtime_error(s.str());
            }
        }

        /// Return the device in which the pointer was allocated

        inline int getPtrDevice(const void *x) {
            struct hipPointerAttribute_t ptr_attr;
            if (hipPointerGetAttributes(&ptr_attr, x) != hipSuccess) return CPU_DEVICE_ID;

            if (ptr_attr.memoryType != hipMemoryTypeDevice)
                return CPU_DEVICE_ID;
            return ptr_attr.device;
        }

        inline const char *hipblasStatusToStr(hipblasStatus_t status) {
            // clang-format off
            if (status == HIPBLAS_STATUS_SUCCESS         ) return "HIPBLAS_STATUS_SUCCESS";
            if (status == HIPBLAS_STATUS_NOT_INITIALIZED ) return "HIPBLAS_STATUS_NOT_INITIALIZED";
            if (status == HIPBLAS_STATUS_ALLOC_FAILED    ) return "HIPBLAS_STATUS_ALLOC_FAILED";
            if (status == HIPBLAS_STATUS_INVALID_VALUE   ) return "HIPBLAS_STATUS_INVALID_VALUE";
            if (status == HIPBLAS_STATUS_ARCH_MISMATCH   ) return "HIPBLAS_STATUS_ARCH_MISMATCH";
            if (status == HIPBLAS_STATUS_MAPPING_ERROR   ) return "HIPBLAS_STATUS_MAPPING_ERROR";
            if (status == HIPBLAS_STATUS_EXECUTION_FAILED) return "HIPBLAS_STATUS_EXECUTION_FAILED";
            if (status == HIPBLAS_STATUS_INTERNAL_ERROR  ) return "HIPBLAS_STATUS_INTERNAL_ERROR";
            if (status == HIPBLAS_STATUS_NOT_SUPPORTED   ) return "HIPBLAS_STATUS_NOT_SUPPORTED";
            // clang-format on
            return "(unknown error code)";
        }

        inline void hipblasCheck(hipblasStatus_t status) {
            if (status != HIPBLAS_STATUS_SUCCESS) {
                std::stringstream s;
                s << "HIPBLAS error: " << hipblasStatusToStr(status);
                throw std::runtime_error(s.str());
            }
        }

        struct Hip {
            int device;
            hipblasHandle_t hipblasHandle;
            /// Optional function for allocating memory on devices
            Allocator alloc;
            /// Optional function for deallocating memory on devices
            Deallocator dealloc;
        };

        /// Return a device identification
        inline int deviceId(Hip hip) { return hip.device; }
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
#elif defined(SUPERBBLAS_USE_HIP)
        std::shared_ptr<hipblasHandle_t> hipblasHandle;
#endif

    public:
        Context(enum platform plat, int device, Allocator alloc = Allocator(),
                Deallocator dealloc = Deallocator())
            : plat(plat), device(device), alloc(alloc), dealloc(dealloc) {

#ifdef SUPERBBLAS_USE_CUDA
            if (plat == CUDA) {
                cublasHandle =
                    std::shared_ptr<cublasHandle_t>(new cublasHandle_t, [](cublasHandle_t *p) {
                        detail::cublasCheck(cublasDestroy(*p));
                        delete p;
                    });
                detail::cublasCheck(cublasCreate(cublasHandle.get()));
            }
#elif defined(SUPERBBLAS_USE_HIP)
            if (plat == GPUAMD) {
                hipblasHandle =
                    std::shared_ptr<hipblasHandle_t>(new hipblasHandle_t, [](hipblasHandle_t *p) {
                        detail::hipblasCheck(hipblasDestroy(*p));
                        delete p;
                    });
                detail::hipblasCheck(hipblasCreate(hipblasHandle.get()));
            }
#endif
        }

        detail::Cpu toCpu() const { return detail::Cpu(); }

#ifdef SUPERBBLAS_USE_CUDA
        detail::Cuda toCuda() const { return detail::Cuda{device, *cublasHandle, alloc, dealloc}; }
#elif defined(SUPERBBLAS_USE_HIP)
        detail::Hip toHip() const { return detail::Hip{device, *hipblasHandle, alloc, dealloc}; }
#else
        void toCuda() const { throw std::runtime_error("Cuda: unsupported platform"); }
        void toHip() const { throw std::runtime_error("HIP: unsupported platform"); }
#endif
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
    inline Context createHipContext(int device = 0, Allocator alloc = Allocator(),
                                     Deallocator dealloc = Deallocator()) {
        return Context{GPUAMD, device, alloc, dealloc};
    }

    /// Return if `T` is a supported type
    template <typename T> struct supported_type {
        static constexpr bool value = detail::supported_type<T>::value;
    };
}

#endif // __SUPERBBLAS_PLATFORM__
