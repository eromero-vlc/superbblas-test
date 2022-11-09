#ifndef __SUPERBBLAS_PLATFORM__
#define __SUPERBBLAS_PLATFORM__

#include "superbblas_lib.h"
#include <complex>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef __CUDACC__
#    define __HOST__ __host__
#    define __DEVICE__ __device__
#    ifndef SUPERBBLAS_USE_CUDA
#        define SUPERBBLAS_USE_CUDA
#    endif
#elif defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
#    define __HOST__ __host__
#    define __DEVICE__ __device__
#    ifndef SUPERBBLAS_USE_HIP
#        define SUPERBBLAS_USE_HIP
#    endif
#else
#    define __HOST__
#    define __DEVICE__
#endif

#ifdef SUPERBBLAS_USE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#    include <cusolverDn.h>
#    include <cusparse.h>
#endif

#ifdef SUPERBBLAS_USE_HIP
#    include <hip/hip_runtime_api.h>
#    include <hipblas.h>
#    include <hipsolver.h>
#    include <hipsparse.h>
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

#if defined(SUPERBBLAS_USE_CUDA) || defined(SUPERBBLAS_USE_HIP)
#    define SUPERBBLAS_USE_GPU
#endif

namespace superbblas {

    /// Where the data is

    enum platform {
        CPU,  ///< tradicional CPUs
        CUDA, ///< NVIDIA CUDA
        HIP   ///< AMD GPU
    };

    /// Default value in `Context`

    constexpr int CPU_DEVICE_ID = -1;

    /// Default GPU platform
#ifdef SUPERBBLAS_USE_CUDA
    const platform GPU = platform::CUDA;
#elif defined(SUPERBBLAS_USE_HIP)
    const platform GPU = platform::HIP;
#endif

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

        /// Set the current device as the one passed
        /// \param cuda: context
        inline void setDevice(Cpu) {}

        /// Return a string identifying the platform
        inline std::string platformToStr(Cpu) { return "CPU"; }

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
#    if CUDART_VERSION >= 11400
            return cublasGetStatusName(status);
#    else
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
#    endif
        }

        inline void cublasCheck(cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::stringstream s;
                s << "CUBLAS error: " << cublasStatusToStr(status);
                throw std::runtime_error(s.str());
            }
        }

        inline void cusparseCheck(cusparseStatus_t status) {
            if (status != CUSPARSE_STATUS_SUCCESS) {
                std::string str = "(unknown)";
                if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
                    str = "CUSPARSE_STATUS_NOT_INITIALIZED";
                if (status == CUSPARSE_STATUS_ALLOC_FAILED) str = "CUSPARSE_STATUS_ALLOC_FAILED";
                if (status == CUSPARSE_STATUS_INVALID_VALUE) str = "CUSPARSE_STATUS_INVALID_VALUE";
                if (status == CUSPARSE_STATUS_ARCH_MISMATCH) str = "CUSPARSE_STATUS_ARCH_MISMATCH";
                if (status == CUSPARSE_STATUS_EXECUTION_FAILED)
                    str = "CUSPARSE_STATUS_EXECUTION_FAILED";
                if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
                    str = "CUSPARSE_STATUS_INTERNAL_ERROR";
                if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
                    str = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                if (status == CUSPARSE_STATUS_NOT_SUPPORTED) str = "CUSPARSE_STATUS_NOT_SUPPORTED";
                if (status == CUSPARSE_STATUS_INSUFFICIENT_RESOURCES)
                    str = "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";

                std::stringstream ss;
                ss << "cuSparse function returned error " << str;
                throw std::runtime_error(ss.str());
            }
        }

        inline void cusolverCheck(cusolverStatus_t status) {
            if (status != CUSOLVER_STATUS_SUCCESS) {
                std::string str = "(unknown)";

                if (status == CUSOLVER_STATUS_NOT_INITIALIZED)
                    str = "CUSOLVER_STATUS_NOT_INITIALIZED";
                if (status == CUSOLVER_STATUS_ALLOC_FAILED) str = "CUSOLVER_STATUS_ALLOC_FAILED";
                if (status == CUSOLVER_STATUS_INVALID_VALUE) str = "CUSOLVER_STATUS_INVALID_VALUE";
                if (status == CUSOLVER_STATUS_ARCH_MISMATCH) str = "CUSOLVER_STATUS_ARCH_MISMATCH";
                if (status == CUSOLVER_STATUS_EXECUTION_FAILED)
                    str = "CUSOLVER_STATUS_EXECUTION_FAILED";
                if (status == CUSOLVER_STATUS_INTERNAL_ERROR)
                    str = "CUSOLVER_STATUS_INTERNAL_ERROR";
                if (status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
                    str = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                std::stringstream ss;
                ss << "cuSolver function returned error " << str;
                throw std::runtime_error(ss.str());
            }
        }

        struct Cuda {
            int device;
            cudaStream_t stream;
            cublasHandle_t cublasHandle;
            cusparseHandle_t cusparseHandle;
            cusolverDnHandle_t cusolverDnHandle;
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

        /// Return the associated cuda stream
        inline const cudaStream_t &getStream(const Cuda &cuda) { return cuda.stream; }
        inline cudaStream_t getStream(const Cpu &) { return 0; }

        /// Return a string identifying the platform
        inline std::string platformToStr(Cuda) { return "CUDA"; }

#elif defined(SUPERBBLAS_USE_HIP)
        inline void hipCheck(hipError_t err) {
            if (err != hipSuccess) {
                std::stringstream s;
                s << "HIP error: " << hipGetErrorName(err) << ": " << hipGetErrorString(err);
                throw std::runtime_error(s.str());
            }
        }

        inline void hipsparseCheck(hipsparseStatus_t status) {
            std::string str = "(unknown)";
            if (status == HIPSPARSE_STATUS_NOT_INITIALIZED)
                str = "HIPSPARSE_STATUS_NOT_INITIALIZED";
            if (status == HIPSPARSE_STATUS_ALLOC_FAILED) str = "HIPSPARSE_STATUS_ALLOC_FAILED";
            if (status == HIPSPARSE_STATUS_INVALID_VALUE) str = "HIPSPARSE_STATUS_INVALID_VALUE";
            if (status == HIPSPARSE_STATUS_ARCH_MISMATCH) str = "HIPSPARSE_STATUS_ARCH_MISMATCH";
            if (status == HIPSPARSE_STATUS_MAPPING_ERROR) str = "HIPSPARSE_STATUS_MAPPING_ERROR";
            if (status == HIPSPARSE_STATUS_EXECUTION_FAILED)
                str = "HIPSPARSE_STATUS_EXECUTION_FAILED";
            if (status == HIPSPARSE_STATUS_INTERNAL_ERROR) str = "HIPSPARSE_STATUS_INTERNAL_ERROR";
            if (status == HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
                str = "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            if (status == HIPSPARSE_STATUS_ZERO_PIVOT) str = "HIPSPARSE_STATUS_ZERO_PIVOT";
            if (status == HIPSPARSE_STATUS_NOT_SUPPORTED) str = "HIPSPARSE_STATUS_NOT_SUPPORTED";
            if (status == HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES)
                str = "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES";

            if (status != HIPSPARSE_STATUS_SUCCESS) {
                std::stringstream ss;
                ss << "hipSPARSE function returned error " << str;
                throw std::runtime_error(ss.str());
            }
        }

        /// Return the device in which the pointer was allocated

        inline int getPtrDevice(const void *x) {
            struct hipPointerAttribute_t ptr_attr;
            if (hipPointerGetAttributes(&ptr_attr, x) != hipSuccess) return CPU_DEVICE_ID;

            if (ptr_attr.memoryType != hipMemoryTypeDevice) return CPU_DEVICE_ID;
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

        inline void hipsolverCheck(hipsolverStatus_t status) {
            if (status != HIPSOLVER_STATUS_SUCCESS) {
                std::string str = "(unknown)";
                if (status == HIPSOLVER_STATUS_NOT_INITIALIZED)
                    str = "HIPSOLVER_STATUS_NOT_INITIALIZED";
                if (status == HIPSOLVER_STATUS_ALLOC_FAILED) str = "HIPSOLVER_STATUS_ALLOC_FAILED";
                if (status == HIPSOLVER_STATUS_INVALID_VALUE)
                    str = "HIPSOLVER_STATUS_INVALID_VALUE";
                if (status == HIPSOLVER_STATUS_MAPPING_ERROR)
                    str = "HIPSOLVER_STATUS_MAPPING_ERROR";
                if (status == HIPSOLVER_STATUS_EXECUTION_FAILED)
                    str = "HIPSOLVER_STATUS_EXECUTION_FAILED";
                if (status == HIPSOLVER_STATUS_INTERNAL_ERROR)
                    str = "HIPSOLVER_STATUS_INTERNAL_ERROR";
                if (status == HIPSOLVER_STATUS_NOT_SUPPORTED)
                    str = "HIPSOLVER_STATUS_NOT_SUPPORTED";
                if (status == HIPSOLVER_STATUS_ARCH_MISMATCH)
                    str = "HIPSOLVER_STATUS_ARCH_MISMATCH";
                if (status == HIPSOLVER_STATUS_HANDLE_IS_NULLPTR)
                    str = "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
                if (status == HIPSOLVER_STATUS_INVALID_ENUM) str = "HIPSOLVER_STATUS_INVALID_ENUM";
                if (status == HIPSOLVER_STATUS_UNKNOWN) str = "HIPSOLVER_STATUS_UNKNOWN";

                std::stringstream ss;
                ss << "hipSolver function returned error " << str;
                throw std::runtime_error(ss.str());
            }
        }

        struct Hip {
            int device;
            hipStream_t stream;
            hipblasHandle_t hipblasHandle;
            hipsparseHandle_t hipsparseHandle;
            hipsolverDnHandle_t hipsolverDnHandle;
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
        inline int deviceId(Hip hip) { return hip.device; }

        /// Set the current device as the one passed
        /// \param hip: context

        inline void setDevice(Hip hip) {
            int currentDevice;
            hipCheck(hipGetDevice(&currentDevice));
            if (currentDevice != deviceId(hip)) hipCheck(hipSetDevice(deviceId(hip)));
        }

        /// Return the associated cuda stream
        inline const hipStream_t &getStream(const Hip &hip) { return hip.stream; }
        inline hipStream_t getStream(const Cpu &) { return 0; }

        /// Return a string identifying the platform
        inline std::string platformToStr(Hip) { return "HIP"; }

#else
        /// Return the device in which the pointer was allocated

        inline int getPtrDevice(const void *) { return CPU_DEVICE_ID; }
#endif

        // struct Gpuamd {int device; };

        /// Return if `T` is a supported type
        template <typename T> struct supported_type { static constexpr bool value = false; };
        template <> struct supported_type<int> { static constexpr bool value = true; };
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

#ifdef SUPERBBLAS_USE_CUDA
        using Gpu = Cuda;
#elif defined(SUPERBBLAS_USE_HIP)
        using Gpu = Hip;
#else
        using Gpu = void;
#endif
    }

    class Context {
    public:
        enum platform plat; ///< platform where the data is

        /// If `plat` is `CPU`, then `DEFAULT_DEVICE` means to use all the threads on an OpenMP
        /// fashion. If `plat` is `CUDA` and `HIP`, the value is the device identification.
        int device;

    private:
        /// Optional function for allocating memory on devices
        const Allocator alloc;

        /// Optional function for deallocating memory on devices
        const Deallocator dealloc;

#ifdef SUPERBBLAS_USE_CUDA
        std::shared_ptr<cudaStream_t> stream;
        std::shared_ptr<cublasHandle_t> cublasHandle;
        std::shared_ptr<cusparseHandle_t> cusparseHandle;
        std::shared_ptr<cusolverDnHandle_t> cusolverDnHandle;
#elif defined(SUPERBBLAS_USE_HIP)
        std::shared_ptr<hipStream_t> stream;
        std::shared_ptr<hipblasHandle_t> hipblasHandle;
        std::shared_ptr<hipsparseHandle_t> hipsparseHandle;
        std::shared_ptr<hipsolverDnHandle_t> hipsolverDnHandle;
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
                const auto this_stream = stream =
                    std::shared_ptr<cudaStream_t>(new cudaStream_t, [](cudaStream_t *p) {
                        detail::cudaCheck(cudaStreamDestroy(*p));
                        delete p;
                    });
                detail::cudaCheck(cudaStreamCreate(stream.get()));
                cublasHandle = std::shared_ptr<cublasHandle_t>(
                    new cublasHandle_t, [this_stream](cublasHandle_t *p) {
                        detail::cublasCheck(cublasDestroy(*p));
                        delete p;
                    });
                detail::cublasCheck(cublasCreate(cublasHandle.get()));
                detail::cublasCheck(cublasSetStream(*cublasHandle, *stream));
                cusparseHandle = std::shared_ptr<cusparseHandle_t>(
                    new cusparseHandle_t, [this_stream](cusparseHandle_t *p) {
                        detail::cusparseCheck(cusparseDestroy(*p));
                        delete p;
                    });
                detail::cusparseCheck(cusparseCreate(cusparseHandle.get()));
                detail::cusparseCheck(cusparseSetStream(*cusparseHandle, *stream));
                cusolverDnHandle = std::shared_ptr<cusolverDnHandle_t>(
                    new cusolverDnHandle_t, [this_stream](cusolverDnHandle_t *p) {
                        detail::cusolverCheck(cusolverDnDestroy(*p));
                        delete p;
                    });
                detail::cusolverCheck(cusolverDnCreate(cusolverDnHandle.get()));
                detail::cusolverCheck(cusolverDnSetStream(*cusolverDnHandle, *stream));
            }
#elif defined(SUPERBBLAS_USE_HIP)
            if (plat == HIP) {
                int currentDevice = -1;
                detail::hipCheck(hipGetDevice(&currentDevice));
                if (currentDevice != device) detail::hipCheck(hipSetDevice(device));
                const auto this_stream = stream =
                    std::shared_ptr<hipStream_t>(new hipStream_t, [](hipStream_t *p) {
                        detail::hipCheck(hipStreamDestroy(*p));
                        delete p;
                    });
                detail::hipCheck(hipStreamCreate(stream.get()));
                hipblasHandle = std::shared_ptr<hipblasHandle_t>(
                    new hipblasHandle_t, [this_stream](hipblasHandle_t *p) {
                        detail::hipblasCheck(hipblasDestroy(*p));
                        delete p;
                    });
                detail::hipblasCheck(hipblasCreate(hipblasHandle.get()));
                detail::hipblasCheck(hipblasSetStream(*hipblasHandle, *stream));
                hipsparseHandle = std::shared_ptr<hipsparseHandle_t>(
                    new hipsparseHandle_t, [this_stream](hipsparseHandle_t *p) {
                        detail::hipsparseCheck(hipsparseDestroy(*p));
                        delete p;
                    });
                detail::hipsparseCheck(hipsparseCreate(hipsparseHandle.get()));
                detail::hipsparseCheck(hipsparseSetStream(*hipsparseHandle, *stream));
                hipsolverDnHandle = std::shared_ptr<hipsolverDnHandle_t>(
                    new hipsolverDnHandle_t, [this_stream](hipsolverDnHandle_t *p) {
                        detail::hipsolverCheck(hipsolverDnDestroy(*p));
                        delete p;
                    });
                detail::hipsolverCheck(hipsolverDnCreate(hipsolverDnHandle.get()));
                detail::hipsolverCheck(hipsolverDnSetStream(*hipsolverDnHandle, *stream));
            }
#endif
        }

        detail::Cpu toCpu(Session session) const { return detail::Cpu{session}; }

#ifdef SUPERBBLAS_USE_CUDA
        detail::Cuda toCuda(Session session) const {
            return detail::Cuda{device, *stream, *cublasHandle, *cusparseHandle, *cusolverDnHandle,
                                alloc,  dealloc, session};
        }

        detail::Cuda toGpu(Session session) const { return toCuda(session); }

#elif defined(SUPERBBLAS_USE_HIP)
        detail::Hip toHip(Session session) const {
            return detail::Hip{
                device, *stream, *hipblasHandle, *hipsparseHandle, *hipsolverDnHandle,
                alloc,  dealloc, session};
        }

        detail::Hip toGpu(Session session) const { return toHip(session); }
#else
        void toGpu(Session) const {
            throw std::runtime_error("Compiled without support for Cuda or HIP");
        }
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

    /// Return a HIP context
    /// \param device: device ID
    inline Context createHipContext(int device = 0, Allocator alloc = Allocator(),
                                    Deallocator dealloc = Deallocator()) {
        return Context{HIP, device, alloc, dealloc};
    }

    /// Return a CUDA or HIP context
    /// \param device: device ID
    inline Context createGpuContext(int device = 0, Allocator alloc = Allocator(),
                                    Deallocator dealloc = Deallocator()) {
#ifdef SUPERBBLAS_USE_CUDA
        return createCudaContext(device, alloc, dealloc);
#elif defined(SUPERBBLAS_USE_HIP)
        return createHipContext(device, alloc, dealloc);
#else
        (void)device;
        (void)alloc;
        (void)dealloc;
        throw std::runtime_error("Compiled without support for Cuda or HIP");
#endif
    }

    /// Return if `T` is a supported type
    template <typename T> struct supported_type {
        static constexpr bool value = detail::supported_type<T>::value;
    };
}

#endif // __SUPERBBLAS_PLATFORM__
