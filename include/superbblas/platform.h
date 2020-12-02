#ifndef __SUPERBBLAS_PLATFORM__
#define __SUPERBBLAS_PLATFORM__

#include <memory>
#include <sstream>
#include <stdexcept>

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
#endif


namespace superbblas {

    /// Where the data is

    enum platform {
        CPU,     ///< tradicional CPUs
        CUDA,    ///< NVIDIA CUDA
        GPUAMD   ///< AMD GPU
    };

    /// Default value in `Context`

    constexpr int DEFAULT_DEVICE = -1;

    /// Platform and device information of data

    namespace detail {
        struct Cpu {};

#ifdef SUPERBBLAS_USE_CUDA
        inline void cublasCheck(cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::stringstream s;
                s << "CUBLAS error " << status;
                throw std::runtime_error(s.str());
            }
        }

        struct Cuda {
            int device;
            std::shared_ptr<cublasHandle_t> cublasHandle;
            Cuda(int device)
                : device(device), cublasHandle(new cublasHandle_t, [](cublasHandle_t *p) {
                      cublasCheck(cublasDestroy(*p));
                      delete p;
                  }) {
                cublasCheck(cublasCreate(cublasHandle.get()));
            }
        };
#endif

//        struct Gpuamd {int device; };
    }

    struct Context {
        enum platform plat;     ///< platform where the data is

        /// If `plat` is `CPU`, then `DEFAULT_DEVICE` means to use all the threads on an OpenMP
        /// fashion. If `plat` is `CUDA` and `GPUAMD`, the value is the device identification.
        int device;             
 
        detail::Cpu toCpu() const { return detail::Cpu(); }
#ifdef SUPERBBLAS_USE_CUDA
        detail::Cuda toCuda() const { return detail::Cuda{device}; }
#else
        void toCuda() const { throw std::runtime_error("Cuda: unsupported platform"); }
#endif
        void toGpuamd() const { throw std::runtime_error("Gpuamd: unsupported platform"); }
    };

    /// Return a CPU context
    inline Context createCpuContext() { return Context{CPU, 0}; }

    /// Return a CUDA context
    /// \param device: device ID
    inline Context createCudaContext(int device = 0) { return Context{CUDA, device}; }

    /// Return a GPUAMD context
    /// \param device: device ID
    inline Context createGpuamdContext(int device = 0) { return Context{GPUAMD, device}; }
}

#endif // __SUPERBBLAS_PLATFORM__
