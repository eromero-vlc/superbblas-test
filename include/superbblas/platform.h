#ifndef __SUPERBBLAS_PLATFORM__
#define __SUPERBBLAS_PLATFORM__

#ifdef __CUDA_ARCH__
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
        struct Cuda { int device; };
        struct Gpuamd {int device; };
    }

    struct Context {
        enum platform plat;     ///< platform where the data is

        /// If `plat` is `CPU`, then `DEFAULT_DEVICE` means to use all the threads on an OpenMP
        /// fashion. If `plat` is `CUDA` and `GPUAMD`, the value is the device identification.
        int device;             
 
        detail::Cpu toCpu() const { return detail::Cpu(); }
        detail::Cuda toCuda() const { return detail::Cuda{device}; }
        detail::Gpuamd toGpuamd() const { return detail::Gpuamd{device}; }
    };

    Context createCpuContext() { return Context{CPU, 0}; }
    Context createCudaContext(int device = 0) { return Context{CUDA, device}; }
    Context createGpuamdContext(int device = 0) { return Context{GPUAMD, device}; }
}

#endif // __SUPERBBLAS_PLATFORM__
