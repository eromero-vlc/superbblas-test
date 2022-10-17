//#include "superbblas.h"
#include "superbblas/tensor.h"
#include <array>
#include <iostream>
#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t N> using ACoor = std::array<IndexType, N>;

/// Return the number of vertices in a lattice
/// \param dim: lattice dimensions

template <std::size_t Nd> std::size_t volume(const ACoor<Nd> &dim) {
    if (dim.size() <= 0) return 0;

    std::size_t vol = dim[0];
    for (std::size_t i = 1; i < dim.size(); ++i) vol *= dim[i];
    return vol;
}

template <std::size_t N, typename T, typename XPU> struct tensor {
    ACoor<N> dim;          ///< global dimensions
    vector<T, XPU> v;     ///< data

    /// Constructor with a partition
    tensor(const ACoor<N> &dim, XPU xpu)
        : dim(dim), v(vector<T, XPU>(volume(dim), xpu)) {}
};

// Dummy initialization of a tensor
template <std::size_t N, typename T, typename XPU> void dummyFill(tensor<N, T, XPU> &t) {
    vector<T, Cpu> v(t.v.size(), Cpu{});
    for (unsigned int i = 0, vol = v.size(); i < vol; i++) v[i] = i;
    copy_n(v.data(), v.ctx(), v.size(), t.v.data(), t.v.ctx());
}

constexpr std::size_t Nd = 7;          // xyztscn
constexpr unsigned int nS = 4, nC = 3; // length of dimension spin and color dimensions
constexpr unsigned int X = 0, Y = 1, Z = 2, T = 3, S = 4, C = 5, N = 6;

template <typename XPU>
void test(ACoor<Nd> dim, Context ctx, XPU xpu, unsigned int nrep) {

    using Scalar = std::complex<float>;
    using ScalarD = std::complex<double>;

    // Create tensor t0 of Nd-1 dims: a lattice color vector
    const ACoor<Nd - 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
    tensor<Nd - 1, Scalar, XPU> t0(dim0, xpu);
    dummyFill(t0);

    // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
    const ACoor<Nd> dim1 = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z], dim[C]};   // tnsxyzc
    tensor<Nd, Scalar, XPU> t1(dim1, xpu);

    const bool is_cpu = deviceId(xpu) == CPU_DEVICE_ID;
    std::cout << ">>> " << (is_cpu ? "CPU" : "GPU") << " tests:" << std::endl;

    std::size_t local_vol0 = volume(t0.dim);
    std::size_t local_vol1 = volume(t1.dim);
    std::cout << "Maximum number of elements in a tested tensor: " << local_vol1
              << " ( " << local_vol1 * 1.0 * sizeof(Scalar) / 1024 / 1024 << " MiB)" << std::endl;

    resetTimings();

    // Copy tensor t0 into tensor 1 (for reference)
    double tref = 0.0;
    {
        sync(xpu);
        vector<Scalar, XPU> aux(local_vol0 * dim[N], xpu);
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            for (int n = 0; n < dim[N]; ++n) {
                copy_n(t0.v.data(), t0.v.ctx(), local_vol0, aux.data() + local_vol0 * n, aux.ctx());
            }
        }
        sync(xpu);
        t = w_time() - t;
        std::cout << "Time in dummy copying from xyzts to tnsxyzc " << t / nrep << std::endl;
        tref = t / nrep; // time in copying a whole tensor with size dim1
    }

    // Copy tensor t0 into each of the c components of tensor 1
    {
        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            for (int n = 0; n < dim[N]; ++n) {
                const ACoor<Nd - 1> from0 = {};
                const ACoor<Nd> from1 = {0, n};
                local_copy(1.0, "xyztsc", from0.data(), dim0.data(), dim0.data(), t0.v.data(),
                           nullptr, ctx, "tnsxyzc", from1.data(), dim1.data(), t1.v.data(),
                           nullptr, ctx, SlowToFast, Copy);
            }
        }
        t = w_time() - t;
        std::cout << "Time in copying/permuting from xyztsc to tnsxyzc " << t / nrep
                  << " (overhead " << t / nrep / tref << " )" << std::endl;
    }

    // Copy tensor t0 into each of the c components of tensor 1 in double
    {
        tensor<Nd, ScalarD, XPU> t1(dim1, xpu);
        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            for (int n = 0; n < dim[N]; ++n) {
                const ACoor<Nd - 1> from0 = {};
                const ACoor<Nd> from1 = {0, n};
                local_copy(1.0, "xyztsc", from0.data(), dim0.data(), dim0.data(), t0.v.data(),
                           nullptr, ctx, "tnsxyzc", from1.data(), dim1.data(), t1.v.data(),
                           nullptr, ctx, SlowToFast, Copy);
            }
        }
        sync(xpu);
        t = w_time() - t;
        std::cout << "Time in copying/permuting from xyztsc (single) to tnsxyzc (double) "
                  << t / nrep << " (overhead " << t / nrep / tref << " )" << std::endl;
    }

    // Shift tensor 1 on the z-direction and store it on tensor 2
    tensor<Nd, Scalar, XPU> t2(dim1, xpu);
    {
        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            const ACoor<Nd> from0 = {};
            ACoor<Nd> from1 = {};
            from1[4] = 1; // Displace one on the z-direction
            local_copy(1.0, "tnsxyzc", from0.data(), dim1.data(), dim1.data(), t1.v.data(), nullptr,
                       ctx, "tnsxyzc", from1.data(), dim1.data(), t2.v.data(), nullptr, ctx,
                       SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        std::cout << "Time in shifting " << t / nrep << std::endl;
    }

    // Create tensor t3 of 5 dims
    {
        const ACoor<5> dimc = {dim[T], dim[N], dim[S], dim[N], dim[S]}; // tnsns
        tensor<5, Scalar, XPU> tc(dimc, xpu);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            local_contraction<Scalar>(1.0, dim1.data(), "tnsxyzc", false, t1.v.data(), ctx,
                                      dim1.data(), "tNSxyzc", false, t2.v.data(), ctx, 0.0,
                                      dimc.data(), "tNSns", tc.v.data(), ctx, SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        std::cout << "Time in contracting xyzs " << t / nrep << std::endl;
    }

    reportTimings(std::cout);
    reportCacheUsage(std::cout);
}

int main(int argc, char **argv) {
    ACoor<Nd> dim = {16, 16, 16, 32, nS, nC, 64}; // xyztscn
    unsigned int nrep = getDebugLevel() == 0 ? 10 : 1;

    // Get options
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d %d %d", &dim[X], &dim[Y], &dim[Z], &dim[T],
                       &dim[N]) != 5) {
                std::cerr << "--dim= should follow 5 numbers, for instance -dim='2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
        } else if (std::strncmp("--reps=", argv[i], 7) == 0) {
            if (sscanf(argv[i] + 7, "%d", &nrep) != 1) {
                std::cerr << "--reps= should follow one number" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--dim='x y z t n'] [--reps=r] [--help]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // Show lattice dimensions arrangement
    std::cout << "Testing lattice dimensions xyzt= " << dim[X] << " " << dim[Y] << " " << dim[Z]
              << " " << dim[T] << " spin-color= " << dim[S] << " " << dim[C]
              << "  num_vecs= " << dim[N] << std::endl;

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    std::cout << "Tests with " << num_threads << " threads" << std::endl;

    {
        Context ctx = createCpuContext();
        test(dim, ctx, ctx.toCpu(0), nrep);
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext();
        test(dim, ctx, ctx.toGpu(0), nrep);
    }
#endif

    // Clear internal superbblas caches
    clearCaches();

    return 0;
}
