#include "superbblas.h"
#include <iostream>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef SUPERBBLAS_USE_CUDA
#    include <thrust/device_vector.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <std::size_t Nd> PartitionStored<Nd> dist_tensor_on_root(Coor<Nd> dim, int nprocs) {
    PartitionStored<Nd> fs(nprocs);
    if (1 <= nprocs) fs[0][1] = dim;
    return fs;
}

template <std::size_t N, typename T, typename XPU> struct tensor {
    Coor<N> dim;          ///< global dimensions
    PartitionStored<N> p; ///< partition
    vector<T, XPU> v;     ///< data
    int rank;             ///< rank of the current process

    /// Constructor with a partition
    tensor(const Coor<N> &dim, const PartitionStored<N> &p, int rank, XPU xpu)
        : dim(dim), p(p), v(vector<T, XPU>(volume(p[rank][1]), xpu)), rank(rank) {}

    /// Constructor for a distributed tensor
    tensor(const Coor<N> &dim, const Coor<N> &procs, int rank, XPU xpu)
        : tensor(dim, basic_partitioning(dim, procs), rank, xpu) {}

    /// Constructor for a distributed tensor with power
    tensor(const Coor<N> &dim, const Coor<N> &procs, const Coor<N> &power, int rank, XPU xpu)
        : tensor(dim, basic_partitioning(dim, procs, -1, false, power), rank, xpu) {}

    /// Constructor for a tensor with support only on the root process
    tensor(const Coor<N> &dim, int nprocs, int rank, XPU xpu)
        : tensor(dim, dist_tensor_on_root(dim, nprocs), rank, xpu) {}
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
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, Context ctx, XPU xpu, unsigned int nrep) {

    using Scalar = std::complex<float>;
    using ScalarD = std::complex<double>;

    // Create tensor t0 of Nd-1 dims: a lattice color vector
    const Coor<Nd - 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
    const Coor<Nd - 1> procs0 = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsc
    tensor<Nd - 1, Scalar, XPU> t0(dim0, procs0, rank, xpu);
    dummyFill(t0);

    // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
    const Coor<Nd> dim1 = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z], dim[C]};   // tnsxyzc
    const Coor<Nd> procs1 = {procs[T], procs[N], 1, procs[X], procs[Y], procs[Z], 1}; // tnsxyzc
    tensor<Nd, Scalar, XPU> t1(dim1, procs1, rank, xpu);

    const bool is_cpu = deviceId(xpu) == CPU_DEVICE_ID;
    if (rank == 0) std::cout << ">>> " << (is_cpu ? "CPU" : "GPU") << " tests:" << std::endl;

    std::size_t local_vol0 = volume(t0.p[rank][1]);
    std::size_t local_vol1 = volume(t1.p[rank][1]);
    if (rank == 0)
        std::cout << "Maximum number of elements in a tested tensor per process: " << local_vol1
                  << " ( " << local_vol1 * 1.0 * sizeof(Scalar) / 1024 / 1024 << " MiB)"
                  << std::endl;

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
        if (rank == 0)
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
                const Coor<Nd - 1> from0 = {};
                const Coor<Nd> from1 = {0, n};
                Scalar *ptr0 = t0.v.data(), *ptr1 = t1.v.data();
                copy(1.0, t0.p.data(), 1, "xyztsc", from0, dim0, dim0, (const Scalar **)&ptr0,
                     nullptr, &ctx, t1.p.data(), 1, "tnsxyzc", from1, dim1, &ptr1, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                     MPI_COMM_WORLD,
#endif
                     SlowToFast, Copy);
            }
        }
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in copying/permuting from xyztsc to tnsxyzc " << t / nrep
                      << " (overhead " << t / nrep / tref << " )" << std::endl;
    }

    // Copy tensor t0 into each of the c components of tensor 1 in double
    {
        tensor<Nd, ScalarD, XPU> t1(dim1, procs1, rank, xpu);
        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            for (int n = 0; n < dim[N]; ++n) {
                const Coor<Nd - 1> from0 = {};
                const Coor<Nd> from1 = {0, n};
                Scalar *ptr0 = t0.v.data();
                ScalarD *ptr1 = t1.v.data();
                copy(1.0, t0.p.data(), 1, "xyztsc", from0, dim0, dim0, (const Scalar **)&ptr0,
                     nullptr, &ctx, t1.p.data(), 1, "tnsxyzc", from1, dim1, &ptr1, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                     MPI_COMM_WORLD,
#endif
                     SlowToFast, Copy);
            }
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in copying/permuting from xyztsc (single) to tnsxyzc (double) "
                      << t / nrep << " (overhead " << t / nrep / tref << " )" << std::endl;
    }

    // Shift tensor 1 on the z-direction and store it on tensor 2
    tensor<Nd, Scalar, XPU> t2(dim1, procs1, rank, xpu);
    {
        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            const Coor<Nd> from0 = {};
            Coor<Nd> from1 = {};
            from1[4] = 1; // Displace one on the z-direction
            Scalar *ptr0 = t1.v.data(), *ptr1 = t2.v.data();
            copy(1.0, t1.p.data(), 1, "tnsxyzc", from0, dim1, dim1, (const Scalar **)&ptr0, nullptr,
                 &ctx, t2.p.data(), 1, "tnsxyzc", from1, dim1, &ptr1, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in shifting " << t / nrep << std::endl;
    }

    // Create tensor t3 of 5 dims
    {
        const Coor<5> dimc = {dim[T], dim[N], dim[S], dim[N], dim[S]}; // tnsns
        tensor<5, Scalar, XPU> tc(dimc, volume(procs), rank, xpu);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            Scalar *ptr0 = t1.v.data(), *ptr1 = t2.v.data(), *ptrc = tc.v.data();
            contraction(Scalar{1.0}, t1.p.data(), dim1, 1, "tnsxyzc", false, (const Scalar **)&ptr0,
                        &ctx, t2.p.data(), dim1, 1, "tNSxyzc", false, (const Scalar **)&ptr1, &ctx,
                        Scalar{0.0}, tc.p.data(), dimc, 1, "tNSns", &ptrc, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in contracting xyzs " << t / nrep << std::endl;
    }

    // Copy halos
    {
	const int power = 1;
        const Coor<Nd> ext = {power, 0, 0, power, power, power, 0}; // tnsxyzc
        tensor<Nd, Scalar, XPU> th(dim1, procs1, ext, rank, xpu);
        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) t = w_time();
            const Coor<Nd> from0 = {};
            Coor<Nd> from1 = {};
            Scalar *ptr1 = t1.v.data(), *ptrh = th.v.data();
            copy(1.0, t1.p.data(), 1, "tnsxyzc", from0, dim1, dim1, (const Scalar **)&ptr1, nullptr,
                 &ctx, th.p.data(), 1, "tnsxyzc", from1, dim1, &ptrh, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in copying halos in " << t / nrep << std::endl;

        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) t = w_time();
            const Coor<Nd> from0 = {};
            Coor<Nd> from1 = {};
            Scalar *ptrh = th.v.data(), *ptr1 = t1.v.data();
            copy(1.0, th.p.data(), 1, "tnsxyzc", from1, dim1, dim1, (const Scalar **)&ptrh, nullptr,
                 &ctx, t1.p.data(), 1, "tnsxyzc", from0, dim1, &ptr1, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in copying halos out " << t / nrep << std::endl;
    }

    // Copy halos
    {
	const int power = 1;
        const Coor<Nd> ext = {power, 0, 0, power, power, power, 0}; // tnsxyzc
        tensor<Nd, int, XPU> t1(dim1, procs1, rank, xpu);
        tensor<Nd, int, XPU> th(dim1, procs1, ext, rank, xpu);
        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            const Coor<Nd> from0 = {};
            Coor<Nd> from1 = {};
            int *ptr1 = t1.v.data(), *ptrh = th.v.data();
            copy(1, t1.p.data(), 1, "tnsxyzc", from0, dim1, dim1, (const int **)&ptr1, nullptr,
                 &ctx, th.p.data(), 1, "tnsxyzc", from1, dim1, &ptrh, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in copying halos in for integers " << t / nrep << std::endl;

        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            const Coor<Nd> from0 = {};
            Coor<Nd> from1 = {};
            int *ptrh = th.v.data(), *ptr1 = t1.v.data();
            copy(1, th.p.data(), 1, "tnsxyzc", from1, dim1, dim1, (const int **)&ptrh, nullptr,
                 &ctx, t1.p.data(), 1, "tnsxyzc", from0, dim1, &ptr1, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in copying halos out for integers " << t / nrep << std::endl;
    }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);
}

int main(int argc, char **argv) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc;
    (void)argv;
    nprocs = 1;
    rank = 0;
#endif

    Coor<Nd> dim = {16, 16, 16, 32, nS, nC, 64}; // xyztscn
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1};
    unsigned int nrep = getDebugLevel() == 0 ? 10 : 1;

    // Get options
    bool procs_was_set = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d %d %d", &dim[X], &dim[Y], &dim[Z], &dim[T],
                       &dim[N]) != 5) {
                std::cerr << "--dim= should follow 5 numbers, for instance -dim='2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
        } else if (std::strncmp("--procs=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, "%d %d %d %d", &procs[X], &procs[Y], &procs[Z], &procs[T]) !=
                4) {
                std::cerr << "--procs= should follow 4 numbers, for instance --procs='2 2 2 2'"
                          << std::endl;
                return -1;
            }
            if (detail::volume(procs) != (std::size_t)nprocs) {
                std::cerr << "The total number of processes set by the option `--procs=` should "
                             "match the number of processes"
                          << std::endl;
                return -1;
            }
            procs_was_set = true;
        } else if (std::strncmp("--reps=", argv[i], 7) == 0) {
            if (sscanf(argv[i] + 7, "%d", &nrep) != 1) {
                std::cerr << "--reps= should follow one number" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--dim='x y z t n'] [--procs='x y z t n'] [--reps=r] [--help]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // If --procs isn't set, put all processes on the first dimension
    if (!procs_was_set) procs[X] = nprocs;

    // Show lattice dimensions and processes arrangement
    if (rank == 0) {
        std::cout << "Testing lattice dimensions xyzt= " << dim[X] << " " << dim[Y] << " " << dim[Z]
                  << " " << dim[T] << " spin-color= " << dim[S] << " " << dim[C]
                  << "  num_vecs= " << dim[N] << std::endl;
        std::cout << "Processes arrangement xyzt= " << procs[X] << " " << procs[Y] << " "
                  << procs[Z] << " " << procs[T] << std::endl;
    }

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    if (rank == 0) std::cout << "Tests with " << num_threads << " threads" << std::endl;

    {
        Context ctx = createCpuContext();
        test(dim, procs, rank, ctx, ctx.toCpu(0), nrep);
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext();
        test(dim, procs, rank, ctx, ctx.toGpu(0), nrep);
    }
#endif

    // Clear internal superbblas caches
    clearCaches();

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
