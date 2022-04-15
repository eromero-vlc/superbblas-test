#include "superbblas.h"
#include <vector>
#include <algorithm>
#include <iostream>
#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

constexpr std::size_t Nd = 7; // xyztscn
constexpr unsigned int X = 0, Y = 1, Z = 2, T = 3, S = 4, C = 5, N = 6;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

// Return a vector of all ones
template <typename T, typename XPU> vector<T, XPU> ones(std::size_t size, XPU xpu) {
    vector<T, Cpu> r(size, Cpu{});
    for (std::size_t i = 0; i < size; ++i) r[i] = 1.0;
    return makeSure(r, xpu);
}

// Return a vector of all ones
template <typename T, typename XPU> vector<T, XPU> laplacian(std::size_t n, std::size_t size, XPU xpu) {
    vector<T, Cpu> r(size, Cpu{});
    if (size % (n * n) != 0)
        throw std::runtime_error("Unsupported the creation of partial square matrices");
    for (std::size_t i = 0; i < size; ++i) r[i] = 0;
    for (std::size_t k = 0, K = size / (n * n); k < K; ++k) {
        for (std::size_t i = 0; i < n; ++i) r[k * n * n + i * n + i] = -2;
        for (std::size_t i = 0; i < n - 1; ++i) r[k * n * n + (i + 1) * n + i] = 1;
        for (std::size_t i = 1; i < n; ++i) r[k * n * n + (i - 1) * n + i] = 1;
    }
    return makeSure(r, xpu);
}

template <typename Q, typename XPU>
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, Context ctx, XPU xpu,
          unsigned int nrep = 10) {

    // Create tensor t0 of Nd dims: a lattice color vector
    const Coor<Nd + 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T],
                               dim[S], dim[C], dim[S], dim[C]}; // xyztscSC
    const Coor<Nd + 1> procs0 = {procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1, 1}; // xyztscSC
    PartitionStored<Nd + 1> p0 = basic_partitioning(dim0, procs0);
    const Coor<Nd + 1> local_size0 = p0[rank][1];
    std::size_t vol0 = detail::volume(local_size0);
    vector<Q, XPU> t0 = laplacian<Q>(dim[S] * dim[C] * dim[N], vol0, xpu);

    const bool is_cpu = deviceId(xpu) == CPU_DEVICE_ID;
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    if (rank == 0)
        std::cout << ">>> " << (is_cpu ? "CPU" : "GPU") << " tests with " << num_threads
                  << " threads" << std::endl;

    if (rank == 0)
        std::cout << "Maximum number of elements in a tested tensor per process: " << vol0 << " ( "
                  << vol0 * 1.0 * sizeof(Q) / 1024 / 1024 << " MiB)" << std::endl;

    // Copy tensor t0 into each of the c components of tensor 1
    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            for (int n = 0; n < dim[N]; ++n) {
                Q *ptr0 = t0.data();
                cholesky<Nd + 1, Q>(p0.data(), 1, "xyztscSC", (Q **)&ptr0, "sc", "SC", &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                    MPI_COMM_WORLD,
#endif
                                    SlowToFast);
            }
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in cholesky " << t / nrep << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    resetTimings();

    // Create tensors tx and ty of Nd dims: a lattice color vector
    const Coor<Nd> dimx = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C], dim[N]}; // xyztscn
    const Coor<Nd> procsx = {procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}; // xyztscn
    PartitionStored<Nd> px = basic_partitioning(dimx, procsx);
    const Coor<Nd> local_sizex = px[rank][1];
    std::size_t volx = detail::volume(local_sizex);
    vector<Q, XPU> tx = ones<Q>(volx, xpu);
    vector<Q, XPU> ty(volx, xpu);

    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            for (int n = 0; n < dim[N]; ++n) {
                Q *ptr0 = t0.data();
                Q *ptrx = tx.data();
                Q *ptry = ty.data();
                trsm<Nd + 1, Nd, Nd, Q>(p0.data(), 1, "xyztscSC", (const Q **)&ptr0, "sc", "SC",
                                        &ctx, px.data(), 1, "xyztscn", (const Q **)&ptrx, &ctx,
                                        px.data(), 1, "xyztSCn", (Q **)&ptry, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                        MPI_COMM_WORLD,
#endif
                                        SlowToFast);
            }
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in trsm " << t / nrep << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

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

    Coor<Nd> dim = {16, 16, 16, 32, 1, 12, 64}; // xyztscn
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1};

    // Get options
    bool procs_was_set = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d %d %d %d", &dim[X], &dim[Y], &dim[Z], &dim[T],
                       &dim[N], &dim[C]) != 6) {
                std::cerr << "--dim= should follow 6 numbers, for instance -dim='2 2 2 2 2 2'"
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
         } else if(std::strncmp("--help", argv[i], 6) == 0) {
             std::cout << "Commandline option:\n  " << argv[0]
                       << " [--dim='x y z t n b'] [--procs='x y z t n c'] [--help]"
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

    {
        Context ctx = createCpuContext();
        test<std::complex<float>, Cpu>(dim, procs, rank, ctx, ctx.toCpu(0));
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext();
        test<std::complex<float>, Gpu>(dim, procs, rank, ctx, ctx.toGpu(0));
    }
#endif

    // Clear internal superbblas caches
    clearCaches();

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
