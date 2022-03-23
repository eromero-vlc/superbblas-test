#include "superbblas.h"
#include <vector>
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

/// Create a 4D lattice with dimensions tzyxsc
template <typename T, typename XPU>
std::pair<BSR_handle *, vector<T, XPU>> create_lattice(const PartitionStored<6> &pi, int rank,
                                                       const Coor<6> op_dim, Context ctx, XPU xpu) {
    Coor<6> from = pi[rank][0]; // first nonblock dimensions of the RSB image
    Coor<6> dimi = pi[rank][1]; // nonblock dimensions of the RSB image
    dimi[4] = dimi[5] = 1;
    std::size_t voli = volume(dimi);
    vector<IndexType, Cpu> ii(voli, Cpu{});

    // Compute how many neighbors
    int neighbors = 1;
    for (int dim = 0; dim < 4; ++dim) {
        int d = op_dim[dim];
        if (d <= 0) {
            neighbors = 0;
            break;
        }
        if (d > 1) neighbors++;
        if (d > 2) neighbors++;
    }

    // Compute the domain ranges
    PartitionStored<6> pd = pi;
    for (auto &i : pd) {
        for (int dim = 0; dim < 4; ++dim) {
            i[1][dim] = std::min(op_dim[dim], i[1][dim] + 2);
            if (i[1][dim] < op_dim[dim])
                i[0][dim]--;
            else
                i[0][dim] = 0;
        }
        i[0] = normalize_coor(i[0], op_dim);
    }

    // Compute the coordinates for all nonzeros
    for (auto &i : ii) i = neighbors;
    vector<Coor<6>, Cpu> jj(neighbors * voli, Cpu{});
    Coor<6> stride = get_strides<6>(dimi, SlowToFast);
    for (std::size_t i = 0, j = 0; i < voli; ++i) {
        Coor<6> c = index2coor(i, dimi, stride) + from;
        jj[j++] = c;
        for (int dim = 0; dim < 4; ++dim) {
            if (op_dim[dim] == 1) continue;
            for (int dir = -1; dir < 2; dir += 2) {
                Coor<6> c0 = c;
                c0[dim] += dir;
                jj[j++] = normalize_coor(c0 - pd[rank][0], op_dim);
                if (op_dim[dim] <= 2) break;
            }
        }
    }

    // Number of nonzeros
    std::size_t vol_data = voli * neighbors * op_dim[4] * op_dim[5] * op_dim[4] * op_dim[5];
    if (rank == 0)
        std::cout << "Size of the sparse tensor per process: "
                  << vol_data * 1.0 * sizeof(T) / 1024 / 1024 << " MiB" << std::endl;

    Coor<6> block{{1, 1, 1, 1, op_dim[4], op_dim[5]}};
    BSR_handle *bsrh = nullptr;
    vector<int, XPU> ii_xpu = makeSure(ii, xpu);
    vector<Coor<6>, XPU> jj_xpu = makeSure(jj, xpu);
    vector<T, XPU> data_xpu = ones<T>(vol_data, xpu);
    IndexType *iiptr = ii_xpu.data();
    Coor<6> *jjptr = jj_xpu.data();
    T *dataptr = data_xpu.data();
    create_bsr<6, 6, T>(pi.data(), pd.data(), 1, block, block, false, &iiptr, &jjptr,
                        (const T **)&dataptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast, &bsrh);
    return {bsrh, data_xpu};
}

template <typename Q, typename XPU>
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, int max_power, Context ctx, XPU xpu,
          unsigned int nrep = 10) {

    // Create a lattice operator of Nd-1 dims
    const Coor<Nd - 1> dimo = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
    const Coor<Nd - 1> procso = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsc
    PartitionStored<Nd - 1> po = basic_partitioning(dimo, procso);
    auto op_pair = create_lattice<Q>(po, rank, dimo, ctx, xpu);
    BSR_handle *op = op_pair.first;

    // Create tensor t0 of Nd dims: an input lattice color vector
    const Coor<Nd> dim0 = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C], dim[N]}; // xyztscn
    const Coor<Nd> procs0 = {procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1};      // nxyztscn
    PartitionStored<Nd> p0 = basic_partitioning(dim0, procs0);
    const Coor<Nd> local_size0 = p0[rank][1];
    std::size_t vol0 = detail::volume(local_size0);
    vector<Q, XPU> t0 = ones<Q>(vol0, xpu);

    // Create tensor t1 of Nd+1 dims: an output lattice color vector
    const Coor<Nd + 1> dim1_rm = {max_power, dim[X], dim[Y], dim[Z],
                                  dim[T],    dim[S], dim[C], dim[N]};                    // pxyztscn
    const Coor<Nd + 1> procs1_rm = {1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}; // pxyztscn
    PartitionStored<Nd + 1> p1_rm = basic_partitioning(dim1_rm, procs1_rm);
    const Coor<Nd + 1> dim1_cm = {max_power, dim[N], dim[X], dim[Y],
                                  dim[Z],    dim[T], dim[S], dim[C]};                    // pnxyztsc
    const Coor<Nd + 1> procs1_cm = {1, 1, procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // pnxyztsc
    PartitionStored<Nd + 1> p1_cm = basic_partitioning(dim1_cm, procs1_cm);
    std::size_t vol1 = detail::volume(p1_rm[rank][1]);
    vector<Q, XPU> t1 = ones<Q>(vol1, xpu);

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
        std::cout << "Maximum number of elements in a tested tensor per process: " << vol1 << " ( "
                  << vol1 * 1.0 * sizeof(Q) / 1024 / 1024 << " MiB)" << std::endl;

    // Copy tensor t0 into each of the c components of tensor 1
    {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            for (int n = 0; n < dim[N]; ++n) {
                Q *ptr0 = t0.data(), *ptr1 = t1.data();
                bsr_krylov<Nd - 1, Nd - 1, Nd, Nd + 1, Q>(
                    op, "xyztsc", "XYZTSC", p0.data(), 1, "XYZTSCn", (const Q **)&ptr0,
                    is_cpu ? p1_rm.data() : p1_cm.data(), is_cpu ? "pxyztscn" : "pnxyztsc", 'p',
                    &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast, Copy);
                }
            }
            t = w_time() - t;
            if (rank == 0) std::cout << "Time in mavec " << t / nrep << std::endl;
        }

        destroy_bsr(op);

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
    int max_power = 1;

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
        } else if (std::strncmp("--power=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, "%d", &max_power) != 1) {
                std::cerr << "--power= should follow a number, for instance --power=3" << std::endl;
                return -1;
            }
            if (max_power < 1) {
                std::cerr << "The power should greater than zero" << std::endl;
                return -1;
            }
         } else if(std::strncmp("--help", argv[i], 6) == 0) {
             std::cout << "Commandline option:\n  " << argv[0]
                       << " [--dim='x y z t n b'] [--procs='x y z t n b'] [--power=p] [--help]"
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
        std::cout << "Max power " << max_power << std::endl;
    }

    {
        Context ctx = createCpuContext();
        test<std::complex<float>, Cpu>(dim, procs, rank, max_power, ctx, ctx.toCpu(0));
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext();
        test<std::complex<float>, Gpu>(dim, procs, rank, max_power, ctx, ctx.toGpu(0));
    }
#endif

    // Clear internal superbblas caches
    clearCaches();

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
