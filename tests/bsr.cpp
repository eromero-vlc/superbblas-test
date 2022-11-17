#include "superbblas.h"
#include <algorithm>
#include <iostream>
#include <vector>
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
    Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
    Coor<6, std::size_t> strided = get_strides<std::size_t>(pd[rank][1], SlowToFast);
    for (std::size_t i = 0, j = 0; i < voli; ++i) {
        std::size_t j0 = j;
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
        std::sort(&jj[j0], &jj[j], [&](const Coor<6> &a, const Coor<6> &b) {
            return coor2index(a, pd[rank][1], strided) < coor2index(b, pd[rank][1], strided);
        });
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
    create_bsr<6, 6, T>(pi.data(), op_dim, pd.data(), op_dim, 1, block, block, false, &iiptr,
                        &jjptr, (const T **)&dataptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast, &bsrh);
    return {bsrh, data_xpu};
}

/// Extend the region one element in each direction
std::array<Coor<6>, 2> extend(std::array<Coor<6>, 2> fs, const Coor<6> &dim) {
    for (int i = 0; i < 4; ++i) {
        fs[1][i] = std::min(dim[i], fs[1][i] + 2);
        if (fs[1][i] < dim[i])
            fs[0][i]--;
        else
            fs[0][i] = 0;
    }
    fs[0] = normalize_coor(fs[0], dim);
    return fs;
}

/// Create a 4D lattice with dimensions tzyxsc
template <typename T, typename XPU>
std::pair<std::vector<BSR_handle *>, std::vector<vector<T, XPU>>>
create_lattice_split(const PartitionStored<6> &pi, int rank, const Coor<6> op_dim, Context ctx,
                     XPU xpu) {
    // Compute the domain ranges
    PartitionStored<6> pd = pi;
    for (auto &i : pd) i = extend(i, op_dim);

    // Split the local part into the halo and the core
    PartitionStored<6> zero_part(pd.size());
    std::vector<PartitionStored<6>> pd_s(6, zero_part);
    for (unsigned int i = 0; i < pd.size(); ++i) {
        auto parts = make_hole(normalize_coor(pi[i][0] - pd[i][0], pd[i][1]), pi[i][1], pd[i][1]);
        for (unsigned int j = 0; j < parts.size(); ++j)
            pd_s[j][i] = volume(parts[j][1]) == 0
                             ? std::array<Coor<6>, 2>{Coor<6>{{}}, Coor<6>{{}}}
                             : std::array<Coor<6>, 2>{
                                   normalize_coor(parts[j][0] + pd[i][0], op_dim), parts[j][1]};
    }
    {
        std::vector<PartitionStored<6>> pd_s_aux;
        for (const auto &p : pd_s)
            if (p != zero_part) pd_s_aux.push_back(p);
        pd_s = pd_s_aux;
    }
    pd_s.push_back(pi);
    std::vector<PartitionStored<6>> pi_s(pd_s.size(), zero_part);
    for (unsigned int i = 0; i < pd.size(); ++i) {
        for (unsigned int p = 0; p < pd_s.size(); ++p) {
            auto fs = extend(pd_s[p][i], op_dim);
            intersection(pi[i][0], pi[i][1], fs[0], fs[1], op_dim, pi_s[p][i][0], pi_s[p][i][1]);
            if (volume(pi_s[p][i][1]) == 0 || volume(pd_s[p][i][1]))
                pi_s[p][i][0] = pi_s[p][i][1] = pd_s[p][i][0] = pd_s[p][i][1] = Coor<6>{{}};
        }
    }

    // Compute the coordinates for all nonzeros
    std::vector<BSR_handle *> bsrh_s;
    std::vector<vector<T, XPU>> data_s;
    std::size_t total_vol_data = 0;
    for (unsigned int p = 0; p < pd_s.size(); ++p) {
        // Compute how many neighbors
        int neighbors = 1;
        Coor<6> dimd = pd_s[p][rank][1];
        for (int dim = 0; dim < 4; ++dim) {
            int d = dimd[dim];
            if (d <= 0) {
                neighbors = 0;
                break;
            }
            if (d > 1) neighbors++;
            if (d > 2) neighbors++;
        }

        // Allocate and fill the row indices
        Coor<6> dimi = pi_s[p][rank][1]; // nonblock dimensions of the RSB image
        dimi[4] = dimi[5] = 1;
        std::size_t voli = volume(dimi);
        vector<IndexType, Cpu> ii(voli, Cpu{});
        for (auto &i : ii) i = neighbors;

        // Allocate and fill the column indices
        vector<Coor<6>, Cpu> jj(voli * neighbors, Cpu{});
        Coor<6> from = pi_s[p][rank][0]; // first nonblock dimensions of the RSB image
        Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
        Coor<6, std::size_t> strided = get_strides<std::size_t>(dimd, SlowToFast);
        for (std::size_t i = 0, j = 0; i < voli; ++i) {
            std::size_t j0 = j;
            Coor<6> c = index2coor(i, dimi, stride) + from;
            jj[j++] = c;
            for (int dim = 0; dim < 4; ++dim) {
                if (dimd[dim] == 1) continue;
                for (int dir = -1; dir < 2; dir += 2) {
                    Coor<6> c0 = c;
                    c0[dim] += dir;
                    jj[j++] = normalize_coor(c0 - pd_s[p][rank][0], op_dim);
                    if (dimd[dim] <= 2) break;
                }
            }
            std::sort(&jj[j0], &jj[j], [&](const Coor<6> &a, const Coor<6> &b) {
                return coor2index(a, dimd, strided) < coor2index(b, dimd, strided);
            });
        }

        std::size_t vol_data = voli * neighbors * op_dim[4] * op_dim[5] * op_dim[4] * op_dim[5];
        Coor<6> block{{1, 1, 1, 1, op_dim[4], op_dim[5]}};
        BSR_handle *bsrh = nullptr;
        vector<int, XPU> ii_xpu = makeSure(ii, xpu);
        vector<Coor<6>, XPU> jj_xpu = makeSure(jj, xpu);
        vector<T, XPU> data_xpu = ones<T>(vol_data, xpu);
        IndexType *iiptr = ii_xpu.data();
        Coor<6> *jjptr = jj_xpu.data();
        T *dataptr = data_xpu.data();
        create_bsr<6, 6, T>(pi_s[p].data(), op_dim, pd_s[p].data(), op_dim, 1, block, block, false,
                            &iiptr, &jjptr, (const T **)&dataptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                            MPI_COMM_WORLD,
#endif
                            SlowToFast, &bsrh);
        bsrh_s.push_back(bsrh);
        data_s.push_back(data_xpu);
        total_vol_data += vol_data;
    }

    // Number of nonzeros
    if (rank == 0)
        std::cout << "Size of the sparse tensor per process: "
                  << total_vol_data * 1.0 * sizeof(T) / 1024 / 1024 << " MiB" << std::endl;

    return {bsrh_s, data_s};
}

template <typename Q, typename XPU>
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, int max_power, unsigned int nrep, Context ctx,
          XPU xpu) {

    // Create a lattice operator of Nd-1 dims
    const Coor<Nd - 1> dimo = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
    const Coor<Nd - 1> procso = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsc
    PartitionStored<Nd - 1> po =
        basic_partitioning(dimo, procso, -1, false,
                           {{max_power - 1, max_power - 1, max_power - 1, max_power - 1, 0, 0}});
    auto op_pair = create_lattice<Q>(po, rank, dimo, ctx, xpu);
    BSR_handle *op = op_pair.first;

    // Create tensor t0 of Nd dims: an input lattice color vector
    const Coor<Nd + 1> dim0 = {1,      dim[X], dim[Y], dim[Z],
                               dim[T], dim[S], dim[C], dim[N]};                       // pxyztscn
    const Coor<Nd + 1> procs0 = {1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}; // pxyztscn
    PartitionStored<Nd + 1> p0 = basic_partitioning(dim0, procs0);
    const Coor<Nd + 1> local_size0 = p0[rank][1];
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
    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            Q *ptr0 = t0.data(), *ptr1 = t1.data();
            bsr_krylov<Nd - 1, Nd - 1, Nd + 1, Nd + 1, Q>(
                Q{1}, op, "xyztsc", "XYZTSC", p0.data(), 1, "pXYZTSCn", {{}}, dim0, dim0,
                (const Q **)&ptr0, Q{0}, is_cpu ? p1_rm.data() : p1_cm.data(),
                is_cpu ? "pxyztscn" : "pnxyztsc", {{}}, is_cpu ? dim1_rm : dim1_cm,
                is_cpu ? dim1_rm : dim1_cm, 'p', &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                MPI_COMM_WORLD,
#endif
                SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in mavec per rhs: " << t / nrep / dim[N] << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    destroy_bsr(op);

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    // Create split tensor
    auto op_pair_s = create_lattice_split<Q>(po, rank, dimo, ctx, xpu);

    // Copy tensor t0 into each of the c components of tensor 1
    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            Q *ptr0 = t0.data(), *ptr1 = t1.data();

            // Set the output tensor to zero
            copy(0, is_cpu ? p1_rm.data() : p1_cm.data(), 1, is_cpu ? "pxyztscn" : "pnxyztsc", {{}},
                 is_cpu ? dim1_rm : dim1_cm, is_cpu ? dim1_rm : dim1_cm, (const Q **)&ptr1, nullptr,
                 &ctx, is_cpu ? p1_rm.data() : p1_cm.data(), 1, is_cpu ? "pxyztscn" : "pnxyztsc",
                 {{}}, is_cpu ? dim1_rm : dim1_cm, &ptr1, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);

            // Do the contractions on each part
            std::vector<Request> r(op_pair_s.first.size());
            for (unsigned int p = 0; p < op_pair_s.first.size(); ++p) {
                bsr_krylov<Nd - 1, Nd - 1, Nd + 1, Nd + 1, Q>(
                    Q{1}, op_pair_s.first[p], "xyztsc", "XYZTSC", p0.data(), 1, "pXYZTSCn", {{}},
                    dim0, dim0, (const Q **)&ptr0, Q{1}, is_cpu ? p1_rm.data() : p1_cm.data(),
                    is_cpu ? "pxyztscn" : "pnxyztsc", {{}}, is_cpu ? dim1_rm : dim1_cm,
                    is_cpu ? dim1_rm : dim1_cm, 'p', &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast, &r[p]);
            }
            for (const auto &ri : r) wait(ri);
        }

        sync(xpu);
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in mavec per rhs (split): " << t / nrep / dim[N] << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    for (const auto op : op_pair_s.first) destroy_bsr(op);

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
    int nrep = getDebugLevel() == 0 ? 10 : 1;

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
        } else if (std::strncmp("--rep=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d", &nrep) != 1) {
                std::cerr << "--rep= should follow a number, for instance --rep=3" << std::endl;
                return -1;
            }
            if (nrep < 1) {
                std::cerr << "The rep should greater than zero" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--dim='x y z t n b'] [--procs='x y z t n b'] [--power=p] [--help]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // If --procs isn't set, distributed the processes automatically
    if (!procs_was_set) procs = partitioning_distributed_procs("xyztscn", dim, "xyzt", nprocs);

    // Show lattice dimensions and processes arrangement
    if (rank == 0) {
        std::cout << "Testing lattice dimensions xyzt= " << dim[X] << " " << dim[Y] << " " << dim[Z]
                  << " " << dim[T] << " spin-color= " << dim[S] << " " << dim[C]
                  << "  num_vecs= " << dim[N] << std::endl;
        std::cout << "Processes arrangement xyzt= " << procs[X] << " " << procs[Y] << " "
                  << procs[Z] << " " << procs[T] << std::endl;
        std::cout << "Max power " << max_power << std::endl;
        std::cout << "Repetitions " << nrep << std::endl;
    }

    {
        Context ctx = createCpuContext();
        test<std::complex<double>, Cpu>(dim, procs, rank, max_power, nrep, ctx, ctx.toCpu(0));
        clearCaches();
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext(rank % getGpuDevicesCount());
        test<float, Gpu>(dim, procs, rank, max_power, nrep, ctx, ctx.toGpu(0));
        test<std::complex<double>, Gpu>(dim, procs, rank, max_power, nrep, ctx, ctx.toGpu(0));
        clearCaches();
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
