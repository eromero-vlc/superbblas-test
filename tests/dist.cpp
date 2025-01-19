#include "superbblas.h"
#include <iostream>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <std::size_t Nd> PartitionStored<Nd> dist_tensor_on_root(Coor<Nd> dim, int nprocs) {
    PartitionStored<Nd> fs(nprocs);
    if (1 <= nprocs) fs[0][1] = dim;
    return fs;
}

constexpr std::size_t Nd = 7;          // xyztscn
constexpr unsigned int nS = 4, nC = 3; // length of dimension spin and color dimensions
constexpr unsigned int X = 0, Y = 1, Z = 2, T = 3, S = 4, C = 5, N = 6;
static const char *labels = "xyztscn";
template <std::size_t N> using Labels = std::array<unsigned int, N>;

template <std::size_t N> Coor<N> get_dim(const Labels<N> &l, const Coor<Nd> &dim) {
    Coor<N> r;
    for (std::size_t i = 0; i < N; ++i) r[i] = dim[l[i]];
    return r;
}

template <std::size_t N> Order<N + 1> get_order(const Labels<N> &l) {
    Order<N + 1> r{{}};
    for (std::size_t i = 0; i < N; ++i) r[i] = labels[l[i]];
    return r;
}

template <std::size_t N> Order<N + 1> get_order_from_str(const char *s) {
    Order<N + 1> r{{}};
    std::copy_n(s, N, r.begin());
    return r;
}

template <std::size_t N>
std::array<Coor<N>, 2> extend(std::array<Coor<N>, 2> fs, const Coor<N> &power, const Coor<N> &dim) {
    for (unsigned int i = 0; i < N; ++i) {
        fs[1][i] = std::min(dim[i], fs[1][i] + power[i] * 2);
        if (fs[1][i] < dim[i])
            fs[0][i] -= power[i];
        else
            fs[0][i] = 0;
    }
    fs[0] = normalize_coor(fs[0], dim);
    return fs;
}

/// Extend the support for all regions, one element in each direction
template <std::size_t N>
PartitionStored<N> extend(const PartitionStored<N> &p, const Coor<N> &power, const Coor<N> &dim) {
    PartitionStored<N> r = p;
    for (auto &i : r) i = extend(i, power, dim);
    return r;
}

template <std::size_t N, typename T, typename XPU> struct tensor {
    Coor<N> dim;                   ///< global dimensions
    PartitionStored<N> p;          ///< partition
    std::vector<vector<T, XPU>> v; ///< data
    std::vector<T *> ptr;          ///< data pointers
    int rank;                      ///< rank of the current process
    Order<N + 1> order;            ///< labels

    /// Constructor with a partition
    tensor(const Coor<N> &dim, const PartitionStored<N> &p, int rank, const std::vector<XPU> &xpus,
           const Order<N + 1> &order = {})
        : dim(dim), p(p), rank(rank), order(order) {
        if (p.size() % xpus.size() != 0) throw std::runtime_error("Invalid number of components");
        if (std::size_t(rank) >= p.size() / xpus.size())
            throw std::runtime_error("Invalid rank value");
        for (unsigned int i = 0; i < xpus.size(); ++i) {
            v.push_back(vector<T, XPU>(volume(p.at(xpus.size() * rank + i).at(1)), xpus.at(i)));
        }
    }

    /// Constructor for a distributed tensor
    tensor(const Coor<N> &dim, const Coor<N> &procs, int nprocs, int rank,
           const std::vector<XPU> &xpus, const Order<N + 1> &order = {})
        : tensor(dim, basic_partitioning(order.data(), dim, procs, "xyzt", nprocs, xpus.size()),
                 rank, xpus, order) {}

    /// Constructor for a distributed tensor with power
    tensor(const Coor<N> &dim, const Coor<N> &procs, const Coor<N> &power, int nprocs, int rank,
           const std::vector<XPU> &xpus, const Order<N + 1> &order = {})
        : tensor(dim,
                 extend(basic_partitioning(order.data(), dim, procs, "xyzt", nprocs, xpus.size()),
                        power, dim),
                 rank, xpus, order) {}

    tensor(const Coor<N> &dim, int nprocs, int rank, const std::vector<XPU> &xpus,
           const Order<N + 1> &order = {})
        : tensor(dim, dist_tensor_on_root(dim, nprocs * xpus.size()), rank, xpus, order) {}

    /// Constructor giving labels and a new order
    tensor(const Labels<N> &l, const Coor<Nd> &d, int nprocs, int rank,
           const std::vector<XPU> &xpus, const char *order_str)
        : tensor(get_dim(l, d), nprocs, rank, xpus, order(get_order_from_str<N>(order_str))) {}

    /// Constructor giving labels
    tensor(const Labels<N> &l, const Coor<Nd> &d, int nprocs, int rank,
           const std::vector<XPU> &xpus)
        : tensor(get_dim(l, d), nprocs, rank, xpus, get_order(l)) {}

    /// Constructor giving labels and procs
    tensor(const Labels<N> &l, const Coor<Nd> &d, const Coor<Nd> &procs, int nprocs, int rank,
           const std::vector<XPU> &xpus)
        : tensor(get_dim(l, d), get_dim(l, procs), nprocs, rank, xpus, get_order(l)) {}

    /// Constructor giving labels and procs and power
    tensor(const Labels<N> &l, const Coor<Nd> &d, const Coor<Nd> &procs, const Coor<Nd> &power,
           int nprocs, int rank, const std::vector<XPU> &xpus)
        : tensor(get_dim(l, d), get_dim(l, procs), get_dim(l, power), nprocs, rank, xpus,
                 get_order(l)) {}

    /// Destructor for a tensor with support only on the root process
    void release() {
        v.clear();
        ptr.clear();
    }

    /// Return an array of pointers
    T **data() {
        if (ptr.size() != v.size()) {
            for (const auto &it : v) ptr.push_back(it.data());
        }
        return ptr.data();
    }
};

// Dummy initialization of a tensor
template <typename T, typename XPU> void dummyFill(vector<T, XPU> &t) {
    vector<T, Cpu> v(t.size(), Cpu{});
    for (unsigned int i = 0, vol = v.size(); i < vol; i++) v[i] = i;
    copy_n(v.data(), v.ctx(), v.size(), t.data(), t.ctx());
}

template <std::size_t N, typename T, typename XPU> void dummyFill(tensor<N, T, XPU> &t) {
    for (auto &v : t.v) dummyFill(v);
}

void test_distribution() {
    {
        Coor<5> dim{4, 4, 4, 4, 3};
        Coor<5> procs = partitioning_distributed_procs("xyztc", dim, "xyzt", 6);
        if (procs != Coor<5>{3, 2, 1, 1, 1})
            throw std::runtime_error("Unexpected result for partitioning_distributed_procs");
        basic_partitioning("xyztc", dim, procs, "xyzt", 6);
    }
    {
        Coor<5> dim{4, 4, 4, 4, 3};
        Coor<5> procs = partitioning_distributed_procs("xyztc", dim, "xyzt", 7);
        if (procs != Coor<5>{3, 2, 1, 1, 1})
            throw std::runtime_error("Unexpected result for partitioning_distributed_procs");
        basic_partitioning("xyztc", dim, procs, "xyzt", 7);
    }
    {
        Coor<5> dim{4, 4, 4, 1, 3};
        Coor<5> procs = partitioning_distributed_procs("xyztc", dim, "tzyx", 32);
        if (procs != Coor<5>{2, 4, 4, 1, 1})
            throw std::runtime_error("Unexpected result for partitioning_distributed_procs");
        basic_partitioning("xyztc", dim, procs, "tzyx", 32);
    }
}

template <std::size_t N>
void test_make_hole(const Coor<N> &from, const Coor<N> &size, const Coor<N> &hole_from,
                    const Coor<N> &hole_size, const Coor<N> &dim) {

    auto r = make_hole(from, size, hole_from, hole_size, dim);

    for (const auto &it : r) {
        // Make sure that the resulting range has fully support on (from, size)
        if (volume(intersection<false>(it[0], it[1], from, size, dim)) != volume(it[1]))
            throw std::runtime_error("Unexpected result in `subtract_range`");

        // Make sure that the resulting range has no support on (hole_from, hole_size)
        if (volume(intersection<false>(it[0], it[1], hole_from, hole_size, dim)) != 0)
            throw std::runtime_error("Unexpected result in `subtract_range`");
    }

    // Check that the resulting ranges have no overlap
    for (std::size_t i = 0; i < r.size() - 1; ++i) {
        From_size<N> ri(r.size() - i - 1);
        std::copy(r.begin() + i + 1, r.end(), ri.begin());
        if (volume(intersection<false>(ri, r[i][0], r[i][1], dim)) != 0)
            throw std::runtime_error("Unexpected result in `subtract_range`");
    }

    // Check the resulting ranges together with the hole covers the region (from, size)
    if (volume(r) + volume(intersection<false>(from, size, hole_from, hole_size, dim)) != volume(size))
        throw std::runtime_error("Unexpected result in `subtract_range`");
}

template <typename T, typename XPU>
void test_gemm(std::size_t m, std::size_t n, std::size_t k, std::size_t batch_size, const T *a,
               const T *b, T *c, char transa, char transb, int rank, XPU xpu,
               std::size_t nreps = 10) {
    bool ta = (transa != 'n');
    bool tb = (transb != 'n');
    double t = 0;
    for (std::size_t rep = 0; rep <= nreps; ++rep) {
        if (rep == 1) {
            sync(xpu);
            t = w_time();
        }
        superbblas::detail::xgemm_batch_strided(transa, transb, m, n, k, T{1}, a, !ta ? m : k,
                                                m * k, b, !tb ? k : n, n * k, T{0}, c, m, m * n,
                                                batch_size, xpu);
    }
    sync(xpu);
    t = w_time() - t;
    if (rank == 0)
        std::cout << "Time in contracting " << transa << transb << " "
                  << m * n * k * batch_size * nreps / t / 1e9 << " GFLOPS" << std::endl;
}

template <typename T, typename XPU>
void test_gemm(std::size_t m, std::size_t n, std::size_t k, std::size_t batch_size, int rank,
               XPU xpu, std::size_t nreps = 10) {
    vector<T, XPU> a(m * k * batch_size, xpu);
    vector<T, XPU> b(n * k * batch_size, xpu);
    vector<T, XPU> c(m * n * batch_size, xpu);
    dummyFill(a);
    dummyFill(b);

    test_gemm(m, n, k, batch_size, a.data(), b.data(), c.data(), 'n', 'n', rank, xpu, nreps);
    test_gemm(m, n, k, batch_size, a.data(), b.data(), c.data(), 't', 'n', rank, xpu, nreps);
    test_gemm(m, n, k, batch_size, a.data(), b.data(), c.data(), 'n', 't', rank, xpu, nreps);
    test_gemm(m, n, k, batch_size, a.data(), b.data(), c.data(), 't', 't', rank, xpu, nreps);
    test_gemm(m, n, k, batch_size, a.data(), b.data(), c.data(), 'c', 'n', rank, xpu, nreps);
    test_gemm(m, n, k, batch_size, a.data(), b.data(), c.data(), 'c', 't', rank, xpu, nreps);
}

template <typename XPU> void sync(const std::vector<XPU> &xpus) {
    int numDevices = getGpuDevicesCount() + 1;
    std::vector<bool> already_sync(numDevices, false);
    for (const auto xpu : xpus) {
        if (already_sync.at(deviceId(xpu) + 1)) continue;
        sync(xpu);
        already_sync.at(deviceId(xpu) + 1) = true;
    }
}

template <typename XPU>
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, int nprocs, const std::vector<Context> &ctx,
          const std::vector<XPU> &xpu, unsigned int nrep) {

    using Scalar = std::complex<float>;
    using ScalarD = std::complex<double>;

    resetTimings();

    // Copy tensor t0 into tensor 1 (for reference)
    double tref = 0.0;
    {
        tensor<Nd - 1, Scalar, XPU> t0({X, Y, Z, T, S, C}, dim, procs, nprocs, rank,
                                       std::vector<XPU>(1, xpu.at(0)));
        dummyFill(t0);

        const bool is_cpu = deviceId(xpu.at(0)) == CPU_DEVICE_ID;
        if (rank == 0) std::cout << ">>> " << (is_cpu ? "CPU" : "GPU") << " tests:" << std::endl;

        std::size_t local_vol0 = volume(t0.p[rank][1]);
        if (rank == 0)
            std::cout << "Maximum number of elements in a tested tensor per process: " << local_vol0
                      << " ( " << local_vol0 * 1.0 * sizeof(Scalar) / 1024 / 1024 << " MiB)"
                      << std::endl;

        sync(xpu.at(0));
        vector<Scalar, XPU> aux(local_vol0 * dim[N], xpu.at(0));
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            for (int n = 0; n < dim[N]; ++n) {
                copy_n(t0.v.at(0).data(), t0.v.at(0).ctx(), local_vol0, aux.data() + local_vol0 * n,
                       aux.ctx());
            }
        }
        sync(xpu.at(0));
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in dummy copying from xyzts to tnsxyzc " << t / nrep << " ( "
                      << local_vol0 * dim[N] * sizeof(Scalar) * nrep / t / 1e9 << " GBYTES/s)"
                      << std::endl;
        tref = t / nrep; // time in copying a whole tensor with size dim1
    }

    // Copy tensor t0 into each of the c components of tensor 1
    {
        tensor<Nd - 1, Scalar, XPU> t0({X, Y, Z, T, S, C}, dim, procs, nprocs, rank, xpu);
        tensor<Nd, Scalar, XPU> t1(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        dummyFill(t0);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            for (int n = 0; n < dim[N]; ++n) {
                const Coor<Nd - 1> from0 = {{}};
                const Coor<Nd> from1 = {0, n};
                copy(1.0, t0.p.data(), ctx.size(), t0.order.data(), from0, t0.dim, t0.dim,
                     (const Scalar **)t0.data(), nullptr, ctx.data(), t1.p.data(), ctx.size(),
                     t1.order.data(), from1, t1.dim, t1.data(), nullptr, ctx.data(),
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
        if (rank == 0) reportTimings(std::cout);
        if (rank == 0) reportCacheUsage(std::cout);
    }

	return; /// temp!

    // Copy tensor t0 into each of the c components of tensor 1 in double
    {
        tensor<Nd - 1, Scalar, XPU> t0({X, Y, Z, T, S, C}, dim, procs, nprocs, rank, xpu);
        tensor<Nd, ScalarD, XPU> t1(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        dummyFill(t0);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            for (int n = 0; n < dim[N]; ++n) {
                const Coor<Nd - 1> from0 = {{}};
                const Coor<Nd> from1 = {0, n};
                copy(1.0, t0.p.data(), ctx.size(), t0.order.data(), from0, t0.dim, t0.dim,
                     (const Scalar **)t0.data(), nullptr, ctx.data(), t1.p.data(), ctx.size(),
                     t1.order.data(), from1, t1.dim, t1.data(), nullptr, ctx.data(),
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
    {
        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        tensor<Nd, Scalar, XPU> t1(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        tensor<Nd, Scalar, XPU> t2(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        dummyFill(t1);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            const Coor<Nd> from0 = {{}};
            Coor<Nd> from1 = {{}};
            from1[4] = 1; // Displace one on the z-direction
            copy(1.0, t1.p.data(), ctx.size(), t1.order.data(), from0, t1.dim, t1.dim,
                 (const Scalar **)t1.data(), nullptr, ctx.data(), t2.p.data(), ctx.size(),
                 t2.order.data(), from1, t2.dim, t2.data(), nullptr, ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in shifting " << t / nrep << std::endl;
    }

    {
        tensor<Nd, Scalar, XPU> t1(Labels<Nd>{T, S, X, Y, Z, C, N}, dim, procs, nprocs, rank,
                                   std::vector<XPU>(1, xpu.at(0)));

        auto local_size = t1.p[rank][1];
        int k = local_size[2] * local_size[3] * local_size[4] * dim[S] * dim[C]; // xyzsc
        int tt = local_size[0];                                                  // t
        std::vector<int> n_sizes;
        for (int n : std::vector<int>{1, 2, 3, 4, 12})
            if (n <= dim[N]) n_sizes.push_back(n);
        for (int n = 8; n <= dim[N]; n *= 2) n_sizes.push_back(n);

        for (int n : n_sizes) {
            if (rank == 0)
                std::cout << "*) [inner product] results for m,n,k,batch_size: " << n << "," << n
                          << "," << k << "," << tt << std::endl;
            test_gemm<Scalar>(n, n, k, tt, rank, xpu.at(0), nrep);
        }

        for (int n : n_sizes) {
            if (rank == 0)
                std::cout << "*) [update] results for m,n,k,batch_size: " << k << "," << n << ","
                          << n << "," << tt << std::endl;
            test_gemm<Scalar>(k, n, n, tt, rank, xpu.at(0), nrep);
        }
    }

    // Create tensor t3 of 5 dims
    {
        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        tensor<Nd, Scalar, XPU> t1(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        tensor<Nd, Scalar, XPU> t2(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        tensor<5, Scalar, XPU> tc({T, N, S, N, S}, dim, nprocs, rank, xpu);
        dummyFill(t1);
        dummyFill(t2);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                resetTimings();
                sync(xpu);
                t = w_time();
            }
            contraction(Scalar{1.0}, t1.p.data(), {{}}, t1.dim, t1.dim, ctx.size(), t1.order.data(),
                        false, (const Scalar **)t1.data(), ctx.data(), t2.p.data(), {{}}, t2.dim,
                        t2.dim, ctx.size(), "tNSxyzc", false, (const Scalar **)t2.data(),
                        ctx.data(), Scalar{0.0}, tc.p.data(), {{}}, tc.dim, tc.dim, ctx.size(),
                        "tNSns", tc.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in contracting xyz in column major " << t / nrep << std::endl;
        if (rank == 0) reportTimings(std::cout);
    }
    {
        tensor<Nd, Scalar, XPU> t1(Labels<Nd>{T, S, X, Y, Z, C, N}, dim, procs, nprocs, rank, xpu);
        tensor<Nd, Scalar, XPU> t2(Labels<Nd>{T, S, X, Y, Z, C, N}, dim, procs, nprocs, rank, xpu);
        tensor<5, Scalar, XPU> tc({T, N, S, N, S}, dim, nprocs, rank, xpu);
        dummyFill(t1);
        dummyFill(t2);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                resetTimings();
                sync(xpu);
                t = w_time();
            }
            contraction(Scalar{1.0}, t1.p.data(), {{}}, t1.dim, t1.dim, ctx.size(), t1.order.data(),
                        false, (const Scalar **)t1.data(), ctx.data(), t2.p.data(), {{}}, t2.dim,
                        t2.dim, ctx.size(), "tSxyzcN", false, (const Scalar **)t2.data(),
                        ctx.data(), Scalar{0.0}, tc.p.data(), {{}}, tc.dim, tc.dim, ctx.size(),
                        "tNSns", tc.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in contracting xyz in row major " << t / nrep << std::endl;
        if (rank == 0) reportTimings(std::cout);
    }

    // Contraction gpu and cpu and contraction of tensors with different number of components
    if (!std::is_same<Cpu, XPU>::value || ctx.size() > 1) {
        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        Context ctx_cpu = createCpuContext();
        Cpu cpu = ctx_cpu.toCpu(0);
        tensor<Nd, Scalar, XPU> t1(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        tensor<Nd, Scalar, Cpu> t2(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank,
                                   std::vector<Cpu>(1, cpu));
        tensor<5, Scalar, XPU> tc({T, N, S, N, S}, dim, nprocs, rank, xpu);
        dummyFill(t1);
        dummyFill(t2);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                resetTimings();
                sync(xpu);
                t = w_time();
            }
            contraction(Scalar{1.0}, t1.p.data(), {{}}, t1.dim, t1.dim, ctx.size(), t1.order.data(),
                        false, (const Scalar **)t1.data(), ctx.data(), t2.p.data(), {{}}, t2.dim,
                        t2.dim, 1, "tNSxyzc", false, (const Scalar **)t2.data(), &ctx_cpu,
                        Scalar{0.0}, tc.p.data(), {{}}, tc.dim, tc.dim, ctx.size(), "tNSns",
                        tc.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in contracting xyz in column major (gpu x cpu -> gpu) " << t / nrep
                      << std::endl;
        if (rank == 0) reportTimings(std::cout);
    }

    // Copy halos
    {
        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        tensor<Nd, Scalar, XPU> t1(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, nprocs, rank, xpu);
        dummyFill(t1);

        const int power = 1;
        const Coor<Nd> ext = {power, power, power, power, 0, 0, 0}; // xyztscn
        tensor<Nd, Scalar, XPU> th(Labels<Nd>{T, N, S, X, Y, Z, C}, dim, procs, ext, nprocs, rank,
                                   xpu);

        double t = 0;
        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            const Coor<Nd> from0 = {{}};
            Coor<Nd> from1 = {{}};
            copy(1.0, t1.p.data(), ctx.size(), t1.order.data(), from0, t1.dim, t1.dim,
                 (const Scalar **)t1.data(), nullptr, ctx.data(), th.p.data(), ctx.size(),
                 th.order.data(), from1, th.dim, th.data(), nullptr, ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in copying halos in " << t / nrep << std::endl;

        for (unsigned int rep = 0; rep <= nrep; ++rep) {
            if (rep == 1) {
                sync(xpu);
                t = w_time();
            }
            const Coor<Nd> from0 = {{}};
            Coor<Nd> from1 = {{}};
            copy(1.0, th.p.data(), ctx.size(), th.order.data(), from1, th.dim, th.dim,
                 (const Scalar **)th.data(), nullptr, ctx.data(), t1.p.data(), ctx.size(),
                 t1.order.data(), from0, t1.dim, t1.data(), nullptr, ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in copying halos out " << t / nrep << std::endl;
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

    test_distribution();

    test_make_hole<1>({1}, {4}, {2}, {3}, {8});
    test_make_hole<1>({1}, {3}, {2}, {4}, {8});
    test_make_hole<2>(Coor<2>{1, 1}, Coor<2>{4, 4}, Coor<2>{2, 2}, Coor<2>{1, 1}, Coor<2>{5, 5});
    test_make_hole<2>(Coor<2>{1, 1}, Coor<2>{4, 4}, Coor<2>{4, 4}, Coor<2>{3, 3}, Coor<2>{5, 5});

    Coor<Nd> dim = {16, 16, 16, 32, nS, nC, 64}; // xyztscn
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1};
    unsigned int nrep = getDebugLevel() == 0 ? 10 : 1;
    int ncomponents = 0;

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
        } else if (std::strncmp("--components=", argv[i], 13) == 0) {
            if (sscanf(argv[i] + 13, "%d", &ncomponents) != 1) {
                std::cerr << "--components= should follow a number, for instance --components=2"
                          << std::endl;
                return -1;
            }
            if (ncomponents < 0) {
                std::cerr << "The number of components shouldn't be negative" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--reps=", argv[i], 7) == 0) {
            if (sscanf(argv[i] + 7, "%d", &nrep) != 1) {
                std::cerr << "--reps= should follow one number" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout
                << "Commandline option:\n  " << argv[0]
                << " [--dim='x y z t n'] [--procs='x y z t n'] [--components=c] [--reps=r] [--help]"
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
    }

    // Set the default number of components
    ncomponents = ncomponents == 0 ? 1 : ncomponents;

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    if (rank == 0)
        std::cout << "Tests with " << num_threads << " threads and " << ncomponents << " components"
                  << std::endl;

    {
        std::vector<Context> ctx;
        for (int i = 0; i < ncomponents; ++i) ctx.push_back(createCpuContext());
        std::vector<Cpu> xpus;
        for (const auto &i : ctx) xpus.push_back(i.toCpu(0));
        test(dim, procs, rank, nprocs, ctx, xpus, nrep);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext(rank % getGpuDevicesCount());
        test(dim, procs, rank, nprocs, ctx, ctx.toGpu(0), nrep);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
