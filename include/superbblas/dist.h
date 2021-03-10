#ifndef __SUPERBBLAS_DIST__
#define __SUPERBBLAS_DIST__

#include "tensor.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef SUPERBBLAS_USE_MPI
#    include "mpi.h"
#endif // SUPERBBLAS_USE_MPI

namespace superbblas {

    /// First coordinate and size of a range of coordinates supported on a process/component.
    /// See ConstPartition.

    template <std::size_t N> using PartitionItem = std::array<Coor<N>, 2>;

    /// Distribution of the elements of a N-dimension tensor among the processes/components
    ///
    /// This structure is a three-dimensional array with dimensions [P][2][N], where P is the number of
    /// processes/components and the tensors have N dimensions. The values of the array indicates that
    /// the i-th process/component stores all elements with coordinates c such that for all j dimensions:
    ///    (*this)[i][0][j] <= c[j] <= (*this)[i][0][j] + (*this)[i][1][j],      (mod dim[j]),
    /// where dim are tensor dimensions. In other words, each process/components stores a continuum range
    /// of coordinates such that the first coordinate at the j-dimension is [i][0][j] and
    /// stores up to [i][1][j] elements in that j dimension.

    template <std::size_t N> using ConstPartition = const PartitionItem<N> *;

    namespace detail {

        /// Type use in MPI calls to indicate cardinality and displacements
        using MpiInt = int;

        /// Set a user-defined MPI type that allows to send up 256 GiB in a single calls
        /// MPI type with this size; the maximum package size is
        constexpr std::size_t MpiTypeSize = 64;

        /// First coordinate and size of a range
        template <std::size_t N> using From_size_item = PartitionItem<N>;
        /// List of ranges
        template <std::size_t N> using From_size = vector<const From_size_item<N>, Cpu>;
        template <std::size_t N> using From_size_out = vector<From_size_item<N>, Cpu>;
        template <std::size_t N> using From_size_vector = std::vector<From_size_item<N>>;
        /// From_size iterator
        template <std::size_t N> using From_size_iterator = const From_size_item<N> *;

        template <std::size_t Nd0, std::size_t Nd1>
        using PairPerms = std::tuple<Coor<Nd0>, Coor<Nd1>>;

        //
        // Auxiliary functions
        //

        /// Return the global dimensions of a tensor from its partitioning
        /// \param p: partitioning of the tensor in consecutive ranges

        template <std::size_t N> Coor<N> get_dim(From_size_iterator<N> p, std::size_t n) {
            Coor<N> r = {};
            for (std::size_t j = 0; j < n; j++) r = max_each(r, p[j][0] + p[j][1]);
            return r;
        }

        /// Return the global dimensions of a tensor from its partitioning
        /// \param p: partitioning of the tensor in consecutive ranges

        template <std::size_t N> Coor<N> get_dim(const From_size<N> &p) {
            return get_dim<N>(p.begin(), p.size());
        }

        /// Return the total volume in a partition
        /// \param p: partitioning of the tensor in consecutive ranges

        template <std::size_t N> std::size_t get_volume(From_size_iterator<N> p, std::size_t n) {
            std::size_t r = 0;
            for (std::size_t j = 0; j < n; j++) r += volume<N>(p[j][1]);
            return r;
        }

        /// Return the total volume in a partition
        /// \param p: partitioning of the tensor in consecutive ranges

        template <std::size_t N> std::size_t get_volume(const From_size<N> &p) {
            return get_volume<N>(p.begin(), p.size());
        }

        /// Return the permutations associated to two order

        template <std::size_t Nd0, std::size_t Nd1>
        PairPerms<Nd0, Nd1> get_perms(const Order<Nd0> &o0, const Order<Nd1> &o1) {
            return PairPerms<Nd0, Nd1>{find_permutation<Nd1, Nd0>(o1, o0),
                                       find_permutation<Nd0, Nd1>(o0, o1)};
        }

        /// Output of `send` and input of `wait`
        using Request = std::function<void(void)>;

        /// Wait until an operation started by `send` finishes
        /// \param request:

        inline void wait(Request request) { request(); }

#ifdef SUPERBBLAS_USE_MPI
        /// Communicator
        struct MpiComm {
            unsigned int nprocs; ///< Number of processes
            unsigned int rank;   ///< Process id
            MPI_Comm comm;       ///< MPI communicator
        };

        /// Throw exception if MPI reports an error
        /// \param error: MPI returned error

        inline void MPI_check(int error) {
            if (error == MPI_SUCCESS) return;

#    define CHECK_AND_THROW(ERR)                                                                   \
        if (error == ERR) throw std::runtime_error("MPI error: " #ERR);

            CHECK_AND_THROW(MPI_ERR_BUFFER);
            CHECK_AND_THROW(MPI_ERR_COUNT);
            CHECK_AND_THROW(MPI_ERR_TYPE);
            CHECK_AND_THROW(MPI_ERR_TAG);
            CHECK_AND_THROW(MPI_ERR_COMM);
            CHECK_AND_THROW(MPI_ERR_RANK);
            CHECK_AND_THROW(MPI_ERR_ROOT);
            CHECK_AND_THROW(MPI_ERR_GROUP);
            CHECK_AND_THROW(MPI_ERR_OP);
            CHECK_AND_THROW(MPI_ERR_TOPOLOGY);
            CHECK_AND_THROW(MPI_ERR_DIMS);
            CHECK_AND_THROW(MPI_ERR_ARG);
            CHECK_AND_THROW(MPI_ERR_UNKNOWN);
            CHECK_AND_THROW(MPI_ERR_TRUNCATE);
            CHECK_AND_THROW(MPI_ERR_OTHER);
            CHECK_AND_THROW(MPI_ERR_INTERN);
            CHECK_AND_THROW(MPI_ERR_IN_STATUS);
            CHECK_AND_THROW(MPI_ERR_PENDING);
            CHECK_AND_THROW(MPI_ERR_REQUEST);
            CHECK_AND_THROW(MPI_ERR_LASTCODE);
#    undef CHECK_AND_THROW
        }

        /// Return a communicator for a MPI_Comm
        inline MpiComm get_comm(MPI_Comm comm) {
            int nprocs, rank;
            MPI_check(MPI_Comm_size(comm, &nprocs));
            MPI_check(MPI_Comm_rank(comm, &rank));
            return MpiComm{(unsigned int)nprocs, (unsigned int)rank, comm};
        }

#endif // SUPERBBLAS_USE_MPI

        /// Communicator
        struct SelfComm {
            unsigned int nprocs; ///< Number of processes
            unsigned int rank;   ///< Process id
        };

        /// Return a communicator for a MPI_Comm
        inline SelfComm get_comm() { return SelfComm{1u, 0u}; }

#ifdef SUPERBBLAS_USE_MPI
        /// Return the MPI_datatype for a type returned by `NativeMpiDatatype`
        inline MPI_Datatype get_mpi_datatype() {
            if (MpiTypeSize == sizeof(char)) return MPI_CHAR;
            if (MpiTypeSize == sizeof(float)) return MPI_FLOAT;
            if (MpiTypeSize == sizeof(double)) return MPI_DOUBLE;
            MPI_Datatype t;
            MPI_check(MPI_Type_contiguous(MpiTypeSize, MPI_CHAR, &t));
            MPI_check(MPI_Type_commit(&t));
            return t;
        }
#endif // SUPERBBLAS_USE_MPI

        /// Component of a tensor
        template <std::size_t Nd, typename T, typename XPU> struct Component {
            vector<T, XPU> it;        ///< data
            Coor<Nd> dim;             ///< dimension of the tensor
            unsigned int componentId; ///< Component Id

            template <typename Q = T, typename = typename std::enable_if<std::is_same<
                                          Q, typename std::remove_const<Q>::type>::value>::type>
            operator Component<Nd, const Q, XPU>() const {
                return {it, dim, componentId};
            }
        };

        /// A tensor composed of several components
        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        using Components_tmpl =
            std::pair<std::vector<Component<Nd, T, XPU0>>, std::vector<Component<Nd, T, XPU1>>>;

#ifdef SUPERBBLAS_USE_CUDA
        /// A tensor composed of several CPU and CUDA elements
        template <std::size_t Nd, typename T> using Components = Components_tmpl<Nd, T, Cuda, Cpu>;
#else
        /// A tensor composed of only of CPU components
        template <std::size_t Nd, typename T> using Components = Components_tmpl<Nd, T, Cpu, Cpu>;
#endif // SUPERBBLAS_USE_CUDA

        template <std::size_t Nd, typename T, typename Comm>
        Components<Nd, T> get_components(T **v, const Context *ctx, unsigned int ncomponents,
                                         From_size_iterator<Nd> p, Comm comm) {
            // Get components on the local process
            From_size_iterator<Nd> fs = p + comm.rank * ncomponents;

            Components<Nd, T> r;
            for (unsigned int i = 0; i < ncomponents; ++i) {
                switch (ctx[i].plat) {
#ifdef SUPERBBLAS_USE_CUDA
                case CPU:
                    r.second.push_back(
                        Component<Nd, T, Cpu>{to_vector(v[i], ctx[i].toCpu()), fs[i][1], i});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
                case CUDA:
                    r.first.push_back(
                        Component<Nd, T, Cuda>{to_vector(v[i], ctx[i].toCuda()), fs[i][1], i});
                    assert(!v[i] || getPtrDevice(v[i]) == ctx[i].device);
                    break;
#else // SUPERBBLAS_USE_CUDA
                case CPU:
                    r.first.push_back(
                        Component<Nd, T, Cpu>{to_vector(v[i], ctx[i].toCpu()), fs[i][1], i});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
#endif
                default: throw std::runtime_error("Unsupported platform");
                }
            }
            return r;
        }

        /// Return a const version of `Component_tmpl`

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        Components_tmpl<Nd, const T, XPU0, XPU1>
        toConst(const Components_tmpl<Nd, T, XPU0, XPU1> &c) {
            return {std::vector<Component<Nd, const T, XPU0>>(c.first.begin(), c.first.end()),
                    std::vector<Component<Nd, const T, XPU1>>(c.second.begin(), c.second.end())};
        }

        /// Print a message in the standard error
        /// \param comm: a communicator
        /// \param msg: thing to print

        template <typename Comm, typename Msg> void print(const Comm &comm, const Msg msg) {
            std::cerr << "[" << comm.rank << "] " << msg << std::endl;
            std::cerr.flush();
        }

        template <typename Ostream, typename T, std::size_t N>
        Ostream &operator<<(Ostream &s, std::array<T, N> v) {
            s << "{";
            for (const auto &i : v) s << " " << i;
            s << "}";
            return s;
        }

        /// Print a vector in the standard error
        /// \param comm: a communicator
        /// \param v: vector print
        /// \param name: name to prefix the print

        template <typename Comm, typename Vector>
        void print(const Comm &comm, const Vector &v, std::string name) {
            std::cerr << "[" << comm.rank << "] "
                      << " " << name << ":";
            for (const auto &i : v) std::cerr << " " << i;
            std::cerr << std::endl;
            std::cerr.flush();
        }

        /// Vectors used in MPI communications
        template <typename T> struct PackedValues {
            std::vector<T> buf;         ///< pointer to data
            std::vector<MpiInt> counts; ///< number of items send/receive for rank i
            std::vector<MpiInt> displ;  ///< index of the first element to send/receive for rank i
        };

#ifdef SUPERBBLAS_USE_MPI
        /// Communication barrier

        inline void barrier(MpiComm comm) { MPI_Barrier(comm.comm); }

        /// Allocate buffers and prepare arrays from a list of ranges to be used in a MPI communication
        /// \param ranges: iterator over a list of tensor ranges to be packed
        /// \param nranges: number of elements in the list
        /// \param ncomponents: comm.nprocs * ncomponents == the length of each element in `ranges`
        /// \param comm: communicator

        template <std::size_t Nd, typename T>
        PackedValues<T> prepare_pack(const From_size<Nd> *ranges, unsigned int nranges,
                                     unsigned int ncomponents, MpiComm comm) {

            // Allocate PackedValues
            static_assert(MpiTypeSize % sizeof(T) == 0,
                          "Please change MpiTypeSize to be a power of two!");
            PackedValues<T> r{std::vector<T>(), std::vector<MpiInt>(comm.nprocs),
                              std::vector<MpiInt>(comm.nprocs)};

            // Prepare counts and displ
            std::size_t n = 0; // accumulate total number of T elements
            int d = 0;         // accumulate total number of MpiT elements
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                std::size_t n_rank = 0;  // total number of T elements in rank
                if (rank != comm.rank) { // Skip the communications of the local rank
                    // Compute the total number of T elements for rank i
                    for (unsigned int irange = 0; irange < nranges; ++irange) {
                        assert(ranges[irange].size() == comm.nprocs * ncomponents);
                        for (unsigned int componentId = 0; componentId < ncomponents;
                             ++componentId) {
                            n_rank +=
                                volume<Nd>(ranges[irange][rank * ncomponents + componentId][1]);
                        }
                    }
                }
                n += (n_rank * sizeof(T) + MpiTypeSize - 1) / MpiTypeSize * MpiTypeSize / sizeof(T);
                r.counts[rank] = (n_rank * sizeof(T) + MpiTypeSize - 1) / MpiTypeSize;
                r.displ[rank] = d;
                d += r.counts[rank];
            }
            if (d * MpiTypeSize != n * sizeof(T))
                throw std::runtime_error(
                    "Exceeded the maximum package size: increase `MpiTypeSize`");
            r.buf.resize(n);

            return r;
        }

        /// Pack a list of subtensors contiguously in memory
        /// \param o0: dimension labels for the origin tensor
        /// \param fs: a From_size iterator
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param ncomponents1: comm.nprocs * ncomponents1 == fs.size()
        /// \param comm: communicator
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0>
        void pack(const Order<Nd0> &o0, const From_size<Nd0> &fs, const Coor<Nd0> &dim0,
                  vector<const T, XPU0> v0, const Order<Nd1> &o1,
                  typename Indices<Cpu>::iterator disp1, std::vector<Q> &v1,
                  unsigned int ncomponents1, MpiComm comm, CoorOrder co) {

            assert(fs.size() == comm.nprocs * ncomponents1);

            // Get the volume of communicated data without the local part
            std::size_t vol = get_volume<Nd0>(fs) -
                              get_volume<Nd0>(fs.begin() + comm.rank * ncomponents1, ncomponents1);

            // Find indices on cache
            using pointer_perm = std::tuple<From_size<Nd0>, PairPerms<Nd0, Nd1>, int, CoorOrder>;
            using PairIndices = std::pair<Indices<XPU0>, Indices<Cpu>>;
            static std::unordered_map<pointer_perm, PairIndices, TupleHash<pointer_perm>> cache(16);
            pointer_perm key{fs, get_perms(o0, o1), deviceId(v0.ctx()), co};
            auto it = cache.find(key);

            // If they are not, compute the permutation vectors
            if (it == cache.end()) {
                tracker _t("comp. pack permutation");

                Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
                Indices<Cpu> indices0(vol), indices1(vol);
                for (std::size_t i = 0, n = 0; i < fs.size(); ++i) {
                    // Skip the communications of the local rank
                    if (i / ncomponents1 == comm.rank) continue;

                    // Compute the permutation so that the subtensors are packed on the natural
                    // order on the destination; in other words, apply the permutation before
                    // doing the MPI call
                    Coor<Nd0> fromi = fs[i][0], sizei = fs[i][1];
                    Coor<Nd1> sizei1 = reorder_coor<Nd0, Nd1>(sizei, perm0, 1);
                    Indices<Cpu> indices = get_permutation_origin<Nd0, Nd1>(
                        o0, fromi, sizei, dim0, o1, {}, sizei1, Cpu{}, co);
                    assert(indices.size() + n <= vol);
                    std::copy_n(indices.begin(), indices.size(), indices0.begin() + n);
                    indices = get_permutation_destination<Nd0, Nd1>(o0, fromi, sizei, dim0, o1, {},
                                                                    sizei1, Cpu{}, co);
                    assert(indices.size() + n <= vol);
                    std::transform(indices.begin(), indices.end(), indices1.begin() + n,
                                   [=](IndexType d) { return d + disp1[i]; });

                    n += indices.size();
                    assert(n <= vol);
                    assert(i != fs.size() - 1 || n == vol);
                }
                Indices<XPU0> indices0_xpu(indices0.size(), v0.ctx());
                copy_n<IndexType, IndexType>(indices0.data(), Cpu{}, indices0.size(),
                                             indices0_xpu.data(), v0.ctx(), EWOp::Copy{});
                cache[key] = PairIndices{indices0_xpu, indices1};
                it = cache.find(key);
            }

            // Do the copy
            copy_n<IndexType, T, Q>(1.0, v0.data(), it->second.first.begin(), v0.ctx(), vol,
                                    v1.data(), it->second.second.begin(), Cpu{}, EWOp::Copy{});
        }

        /// Pack a list of ranges to be used in a MPI communication
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param ncomponents0: number of elements in toSend and v
        /// \param v: vector containing the values to send
        /// \param o0: dimension labels for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param ncomponents1: number of components on the receiving tensor
        /// \param comm: communicator
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1>
        PackedValues<Q> pack(const std::vector<From_size<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v,
                             const Order<Nd0> &o0, const Order<Nd1> &o1, unsigned int ncomponents1,
                             MpiComm comm, CoorOrder co) {
            unsigned int ncomponents0 = toSend.size();
            PackedValues<Q> r =
                prepare_pack<Nd0, Q>(toSend.data(), ncomponents0, ncomponents1, comm);

            Indices<Cpu> buf_disp(comm.nprocs);
            std::size_t n = 0; // accumulate total number of Q elements
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                std::size_t n_rank = 0;  // total number of Q elements in rank
                if (rank != comm.rank) { // Skip the local communications
                    // Compute the total number of Q elements for rank i
                    for (unsigned int irange = 0; irange < ncomponents0; ++irange) {
                        for (unsigned int componentId1 = 0; componentId1 < ncomponents1;
                             ++componentId1) {
                            n_rank +=
                                volume<Nd0>(toSend[irange][rank * ncomponents1 + componentId1][1]);
                        }
                    }
                }
                buf_disp[rank] = n;
                n += n_rank;
            }
            assert(r.buf.size() == n);

            for (unsigned int componentId0 = 0; componentId0 < ncomponents0; ++componentId0) {
                for (const Component<Nd0, const T, XPU0> &c : v.first) {
                    if (c.componentId == componentId0) {
                        pack<Nd0, Nd1, T, Q>(o0, toSend[componentId0], c.dim, c.it, o1,
                                             buf_disp.begin(), r.buf, ncomponents1, comm, co);
                        for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                            if (rank != comm.rank)
                                buf_disp[rank] += volume<Nd0>(toSend[componentId0][rank][1]);
                        }
                    }
                }
                for (const Component<Nd0, const T, XPU1> &c : v.second) {
                    if (c.componentId == componentId0) {
                        pack<Nd0, Nd1, T, Q>(o0, toSend[componentId0], c.dim, c.it, o1,
                                             buf_disp.begin(), r.buf, ncomponents1, comm, co);
                        for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                            if (rank != comm.rank)
                                buf_disp[rank] += volume<Nd0>(toSend[componentId0][rank][1]);
                        }
                    }
                }
            }
            return r;
        }

        /// Return an order with values 0, 1, 2, ..., N-1

        template <std::size_t N> Order<N> trivial_order() {
            Order<N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = (char)i;
            return r;
        }

        /// Unpack and copy packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param ncomponents0: number of components on the origin tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to packed tensors

        template <std::size_t Nd, typename T, typename XPU>
        void unpack(const PackedValues<T> &r, const From_size<Nd> &toReceive,
                    const Component<Nd, T, XPU> &v, unsigned int ncomponents0, MpiComm comm,
                    EWOp::Copy, CoorOrder co, typename elem<T>::type alpha) {

            // Find indices on cache
            using pointer_dev = std::tuple<From_size_iterator<Nd>, int, CoorOrder>;
            static std::unordered_map<pointer_dev, std::array<Indices<XPU>, 2>,
                                      TupleHash<pointer_dev>>
                cache(16);
            pointer_dev key{toReceive.data(), deviceId(v.it.ctx()), co};
            auto it = cache.find(key);

            // If they are not, compute the permutation vectors
            std::size_t vol = r.buf.size();
            if (it == cache.end()) {
                tracker _t("comp. unpack permutation for copy");

		// Check whether the packages has extra elements to fill up to MpiTypeSize
                bool pre_index = false;
                for (std::size_t i = 0, n = 0; i < comm.nprocs * ncomponents0; ++i) {
                    Coor<Nd> sizei = toReceive[i][1];
                    std::size_t voli = volume(sizei);
                    if (n != r.displ[i / ncomponents0] * (MpiTypeSize / sizeof(T)))
                        pre_index = true;
                    n += voli;
                }

                // Compute the destination index for all received elements
                Indices<Cpu> indices0(pre_index ? vol : 0), indices1(vol);
                Order<Nd> o = trivial_order<Nd>();
                for (std::size_t i = 0, n = 0, n_pre = 0; i < comm.nprocs * ncomponents0; ++i) {
                    if (i / ncomponents0 == comm.rank) continue;
                    Coor<Nd> fromi = toReceive[i][0], sizei = toReceive[i][1];
                    std::size_t voli = volume<Nd>(sizei);

                    if (pre_index) {
                        IndexType disp = r.displ[i / ncomponents0] * (MpiTypeSize / sizeof(T));
                        if (i % ncomponents0 == 0) n_pre = 0;
                        for (std::size_t j = 0; j < voli; ++j) indices0[n + j] = n_pre++ + disp;
                    }

                    Indices<Cpu> indices = get_permutation_destination<Nd, Nd>(
                        o, {}, sizei, sizei, o, fromi, v.dim, Cpu{}, co);
                    std::copy_n(indices.begin(), voli, indices1.begin() + n);
                    n += voli;
                    assert(n <= vol);
                }

                // Copy indices1_cmp to the same device as the destination tensor
                Indices<XPU> indices0_xpu(indices0.size(), v.it.ctx());
                copy_n<IndexType, IndexType>(indices0.data(), Cpu{}, indices0.size(),
                                             indices0_xpu.data(), v.it.ctx(), EWOp::Copy{});
                Indices<XPU> indices1_xpu(indices1.size(), v.it.ctx());
                copy_n<IndexType, IndexType>(indices1.data(), Cpu{}, indices1.size(),
                                             indices1_xpu.data(), v.it.ctx(), EWOp::Copy{});

                // Cache this effort
                cache[key] = {indices0_xpu, indices1_xpu};
                it = cache.find(key);
            }

            // Do the copy
            if (it->second[0].size() == 0)
                copy_n<IndexType, T, T>(alpha, r.buf.data(), Cpu{}, vol, v.it.data(),
                                        it->second[1].begin(), v.it.ctx(), EWOp::Copy{});
            else
                copy_n<IndexType, T, T>(alpha, r.buf.data(), it->second[0].begin(), Cpu{}, vol,
                                        v.it.data(), it->second[1].begin(), v.it.ctx(),
                                        EWOp::Copy{});
        }

        /// Unpack and sum-reduce packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param ncomponents0: number of components on the origin tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to packed tensors

        template <std::size_t Nd, typename T, typename XPU>
        void unpack(const PackedValues<T> &r, const From_size<Nd> &toReceive,
                    const Component<Nd, T, XPU> &v, unsigned int ncomponents0, MpiComm comm,
                    EWOp::Add, CoorOrder co, typename elem<T>::type alpha) {

            // Find indices on cache
            using pointer_dev = std::tuple<From_size_iterator<Nd>, int, CoorOrder>;
            using PermPermreduceIndices = std::tuple<Indices<Cpu>, Indices<Cpu>, Indices<XPU>>;
            static std::unordered_map<pointer_dev, PermPermreduceIndices, TupleHash<pointer_dev>>
                cache(16);
            pointer_dev key{toReceive.data(), deviceId(v.it.ctx()), co};
            auto it = cache.find(key);

            // If they are not, compute the permutation vectors
            std::size_t vol = r.buf.size();
            if (it == cache.end()) {
                tracker _t("comp. unpack permutation for add");

		// Check whether the packages has extra elements to fill up to MpiTypeSize
                bool pre_index = false;
                for (std::size_t i = 0, n = 0; i < comm.nprocs * ncomponents0; ++i) {
                    Coor<Nd> sizei = toReceive[i][1];
                    std::size_t voli = volume(sizei);
                    if (n != r.displ[i / ncomponents0] * (MpiTypeSize / sizeof(T)))
                        pre_index = true;
                    n += voli;
                }
                assert(!pre_index);

                // Compute the destination index for all received elements
                Indices<Cpu> indices1(vol);
                Order<Nd> o = trivial_order<Nd>();
                for (std::size_t i = 0, n = 0; i < comm.nprocs * ncomponents0; ++i) {
                    if (i / ncomponents0 == comm.rank) continue;
                    Coor<Nd> fromi = toReceive[i][0], sizei = toReceive[i][1];
                    std::size_t voli = volume<Nd>(sizei);
                    Indices<Cpu> indices = get_permutation_destination<Nd, Nd>(
                        o, {}, sizei, sizei, o, fromi, v.dim, Cpu{}, co);
                    std::copy_n(indices.begin(), voli, indices1.begin() + n);
                    n += voli;
                    assert(n <= vol);
                }

                // Rearrange together the received elements with the same destination
                Indices<Cpu> perm(vol);
                for (std::size_t i = 0; i < vol; ++i) perm[i] = i;
                std::sort(perm.begin(), perm.end(), [&](const IndexType &a, const IndexType &b) {
                    return (indices1[a] < indices1[b]
                                ? true
                                : (indices1[a] == indices1[b] ? a < b : false));
                });

                // Count how many distinct destination indices there are
                std::size_t perm_distinct_size = (vol == 0 ? 1 : 2);
                for (std::size_t i = 1; i < vol; ++i)
                    if (indices1[perm[i]] != indices1[perm[i - 1]]) perm_distinct_size++;

                // Compute where each distinct destination index starts
                Indices<Cpu> perm_distinct(perm_distinct_size);
                std::size_t perm_distinct_i = 0;
                if (vol > 0) perm_distinct[perm_distinct_i++] = 0;
                for (std::size_t i = 1; i < vol; ++i) {
                    if (indices1[perm[i]] != indices1[perm[i - 1]])
                        perm_distinct[perm_distinct_i++] = i;
                }
                perm_distinct[perm_distinct_i++] = vol;

                // Compute the destination indices for each group
                Indices<Cpu> indices1_cmp(perm_distinct_size - 1);
                for (std::size_t i = 0; i < perm_distinct_size - 1; ++i)
                    indices1_cmp[i] = indices1[perm[perm_distinct[i]]];

                // Copy indices1_cmp to the same device as the destination tensor
                Indices<XPU> indices1_xpu(perm_distinct_size - 1, v.it.ctx());
                copy_n<IndexType>(indices1_cmp.data(), Cpu{}, perm_distinct_size - 1,
                                  indices1_xpu.data(), v.it.ctx(), EWOp::Copy{});

                // Cache this effort
                cache[key] = std::make_tuple(perm, perm_distinct, indices1_xpu);
                it = cache.find(key);
            }

            // Do the copy
            copy_reduce_n<IndexType, T>(alpha, r.buf.data(), Cpu{}, std::get<0>(it->second).begin(),
                                        std::get<1>(it->second).begin(),
                                        std::get<1>(it->second).size(), Cpu{}, v.it.data(),
                                        std::get<2>(it->second).begin(), v.it.ctx());
        }

        /// Unpack and copy packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param ncomponents0: number of components on the origin tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to packed tensors

        template <std::size_t Nd, typename T, typename XPU, typename EWOP>
        void unpack_alt(const PackedValues<T> &r, const From_size<Nd> &toReceive,
                        const Component<Nd, T, XPU> &v, unsigned int ncomponents0, MpiComm comm,
                        EWOP, CoorOrder co, typename elem<T>::type alpha) {

            tracker _t("unpack for add (alt)");

            // Compute the destination index for all received elements
            Order<Nd> o = trivial_order<Nd>();
            for (std::size_t i = 0, n = 0; i < comm.nprocs * ncomponents0; ++i) {
                if (i / ncomponents0 == comm.rank) continue;
                Coor<Nd> fromi = toReceive[i][0], sizei = toReceive[i][1];
                std::size_t voli = volume<Nd>(sizei);
                std::shared_ptr<Indices<XPU>> indices1;
                IndexType disp;
                get_permutation_destination_cache<Nd, Nd>(o, {}, sizei, sizei, o, fromi, v.dim,
                                                          v.it.ctx(), indices1, disp, co);
                // Do the addition
                const T *data =
                    r.buf.data() + r.displ[i / ncomponents0] * (MpiTypeSize / sizeof(T));
                if (i % ncomponents0 == 0) n = 0;
                copy_n<IndexType, T, T>(alpha, data, Cpu{}, voli, v.it.data() + disp,
                                        indices1->begin(), v.it.ctx(), EWOP{});
                n += voli;
            }
        }

        /// Asynchronous sending and receiving
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param ncomponents1: number of components on the destination tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1, typename XPUr, typename EWOp>
        Request send_receive(const Order<Nd0> &o0, const std::vector<From_size<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, From_size<Nd1> toReceive,
                             const Component<Nd1, Q, XPUr> &v1, unsigned int ncomponents1,
                             MpiComm comm, EWOp ewop, CoorOrder co, typename elem<T>::type alpha) {

            tracker _t("packing");

            if (comm.nprocs <= 1) return [] {};

            // Pack v0 and prepare for receiving data from other processes
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            std::shared_ptr<PackedValues<Q>> v0ToSend = std::make_shared<PackedValues<Q>>(
                pack<Nd0, Nd1, T, Q>(toSend, v0, o0, o1, ncomponents1, comm, co));
            std::shared_ptr<PackedValues<Q>> v1ToReceive = std::make_shared<PackedValues<Q>>(
                prepare_pack<Nd1, Q>(&toReceive, 1, ncomponents0, comm));

            // Do the MPI communication
            static MPI_Datatype dtype = get_mpi_datatype();
            MPI_Request r = MPI_REQUEST_NULL;
            assert(v0ToSend->counts.size() == comm.nprocs);
            assert(v0ToSend->displ.size() == comm.nprocs);
            assert(v1ToReceive->counts.size() == comm.nprocs);
            assert(v1ToReceive->displ.size() == comm.nprocs);
            int dtype_size = 0;
            MPI_check(MPI_Type_size(dtype, &dtype_size));
            (void)dtype_size;
            assert((std::size_t)dtype_size == MpiTypeSize);
            assert((v0ToSend->displ.back() + v0ToSend->counts.back()) * MpiTypeSize <=
                   v0ToSend->buf.size() * sizeof(Q));
            assert((v1ToReceive->displ.back() + v1ToReceive->counts.back()) * MpiTypeSize <=
                   v1ToReceive->buf.size() * sizeof(Q));
            assert(v0ToSend->counts[comm.rank] == 0);
            assert(v1ToReceive->counts[comm.rank] == 0);
            if (getUseAsyncAlltoall()) {
                // NOTE: detected hung of MPI_Ialltoallv in some cases; still exploring the source of the problem
                MPI_check(MPI_Ialltoallv(v0ToSend->buf.data(), v0ToSend->counts.data(),
                                         v0ToSend->displ.data(), dtype, v1ToReceive->buf.data(),
                                         v1ToReceive->counts.data(), v1ToReceive->displ.data(),
                                         dtype, comm.comm, &r));
            } else {
                MPI_check(MPI_Alltoallv(v0ToSend->buf.data(), v0ToSend->counts.data(),
                                        v0ToSend->displ.data(), dtype, v1ToReceive->buf.data(),
                                        v1ToReceive->counts.data(), v1ToReceive->displ.data(),
                                        dtype, comm.comm));
                v0ToSend.reset();
            }

            // Do this later
            return [=] {
                // Wait for the MPI communication to finish
                if (getUseAsyncAlltoall()) {
                    tracker _t("alltoall");
                    MPI_Request r0 = r; // this copy avoid compiler warnings
                    MPI_check(MPI_Wait(&r0, MPI_STATUS_IGNORE));
                }

                // Do this copy is unnecessary, but v0ToSend needs to be captured to avoid
                // being released until this point
                std::shared_ptr<PackedValues<Q>> v0ToSend_dummy = v0ToSend;

                // Copy back to v1
                tracker _t("unpacking");
                if (getUnpackAlt())
                    unpack_alt<Nd1>(*v1ToReceive, toReceive, v1, ncomponents0, comm, ewop, co,
                                    Q(alpha));
                else
                    unpack<Nd1>(*v1ToReceive, toReceive, v1, ncomponents0, comm, ewop, co,
                                Q(alpha));
            };
        }
#else

        inline void barrier(SelfComm) {}

#endif // SUPERBBLAS_USE_MPI

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param ncomponents1: number of components on the destination tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1, typename XPUr, typename EWOp>
        Request send_receive(const Order<Nd0> &o0, const std::vector<From_size<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, From_size<Nd1> toReceive,
                             const Component<Nd1, Q, XPUr> &v1, unsigned int ncomponents1,
                             SelfComm comm, EWOp ewop, CoorOrder co, typename elem<T>::type alpha) {
            (void)o0;
            (void)toSend;
            (void)v0;
            (void)o1;
            (void)toReceive;
            (void)v1;
            (void)ncomponents1;
            (void)ewop;
            (void)co;
            (void)alpha;
            if (comm.nprocs <= 1) return [] {};
            throw std::runtime_error("Unsupported SelfComm with nprocs > 1");
        }

        /// Return coor[i] % dim[i]
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions

        template <std::size_t Nd>
        Coor<Nd> normalize_coor(const Coor<Nd> &coor, const Coor<Nd> &dim) {
            Coor<Nd> r;
            for (std::size_t j = 0; j < Nd; j++)
                r[j] = (coor[j] + dim[j] * (coor[j] < 0 ? -coor[j] / dim[j] + 1 : 0)) % dim[j];
            return r;
        }

        /// Return the intersection of two 1D ranges for a NOT toroidal lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param fromr: first coordinate of the resulting range
        /// \param sizer: size of the resulting range

        inline void intersection(IndexType from0, IndexType size0, IndexType from1, IndexType size1,
                                 IndexType dim, IndexType &fromr, IndexType &sizer) {
            fromr = from0 + std::min(std::max(from1 - from0, 0), size0);
            sizer = from0 + std::min(std::max(from1 + size1 - from0, 0), size0) - fromr;
            fromr = (fromr + dim) % dim;
            if (sizer == dim) fromr = from0;
        }

        /// Return the intersection between two ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param fromr0: first coordinate of the first resulting range
        /// \param sizer0: size of the first resulting range
        /// \param fromr1: first coordinate of the second resulting range
        /// \param sizer1: size of the second resulting range

        template <std::size_t Nd>
        From_size_vector<Nd> intersection(const Coor<Nd> &from0, const Coor<Nd> &size0,
                                          const Coor<Nd> &from1, const Coor<Nd> &size1,
                                          const Coor<Nd> &dim) {

            From_size_vector<Nd> r;
            r.push_back({Coor<Nd>{}, Coor<Nd>{}});
            for (std::size_t i = 0; i < Nd; ++i) {
                //
                // Compute the subintervals for the dimension ith
                //
                IndexType fromr0 = 0, sizer0 = 0, fromr1 = 0, sizer1 = 0;

                // Proceed with easy cases: if one of the ranges in the whole lattice
                if (size1[i] == dim[i]) {
                    fromr0 = from0[i], sizer0 = size0[i];
                } else if (size0[i] == dim[i]) {
                    fromr0 = from1[i], sizer0 = size1[i];

                    // Proceed with the general case
                } else {
                    intersection(from0[i], size0[i], from1[i], size1[i], dim[i], fromr0, sizer0);
                    intersection(from0[i], size0[i], from1[i] + dim[i], size1[i], dim[i], fromr1,
                                 sizer1);
                }
                From_size_vector<Nd> q;
                for (const auto &fs : r) {
                    if (sizer0 > 0) {
                        Coor<Nd> fromi = fs[0], sizei = fs[1];
                        fromi[i] = fromr0;
                        sizei[i] = sizer0;
                        q.push_back({fromi, sizei});
                    }
                    if (sizer1 > 0) {
                        Coor<Nd> fromi = fs[0], sizei = fs[1];
                        fromi[i] = fromr1;
                        sizei[i] = sizer1;
                        q.push_back({fromi, sizei});
                    }
                }
                r = q;
            }

            return r;
        }

        /// Return the intersection between two ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param fromr: first coordinate of the first resulting range
        /// \param sizer: size of the first resulting range

        template <std::size_t Nd>
        void intersection(const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &from1,
                          const Coor<Nd> &size1, const Coor<Nd> &dim, Coor<Nd> &fromr,
                          Coor<Nd> &sizer) {
            From_size_vector<Nd> fs = intersection<Nd>(from0, size0, from1, size1, dim);
            if (fs.size() == 0) {
                fromr = Coor<Nd>{};
                sizer = Coor<Nd>{};
            } else if (fs.size() == 1) {
                fromr = fs[0][0];
                sizer = (volume(fs[0][1]) > 0 ? fs[0][1] : Coor<Nd>{});
            } else {
                throw std::runtime_error("Not supported complex overlap of intervals");
            }
        }

        /// Translate a range from one coordinate lattice to another
        /// \param rfrom0: first coordinate of the range to translate
        /// \param rsize0: size of the range to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice
        /// \param perm: permutation of the coordinates
        /// \param fromr: first coordinate of input range into the destination lattice
        /// \param sizer: size of the input range on the destination lattice

        template <std::size_t Nd0, std::size_t Nd1>
        void translate_range(const Coor<Nd0> &rfrom0, const Coor<Nd0> &rsize0,
                             const Coor<Nd0> &from0, const Coor<Nd0> &dim0, const Coor<Nd1> &from1,
                             const Coor<Nd1> &dim1, const Coor<Nd1> perm, Coor<Nd1> &fromr,
                             Coor<Nd1> &sizer) {
            fromr = normalize_coor<Nd1>(
                reorder_coor<Nd0, Nd1>(normalize_coor<Nd0>(rfrom0 - from0 + dim0, dim0), perm) +
                    from1,
                dim1);
            sizer = reorder_coor<Nd0, Nd1>(rsize0, perm, 1);
            if (volume(sizer) == 0) sizer = Coor<Nd1>{};
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: dimension labels for the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param indices0: coordinate in origin tensor that are going to send to each process
        /// \param indices1: coordinate in destination tensor that are going to receive from each process
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <std::size_t Nd0, std::size_t Nd1>
        From_size<Nd0> get_indices_to_send(From_size<Nd0> p0, unsigned int from_rank,
                                           const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                           const Coor<Nd0> &size0, From_size<Nd1> p1,
                                           unsigned int componentId1, unsigned int ncomponents1,
                                           const Order<Nd1> &o1, const Coor<Nd1> &from1) {

            tracker _t("comp. tensor overlaps");

            // Get the global dimensions of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Find partition on cache
            using Key = std::tuple<From_size<Nd0>, unsigned int, Coor<Nd0>, Coor<Nd0>,
                                   From_size<Nd1>, unsigned int, Coor<Nd1>, PairPerms<Nd0, Nd1>>;
            static std::unordered_map<Key, From_size<Nd0>, TupleHash<Key>> cache(16);
            Key key{p0, from_rank, from0, size0, p1, componentId1, from1, get_perms(o0, o1)};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;

            // Restrict the local range in v0 to the range from0, size0
            Coor<Nd0> local_from0 = p0[from_rank][0];
            Coor<Nd0> local_size0 = p0[from_rank][1];
            Coor<Nd0> rlocal_from0, rlocal_size0;
            intersection<Nd0>(from0, size0, local_from0, local_size0, dim0, rlocal_from0,
                              rlocal_size0);

            // Translate the restricted range to the destination lattice
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> rfrom1, rsize1;
            translate_range(rlocal_from0, rlocal_size0, from0, dim0, from1, dim1, perm0, rfrom1,
                            rsize1);

            // Compute the indices
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            unsigned int nprocs = p1.size() / ncomponents1;
            From_size_out<Nd0> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd1> &local_from1 = p1[i * ncomponents1 + componentId1][0];
                const Coor<Nd1> &local_size1 = p1[i * ncomponents1 + componentId1][1];
                Coor<Nd1> fromi, sizei;
                intersection<Nd1>(rfrom1, rsize1, local_from1, local_size1, dim1, fromi, sizei);
                translate_range(fromi, sizei, from1, dim1, from0, dim0, perm1, r[i][0], r[i][1]);
                r[i][0] = normalize_coor(r[i][0] - local_from0, dim0);
            }
            cache[key] = r;

            return r;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: dimension labels for the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param indices0: coordinate in origin tensor that are going to send to each process
        /// \param indices1: coordinate in destination tensor that are going to receive from each process
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <std::size_t Nd0, std::size_t Nd1>
        From_size<Nd1> get_indices_to_receive(const From_size<Nd0> &p0, const Order<Nd0> &o0,
                                              const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                                              const From_size<Nd1> &p1, unsigned int to_rank,
                                              const Order<Nd1> &o1, const Coor<Nd1> &from1) {

            tracker _t("comp. tensor overlaps");

            // Get the global dimensions of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Find partition on cache
            using Key = std::tuple<From_size<Nd0>, Coor<Nd0>, Coor<Nd0>, From_size<Nd1>,
                                   unsigned int, Coor<Nd1>, PairPerms<Nd0, Nd1>>;
            static std::unordered_map<Key, From_size<Nd1>, TupleHash<Key>> cache(16);
            Key key{p0, from0, size0, p1, to_rank, from1, get_perms(o0, o1)};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;

            // Restrict the local range in v1 to the range from1, size1
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1); // size in the destination
            Coor<Nd1> local_from1 = p1[to_rank][0];
            Coor<Nd1> local_size1 = p1[to_rank][1];
            Coor<Nd1> rlocal_from1, rlocal_size1;
            intersection<Nd1>(from1, size1, local_from1, local_size1, dim1, rlocal_from1,
                              rlocal_size1);

            // Translate the restricted range to the origin lattice
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            Coor<Nd0> rfrom0, rsize0;
            translate_range(rlocal_from1, rlocal_size1, from1, dim1, from0, dim0, perm1, rfrom0,
                            rsize0);

            // Compute the indices
            unsigned int nprocs = p0.size();
            From_size_out<Nd1> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd0> &local_from0 = p0[i][0];
                const Coor<Nd0> &local_size0 = p0[i][1];
                Coor<Nd0> fromi, sizei;
                intersection<Nd0>(rfrom0, rsize0, local_from0, local_size0, dim0, fromi, sizei);
                translate_range(fromi, sizei, from0, dim0, from1, dim1, perm0, r[i][0], r[i][1]);
                r[i][0] = normalize_coor(r[i][0] - local_from1, dim1);
            }
            cache[key] = r;

            return r;
        }

        /// Check that dim0 and dim1 have the same dimensions
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: first coordinate not to copy from the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor

        template <std::size_t Nd0, std::size_t Nd1>
        bool check_equivalence(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                               const Coor<Nd1> dim1) {

            if (volume(dim0) == 0 && volume(dim1) == 0) return true;
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> new_dim1 = reorder_coor<Nd0, Nd1>(dim0, perm0, 1);
            return new_dim1 == dim1;
        }

        namespace ns_copy_test {
            enum MockFilling { FillWithIndices, FillWithZeros };

            /// Return a vector with the global indices of the elements that contains
            /// \param from: first coordinate of the component
            /// \param size: number of elements to copy in each dimension
            /// \param dim: global dimension size of the tensor
            /// \param co: coordinate linearization order
            /// \param mf: either fill with indices or zeros

            template <std::size_t Nd>
            vector<IndexType, Cpu> get_mock_components(const Coor<Nd> &from, const Coor<Nd> &size,
                                                       const Coor<Nd> &dim, Cpu, CoorOrder co,
                                                       MockFilling mf) {
                std::size_t vol = volume(size);
                vector<IndexType, Cpu> r(vol);

                if (mf == FillWithIndices) {
                    Coor<Nd> local_stride = get_strides(size, co);
                    Coor<Nd> stride = get_strides(dim, co);
#ifdef _OPENMP
#    pragma omp parallel for
#endif
                    for (std::size_t i = 0; i < vol; ++i)
                        r[i] = coor2index(
                            normalize_coor(index2coor<Nd>(i, size, local_stride) + from, dim), dim,
                            stride);
                } else {
                    zero_n(r.data(), vol, r.ctx());
                }

                return r;
            }

            /// Return a vector with the global indices of the elements that contains
            /// \param from: first coordinate of the component
            /// \param size: number of elements to copy in each dimension
            /// \param dim: global dimension size of the tensor
            /// \param co: coordinate linearization order
            /// \param mf: either fill with indices or zeros

            template <std::size_t Nd, typename XPU,
                      typename std::enable_if<!std::is_same<Cpu, XPU>::value, bool>::type = true>
            vector<IndexType, XPU> get_mock_components(const Coor<Nd> &from, const Coor<Nd> &size,
                                                       const Coor<Nd> &dim, XPU xpu, CoorOrder co,
                                                       MockFilling mf) {
                std::size_t vol = volume(size);
                vector<IndexType, XPU> r(vol, xpu);
                vector<IndexType, Cpu> r_host = get_mock_components(from, size, dim, Cpu{}, co, mf);
                copy_n<IndexType>(r_host.data(), r_host.ctx(), vol, r.data(), r.ctx(),
                                  EWOp::Copy{});
                return r;
            }

            template <typename T>
            using mockIndexType = typename std::conditional<std::is_const<T>::value,
                                                            const IndexType, IndexType>::type;

            /// Return a tensor with the same shape as the given one but where each element has its index
            /// \param p0: partitioning of the origin tensor in consecutive ranges
            /// \param o0: dimension labels for the origin tensor
            /// \param from0: first coordinate to copy from the origin tensor
            /// \param size0: number of elements to copy in each dimension
            /// \param v0: data for the origin tensor
            /// \param p1: partitioning of the destination tensor in consecutive ranges
            /// \param o1: dimension labels for the destination tensor
            /// \param dim1: dimension size for the destination tensor
            /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
            /// \param v1: data for the destination tensor
            /// \param comm: communicator context
            /// \param ewop: either to copy or to add the origin values into the destination values
            /// \param co: coordinate linearization order

            template <std::size_t Nd, typename T, typename Comm, typename XPU0, typename XPU1>
            Components_tmpl<Nd, mockIndexType<T>, XPU0, XPU1>
            get_mock_components(const From_size<Nd> &p, const Components_tmpl<Nd, T, XPU0, XPU1> &v,
                                CoorOrder co, MockFilling mf, Comm comm) {
                Components_tmpl<Nd, mockIndexType<T>, XPU0, XPU1> r;
                Coor<Nd> dim = get_dim(p);
                unsigned int ncomponents = v.first.size() + v.second.size();
                for (const Component<Nd, T, XPU0> &c : v.first) {
                    r.first.push_back(Component<Nd, IndexType, XPU0>{
                        get_mock_components(p[c.componentId + comm.rank * ncomponents][0], c.dim,
                                            dim, c.it.ctx(), co, mf),
                        c.dim, c.componentId});
                }
                for (const Component<Nd, T, XPU1> &c : v.second) {
                    r.second.push_back(Component<Nd, IndexType, XPU1>{
                        get_mock_components(p[c.componentId + comm.rank * ncomponents][0], c.dim,
                                            dim, c.it.ctx(), co, mf),
                        c.dim, c.componentId});
                }
                return r;
            }

            /// Test to copy the content of plural tensor v0 into v1
            /// \param p0: partitioning of the origin tensor in consecutive ranges
            /// \param from0: first coordinate to copy from the origin tensor
            /// \param size0: number of elements to copy in each dimension
            /// \param dim0: number of elements on the origin tensor on each dimension
            /// \param o0: dimension labels for the origin tensor
            /// \param o1: dimension labels for the destination tensor
            /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
            /// \param dim1: dimension size for the destination tensor
            /// \param v: data to check
            /// \param local_from1: first coordinate of the destination tensor
            /// \param co: coordinate linearization order

            template <std::size_t Nd0, std::size_t Nd1, typename XPU, typename EWOP>
            void test_copy_check(const From_size<Nd0> &p, const Coor<Nd0> &from0,
                                 const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                 const Order<Nd0> &o0, const Coor<Nd1> &from1,
                                 const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                                 const Component<Nd1, IndexType, XPU> &v,
                                 const Coor<Nd1> &local_from1, EWOP, CoorOrder co) {

                Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
                Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
                Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
                std::size_t vol = volume(v.dim);
                Coor<Nd1> local_stride1 = get_strides<Nd1>(v.dim, co);
                Coor<Nd0> stride0 = get_strides<Nd0>(dim0, co);
                vector<IndexType, Cpu> v_host = toCpu(v.it);

#ifdef _OPENMP
#    pragma omp parallel for
#endif
                for (std::size_t i = 0; i < vol; ++i) {
                    Coor<Nd1> c1 =
                        normalize_coor(index2coor(i, v.dim, local_stride1) + local_from1, dim1);
                    IndexType true_val = 0;
                    if (is_in_interval(from1, size1, dim1, c1)) {
                        Coor<Nd0> c0 =
                            normalize_coor(reorder_coor(c1 - from1, perm1) + from0, dim0);
                        true_val = coor2index(c0, dim0, stride0);
                        int rep = 0;
                        for (const auto &fs : p)
                            if (is_in_interval(fs[0], fs[1], dim0, c0)) ++rep;
                        if (std::is_same<EWOp::Add, EWOP>::value)
                            true_val *= rep;
                        else if (rep == 0)
                            true_val = 0;
                    }
                    if (v_host[i] != true_val)
                        throw std::runtime_error("test_copy_check do not pass!");
                }
            }

            /// Test to copy the content of plural tensor v0 into v1
            /// \param alpha: factor applied to the input tensors
            /// \param p0: partitioning of the origin tensor in consecutive ranges
            /// \param o0: dimension labels for the origin tensor
            /// \param from0: first coordinate to copy from the origin tensor
            /// \param size0: number of elements to copy in each dimension
            /// \param v0: data for the origin tensor
            /// \param p1: partitioning of the destination tensor in consecutive ranges
            /// \param o1: dimension labels for the destination tensor
            /// \param dim1: dimension size for the destination tensor
            /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
            /// \param v1: data for the destination tensor
            /// \param comm: communicator context
            /// \param ewop: either to copy or to add the origin values into the destination values
            /// \param co: coordinate linearization order

            template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                      typename XPU0, typename XPU1, typename EWOP>
            void test_copy(typename elem<T>::type, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                           const Coor<Nd0> &size0, const Order<Nd0> &o0,
                           const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                           const From_size<Nd1> &p1, const Coor<Nd1> &from1, const Order<Nd1> &o1,
                           const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm, EWOP,
                           CoorOrder co) {

                bool trackingTime = getTrackingTime();
                getTrackingTime() = false;

                // Fill the mock input and output tensors
                const Components_tmpl<Nd0, const IndexType, XPU0, XPU1> v0_ =
                    get_mock_components(p0, v0, co, FillWithIndices, comm);
                const Components_tmpl<Nd1, IndexType, XPU0, XPU1> v1_ =
                    get_mock_components(p1, v1, co, FillWithZeros, comm);

                // Copy the indices
                copy(1, p0, from0, size0, o0, v0_, p1, from1, o1, v1_, comm, EWOP{}, co, false);

                // Check that the modified elements on v1_ are what they should be
                unsigned int ncomponents1 = v1.first.size() + v1.second.size();
                Coor<Nd0> dim0 = get_dim<Nd0>(p0);
                Coor<Nd1> dim1 = get_dim<Nd1>(p1);
                for (const Component<Nd1, IndexType, XPU0> &c : v1_.first) {
                    test_copy_check<Nd0, Nd1>(p0, from0, size0, dim0, o0, from1, dim1, o1, c,
                                              p1[c.componentId + comm.rank * ncomponents1][0],
                                              EWOP{}, co);
                }
                for (const Component<Nd1, IndexType, XPU1> &c : v1_.second) {
                    test_copy_check<Nd0, Nd1>(p0, from0, size0, dim0, o0, from1, dim1, o1, c,
                                              p1[c.componentId + comm.rank * ncomponents1][0],
                                              EWOP{}, co);
                }

                getTrackingTime() = trackingTime;
            }
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param alpha: factor applied to the input tensors
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1, typename EWOp>
        void copy(typename elem<T>::type alpha, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                  const Coor<Nd0> &size0, const Order<Nd0> &o0,
                  const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
                  const Coor<Nd1> &from1, const Order<Nd1> &o1,
                  const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm, EWOp ewop, CoorOrder co,
                  bool do_test = true) {

            if (getDebugLevel() >= 2 && do_test) {
                ns_copy_test::test_copy(alpha, p0, from0, size0, o0, v0, p1, from1, o1, v1, comm,
                                        EWOp{}, co);
            }

            tracker _t("distributed copy");

            // Check the dimensions of p0 and p1
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            unsigned int ncomponents1 = v1.first.size() + v1.second.size();

            if (p0.size() != ncomponents0 * comm.nprocs)
                throw std::runtime_error("Invalid number of elements in the tensor distribution");

            if (p1.size() != ncomponents1 * comm.nprocs)
                throw std::runtime_error("Invalid number of elements in the tensor distribution");

            // Check the compatibility of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);
            if (!check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1))
                throw std::runtime_error("Invalid copy operation");

            // Split the work for each receiving component
            std::vector<Request> reqs;
            for (unsigned int i = 0; i < ncomponents1; ++i) {
                for (const Component<Nd1, Q, XPU0> &c : v1.first) {
                    if (c.componentId == i)
                        reqs.push_back(copy<Nd0, Nd1, T, Q>(alpha, p0, from0, size0, o0, v0, p1,
                                                            ncomponents1, from1, o1, c, comm, ewop,
                                                            co));
                }
                for (const Component<Nd1, Q, XPU1> &c : v1.second) {
                    if (c.componentId == i)
                        reqs.push_back(copy<Nd0, Nd1, T, Q>(alpha, p0, from0, size0, o0, v0, p1,
                                                            ncomponents1, from1, o1, c, comm, ewop,
                                                            co));
                }
            }

            // Finish the request
            for (const Request &r : reqs) wait(r);
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param alpha: factor applied to the input tensors
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1>
        void copy(typename elem<T>::type alpha, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                  const Coor<Nd0> &size0, const Order<Nd0> &o0,
                  const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
                  const Coor<Nd1> &from1, const Order<Nd1> &o1,
                  const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm, CopyAdd copyadd,
                  CoorOrder co) {

            if (getDebugLevel() >= 1) {
                barrier(comm);
                for (const auto &i : v1.first) sync(i.it.ctx());
                for (const auto &i : v1.second) sync(i.it.ctx());
            }

            switch (copyadd) {
            case Copy:
                copy(alpha, p0, from0, size0, o0, v0, p1, from1, o1, v1, comm, EWOp::Copy{}, co);
                break;
            case Add:
                copy(alpha, p0, from0, size0, o0, v0, p1, from1, o1, v1, comm, EWOp::Add{}, co);
                break;
            }

            if (getDebugLevel() >= 1) {
                for (const auto &i : v1.first) sync(i.it.ctx());
                for (const auto &i : v1.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param alpha: factor applied to v0
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1, typename XPU, typename EWOp>
        Request copy(typename elem<T>::type alpha, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                     const Coor<Nd0> &size0, const Order<Nd0> &o0,
                     const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
                     unsigned int ncomponents1, const Coor<Nd1> &from1, const Order<Nd1> &o1,
                     const Component<Nd1, Q, XPU> &v1, Comm comm, EWOp ewop, CoorOrder co) {

            // Generate the list of subranges to send from each component from v0 to v1
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            std::vector<From_size<Nd0>> toSend(ncomponents0);

            for (unsigned int i = 0; i < v0.first.size(); ++i) {
                toSend[v0.first[i].componentId] = get_indices_to_send<Nd0, Nd1>(
                    p0, comm.rank * ncomponents0 + v0.first[i].componentId, o0, from0, size0, p1,
                    v1.componentId, ncomponents1, o1, from1);
            }
            for (unsigned int i = 0; i < v0.second.size(); ++i) {
                toSend[v0.second[i].componentId] = get_indices_to_send<Nd0, Nd1>(
                    p0, comm.rank * ncomponents0 + v0.second[i].componentId, o0, from0, size0, p1,
                    v1.componentId, ncomponents1, o1, from1);
            }

            // Generate the list of subranges to receive from each component from v0 to v1
            From_size<Nd1> toReceive = get_indices_to_receive<Nd0, Nd1>(
                p0, o0, from0, size0, p1, v1.componentId + comm.rank * ncomponents1, o1, from1);

            // Do the sending and receiving
            Request mpi_req = send_receive<Nd0, Nd1>(o0, toSend, v0, o1, toReceive, v1,
                                                     ncomponents1, comm, ewop, co, alpha);

            // Do the local copies
            Request local_req = [=] {
                unsigned int ncomponents0 = v0.first.size() + v0.second.size();
                for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                    assert(check_equivalence(
                        o0, toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][1],
                        o1, toReceive[c0.componentId + comm.rank * ncomponents0][1]));
                    local_copy<Nd0, Nd1, T, Q>(
                        alpha, o0,
                        toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][0],
                        toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][1],
                        c0.dim, c0.it, o1, toReceive[c0.componentId + comm.rank * ncomponents0][0],
                        v1.dim, v1.it, ewop, co);
                }
                for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                    assert(check_equivalence(
                        o0, toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][1],
                        o1, toReceive[c0.componentId + comm.rank * ncomponents0][1]));
                    local_copy<Nd0, Nd1, T, Q>(
                        alpha, o0,
                        toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][0],
                        toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][1],
                        c0.dim, c0.it, o1, toReceive[c0.componentId + comm.rank * ncomponents0][0],
                        v1.dim, v1.it, ewop, co);
                }
            };

            return [=] {
                wait(local_req);
                wait(mpi_req);
            };
        }

        /// Return value for the dimensions in o_r matching the given for o0 and o1

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        Coor<Ndo> get_dimensions(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                                 const Coor<Nd1> &dim1, const Order<Ndo> &o_r) {
            std::map<char, IndexType> m;
            for (std::size_t i = 0; i < Nd0; ++i) m[o0[i]] = dim0[i];
            for (std::size_t i = 0; i < Nd1; ++i) {
                auto it = m.find(o1[i]);
                if (it == m.end())
                    m[o1[i]] = dim1[i];
                else if (it->second != dim1[i])
                    throw std::runtime_error("Incompatible distributions for contraction");
            }
            Coor<Ndo> r;
            for (std::size_t i = 0; i < Ndo; ++i) r[i] = m[o_r[i]];
            return r;
        }

        /// Get the output partition
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        From_size<Ndo> get_output_partition(From_size<Nd0> p0, const Order<Nd0> &o0,
                                            From_size<Nd1> p1, const Order<Nd1> &o1,
                                            const Order<Ndo> &o_r) {
            assert(p0.size() == p1.size());

            // Find partition on cache
            using Key = std::tuple<From_size<Nd0>, From_size<Nd1>, PairPerms<Nd0, Nd1>,
                                   PairPerms<Nd0, Ndo>, PairPerms<Nd1, Ndo>>;
            static std::unordered_map<Key, From_size<Ndo>, TupleHash<Key>> cache(16);
            Key key{p0, p1, get_perms(o0, o1), get_perms(o0, o_r), get_perms(o1, o_r)};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;

            // Create partition
            From_size_out<Ndo> pr(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                pr[i][0] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][0], o1, p1[i][0], o_r);
                pr[i][1] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][1], o1, p1[i][1], o_r);
            }
            cache[key] = pr;

            return pr;
        }

        /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
        /// \param alpha: factor on the contraction
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param ncomponents0: number of consecutive components in each MPI rank
        /// \param o0: dimension labels for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param v0: data for the first operator
        /// \param ctx0: context for each data pointer in v0
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param ncomponents1: number of consecutive components in each MPI rank
        /// \param o1: dimension labels for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param v1: data for the second operator
        /// \param ctx1: context for each data pointer in v1
        /// \param beta: factor on the destination tensor
        /// \param pr: partitioning of the resulting tensor in consecutive ranges
        /// \param ncomponentsr: number of consecutive components in each MPI rank
        /// \param o_r: dimension labels for the output operator
        /// \param vr: data for the second operator
        /// \param ctxr: context for each data pointer in vr
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
        ///
        /// The order of the labels should be as following:
        ///
        /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
        /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
        /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
        /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void contraction(T alpha, const From_size<Nd0> &p0, const Order<Nd0> &o0, bool conj0,
                         const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                         const From_size<Nd1> &p1, const Order<Nd1> &o1, bool conj1,
                         const Components_tmpl<Nd1, const T, XPU0, XPU1> &v1, T beta,
                         const From_size<Ndo> &pr, const Order<Ndo> &o_r,
                         const Components_tmpl<Ndo, T, XPU0, XPU1> &vr, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vr.first) sync(i.it.ctx());
                for (const auto &i : vr.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker _t("distributed contraction");

            // Check the compatibility of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);
            Coor<Ndo> dimr = get_dim<Ndo>(pr);
            if (!check_dimensions<Nd0, Nd1, Ndo>(o0, dim0, o1, dim1, o_r, dimr))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (v0.first.size() != v1.first.size() || v0.second.size() != v1.second.size())
                throw std::runtime_error(
                    "the two input tensors should have the same number of components");
            bool unmatch_dev = false;
            for (unsigned int i = 0; i < v0.first.size(); ++i)
                if (deviceId(v0.first[i].it.ctx()) != deviceId(v1.first[i].it.ctx()))
                    unmatch_dev = true;
            for (unsigned int i = 0; i < v0.second.size(); ++i)
                if (deviceId(v0.second[i].it.ctx()) != deviceId(v1.second[i].it.ctx()))
                    unmatch_dev = true;
            if (unmatch_dev)
                throw std::runtime_error(
                    "Each component of the input tensors should be on the same device");

            // Generate the partitioning and the storage for the output tensor
            unsigned int ncomponents = v0.first.size() + v1.second.size();
            From_size<Ndo> pr_ = get_output_partition<Nd0, Nd1, Ndo>(p0, o0, p1, o1, o_r);
            Components_tmpl<Ndo, const T, XPU0, XPU1> vr_;
            std::vector<vector<T, XPU0>> vr0(v0.first.size());
            for (unsigned int i = 0; i < v0.first.size(); ++i) {
                const unsigned int componentId = v0.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Ndo> &dimi = pr_[pi][1];
                vr0[i] = vector<T, XPU0>(volume<Ndo>(dimi), v0.first[i].it.ctx());
                vr_.first.push_back(Component<Ndo, T, XPU0>{vr0[i], dimi, componentId});
                local_contraction<Nd0, Nd1, Ndo, T>(alpha, o0, p0[pi][1], conj0, v0.first[i].it, o1,
                                                    p1[pi][1], conj1, v1.first[i].it, T{0.0}, o_r,
                                                    dimi, vr0[i], co);
            }
            std::vector<vector<T, XPU1>> vr1(v0.second.size());
            for (unsigned int i = 0; i < v0.second.size(); ++i) {
                const unsigned int componentId = v0.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Ndo> &dimi = pr_[pi][1];
                vr1[i] = vector<T, XPU1>(volume<Ndo>(dimi), v0.second[i].it.ctx());
                vr_.second.push_back(Component<Ndo, T, XPU1>{vr1[i], dimi, componentId});
                local_contraction<Nd0, Nd1, Ndo, T>(alpha, o0, p0[pi][1], conj0, v0.second[i].it,
                                                    o1, p1[pi][1], conj1, v1.second[i].it, T{0.0},
                                                    o_r, dimi, vr1[i], co);
            }

            // Scale the output tensor by beta
            copy<Ndo, Ndo, T>(beta, pr, {}, dimr, o_r, toConst(vr), pr, {}, o_r, vr, comm,
                              EWOp::Copy{}, co);

            // Reduce all the subtensors to the final tensor
            copy<Ndo, Ndo, T>(1.0, pr_, {}, dimr, o_r, vr_, pr, {}, o_r, vr, comm, EWOp::Add{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vr.first) sync(i.it.ctx());
                for (const auto &i : vr.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        /// Return a From_size from a partition that can be hashed and stored
        /// \param p: partitioning
        /// \return: From_size

        template <std::size_t Nd>
        From_size<Nd> get_from_size(const PartitionItem<Nd> *p, std::size_t n) {
            static std::unordered_set<From_size<Nd>, TupleHash<From_size<Nd>>> cache(16);
            From_size<Nd> fs = to_vector(p, n);
            auto it = cache.find(fs);
            if (it == cache.end()) it = cache.insert(fs.clone()).first;
            return *it;
        }
    }

    /// Return a partitioning for a tensor of `dim` dimension onto a grid of processes
    /// \param dim1: dimension size for the tensor
    /// \param procs: number of processes on each dimension; the total number of processes is the
    ///               product of all the elements.

    template <std::size_t Nd>
    std::vector<PartitionItem<Nd>> basic_partitioning(Coor<Nd> dim, Coor<Nd> procs) {
        int vol_procs = (int)detail::volume<Nd>(procs);
        std::vector<PartitionItem<Nd>> fs(vol_procs);
        Coor<Nd> stride = detail::get_strides<Nd>(procs, SlowToFast);
        for (int rank = 0; rank < vol_procs; ++rank) {
            Coor<Nd> cproc = detail::index2coor<Nd>(rank, procs, stride);
            for (std::size_t i = 0; i < Nd; ++i) {
                // First coordinate in process with rank 'rank' on dimension 'i'
                fs[rank][0][i] =
                    dim[i] / procs[i] * cproc[i] + std::min(cproc[i], dim[i] % procs[i]);
                // Number of elements in process with rank 'cproc[i]' on dimension 'i'
                fs[rank][1][i] = dim[i] / procs[i] + (dim[i] % procs[i] > cproc[i] ? 1 : 0);
            }
        }
        return fs;
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Copy the content of plural tensor v0 into v1
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param mpicomm: MPI communicator context
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const T **v0,
              const Context *ctx0, const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1,
              const Coor<Nd1> &from1, Q **v1, const Context *ctx1, MPI_Comm mpicomm, CoorOrder co,
              CopyAdd copyadd) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::copy<Nd0, Nd1>(alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs), from0,
                               size0, detail::toArray<Nd0>(o0, "o0"),
                               detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0, comm),
                               detail::get_from_size(p1, ncomponents1 * comm.nprocs), from1,
                               detail::toArray<Nd1>(o1, "o1"),
                               detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm), comm,
                               copyadd, co);
    }
#endif // SUPERBBLAS_USE_MPI

    /// Copy the content of plural tensor v0 into v1
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> from0, const Coor<Nd0> size0, const T **v0,
              const Context *ctx0, const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1,
              const Coor<Nd1> from1, Q **v1, const Context *ctx1, CoorOrder co, CopyAdd copyadd) {

        detail::SelfComm comm = detail::get_comm();

        detail::copy<Nd0, Nd1>(alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs), from0,
                               size0, detail::toArray<Nd0>(o0, "o0"),
                               detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0, comm),
                               detail::get_from_size(p1, ncomponents1 * comm.nprocs), from1,
                               detail::toArray<Nd1>(o1, "o1"),
                               detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm), comm,
                               copyadd, co);
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
    /// \param alpha: factor on the contraction
    /// \param p0: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponents1: number of consecutive components in each MPI rank
    /// \param o1: dimension labels for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param ctx1: context for each data pointer in v1
    /// \param beta: factor on the destination tensor
    /// \param pr: partitioning of the resulting tensor in consecutive ranges
    /// \param ncomponentsr: number of consecutive components in each MPI rank
    /// \param o_r: dimension labels for the output operator
    /// \param vr: data for the second operator
    /// \param ctxr: context for each data pointer in vr
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    ///
    /// The order of the labels should be as following:
    ///
    /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
    /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
    /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
    /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T>
    void contraction(T alpha, const PartitionItem<Nd0> *p0, int ncomponents0, const char *o0,
                     bool conj0, const T **v0, const Context *ctx0, const PartitionItem<Nd1> *p1,
                     int ncomponents1, const char *o1, bool conj1, const T **v1,
                     const Context *ctx1, T beta, const PartitionItem<Ndo> *pr, int ncomponentsr,
                     const char *o_r, T **vr, const Context *ctxr, MPI_Comm mpicomm, CoorOrder co) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::contraction<Nd0, Nd1, Ndo>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs), o0_, conj0,
            detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0, comm),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs), o1_, conj1,
            detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm), beta,
            detail::get_from_size(pr, ncomponentsr * comm.nprocs), o_r_,
            detail::get_components<Ndo>(vr, ctxr, ncomponentsr, pr, comm), comm, co);
    }
#endif // SUPERBBLAS_USE_MPI

    /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
    /// \param alpha: factor on the contraction
    /// \param p0: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponents1: number of consecutive components in each MPI rank
    /// \param o1: dimension labels for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param ctx1: context for each data pointer in v1
    /// \param beta: factor on the destination tensor
    /// \param pr: partitioning of the resulting tensor in consecutive ranges
    /// \param ncomponentsr: number of consecutive components in each MPI rank
    /// \param o_r: dimension labels for the output operator
    /// \param vr: data for the second operator
    /// \param ctxr: context for each data pointer in vr
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    ///
    /// The order of the labels should be as following:
    ///
    /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
    /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
    /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
    /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T>
    void contraction(T alpha, const PartitionItem<Nd0> *p0, int ncomponents0, const char *o0,
                     bool conj0, const T **v0, const Context *ctx0, const PartitionItem<Nd1> *p1,
                     int ncomponents1, const char *o1, bool conj1, const T **v1,
                     const Context *ctx1, T beta, const PartitionItem<Ndo> *pr, int ncomponentsr,
                     const char *o_r, T **vr, const Context *ctxr, CoorOrder co) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::SelfComm comm = detail::get_comm();

        detail::contraction<Nd0, Nd1, Ndo>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs), o0_, conj0,
            detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0, comm),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs), o1_, conj1,
            detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm), beta,
            detail::get_from_size(pr, ncomponentsr * comm.nprocs), o_r_,
            detail::get_components<Ndo>(vr, ctxr, ncomponentsr, pr, comm), comm, co);
    }
}

#endif //  __SUPERBBLAS_DIST__
