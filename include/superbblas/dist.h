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
        /// From_size iterator
        template <std::size_t N> using From_size_iterator = const From_size_item<N> *;

        template <std::size_t Nd0, std::size_t Nd1>
        using PairPerms = std::tuple<Coor<Nd0>, Coor<Nd1>>;

        // Supported types for contractions: the ones supported by superbblas excepting int
        template <typename T> struct supported_type_for_contractions {
            static constexpr bool value = supported_type<T>::value;
        };
        template <> struct supported_type_for_contractions<int> {
            static constexpr bool value = false;
        };

        template <std::size_t N> using Proc_ranges = std::vector<From_size_out<N>>;

        //
        // Auxiliary functions
        //

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

            char s[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(error, s, &len);

#    define CHECK_AND_THROW(ERR)                                                                   \
        if (error == ERR) {                                                                        \
            std::stringstream ss;                                                                  \
            ss << "MPI error: " #ERR ": " << std::string(&s[0], &s[0] + len);                      \
            throw std::runtime_error(ss.str());                                                    \
        }

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
            Mask<XPU> mask_it;        ///< Mask

            template <
                typename Q = T,
                typename std::enable_if<std::is_same<Q, typename std::remove_const<Q>::type>::value,
                                        bool>::type = true>
            operator Component<Nd, const Q, XPU>() const {
                return {it, dim, componentId, mask_it};
            }

            template <typename Q = T,
                      typename std::enable_if<
                          !std::is_same<Q, typename std::remove_const<Q>::type>::value,
                          bool>::type = true>
            operator Component<Nd, typename std::remove_const<Q>::type, XPU>() const {
                return {it, dim, componentId, mask_it};
            }
        };

        /// A tensor composed of several components
        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        using Components_tmpl =
            std::pair<std::vector<Component<Nd, T, XPU0>>, std::vector<Component<Nd, T, XPU1>>>;

#ifdef SUPERBBLAS_USE_GPU
        /// A tensor composed of several CPU and GPU elements
        template <std::size_t Nd, typename T> using Components = Components_tmpl<Nd, T, Gpu, Cpu>;
#else
        /// A tensor composed of only of CPU components
        template <std::size_t Nd, typename T> using Components = Components_tmpl<Nd, T, Cpu, Cpu>;
#endif // SUPERBBLAS_USE_GPU

        template <std::size_t Nd, typename T, typename Comm>
        Components<Nd, T> get_components(T **v, const MaskType **mask, const Context *ctx,
                                         unsigned int ncomponents, From_size_iterator<Nd> p,
                                         Comm comm, Session session) {
            // Get components on the local process
            From_size_iterator<Nd> fs = p + comm.rank * ncomponents;

            Components<Nd, T> r;
            for (unsigned int i = 0; i < ncomponents; ++i) {
                MaskType *maski = mask ? (MaskType *)mask[i] : (MaskType *)nullptr;
                switch (ctx[i].plat) {
#ifdef SUPERBBLAS_USE_GPU
                case CPU:
                    r.second.push_back(Component<Nd, T, Cpu>{
                        to_vector(v[i], volume(fs[i][1]), ctx[i].toCpu(session)), fs[i][1], i,
                        to_vector(maski, volume(fs[i][1]), ctx[i].toCpu(session))});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
                case GPU:
                    r.first.push_back(Component<Nd, T, Gpu>{
                        to_vector(v[i], volume(fs[i][1]), ctx[i].toGpu(session)), fs[i][1], i,
                        to_vector(maski, volume(fs[i][1]), ctx[i].toGpu(session))});
                    assert(!v[i] || getPtrDevice(v[i]) == ctx[i].device);
                    break;
#else // SUPERBBLAS_USE_GPU
                case CPU:
                    r.first.push_back(Component<Nd, T, Cpu>{
                        to_vector(v[i], volume(fs[i][1]), ctx[i].toCpu(session)), fs[i][1], i,
                        to_vector(maski, volume(fs[i][1]), ctx[i].toCpu(session))});
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

        /// Return a non-const version of `Component_tmpl`

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        Components_tmpl<Nd, typename std::remove_const<T>::type, XPU0, XPU1>
        toNonConst(const Components_tmpl<Nd, T, XPU0, XPU1> &c) {
            return {std::vector<Component<Nd, typename std::remove_const<T>::type, XPU0>>(
                        c.first.begin(), c.first.end()),
                    std::vector<Component<Nd, typename std::remove_const<T>::type, XPU1>>(
                        c.second.begin(), c.second.end())};
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

        /// Return an order with values 0, 1, 2, ..., N-1

        template <std::size_t N> Order<N> trivial_order() {
            Order<N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = (char)i;
            return r;
        }

#ifdef SUPERBBLAS_USE_MPI
        /// Communication barrier

        inline void barrier(MpiComm comm) { MPI_Barrier(comm.comm); }

        /// Vectors used in MPI communications
        template <typename T> struct PackedValues {
            vector<T, Cpu> buf;         ///< pointer to data
            std::vector<MpiInt> counts; ///< number of items send/receive for rank i
            std::vector<MpiInt> displ;  ///< index of the first element to send/receive for rank i
        };

        /// Allocate buffers and prepare arrays from a list of ranges to be used in a MPI communication
        /// \param ranges: iterator over a list of tensor ranges to be packed
        /// \param nranges: number of elements in the list
        /// \param ncomponents: comm.nprocs * ncomponents == the length of each element in `ranges`
        /// \param comm: communicator

        template <std::size_t Nd, typename T>
        PackedValues<T> prepare_pack(const std::vector<Proc_ranges<Nd>> &toSend, MpiComm comm) {

            // Allocate PackedValues
            static_assert(MpiTypeSize % sizeof(T) == 0,
                          "Please change MpiTypeSize to be a power of two!");
            PackedValues<T> r{vector<T, Cpu>(), std::vector<MpiInt>(comm.nprocs),
                              std::vector<MpiInt>(comm.nprocs)};

            // Prepare counts and displ
            std::size_t nranges = toSend.size();
            unsigned int ncomponents = 1;
            std::size_t n = 0; // accumulate total number of T elements
            int d = 0;         // accumulate total number of MpiT elements
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                std::size_t n_rank = 0;  // total number of T elements in rank
                if (rank != comm.rank) { // Skip the communications of the local rank
                    // Compute the total number of T elements for rank i
                    for (unsigned int irange = 0; irange < nranges; ++irange) {
                        assert(toSend[irange].size() == comm.nprocs * ncomponents);
                        for (unsigned int componentId = 0; componentId < ncomponents;
                             ++componentId) {
                            for (const auto &fs : toSend[irange][rank * ncomponents + componentId])
                                n_rank += volume<Nd>(fs[1]);
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
            r.buf = vector<T, Cpu>(n, Cpu{}, MpiTypeSize);

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
        void pack(const Order<Nd0> &o0, const Proc_ranges<Nd0> &fs, const Coor<Nd0> &dim0,
                  vector<const T, XPU0> v0, Mask<XPU0> mask0, const Order<Nd1> &o1,
                  typename Indices<Cpu>::iterator disp1, vector<Q, Cpu> &v1,
                  unsigned int ncomponents1, MpiComm comm, CoorOrder co) {

            assert(fs.size() == comm.nprocs * ncomponents1);

            // Find indices on cache
            using Key =
                std::tuple<Proc_ranges<Nd0>, Coor<Nd0>, PairPerms<Nd0, Nd1>, int, CoorOrder>;
            using PairIndices = std::pair<Indices<XPU0>, Indices<Cpu>>;
            struct cache_tag {};
            auto cache = getCache<Key, PairIndices, TupleHash<Key>, cache_tag>(v0.ctx());
            Key key{fs, dim0, get_perms(o0, o1), deviceId(v0.ctx()), co};
            auto it = mask0.size() == 0 ? cache.find(key) : cache.end();

            // If they are not, compute the permutation vectors
            Indices<XPU0> indices0_xpu;
            Indices<Cpu> indices1;
            if (it == cache.end()) {
                tracker<XPU0> _t("comp. pack permutation", v0.ctx());

                // Get the maximum volume of communicated data without the local part
                std::size_t vol = 0;
                for (unsigned int i = 0; i < fs.size(); ++i)
                    if (i / ncomponents1 != comm.rank)
                        for (const auto &it : fs[i]) vol += volume(it[1]);

                Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
                Indices<Cpu> indices0{vol, Cpu{}};
                indices1 = Indices<Cpu>{vol, Cpu{}};
                Mask<Cpu> mask0_cpu = makeSure(mask0, Cpu{});
                std::size_t n = 0;
                for (std::size_t i = 0; i < fs.size(); ++i) {
                    // Skip the communications of the local rank
                    if (i / ncomponents1 == comm.rank) continue;

                    for (const auto &fsi : fs[i]) {
                        // Compute the permutation so that the subtensors are packed on the natural
                        // order on the destination; in other words, apply the permutation before
                        // doing the MPI call
                        Coor<Nd0> fromi = fsi[0], sizei = fsi[1];
                        Coor<Nd1> sizei1 = reorder_coor<Nd0, Nd1>(sizei, perm0, 1);
                        Indices<Cpu> indices0i = get_permutation_origin<Nd0, Nd1>(
                            o0, fromi, sizei, dim0, o1, {}, sizei1, Cpu{}, co);
                        assert(indices0i.size() + n <= vol);
                        Indices<Cpu> indices0i_mask = indices0i;
                        if (mask0_cpu.size() > 0)
                            indices0i_mask = select(indices0i, mask0_cpu.data(), indices0i);
                        std::copy_n(indices0i_mask.begin(), indices0i_mask.size(),
                                    indices0.begin() + n);

                        Indices<Cpu> indices1i_mask = get_permutation_destination<Nd0, Nd1>(
                            o0, fromi, sizei, dim0, o1, {}, sizei1, Cpu{}, co);
                        assert(indices0i.size() == indices1i_mask.size());
                        if (mask0_cpu.size() > 0)
                            indices1i_mask = select(indices0i, mask0_cpu.data(), indices1i_mask);
                        IndexType dispi = disp1[i / ncomponents1];
                        std::transform(indices1i_mask.begin(), indices1i_mask.end(),
                                       indices1.begin() + n,
                                       [=](IndexType d) { return d + dispi; });

                        disp1[i / ncomponents1] += indices1i_mask.size();
                        n += indices1i_mask.size();
                        assert(n <= vol);
                    }
                }
                indices0.resize(n);
                indices1.resize(n);
                indices0_xpu = makeSure(indices0, v0.ctx());

                // The cache trackers consider that all cache entries are on the same device; so just track the
                // indices0_xpu when using gpus
                if (mask0.size() == 0) {
                    std::size_t size =
                        storageSize(indices0_xpu) +
                        (deviceId(v0.ctx()) == CPU_DEVICE_ID ? storageSize(indices1) : 0ul);
                    cache.insert(key, PairIndices{indices0_xpu, indices1}, size);
                }
            } else {
                indices0_xpu = it->second.value.first;
                indices1 = it->second.value.second;
            }

            // Do the copy
            tracker<XPU0> _t("local copy", v0.ctx());
            copy_n<IndexType, T, Q>(1.0, v0.data(), indices0_xpu.begin(), v0.ctx(),
                                    indices0_xpu.size(), v1.data(), indices1.begin(), v1.ctx(),
                                    EWOp::Copy{});
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
        PackedValues<Q> pack(const std::vector<Proc_ranges<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v,
                             const Order<Nd0> &o0, const Order<Nd1> &o1, unsigned int ncomponents1,
                             MpiComm comm, CoorOrder co) {

            tracker<Cpu> _t("prepare and pack", Cpu{});

            unsigned int ncomponents0 = toSend.size();
            PackedValues<Q> r = prepare_pack<Nd0, Q>(toSend, comm);

            Indices<Cpu> buf_disp(comm.nprocs, Cpu{});
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank)
                buf_disp[rank] = r.displ[rank] * (MpiTypeSize / sizeof(Q));

            for (unsigned int componentId0 = 0; componentId0 < ncomponents0; ++componentId0) {
                for (const Component<Nd0, const T, XPU0> &c : v.first)
                    if (c.componentId == componentId0)
                        pack<Nd0, Nd1, T, Q>(o0, toSend[componentId0], c.dim, c.it, c.mask_it, o1,
                                             buf_disp.begin(), r.buf, ncomponents1, comm, co);
                for (const Component<Nd0, const T, XPU1> &c : v.second)
                    if (c.componentId == componentId0)
                        pack<Nd0, Nd1, T, Q>(o0, toSend[componentId0], c.dim, c.it, c.mask_it, o1,
                                             buf_disp.begin(), r.buf, ncomponents1, comm, co);
            }

            // Update the counts when using maskin
            if (v.first.size() > 0 && v.first[0].mask_it.size() > 0)
                for (unsigned int rank = 0; rank < comm.nprocs; ++rank)
                    r.counts[rank] = (buf_disp[rank] * sizeof(Q) + MpiTypeSize - 1) / MpiTypeSize -
                                     r.displ[rank];
            return r;
        }

        /// Vectors used in MPI communications
        template <typename T, typename XPU> struct UnpackedValues : public PackedValues<T> {
            std::vector<std::vector<Indices<XPU>>> indices; ///< indices of the destination elements
            std::vector<std::vector<IndexType>> indices_disp; ///< constant added to all indices
            UnpackedValues(vector<T, Cpu> buf, const std::vector<MpiInt> &counts,
                           const std::vector<MpiInt> &displ,
                           const std::vector<std::vector<Indices<XPU>>> &indices,
                           const std::vector<std::vector<IndexType>> &indices_disp)
                : PackedValues<T>{buf, counts, displ},
                  indices{indices},
                  indices_disp{indices_disp} {}
        };

        /// Allocate buffers for the receiving tensor pieces from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param ncomponents0: number of components on the origin tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order

        template <std::size_t Nd, typename T, typename XPU>
        UnpackedValues<T, XPU> prepare_unpack(const Proc_ranges<Nd> &toReceive,
                                              const Component<Nd, T, XPU> &v, MpiComm comm,
                                              CoorOrder co) {

            tracker<XPU> _t("prepare unpack", v.it.ctx());

            UnpackedValues<T, XPU> r{
                vector<T, Cpu>(),                                         // buf
                std::vector<MpiInt>(comm.nprocs, 0),                      // counts
                std::vector<MpiInt>(comm.nprocs, 0),                      // displ
                std::vector<std::vector<Indices<XPU>>>(toReceive.size()), // indices
                std::vector<std::vector<IndexType>>(toReceive.size())     // indices_disp
            };

            // Compute the destination indices and the total number of elements received from each process
            Order<Nd> o = trivial_order<Nd>();
            std::size_t ncomponents = toReceive.size() / comm.nprocs;
            for (std::size_t i = 0; i < toReceive.size(); ++i) {
                if (i / ncomponents == comm.rank) continue;
                for (const auto &fsi : toReceive[i]) {
                    Coor<Nd> fromi = fsi[0], sizei = fsi[1];
                    Indices<XPU> indices1;
                    IndexType disp;
                    get_permutation_destination_cache<Nd, Nd>(o, {}, sizei, sizei, o, fromi, v.dim,
                                                              v.it.ctx(), indices1, disp, co);

                    // Apply the masks
                    if (v.mask_it.size() > 0)
                        indices1 = select(indices1, v.mask_it.data() + disp, indices1);

                    // Store the number of permutation and the number of elements
                    r.counts[i / ncomponents] +=
                        (indices1.size() * sizeof(T) + MpiTypeSize - 1) / MpiTypeSize;
                    r.indices[i].push_back(indices1);
                    r.indices_disp[i].push_back(disp);
                }
            }

            // Compute the displacements
            r.displ[0] = 0;
            for (std::size_t i = 1; i < comm.nprocs; ++i)
                r.displ[i] = r.displ[i - 1] + r.counts[i - 1];

            // Allocate the buffer
            r.buf = vector<T, Cpu>((r.displ.back() + r.counts.back()) * (MpiTypeSize / sizeof(T)),
                                   Cpu{});

            return r;
        }

        /// Unpack and copy packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to packed tensors

        template <std::size_t Nd, typename T, typename XPU, typename EWOP>
        void unpack(UnpackedValues<T, XPU> &r, const Component<Nd, T, XPU> &v, MpiComm comm, EWOP,
                    typename elem<T>::type alpha) {

            tracker<XPU> _t("unpack", v.it.ctx());

            // Do the addition
            std::size_t ncomponents = r.indices.size() / comm.nprocs;
            for (std::size_t i = 0; i < r.indices.size(); ++i) {
                if (i / ncomponents == comm.rank) continue;
                for (unsigned int j = 0; j < r.indices[i].size(); j++) {
                    const T *data =
                        r.buf.data() + r.displ[i / ncomponents] * (MpiTypeSize / sizeof(T));
                    copy_n<IndexType, T, T>(alpha, data, Cpu{}, r.indices[i][j].size(),
                                            v.it.data() + r.indices_disp[i][j],
                                            r.indices[i][j].begin(), v.it.ctx(), EWOP{});
                    r.displ[i / ncomponents] +=
                        (r.indices[i][j].size() * sizeof(T) + MpiTypeSize - 1) / MpiTypeSize;
                }
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
        Request send_receive(const Order<Nd0> &o0, const std::vector<Proc_ranges<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, const Proc_ranges<Nd1> &toReceive,
                             const Component<Nd1, Q, XPUr> &v1, unsigned int ncomponents1,
                             MpiComm comm, EWOp ewop, CoorOrder co, typename elem<T>::type alpha) {

            tracker<Cpu> _t("packing", Cpu{});

            if (comm.nprocs <= 1) return [] {};

            // Pack v0 and prepare for receiving data from other processes
            std::shared_ptr<PackedValues<Q>> v0ToSend = std::make_shared<PackedValues<Q>>(
                pack<Nd0, Nd1, T, Q>(toSend, v0, o0, o1, ncomponents1, comm, co));
            std::shared_ptr<UnpackedValues<Q, XPUr>> v1ToReceive =
                std::make_shared<UnpackedValues<Q, XPUr>>(prepare_unpack(toReceive, v1, comm, co));

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
                tracker<Cpu> _t("alltoall", Cpu{});
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
                    tracker<Cpu> _t("alltoall", Cpu{});
                    MPI_Request r0 = r; // this copy avoid compiler warnings
                    MPI_check(MPI_Wait(&r0, MPI_STATUS_IGNORE));
                }

                // Do this copy is unnecessary, but v0ToSend needs to be captured to avoid
                // being released until this point
                std::shared_ptr<PackedValues<Q>> v0ToSend_dummy = v0ToSend;

                // Copy back to v1
                unpack(*v1ToReceive, v1, comm, ewop, Q(alpha));
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
        Request send_receive(const Order<Nd0> &o0, const std::vector<Proc_ranges<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, const Proc_ranges<Nd1> &toReceive,
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

        /// Return coor % dim
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions

        inline IndexType normalize_coor(IndexType coor, IndexType dim) {
            return (dim == 0 ? 0 : (coor + dim * (coor < 0 ? -coor / dim + 1 : 0)) % dim);
        }

        /// Return coor[i] % dim[i]
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions

        template <std::size_t Nd>
        Coor<Nd> normalize_coor(const Coor<Nd> &coor, const Coor<Nd> &dim) {
            Coor<Nd> r;
            for (std::size_t j = 0; j < Nd; j++) r[j] = normalize_coor(coor[j], dim[j]);
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
            fromr = from0 + std::min(std::max(from1 - from0, IndexType{0}), size0);
            sizer = from0 + std::min(std::max(from1 + size1 - from0, IndexType{0}), size0) - fromr;
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
        std::pair<std::array<From_size_item<Nd>, 2>, Coor<Nd>>
        intersection_aux(const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &from1,
                         const Coor<Nd> &size1, const Coor<Nd> &dim) {

            std::array<From_size_item<Nd>, 2> grid;
            Coor<Nd> grid_n{};
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
                    if (sizer1 == 0)
                        intersection(from0[i] + dim[i], size0[i], from1[i], size1[i], dim[i],
                                     fromr1, sizer1);
                }
                if (sizer0 > 0) {
                    grid[grid_n[i]][0][i] = fromr0;
                    grid[grid_n[i]++][1][i] = sizer0;
                }
                if (sizer1 > 0) {
                    grid[grid_n[i]][0][i] = fromr1;
                    grid[grid_n[i]++][1][i] = sizer1;
                }
            }
            return {grid, grid_n};
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
            auto p = intersection_aux<Nd>(from0, size0, from1, size1, dim);
            std::size_t vol = volume(p.second);
            if (vol == 0) {
                fromr = Coor<Nd>{};
                sizer = Coor<Nd>{};
            } else if (vol == 1) {
                fromr = p.first[0][0];
                sizer = p.first[0][1];
            } else {
                throw std::runtime_error("Not supported complex overlap of intervals");
            }
        }

        /// Return all ranges resulting from intersecting the two given ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice

        template <std::size_t Nd>
        From_size_out<Nd> intersection(const Coor<Nd> &from0, const Coor<Nd> &size0,
                                       const Coor<Nd> &from1, const Coor<Nd> &size1,
                                       const Coor<Nd> &dim) {
            auto p = intersection_aux<Nd>(from0, size0, from1, size1, dim);
            std::size_t vol = volume(p.second);
            if (vol == 0) {
                return {};
            } else if (vol == 1) {
                From_size_out<Nd> r(1, Cpu{});
                r[0] = p.first[0];
                return r;
            } else {
                From_size_out<Nd> r(vol, Cpu{});
                Coor<Nd> stride = get_strides<Nd>(p.second, FastToSlow);
                for (std::size_t i = 0; i < vol; ++i) {
                    Coor<Nd> c = index2coor<Nd>(i, p.second, stride);
                    for (std::size_t j = 0; j < Nd; ++j) {
                        r[i][0][j] = p.first[c[j]][0][j];
                        r[i][1][j] = p.first[c[j]][1][j];
                    }
                }
                return r;
            }
        }

        /// Return all ranges resulting from intersecting the two given ranges in a periodic lattice
        /// \param fs0: vector of first coordinate and size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice

        template <std::size_t Nd>
        From_size_out<Nd> intersection(const From_size_out<Nd> &fs0, const Coor<Nd> &from1,
                                       const Coor<Nd> &size1, const Coor<Nd> &dim) {
            vector<std::pair<std::array<From_size_item<Nd>, 2>, Coor<Nd>>, Cpu> p(fs0.size(),
                                                                                  Cpu{});
            std::size_t vol = 0;
            for (std::size_t i = 0; i < fs0.size(); ++i) {
                p[i] = intersection_aux<Nd>(fs0[i][0], fs0[i][1], from1, size1, dim);
                vol += volume(p[i].second);
            }
            From_size_out<Nd> r(vol, Cpu{});
            std::size_t ri = 0;
            for (std::size_t i = 0; i < fs0.size(); ++i) {
                Coor<Nd> stride = get_strides<Nd>(p[i].second, FastToSlow);
                for (std::size_t j = 0, j1 = volume(p[i].second); j < j1; ++j) {
                    Coor<Nd> c = index2coor<Nd>(j, p[i].second, stride);
                    for (std::size_t k = 0; k < Nd; ++k) {
                        r[ri][0][k] = p[i].first[c[k]][0][k];
                        r[ri][1][k] = p[i].first[c[k]][1][k];
                    }
                    ++ri;
                }
            }
            return r;
        }

        /// Shift a list of ranges
        /// \param fs0: vector of first coordinate and size of the ranges to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice
        /// \param perm: permutation of the coordinates

        template <std::size_t Nd>
        From_size_out<Nd> shift_ranges(const From_size_out<Nd> &fs, const Coor<Nd> &from,
                                       const Coor<Nd> &to, const Coor<Nd> &dim) {
            From_size_out<Nd> r(fs.size(), Cpu{});
            for (std::size_t i = 0; i < fs.size(); ++i) {
                r[i][0] = normalize_coor(fs[i][0] - from + to, dim);
                r[i][1] = fs[i][1];
            }
            return r;
        }

        /// Total volume of a list of ranges
        /// \param fs: vector of first coordinate and size of the ranges to translate

        template <std::size_t Nd> std::size_t volume(const From_size_out<Nd> &fs) {
            std::size_t vol = 0;
            for (const auto &fsi : fs) vol += volume(fsi[1]);
            return vol;
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

        /// Translate a range from one coordinate lattice to another
        /// \param fs0: vector of first coordinate and size of the ranges to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice
        /// \param perm: permutation of the coordinates

        template <std::size_t Nd0, std::size_t Nd1>
        From_size_out<Nd1> translate_range(const From_size_out<Nd0> &fs0, const Coor<Nd0> &from0,
                                           const Coor<Nd0> &dim0, const Coor<Nd1> &from1,
                                           const Coor<Nd1> &dim1, const Coor<Nd1> perm) {
            From_size_out<Nd1> r(fs0.size(), Cpu{});
            for (std::size_t i = 0; i < fs0.size(); ++i)
                translate_range<Nd0, Nd1>(fs0[i][0], fs0[i][1], from0, dim0, from1, dim1, perm,
                                          r[i][0], r[i][1]);
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
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <std::size_t Nd0, std::size_t Nd1>
        Proc_ranges<Nd0>
        get_indices_to_send(From_size<Nd0> p0, unsigned int from_rank, const Order<Nd0> &o0,
                            const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                            From_size<Nd1> p1, unsigned int componentId1, unsigned int ncomponents1,
                            const Order<Nd1> &o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1) {

            tracker<Cpu> _t("comp. tensor overlaps", p0.ctx());

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Restrict the local range in v0 to the range from0, size0
            Coor<Nd0> local_from0 = p0[from_rank][0];
            Coor<Nd0> local_size0 = p0[from_rank][1];
            From_size_out<Nd0> rlocal0 = intersection(from0, size0, local_from0, local_size0, dim0);

            // Translate the restricted range to the destination lattice
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            From_size_out<Nd1> rfs1 =
                translate_range<Nd0, Nd1>(rlocal0, from0, dim0, from1, dim1, perm0);

            // Compute the indices
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            unsigned int nprocs = p1.size() / ncomponents1;
            Proc_ranges<Nd0> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd1> &local_from1 = p1[i * ncomponents1 + componentId1][0];
                const Coor<Nd1> &local_size1 = p1[i * ncomponents1 + componentId1][1];
                r[i] = shift_ranges(
                    translate_range(intersection<Nd1>(rfs1, local_from1, local_size1, dim1), from1,
                                    dim1, from0, dim0, perm1),
                    local_from0, {}, dim0);
            }

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
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <std::size_t Nd0, std::size_t Nd1>
        Proc_ranges<Nd1> get_indices_to_receive(const From_size<Nd0> &p0, const Order<Nd0> &o0,
                                                const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                                                const Coor<Nd0> &dim0, const From_size<Nd1> &p1,
                                                unsigned int to_rank, const Order<Nd1> &o1,
                                                const Coor<Nd1> &from1, const Coor<Nd1> &dim1) {

            tracker<Cpu> _t("comp. tensor overlaps", p0.ctx());

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Restrict the local range in v1 to the range from1, size1
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1); // size in the destination
            Coor<Nd1> local_from1 = p1[to_rank][0];
            Coor<Nd1> local_size1 = p1[to_rank][1];
            From_size_out<Nd1> rlocal1 =
                intersection<Nd1>(from1, size1, local_from1, local_size1, dim1);

            // Translate the restricted range to the origin lattice
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            From_size_out<Nd0> rfs0 = translate_range(rlocal1, from1, dim1, from0, dim0, perm1);

            // Compute the indices
            unsigned int nprocs = p0.size();
            Proc_ranges<Nd1> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd0> &local_from0 = p0[i][0];
                const Coor<Nd0> &local_size0 = p0[i][1];
                r[i] = shift_ranges(
                    translate_range(intersection<Nd0>(rfs0, local_from0, local_size0, dim0), from0,
                                    dim0, from1, dim1, perm0),
                    local_from1, {}, dim1);
            }

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
                                                       const Coor<Nd> &dim, Cpu cpu, CoorOrder co,
                                                       MockFilling mf) {
                std::size_t vol = volume(size);
                vector<IndexType, Cpu> r(vol, cpu);

                if (mf == FillWithIndices) {
                    Coor<Nd> local_stride = get_strides(size, co);
                    Coor<Nd> stride = get_strides(dim, co);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
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
                copy_n<IndexType>(1, r_host.data(), r_host.ctx(), vol, r.data(), r.ctx(),
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
            get_mock_components(const From_size<Nd> &p, const Coor<Nd> &dim,
                                const Components_tmpl<Nd, T, XPU0, XPU1> &v, CoorOrder co,
                                MockFilling mf, Comm comm) {
                Components_tmpl<Nd, mockIndexType<T>, XPU0, XPU1> r;
                unsigned int ncomponents = v.first.size() + v.second.size();
                for (const Component<Nd, T, XPU0> &c : v.first) {
                    r.first.push_back(Component<Nd, IndexType, XPU0>{
                        get_mock_components(p[c.componentId + comm.rank * ncomponents][0], c.dim,
                                            dim, c.it.ctx(), co, mf),
                        c.dim, c.componentId, c.mask_it});
                }
                for (const Component<Nd, T, XPU1> &c : v.second) {
                    r.second.push_back(Component<Nd, IndexType, XPU1>{
                        get_mock_components(p[c.componentId + comm.rank * ncomponents][0], c.dim,
                                            dim, c.it.ctx(), co, mf),
                        c.dim, c.componentId, c.mask_it});
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
                vector<IndexType, Cpu> v_host = makeSure(v.it, Cpu{});
                vector<MaskType, Cpu> m_host = makeSure(v.mask_it, Cpu{});

#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
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
                        if (m_host.size() > 0 && m_host[i] == 0) true_val = 0;
                    }
                    if (v_host[i] != true_val)
                        throw std::runtime_error("test_copy_check does not pass!");
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
                           const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                           const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                           const From_size<Nd1> &p1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                           const Order<Nd1> &o1, const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1,
                           Comm comm, EWOP, CoorOrder co) {

                bool trackingTime = getTrackingTime();
                getTrackingTime() = false;

                // Fill the mock input and output tensors
                const Components_tmpl<Nd0, const IndexType, XPU0, XPU1> v0_ =
                    get_mock_components(p0, dim0, v0, co, FillWithIndices, comm);
                const Components_tmpl<Nd1, IndexType, XPU0, XPU1> v1_ =
                    get_mock_components(p1, dim1, v1, co, FillWithZeros, comm);

                // Copy the indices
                copy(1, p0, from0, size0, dim0, o0, v0_, p1, from1, dim1, o1, v1_, comm, EWOP{}, co,
                     false);

                // Check that the modified elements on v1_ are what they should be
                unsigned int ncomponents1 = v1.first.size() + v1.second.size();
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
                  const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                  const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
                  const Coor<Nd1> &from1, const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                  const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm, EWOp ewop, CoorOrder co,
                  bool do_test = true) {

            if (getDebugLevel() >= 2 && do_test) {
                ns_copy_test::test_copy(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1, o1,
                                        v1, comm, EWOp{}, co);
            }

            tracker<Cpu> _t("distributed copy", p0.ctx());

            // Check the dimensions of p0 and p1
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            unsigned int ncomponents1 = v1.first.size() + v1.second.size();

            if (p0.size() != ncomponents0 * comm.nprocs)
                throw std::runtime_error("Invalid number of elements in the tensor distribution");

            if (p1.size() != ncomponents1 * comm.nprocs)
                throw std::runtime_error("Invalid number of elements in the tensor distribution");

            // Check the compatibility of the tensors
            if (!check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1))
                throw std::runtime_error("Invalid copy operation");

            // Split the work for each receiving component
            std::vector<Request> reqs;
            for (unsigned int i = 0; i < ncomponents1; ++i) {
                for (const Component<Nd1, Q, XPU0> &c : v1.first) {
                    if (c.componentId == i)
                        reqs.push_back(copy<Nd0, Nd1, T, Q>(alpha, p0, from0, size0, dim0, o0, v0,
                                                            p1, ncomponents1, from1, dim1, o1, c,
                                                            comm, ewop, co));
                }
                for (const Component<Nd1, Q, XPU1> &c : v1.second) {
                    if (c.componentId == i)
                        reqs.push_back(copy<Nd0, Nd1, T, Q>(alpha, p0, from0, size0, dim0, o0, v0,
                                                            p1, ncomponents1, from1, dim1, o1, c,
                                                            comm, ewop, co));
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
                  const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                  const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
                  const Coor<Nd1> &from1, const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                  const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm, CopyAdd copyadd,
                  CoorOrder co) {

            if (getDebugLevel() >= 1) {
                barrier(comm);
                for (const auto &i : v1.first) sync(i.it.ctx());
                for (const auto &i : v1.second) sync(i.it.ctx());
            }

            switch (copyadd) {
            case Copy:
                copy(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1, o1, v1, comm,
                     EWOp::Copy{}, co);
                break;
            case Add:
                copy(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1, o1, v1, comm,
                     EWOp::Add{}, co);
                break;
            }

            if (getDebugLevel() >= 1) {
                for (const auto &i : v1.first) sync(i.it.ctx());
                for (const auto &i : v1.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        /// Return whether the distribution has overlaps with itself
        /// \param p: partitioning of the origin tensor in consecutive ranges
        /// \param from: first coordinate to consider
        /// \param size: number of elements to consider in each dimension

        template <std::size_t Nd>
        bool are_there_repetitions(const From_size<Nd> &p, const Coor<Nd> &from,
                                   const Coor<Nd> &size, const Coor<Nd> &dim) {

            tracker<Cpu> _t("are there repetitions", p.ctx());

            unsigned int nprocs = p.size();
            for (unsigned int i0 = 0; i0 < nprocs; ++i0) {
                // Restrict (from, size) to the p[i0] range
                Coor<Nd> fromi0, sizei0;
                intersection(from, size, p[i0][0], p[i0][1], dim, fromi0, sizei0);
                if (volume(sizei0) == 0) continue;

                // Intersect the range with p[i1] range and return if an overlap exists
                for (unsigned int i1 = i0 + 1; i1 < nprocs; ++i1)
                    if (intersection(p[i1][0], p[i1][1], fromi0, sizei0, dim).size() > 0)
                        return true;
            }

            return false;
        }

        /// Return whether the copy operation may need communications
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param ncomponents: length of p0
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied

        template <std::size_t Nd0, std::size_t Nd1, typename EWOP>
        bool may_need_communications(const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                                     const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                     const Order<Nd0> &o0, const From_size<Nd1> &p1,
                                     const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                                     const Order<Nd1> &o1, EWOP) {

            tracker<Cpu> _t("avoid communications", p0.ctx());

            // If the destination partitioning has repetitions, or the origin partitioning
            // has repetitions together with an add operation, then report that communications are needed
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1); // size in the destination
            if (are_there_repetitions(p1, from1, size1, dim1) ||
                (std::is_same<EWOP, EWOp::Add>::value &&
                 are_there_repetitions(p0, from0, size0, dim0)))
                return true;

            // Simple heuristic if
            unsigned int nprocs = p0.size();
            for (unsigned int i = 0; i < nprocs; ++i) {
                // Restrict (from0, size0) to the p0[i] range
                auto fs0 = intersection(from0, size0, p0[i][0], p0[i][1], dim0);

                // Translate the range to the destination tensor
                auto fs1 = translate_range(fs0, from0, dim0, from1, dim1, perm0);

                // Intersect the range with p1[i] range
                auto rfs1 = intersection(fs1, p1[i][0], p1[i][1], dim1);

                // If it is not a complete map, it means that some elements in p0[i] range
                // will go to other processes
                if (volume(fs0) != volume(rfs1)) return true;
            }

            return false;
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
                  typename XPU0, typename XPU1, typename XPU, typename EWOP>
        Request copy(typename elem<T>::type alpha, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                     const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                     const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
                     unsigned int ncomponents1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                     const Order<Nd1> &o1, const Component<Nd1, Q, XPU> &v1, Comm comm, EWOP ewop,
                     CoorOrder co) {

            // Find precomputed pieces on cache
            using Key = std::tuple<From_size<Nd0>, Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, From_size<Nd1>,
                                   Coor<Nd1>, Coor<Nd1>, PairPerms<Nd0, Nd1>, int, int>;
            struct Value {
                std::vector<Proc_ranges<Nd0>> toSend;
                Proc_ranges<Nd1> toReceive;
                bool need_comms;
            };
            struct cache_tag {};
            auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(p0.ctx());
            Key key{p0,           from0,
                    size0,        dim0,
                    p1,           from1,
                    dim1,         get_perms(o0, o1),
                    ncomponents1, std::is_same<EWOP, EWOp::Add>::value ? 1 : 0};
            auto it = cache.find(key);

            // Generate the list of subranges to send from each component from v0 to v1
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            std::vector<Proc_ranges<Nd0>> toSend;
            Proc_ranges<Nd1> toReceive;
            bool need_comms;
            if (it == cache.end()) {
                toSend = std::vector<Proc_ranges<Nd0>>(ncomponents0);
                for (unsigned int i = 0; i < v0.first.size(); ++i) {
                    toSend[v0.first[i].componentId] = get_indices_to_send<Nd0, Nd1>(
                        p0, comm.rank * ncomponents0 + v0.first[i].componentId, o0, from0, size0,
                        dim0, p1, v1.componentId, ncomponents1, o1, from1, dim1);
                }
                for (unsigned int i = 0; i < v0.second.size(); ++i) {
                    toSend[v0.second[i].componentId] = get_indices_to_send<Nd0, Nd1>(
                        p0, comm.rank * ncomponents0 + v0.second[i].componentId, o0, from0, size0,
                        dim0, p1, v1.componentId, ncomponents1, o1, from1, dim1);
                }

                // Generate the list of subranges to receive from each component from v0 to v1
                toReceive = get_indices_to_receive<Nd0, Nd1>(
                    p0, o0, from0, size0, dim0, p1, v1.componentId + comm.rank * ncomponents1, o1,
                    from1, dim1);

                // Check whether communications can be avoided
                need_comms = may_need_communications(p0, from0, size0, dim0, o0, p1, from1, dim1,
                                                     o1, EWOP{});

                // Save the results
                cache.insert(key, {toSend, toReceive, need_comms}, 0);
            } else {
                toSend = it->second.value.toSend;
                toReceive = it->second.value.toReceive;
                need_comms = it->second.value.need_comms;
            }

            // Do the sending and receiving
            Request mpi_req = [] {};
            if (need_comms)
                mpi_req = send_receive<Nd0, Nd1>(o0, toSend, v0, o1, toReceive, v1, ncomponents1,
                                                 comm, ewop, co, alpha);

            // Do the local copies
            Request local_req = [=] {
                unsigned int ncomponents0 = v0.first.size() + v0.second.size();
                for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                    assert(
                        toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1].size() ==
                        toReceive[c0.componentId + comm.rank * ncomponents0].size());
                    for (unsigned int
                             i = 0,
                             i1 = toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1]
                                      .size();
                         i < i1; ++i) {
                        assert(check_equivalence(
                            o0,
                            toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][i][1],
                            o1, toReceive[c0.componentId + comm.rank * ncomponents0][i][1]));
                        local_copy<Nd0, Nd1, T, Q>(
                            alpha, o0,
                            toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][i][0],
                            toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][i][1],
                            c0.dim, c0.it, c0.mask_it, o1,
                            toReceive[c0.componentId + comm.rank * ncomponents0][i][0], v1.dim,
                            v1.it, v1.mask_it, ewop, co);
                    }
                }
                for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                    assert(
                        toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1].size() ==
                        toReceive[c0.componentId + comm.rank * ncomponents0].size());
                    for (unsigned int
                             i = 0,
                             i1 = toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1]
                                      .size();
                         i < i1; ++i) {
                        assert(check_equivalence(
                            o0,
                            toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][i][1],
                            o1, toReceive[c0.componentId + comm.rank * ncomponents0][i][1]));
                        local_copy<Nd0, Nd1, T, Q>(
                            alpha, o0,
                            toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][i][0],
                            toSend[c0.componentId][v1.componentId + comm.rank * ncomponents1][i][1],
                            c0.dim, c0.it, c0.mask_it, o1,
                            toReceive[c0.componentId + comm.rank * ncomponents0][i][0], v1.dim,
                            v1.it, v1.mask_it, ewop, co);
                    }
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
                                 const Coor<Nd1> &dim1, const Order<Ndo> &o_r,
                                 bool report_inconsistencies = true) {
            std::map<char, IndexType> m;
            for (std::size_t i = 0; i < Nd0; ++i) m[o0[i]] = dim0[i];
            for (std::size_t i = 0; i < Nd1; ++i) {
                auto it = m.find(o1[i]);
                if (it == m.end())
                    m[o1[i]] = dim1[i];
                else if (report_inconsistencies && it->second != dim1[i])
                    throw std::runtime_error("Incompatible distributions for contraction");
            }
            Coor<Ndo> r;
            for (std::size_t i = 0; i < Ndo; ++i) r[i] = m[o_r[i]];
            return r;
        }

        /// Return value for the dimensions in o_r matching the given for o0 and o1

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        From_size_item<Ndo> get_dimensions(const Order<Nd0> &o0, From_size_item<Nd0> fs0,
                                           const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                                           From_size_item<Nd1> fs1, const Order<Ndo> &o_r) {

            for (std::size_t i0 = 0; i0 < Nd0; ++i0) {
                auto s1 = std::find(o1.begin(), o1.end(), o0[i0]);
                if (s1 != o1.end()) {
                    unsigned int i1 = s1 - o1.begin();
                    intersection(fs0[0][i0], fs0[1][i0], fs1[0][i1], fs1[1][i1], dim0[i0],
                                 fs0[0][i0], fs0[1][i0]);
                    fs1[0][i1] = fs0[0][i0];
                    fs1[1][i1] = fs0[1][i0];
                }
            }

            From_size_item<Ndo> fsr;

            for (std::size_t i0 = 0; i0 < Nd0; ++i0) {
                auto sr = std::find(o_r.begin(), o_r.end(), o0[i0]);
                if (sr != o_r.end()) {
                    unsigned int ir = sr - o_r.begin();
                    fsr[0][ir] = fs0[0][i0];
                    fsr[1][ir] = fs0[1][i0];
                }
            }

            for (std::size_t i1 = 0; i1 < Nd1; ++i1) {
                auto sr = std::find(o_r.begin(), o_r.end(), o1[i1]);
                if (sr != o_r.end()) {
                    unsigned int ir = sr - o_r.begin();
                    fsr[0][ir] = fs1[0][i1];
                    fsr[1][ir] = fs1[1][i1];
                }
            }

            return fsr;
        }

        /// Get the output partition
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        std::pair<From_size<Ndo>, Coor<Ndo>>
        get_output_partition(From_size<Nd0> p0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                             From_size<Nd1> p1, const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                             const Order<Ndo> &o_r, bool report_inconsistencies = true) {
            assert(p0.size() == p1.size());

            // Find partition on cache
            using Key = std::tuple<From_size<Nd0>, Coor<Nd0>, From_size<Nd1>, Coor<Nd1>,
                                   PairPerms<Nd0, Nd1>, PairPerms<Nd0, Ndo>, PairPerms<Nd1, Ndo>>;
            struct cache_tag {};
            auto cache =
                getCache<Key, std::pair<From_size<Ndo>, Coor<Ndo>>, TupleHash<Key>, cache_tag>(
                    p0.ctx());
            Key key{p0, dim0, p1, dim1, get_perms(o0, o1), get_perms(o0, o_r), get_perms(o1, o_r)};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Create partition
            From_size_out<Ndo> pr(p0.size(), p0.ctx());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                pr[i][0] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][0], o1, p1[i][0], o_r,
                                                         report_inconsistencies);
                pr[i][1] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][1], o1, p1[i][1], o_r,
                                                         report_inconsistencies);
            }
            Coor<Ndo> dimr =
                get_dimensions<Nd0, Nd1, Ndo>(o0, dim0, o1, dim1, o_r, report_inconsistencies);
            cache.insert(key, {pr, dimr}, storageSize(pr));

            return {pr, dimr};
        }

        /// Zeroed repeated results
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param dim0: dimension size for the operator
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param dim1: dimension size for the second operator
        /// \param o1: dimension labels for the second operator
        /// \param pr: partitioning of the output tensor in consecutive ranges
        /// \param dimr: dimension size for the output operator
        /// \param o_r: dimension labels for the output operator
        /// \param componentId: component to process
        /// \param v: data of the component on the output operator
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T, typename XPU>
        void zeroed_repated_tensor(From_size<Nd0> p0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                                   From_size<Nd1> p1, const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                                   From_size<Ndo> pr, const Coor<Ndo> &dimr, const Order<Ndo> &o_r,
                                   unsigned int componentId, vector<T, XPU> v, CoorOrder co) {

            assert(p0.size() == p1.size() && p1.size() == pr.size());

            for (unsigned int i = 0; i < componentId; ++i) {
                // Intersection of first, second, and output tensors of ith and componentId
                Coor<Nd0> from0, size0;
                intersection(p0[i][0], p0[i][1], p0[componentId][0], p0[componentId][1], dim0,
                             from0, size0);
                Coor<Nd1> from1, size1;
                intersection(p1[i][0], p1[i][1], p1[componentId][0], p1[componentId][1], dim1,
                             from1, size1);
                From_size_item<Ndo> fsr =
                    get_dimensions(o0, {from0, size0}, dim0, o1, {from1, size1}, o_r);
                Coor<Ndo> fromr, sizer;
                intersection(pr[i][0], pr[i][1], fsr[0], fsr[1], dimr, fromr, sizer);
                if (volume(sizer) == 0) continue;

                fromr = normalize_coor(fromr - pr[componentId][0], dimr);
                local_copy<Ndo, Ndo, T, T>(0, o_r, fromr, sizer, pr[componentId][1],
                                           (vector<const T, XPU>)v, {}, o_r, fromr,
                                           pr[componentId][1], v, {}, EWOp::Copy{}, co);
            }
        }

        /// Return a tensor with a given partitioning and ordering
        /// \param p0: partitioning of the input tensor
        /// \param o0: dimension labels for the input tensor
        /// \param v0: input tensor components
        /// \param p1: partitioning of the output tensor in consecutive ranges
        /// \param o1: dimension labels for the output tensor
        /// \param co: coordinate linearization order
        /// \param force_copy: whether to NOT avoid copy if the partition is the same

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        Components_tmpl<N, T, XPU0, XPU1>
        reorder_tensor(const From_size<N> &p0, const Order<N> &o0, const Coor<N> &from0,
                       const Coor<N> &size0, const Coor<N> &dim0,
                       const Components_tmpl<N, T, XPU0, XPU1> &v0, const From_size<N> &p1,
                       const Coor<N> &dim1, const Order<N> &o1, Comm comm, CoorOrder co,
                       bool force_copy = false) {

            // If the two orderings and partitions are equal, return the tensor
            if (!force_copy && from0 == Coor<N>{} && o0 == o1 && p0 == p1) return v0;

            // Allocate the tensor
            unsigned int ncomponents = v0.first.size() + v0.second.size();
            Components_tmpl<N, T, XPU0, XPU1> v1;
            for (unsigned int i = 0; i < v0.first.size(); ++i) {
                const unsigned int componentId = v0.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<N> &dimi = p1[pi][1];
                vector<T, XPU0> v1i(volume(dimi), v0.first[i].it.ctx());
                v1.first.push_back(Component<N, T, XPU0>{v1i, dimi, componentId, {}});
            }
            for (unsigned int i = 0; i < v0.second.size(); ++i) {
                const unsigned int componentId = v0.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<N> &dimi = p1[pi][1];
                vector<T, XPU1> v1i(volume(dimi), v0.second[i].it.ctx());
                v1.second.push_back(Component<N, T, XPU1>{v1i, dimi, componentId, {}});
            }

            // Copy the content of v0 into v1
            copy<N, N, T>(T{1}, p0, from0, size0, dim0, o0, toConst(v0), p1, {}, dim1, o1, v1, comm,
                          EWOp::Copy{}, co);

            return v1;
        }

        /// Check that the given components are compatible
        /// \param v0: components to test
        /// \param v1: components to test

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1>
        bool check_components_compatibility(const Components_tmpl<Nd0, T, XPU0, XPU1> &v0,
                                            const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1) {

            // Check that v0 and v1 have the same components and on the same device
            if (v0.first.size() != v1.first.size() || v0.second.size() != v1.second.size())
                return false;
            bool unmatch_dev = false;
            for (unsigned int i = 0; i < v0.first.size(); ++i)
                if (deviceId(v0.first[i].it.ctx()) != deviceId(v1.first[i].it.ctx()))
                    unmatch_dev = true;
            for (unsigned int i = 0; i < v0.second.size(); ++i)
                if (deviceId(v0.second[i].it.ctx()) != deviceId(v1.second[i].it.ctx()))
                    unmatch_dev = true;
            if (unmatch_dev) return false;

            return true;
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

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void contraction(T alpha, const From_size<Nd0> &p0, const Coor<Nd0> &dim0,
                         const Order<Nd0> &o0, bool conj0,
                         const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                         const From_size<Nd1> &p1, const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                         bool conj1, const Components_tmpl<Nd1, const T, XPU0, XPU1> &v1, T beta,
                         const From_size<Ndo> &pr, const Coor<Ndo> &dimr, const Order<Ndo> &o_r,
                         const Components_tmpl<Ndo, T, XPU0, XPU1> &vr, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vr.first) sync(i.it.ctx());
                for (const auto &i : vr.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed contraction", p0.ctx());

            // Check the compatibility of the tensors
            if (!check_dimensions<Nd0, Nd1, Ndo>(o0, dim0, o1, dim1, o_r, dimr))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (!check_components_compatibility(v0, v1))
                throw std::runtime_error(
                    "contraction: the two input tensors don't have the same number of components "
                    "or they don't follow the same order on the devices");

            // Get the optimal ordering for the output tensor pr_
            Order<Ndo> sug_or;
            {
                Order<Nd0> sug_o0;
                Order<Nd1> sug_o1;
                bool swap_operands;
                suggested_orders_for_contraction(o0, dim0, conj0, o1, dim1, conj1, o_r, dimr,
                                                 sug_o0, sug_o1, sug_or, swap_operands, co);
            }
            Coor<Ndo> sug_dimr = reorder_coor(dimr, find_permutation(o_r, sug_or));

            // Generate the partitioning and the storage for the output tensor
            unsigned int ncomponents = v0.first.size() + v0.second.size();
            From_size<Ndo> pr_ =
                get_output_partition<Nd0, Nd1, Ndo>(p0, dim0, o0, p1, dim1, o1, sug_or).first;
            Components_tmpl<Ndo, const T, XPU0, XPU1> vr_;
            std::vector<vector<T, XPU0>> vr0(v0.first.size());
            for (unsigned int i = 0; i < v0.first.size(); ++i) {
                const unsigned int componentId = v0.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Ndo> &dimi = pr_[pi][1];
                vr0[i] = vector<T, XPU0>(volume<Ndo>(dimi), v0.first[i].it.ctx());
                vr_.first.push_back(Component<Ndo, T, XPU0>{vr0[i], dimi, componentId, {}});
                local_contraction<Nd0, Nd1, Ndo, T>(alpha, o0, p0[pi][1], conj0, v0.first[i].it, o1,
                                                    p1[pi][1], conj1, v1.first[i].it, T{0.0},
                                                    sug_or, dimi, vr0[i], co);
                zeroed_repated_tensor(p0, dim0, o0, p1, dim1, o1, pr_, sug_dimr, sug_or, pi, vr0[i],
                                      co);
            }
            std::vector<vector<T, XPU1>> vr1(v0.second.size());
            for (unsigned int i = 0; i < v0.second.size(); ++i) {
                const unsigned int componentId = v0.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Ndo> &dimi = pr_[pi][1];
                vr1[i] = vector<T, XPU1>(volume<Ndo>(dimi), v0.second[i].it.ctx());
                vr_.second.push_back(Component<Ndo, T, XPU1>{vr1[i], dimi, componentId, {}});
                local_contraction<Nd0, Nd1, Ndo, T>(alpha, o0, p0[pi][1], conj0, v0.second[i].it,
                                                    o1, p1[pi][1], conj1, v1.second[i].it, T{0.0},
                                                    sug_or, dimi, vr1[i], co);
                zeroed_repated_tensor(p0, dim0, o0, p1, dim1, o1, pr_, sug_dimr, sug_or, pi, vr1[i],
                                      co);
            }

            // Scale the output tensor by beta
            copy<Ndo, Ndo, T>(beta, pr, {}, dimr, dimr, o_r, toConst(vr), pr, {}, dimr, o_r, vr,
                              comm, EWOp::Copy{}, co);

            // Scale the output tensor by beta and reduce all the subtensors to the final tensor
            copy<Ndo, Ndo, T>(1, pr_, {}, sug_dimr, sug_dimr, sug_or, vr_, pr, {}, dimr, o_r, vr,
                              comm, EWOp::Add{}, co);

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
        From_size<Nd> get_from_size(const PartitionItem<Nd> *p, std::size_t n, Session session) {
            if (Nd == 0) return to_vector(p = nullptr, n, Cpu{session});
            return clone(to_vector(p, n, Cpu{session}));
        }
    }

    /// Return a partitioning for a tensor of `dim` dimension onto a grid of processes
    /// \param dim1: dimension size for the tensor
    /// \param procs: number of processes on each dimension
    /// \param nprocs: (optional) total number of processes; if not given or it is less than the zero,
    ///                it will be the product of all elements in `procs`
    /// \param replicate: (optional) if true and the total processes of `procs` is one, then replicate
    ///                   the support of the tensor on every process

    template <std::size_t Nd>
    std::vector<PartitionItem<Nd>> basic_partitioning(Coor<Nd> dim, Coor<Nd> procs, int nprocs = -1,
                                                      bool replicate = false) {
        int vol_procs = (int)detail::volume<Nd>(procs);
        if (nprocs >= 0 && vol_procs > nprocs)
            std::runtime_error(
                "The total number of processes from `procs` is greater than `nprocs`");
        std::vector<PartitionItem<Nd>> fs(nprocs < 0 ? vol_procs : nprocs);
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
        if (replicate && vol_procs == 1)
            for (auto &fsi : fs) fsi = fs[0];
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
    /// \param mask0: vector of mask pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param mask1: vector of mask pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
              const T **v0, const MaskType **mask0, const Context *ctx0,
              const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1,
              const Coor<Nd1> &from1, const Coor<Nd1> &dim1, Q **v1, const MaskType **mask1,
              const Context *ctx1, MPI_Comm mpicomm, CoorOrder co, CopyAdd copyadd,
              Session session = 0) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::copy<Nd0, Nd1>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, mask0, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1, dim1,
            detail::toArray<Nd1>(o1, "o1"),
            detail::get_components<Nd1>(v1, mask1, ctx1, ncomponents1, p1, comm, session), comm,
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
    /// \param dim0: dimension size for the origin tensor
    /// \param v0: vector of data pointers for the origin tensor
    /// \param data0: vector of mask pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param dim1: dimension size for the destination tensor
    /// \param v1: vector of data pointers for the origin tensor
    /// \param mask1: vector of mask pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> from0, const Coor<Nd0> size0, const Coor<Nd0> dim0,
              const T **v0, const MaskType **mask0, const Context *ctx0,
              const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1, const Coor<Nd1> from1,
              const Coor<Nd1> dim1, Q **v1, const MaskType **mask1, const Context *ctx1,
              CoorOrder co, CopyAdd copyadd, Session session = 0) {

        detail::SelfComm comm = detail::get_comm();

        detail::copy<Nd0, Nd1>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, mask0, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1, dim1,
            detail::toArray<Nd1>(o1, "o1"),
            detail::get_components<Nd1>(v1, mask1, ctx1, ncomponents1, p1, comm, session), comm,
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
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T alpha, const PartitionItem<Nd0> *p0, const Coor<Nd0> &dim0, int ncomponents0,
                     const char *o0, bool conj0, const T **v0, const Context *ctx0,
                     const PartitionItem<Nd1> *p1, const Coor<Nd1> &dim1, int ncomponents1,
                     const char *o1, bool conj1, const T **v1, const Context *ctx1, T beta,
                     const PartitionItem<Ndo> *pr, const Coor<Ndo> &dimr, int ncomponentsr,
                     const char *o_r, T **vr, const Context *ctxr, MPI_Comm mpicomm, CoorOrder co,
                     Session session = 0) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::contraction<Nd0, Nd1, Ndo>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), dim0, o0_, conj0,
            detail::get_components<Nd0>(v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), dim1, o1_, conj1,
            detail::get_components<Nd1>(v1, nullptr, ctx1, ncomponents1, p1, comm, session), beta,
            detail::get_from_size(pr, ncomponentsr * comm.nprocs, session), dimr, o_r_,
            detail::get_components<Ndo>(vr, nullptr, ctxr, ncomponentsr, pr, comm, session), comm,
            co);
    }

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<!detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T, const PartitionItem<Nd0> *, const Coor<Nd0> &, int, const char *, bool,
                     const T **, const Context *, const PartitionItem<Nd1> *, const Coor<Nd1> &,
                     int, const char *, bool, const T **, const Context *, T,
                     const PartitionItem<Ndo> *, const Coor<Ndo> &, int, const char o_r, T **,
                     const Context *, MPI_Comm, CoorOrder, Session = 0) {
        throw std::runtime_error("contraction: unsupported type");
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
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T alpha, const PartitionItem<Nd0> *p0, const Coor<Nd0> &dim0, int ncomponents0,
                     const char *o0, bool conj0, const T **v0, const Context *ctx0,
                     const PartitionItem<Nd1> *p1, const Coor<Nd1> &dim1, int ncomponents1,
                     const char *o1, bool conj1, const T **v1, const Context *ctx1, T beta,
                     const PartitionItem<Ndo> *pr, const Coor<Ndo> &dimr, int ncomponentsr,
                     const char *o_r, T **vr, const Context *ctxr, CoorOrder co,
                     Session session = 0) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::SelfComm comm = detail::get_comm();

        detail::contraction<Nd0, Nd1, Ndo>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), dim0, o0_, conj0,
            detail::get_components<Nd0>(v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), dim1, o1_, conj1,
            detail::get_components<Nd1>(v1, nullptr, ctx1, ncomponents1, p1, comm, session), beta,
            detail::get_from_size(pr, ncomponentsr * comm.nprocs, session), dimr, o_r_,
            detail::get_components<Ndo>(vr, nullptr, ctxr, ncomponentsr, pr, comm, session), comm,
            co);
    }

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<!detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T, const PartitionItem<Nd0> *, const Coor<Nd0> &, int, const char *, bool,
                     const T **, const Context *, const PartitionItem<Nd1> *, const Coor<Nd1> &,
                     int, const char *, bool, const T **, const Context *, T,
                     const PartitionItem<Ndo> *, const Coor<Ndo> &, int, const char o_r, T **,
                     const Context *, CoorOrder, Session = 0) {
        throw std::runtime_error("contraction: unsupported type");
    }
}

#endif //  __SUPERBBLAS_DIST__
