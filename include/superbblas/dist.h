#ifndef __SUPERBBLAS_DIST__
#define __SUPERBBLAS_DIST__

#include "tensor.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef SUPERBBLAS_CREATING_FLAGS
#    ifdef SUPERBBLAS_USE_MPI
EMIT_define(SUPERBBLAS_USE_MPI)
#    endif
#endif

#ifdef SUPERBBLAS_USE_MPI
#    include "mpi.h"

// If using Open-MPI check if supporting GPU aware API
#    if OMPI_MAJOR_VERSION >= 4 || MPICH_NUMVERSION >= 30201000
#        include "mpi-ext.h"
#    endif // OMPI_MAJOR_VERSION >= 4
#    if defined(SUPERBBLAS_USE_CUDA) &&                                                            \
        (defined(MPIX_CUDA_AWARE_SUPPORT) ||                                                       \
         (defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA))
#        define SUPERBBLAS_TEST_MPI_GPU
#    endif
#    if defined(SUPERBBLAS_USE_HIP) &&                                                             \
        (defined(MPIX_ROCM_AWARE_SUPPORT) ||                                                       \
         (defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM))
#        define SUPERBBLAS_TEST_MPI_GPU
#    endif
#endif // SUPERBBLAS_USE_MPI

#ifdef SUPERBBLAS_CREATING_LIB
#    ifdef SUPERBBLAS_USE_MPI
#        define COMMS detail::SelfComm, detail::MpiComm
#    else
#        define COMMS detail::SelfComm
#    endif

#    ifdef SUPERBBLAS_USE_GPU
#        define XPUS_COMP detail::Gpu detail::Cpu
#    else
#        define XPUS_COMP detail::Cpu detail::Cpu
#    endif

/// Generate template instantiations for copy_request functions with template parameters T and Q

#    define DECL_COPY_REQUEST_T_Q(...)                                                             \
        EMIT REPLACE1(copy_request,                                                                \
                      superbblas::detail::copy_request<Nd, T, Q, Comm, XPU0, XPU1, EWOP>)          \
            REPLACE(Nd, COOR_DIMS) REPLACE_T_Q REPLACE(Comm, COMMS) REPLACE(XPU0 XPU1, XPUS_COMP)  \
                REPLACE_EWOP template __VA_ARGS__;

#else
#    define DECL_COPY_REQUEST_T_Q(...) __VA_ARGS__
#endif

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

    /// Callback to execute to finish an operation
    using Request = std::function<void(void)>;

    /// Wait until the operation is finished
    /// \param request: operation to finish

    inline void wait(const Request &request) {
        if (request) request();
    }

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

#ifdef SUPERBBLAS_USE_MPI
        /// Communicator
        struct MpiComm {
            unsigned int nprocs; ///< Number of processes
            unsigned int rank;   ///< Process id
            MPI_Comm comm;       ///< MPI communicator
        };

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

            Component withNewContext(const XPU &xpu) const {
                return {it.withNewContext(xpu), dim, componentId, mask_it.withNewContext(xpu)};
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
        Ostream &operator<<(Ostream &s, const std::array<T, N> &v) {
            s << "{";
            for (const auto &i : v) s << " " << i;
            s << "}";
            return s;
        }

        template <typename Ostream, typename T>
        Ostream &operator<<(Ostream &s, const vector<T, Cpu> &v) {
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
            for (std::size_t i = 0; i < N; i++) r[i] = (char)(i + 1);
            return r;
        }

        /// Throw an error if not all processes give the same value
        /// \param t: value to test
        /// \param comm: communicator
        ///
        /// NOTE: the no MPI version does nothing

        template <typename T, typename H = Hash<T>>
        void check_consistency(const T &, const SelfComm &) {}

#ifdef SUPERBBLAS_USE_MPI
        /// Communication barrier

        inline void barrier(MpiComm comm) { MPI_check(MPI_Barrier(comm.comm)); }

        template <typename T, typename H = Hash<T>>
        void check_consistency(const T &t, const MpiComm &comm) {
            if (getDebugLevel() == 0 || comm.nprocs == 1) return;
            const std::size_t h0 = H::hash(t) + (std::size_t)comm.nprocs;
            std::size_t h = h0;
            MPI_check(MPI_Bcast(&h, sizeof(h) / sizeof(int), MPI_INT, 0, comm.comm));
            if (h0 != h) std::runtime_error("check_consistency failed!");
        }

        /// Vectors used in MPI communications
        template <typename T, typename XPUbuff> struct PackedValues {
            vector<T, XPUbuff> buf;     ///< pointer to data
            vector<MpiInt, Cpu> counts; ///< number of items send/receive for rank i
            vector<MpiInt, Cpu> displ;  ///< index of the first element to send/receive for rank i
        };

        /// Allocate buffers and prepare arrays from a list of ranges to be used in a MPI communication
        /// \param ranges: iterator over a list of tensor ranges to be packed
        /// \param nranges: number of elements in the list
        /// \param ncomponents: comm.nprocs * ncomponents == the length of each element in `ranges`
        /// \param comm: communicator

        template <typename T, typename XPUbuff, std::size_t Nd>
        PackedValues<T, XPUbuff> prepare_pack(const std::vector<Proc_ranges<Nd>> &toSend,
                                              MpiComm comm, XPUbuff xpu) {

            // Allocate PackedValues
            static_assert(MpiTypeSize % sizeof(T) == 0,
                          "Please change MpiTypeSize to be a power of two!");

            // Prepare counts and displ
            vector<MpiInt, Cpu> counts(comm.nprocs, Cpu{});
            vector<MpiInt, Cpu> displ(comm.nprocs, Cpu{});
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
                counts[rank] = (n_rank * sizeof(T) + MpiTypeSize - 1) / MpiTypeSize;
                displ[rank] = d;
                d += counts[rank];
            }
            if (d * MpiTypeSize != n * sizeof(T))
                throw std::runtime_error(
                    "Exceeded the maximum package size: increase `MpiTypeSize`");

            // NOTE: MPI calls may have problems passing null pointers as buffers
            if (n == 0) n = MpiTypeSize / sizeof(T);

            vector<T, XPUbuff> buf(n, xpu, doCacheAlloc, MpiTypeSize);

            return PackedValues<T, XPUbuff>{buf, counts, displ};
        }

        /// Return the common blocksize given a list of ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param size0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param ncomponents1: number of components
        /// \param comm: communicator
        /// \param co: coordinate linearization order
        /// \param nblock: (out) the first `nblock` dimensions are equivalent to a trivial permutation
        /// \param blocksize: (out) the volume of the first `nblock` dimensions or one at least

        template <std::size_t Nd0, std::size_t Nd1>
        void get_block_size_for_copy_normalize(const Order<Nd0> &o0, const Proc_ranges<Nd0> &toSend,
                                               const Coor<Nd0> &size0, const Order<Nd1> &o1,
                                               unsigned int ncomponents1, MpiComm comm,
                                               CoorOrder co,
                                               // Output
                                               std::size_t &nblock, std::size_t &blocksize) {

            assert(toSend.size() == comm.nprocs * ncomponents1);

            // Quick exit for zero volume
            nblock = 0;
            blocksize = 1;
            if (volume(size0) == 0) return;

            Coor<Nd1> perm0 = find_permutation(o0, o1);
            nblock = std::min(Nd0, Nd1);
            if (co == FastToSlow) {
                for (std::size_t toSendi = 0; toSendi < toSend.size(); ++toSendi) {
                    // Skip the communications of the local rank
                    if (toSendi / ncomponents1 == comm.rank) continue;

                    for (const auto &fs : toSend[toSendi]) {
                        if (volume(fs[1]) == 0) continue;
                        std::size_t i = 0;
                        for (std::size_t i1 = 0; i1 < Nd1; ++i1) {
                            superbblas::IndexType i0 = perm0[i1];
                            if (i0 < 0) continue;
                            if ((std::size_t)i0 != i) break;
                            if (i >= nblock) break;
                            if (fs[0][i0] != 0 || fs[1][i0] != size0[i0]) break;
                            ++i;
                        }
                        nblock = i;
                    }
                }
                for (std::size_t i = 0; i < nblock; ++i) blocksize *= size0[i];
                std::size_t compress_nblock = 0;
                for (std::size_t i = 0; i < nblock; ++i)
                    if (size0[i] > 1) ++compress_nblock;
                nblock = compress_nblock;
            } else {
                for (std::size_t toSendi = 0; toSendi < toSend.size(); ++toSendi) {
                    // Skip the communications of the local rank
                    if (toSendi / ncomponents1 == comm.rank) continue;

                    for (const auto &fs : toSend[toSendi]) {
                        if (volume(fs[1]) == 0) continue;
                        std::size_t i = 0;
                        for (int i1 = (int)Nd1 - 1; i1 >= 0; --i1) {
                            superbblas::IndexType i0 = perm0[i1];
                            if (i0 < 0) continue;
                            if ((std::size_t)i0 != Nd0 - i - 1) break;
                            if (i >= nblock) break;
                            if (fs[0][i0] != 0 || fs[1][i0] != size0[i0]) break;
                            ++i;
                        }
                        nblock = i;
                    }
                }
                for (std::size_t i = 0; i < nblock; ++i) blocksize *= size0[Nd0 - i - 1];
                std::size_t compress_nblock = 0;
                for (std::size_t i = 0; i < nblock; ++i)
                    if (size0[Nd0 - i - 1] > 1) ++compress_nblock;
                nblock = compress_nblock;
            }
        }

        /// Pack a list of subtensors contiguously in memory
        /// \param o0: dimension labels for the origin tensor
        /// \param fs: a From_size iterator
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param comm: communicator
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1, typename T, typename Q,
                  typename XPU0, typename XPUbuff>
        void pack_component(const Order<Nd0> &o0, const Proc_ranges<Nd0> &fs, const Coor<Nd0> &dim0,
                            vector<const T, XPU0> v0, Mask<XPU0> mask0, const Order<Nd1> &o1,
                            Indices<Cpu> &disp1, vector<Q, XPUbuff> &v1, MpiComm comm,
                            CoorOrder co) {

            assert(fs.size() == comm.nprocs);

            // Find indices on cache
            using Key = std::tuple<Proc_ranges<Nd0>, Coor<Nd0>, PairPerms<Nd0, Nd1>, Indices<Cpu>,
                                   int, int, int, CoorOrder>;
            using Value = std::tuple<IndicesT<IndexType, XPU0>, IndicesT<IndexType, XPUbuff>,
                                     size_t, Indices<Cpu>>;
            struct cache_tag {};
            auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(v0.ctx());
            Key key{fs,
                    dim0,
                    get_perms(o0, o1),
                    clone(disp1),
                    comm.rank,
                    deviceId(v0.ctx()),
                    deviceId(v1.ctx()),
                    co};
            auto it = mask0.size() == 0 ? cache.find(key) : cache.end();

            // If they are not, compute the permutation vectors
            IndicesT<IndexType, XPU0> indices0_xpu;
            IndicesT<IndexType, XPUbuff> indices1;
            std::size_t blocksize = 1;
            if (it == cache.end()) {
                tracker<XPU0> _t("comp. pack permutation", v0.ctx());

                // Figure out the common blocksize
                std::size_t nblock = 0;
                if (mask0.size() == 0)
                    get_block_size_for_copy_normalize(o0, fs, dim0, o1, 1, comm, co, nblock,
                                                      blocksize);

                // Get the maximum volume of communicated data without the local part
                std::size_t vol = 0;
                for (unsigned int i = 0; i < fs.size(); ++i)
                    if (i != comm.rank)
                        for (const auto &it : fs[i]) vol += volume(it[1]) / blocksize;

                Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
                IndicesT<IndexType, Cpu> indices0{vol, Cpu{}};
                IndicesT<IndexType, Cpu> indices1_cpu{vol, Cpu{}};
                Mask<Cpu> mask0_cpu = makeSure(mask0, Cpu{});
                std::size_t n = 0;
                for (std::size_t i = 0; i < fs.size(); ++i) {
                    // Skip the communications of the local rank
                    if (i == comm.rank) continue;

                    for (const auto &fsi : fs[i]) {
                        // Compute the permutation so that the subtensors are packed on the natural
                        // order on the destination; in other words, apply the permutation before
                        // doing the MPI call
                        Coor<Nd0> fromi = fsi[0], sizei = fsi[1];
                        Coor<Nd1> sizei1 = reorder_coor<Nd0, Nd1>(sizei, perm0, 1);
                        auto indices0i = get_permutation_origin<IndexType>(
                            o0, fromi, sizei, dim0, o1, {{}}, sizei1, DontAllowImplicitPermutation,
                            Cpu{}, co, nblock);
                        assert(indices0i.first.size() + n <= vol);
                        IndicesT<IndexType, Cpu> indices0i_mask = indices0i.first;
                        IndexType indices0i_disp = indices0i.second;
                        if (mask0_cpu.size() > 0)
                            indices0i_mask = select(
                                indices0i.first, mask0_cpu.data() + indices0i_disp, indices0i_mask);
                        std::transform(indices0i_mask.begin(), indices0i_mask.end(),
                                       indices0.begin() + n,
                                       [=](IndexType d) { return d + indices0i_disp; });

                        auto indices1i = get_permutation_destination<IndexType>(
                            o0, fromi, sizei, dim0, o1, {{}}, sizei1, DontAllowImplicitPermutation,
                            Cpu{}, co, nblock);
                        assert(indices0i.first.size() == indices1i.first.size());
                        IndicesT<IndexType, Cpu> indices1i_mask = indices1i.first;
                        IndexType indices1i_disp = indices1i.second;
                        if (mask0_cpu.size() > 0)
                            indices1i_mask = select(
                                indices0i.first, mask0_cpu.data() + indices0i_disp, indices1i_mask);
                        IndexType dispi = disp1[i] + indices1i_disp;
                        std::transform(indices1i_mask.begin(), indices1i_mask.end(),
                                       indices1_cpu.begin() + n,
                                       [=](IndexType d) { return d + dispi; });

                        disp1[i] += indices1i_mask.size() * blocksize;
                        n += indices1i_mask.size();
                        assert(n <= vol);
                    }
                }
                indices0.resize(n);
                indices1_cpu.resize(n);
                indices0_xpu = makeSure(indices0, v0.ctx());
                indices1 = makeSure(indices1_cpu, v1.ctx());

                // The cache trackers consider that all cache entries are on the same device; so just track the
                // indices0_xpu when using gpus
                if (mask0.size() == 0) {
                    std::size_t size =
                        storageSize(indices0_xpu) +
                        (deviceId(v0.ctx()) == deviceId(v1.ctx()) ? storageSize(indices1) : 0ul);
                    cache.insert(key,
                                 Value{archive(indices0_xpu), archive(indices1), blocksize,
                                       archive(clone(disp1))},
                                 size);
                }
            } else {
                indices0_xpu = std::get<0>(it->second.value);
                indices1 = std::get<1>(it->second.value);
                blocksize = std::get<2>(it->second.value);
                const auto new_disp1 = std::get<3>(it->second.value);
                std::copy_n(new_disp1.data(), new_disp1.size(), disp1.data());
            }

            // Do the copy
            tracker<XPUbuff> _t(std::string("local copy from ") + platformToStr(v0.ctx()) +
                                    std::string(" to ") + platformToStr(v1.ctx()),
                                v1.ctx());
            _t.cost = (double)sizeof(Q) * indices0_xpu.size() * blocksize;
            copy_n_blocking<IndexType, T, Q>(1.0, v0.data(), v0.ctx(), blocksize,
                                             indices0_xpu.begin(), indices0_xpu.ctx(),
                                             indices0_xpu.size(), v1.data(), v1.ctx(),
                                             indices1.begin(), indices1.ctx(), EWOp::Copy{});
        }

        /// Pack a list of ranges to be used in a MPI communication
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param ncomponents0: number of elements in toSend and v
        /// \param v: vector containing the values to send
        /// \param o0: dimension labels for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param comm: communicator
        /// \param co: coordinate linearization order

        template <typename IndexType, typename Q, std::size_t Nd0, std::size_t Nd1, typename T,
                  typename XPU0, typename XPU1, typename XPUbuff>
        PackedValues<Q, XPUbuff> pack(const std::vector<Proc_ranges<Nd0>> &toSend,
                                      const Components_tmpl<Nd0, const T, XPU0, XPU1> &v,
                                      const Order<Nd0> &o0, const Order<Nd1> &o1, MpiComm comm,
                                      XPUbuff xpu, CoorOrder co) {

            tracker<Cpu> _t("prepare and pack", Cpu{});

            unsigned int ncomponents0 = toSend.size();
            PackedValues<Q, XPUbuff> r = prepare_pack<Q>(toSend, comm, xpu);

            Indices<Cpu> buf_disp(comm.nprocs, Cpu{});
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank)
                buf_disp[rank] = r.displ[rank] * (MpiTypeSize / sizeof(Q));

            for (unsigned int componentId0 = 0; componentId0 < ncomponents0; ++componentId0) {
                for (const Component<Nd0, const T, XPU0> &c : v.first)
                    if (c.componentId == componentId0)
                        pack_component<IndexType>(o0, toSend[componentId0], c.dim, c.it, c.mask_it,
                                                  o1, buf_disp, r.buf, comm, co);
                for (const Component<Nd0, const T, XPU1> &c : v.second)
                    if (c.componentId == componentId0)
                        pack_component<IndexType>(o0, toSend[componentId0], c.dim, c.it, c.mask_it,
                                                  o1, buf_disp, r.buf, comm, co);
            }

            // Update the counts when using mask
            if (v.first.size() > 0 && v.first[0].mask_it.size() > 0)
                for (unsigned int rank = 0; rank < comm.nprocs; ++rank)
                    r.counts[rank] = (buf_disp[rank] * sizeof(Q) + MpiTypeSize - 1) / MpiTypeSize -
                                     r.displ[rank];
            return r;
        }

        /// Vectors used in MPI communications
        template <typename IndexType, typename T, typename XPUbuff, typename XPU>
        struct UnpackedValues : public PackedValues<T, XPUbuff> {
            IndicesT<IndexType, XPUbuff> indices_buf; ///< indices of the buffer
            IndicesT<IndexType, XPU> indices;         ///< indices of the destination elements
            IndicesT<IndexType, Cpu> indices_groups;  ///< number of indices to process at once
            std::size_t blocksize;                    ///< blocksize for block copying
            UnpackedValues(const vector<T, XPUbuff> &buf, const vector<MpiInt, Cpu> &counts,
                           const vector<MpiInt, Cpu> &displ,
                           const IndicesT<IndexType, XPUbuff> &indices_buf,
                           const IndicesT<IndexType, XPU> &indices,
                           const IndicesT<IndexType, Cpu> &indices_groups,
                           const std::size_t &blocksize)
                : PackedValues<T, XPUbuff>{buf, counts, displ},
                  indices_buf(indices_buf),
                  indices(indices),
                  indices_groups(indices_groups),
                  blocksize(blocksize) {}
        };

        /// Return whether some ranges to receive overlaps
        /// \param toReceive: list of tensor ranges to receive
        /// \param dim: dimensions of the destination tensor
        /// \param comm: communication

        template <std::size_t Nd>
        bool does_self_intersect(const Proc_ranges<Nd> &toReceive, const Coor<Nd> &dim,
                                 const MpiComm &comm) {

            std::size_t ncomponents = toReceive.size() / comm.nprocs;
            for (std::size_t i = 0; i < toReceive.size(); ++i) {
                if (i / ncomponents == comm.rank) continue;

                for (const auto &fsi : toReceive[i]) {
                    Coor<Nd> fromi = fsi[0], sizei = fsi[1];
                    for (std::size_t j = i + 1; j < toReceive.size(); ++j) {
                        if (j / ncomponents == comm.rank) continue;

                        if (volume(intersection(toReceive[j], fromi, sizei, dim)) > 0) return true;
                    }
                }
            }
            return false;
        }

        /// Allocate buffers for the receiving tensor pieces from a MPI communication
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param xpu: context for the buffer
        /// \param comm: communication
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd, typename T, typename XPU, typename XPUbuff,
                  typename EWOP>
        UnpackedValues<IndexType, T, XPUbuff, XPU>
        prepare_unpack(const Proc_ranges<Nd> &toReceive, const Component<Nd, T, XPU> &v,
                       XPUbuff xpu, const MpiComm &comm, CoorOrder co, EWOP) {

            tracker<XPU> _t("prepare unpack", v.it.ctx());

            // Find indices on cache
            using Key = std::tuple<Proc_ranges<Nd>, Coor<Nd>, int, int, int, int, CoorOrder>;
            using Value = std::tuple<vector<MpiInt, Cpu>,          // counts
                                     vector<MpiInt, Cpu>,          // displ
                                     IndicesT<IndexType, XPUbuff>, // indices for the buffer
                                     IndicesT<IndexType, XPU>,     // indices
                                     IndicesT<IndexType, Cpu>,     // number of indices to process
                                     std::size_t>;                 // blocksize
            struct cache_tag {};
            auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(v.it.ctx());
            Key key{toReceive, v.dim, comm.nprocs, comm.rank, deviceId(xpu), deviceId(v.it.ctx()),
                    co};
            auto it = v.mask_it.size() == 0 ? cache.find(key) : cache.end();

            // If they are not, compute the permutation vectors
            vector<MpiInt, Cpu> counts;
            vector<MpiInt, Cpu> displ;
            IndicesT<IndexType, XPUbuff> indices_buf;
            IndicesT<IndexType, XPU> indices;
            IndicesT<IndexType, Cpu> indices_groups;
            std::size_t blocksize = 1;
            if (it == cache.end()) {
                std::size_t ncomponents = toReceive.size() / comm.nprocs;
                assert(toReceive.size() == comm.nprocs * ncomponents);
                counts = vector<MpiInt, Cpu>(comm.nprocs, Cpu{});
                displ = vector<MpiInt, Cpu>(comm.nprocs, Cpu{});
                auto mask_cpu = makeSure(v.mask_it, Cpu{});

                // Figure out the common blocksize
                std::size_t nblock = 0;
                Order<Nd> o = trivial_order<Nd>();
                if (v.mask_it.size() == 0)
                    get_block_size_for_copy_normalize(o, toReceive, v.dim, o, ncomponents, comm, co,
                                                      nblock, blocksize);

                // Compute the destination indices and the total number of elements received from each process
                std::size_t num_elems = 0;
                for (std::size_t i = 0; i < comm.nprocs; ++i) counts[i] = 0;
                std::vector<std::size_t> n(comm.nprocs);
                std::vector<std::vector<IndicesT<IndexType, Cpu>>> indices0_groups(comm.nprocs);
                for (std::size_t i = 0; i < toReceive.size(); ++i) {
                    if (i / ncomponents == comm.rank) continue;

                    std::vector<IndicesT<IndexType, Cpu>> indices0;
                    for (const auto &fsi : toReceive[i]) {
                        Coor<Nd> fromi = fsi[0], sizei = fsi[1];
                        auto indices1_pair = get_permutation_destination<IndexType>(
                            o, {{}}, sizei, sizei, o, fromi, v.dim, DontAllowImplicitPermutation,
                            Cpu{}, co, nblock);
                        IndicesT<IndexType, Cpu> indices1 = indices1_pair.first;
                        IndexType disp = indices1_pair.second;

                        // Apply the masks
                        if (v.mask_it.size() > 0)
                            indices1 = select(indices1, mask_cpu.data() + disp, indices1);
                        else
                            indices1 = clone(indices1);

                        // Apply the displacement
                        std::for_each(indices1.begin(), indices1.end(),
                                      [=](IndexType &d) { d += disp; });

                        // Store the number of permutation and the number of elements
                        n[i / ncomponents] += indices1.size() * blocksize;
                        num_elems += indices1.size();
                        indices0_groups[i / ncomponents].push_back(indices1);
                    }
                }

                // Compute the counts
                for (std::size_t i = 0; i < comm.nprocs; ++i)
                    counts[i] = (n[i] * sizeof(T) + MpiTypeSize - 1) / MpiTypeSize;

                // Compute the displacements
                displ[0] = 0;
                for (std::size_t i = 1; i < comm.nprocs; ++i)
                    displ[i] = displ[i - 1] + counts[i - 1];

                // Create the permutation for the buffer
                IndicesT<IndexType, Cpu> indices_buf_cpu(num_elems, Cpu{});
                const std::size_t num_T = MpiTypeSize / sizeof(T);
                for (std::size_t i = 0, i_buf = 0, disp_buf = 0; i < indices0_groups.size(); ++i) {
                    std::size_t num_blocks = 0;
                    for (std::size_t ii = 0; ii < indices0_groups[i].size(); ++ii) {
                        for (IndexType j = 0, j1 = indices0_groups[i][ii].size(); j < j1; ++j)
                            indices_buf_cpu[i_buf++] = disp_buf + (num_blocks++) * blocksize;
                    }
                    disp_buf += (num_blocks * blocksize + num_T - 1) / num_T * num_T;
                }
                indices_buf = makeSure(indices_buf_cpu, xpu);

                // Concatenate all indices into a single permutation vector
                IndicesT<IndexType, Cpu> indices_cpu(num_elems, Cpu{});
                {
                    std::size_t i0 = 0;
                    for (const auto &indices0 : indices0_groups) {
                        for (const auto &indices : indices0) {
                            copy_n<IndexType>(indices.data(), Cpu{}, indices.size(),
                                              indices_cpu.data() + i0, Cpu{});
                            i0 += indices.size();
                        }
                    }
                }

                indices = makeSure(indices_cpu, v.it.ctx());

                // If EWOP is addition and the toReceive ranges intersect, then copy_n may result
                // in undefined behaviour as several threads may add on the same destination element
                if (std::is_same<EWOP, EWOp::Add>::value &&
                    does_self_intersect(toReceive, v.dim, comm)) {
                    std::size_t num_groups = 0;
                    for (const auto &indices0 : indices0_groups) num_groups += indices0.size();
                    indices_groups = IndicesT<IndexType, Cpu>(num_groups, Cpu{});
                    std::size_t i0 = 0;
                    for (const auto &indices0 : indices0_groups)
                        for (const auto &indices : indices0) indices_groups[i0++] = indices.size();
                } else {
                    indices_groups = IndicesT<IndexType, Cpu>(1, Cpu{});
                    indices_groups[0] = indices.size();
                }

                if (v.mask_it.size() == 0) {
                    std::size_t size = storageSize(indices_buf) + storageSize(indices);
                    cache.insert(key,
                                 Value{archive(counts), archive(displ), archive(indices_buf),
                                       archive(indices), archive(indices_groups), blocksize},
                                 size);
                }
            } else {
                counts = std::get<0>(it->second.value);
                displ = std::get<1>(it->second.value);
                indices_buf = std::get<2>(it->second.value);
                indices = std::get<3>(it->second.value);
                indices_groups = std::get<4>(it->second.value);
                blocksize = std::get<5>(it->second.value);
            }

            std::size_t buf_count = (displ.back() + counts.back()) * (MpiTypeSize / sizeof(T));

            // NOTE: MPI calls may have problems passing null pointers as buffers
            if (buf_count == 0) buf_count = MpiTypeSize / sizeof(T);

            // Allocate the buffer
            vector<T, XPUbuff> buf(buf_count, xpu, doCacheAlloc, MpiTypeSize);

            return UnpackedValues<IndexType, T, XPUbuff, XPU>{
                buf, counts, displ, indices_buf, indices, indices_groups, blocksize};
        }

        /// Unpack and copy packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to packed tensors

        template <typename IndexType, std::size_t Nd, typename T, typename XPUbuff, typename XPU,
                  typename EWOP>
        void unpack(const UnpackedValues<IndexType, T, XPUbuff, XPU> &r,
                    const Component<Nd, T, XPU> &v, EWOP, typename elem<T>::type alpha) {

            tracker<XPU> _t(std::string("unpack from ") + platformToStr(r.buf.ctx()) +
                                std::string(" to ") + platformToStr(v.it.ctx()),
                            v.it.ctx());

            // Transfer the buffer to the destination device
            IndexType disp = 0;
            for (unsigned int i = 0, i1 = r.indices_groups.size(); i < i1; ++i) {
                copy_n_blocking<IndexType, T, T>(alpha, r.buf.data(), r.buf.ctx(), r.blocksize,
                                                 r.indices_buf.data() + disp, r.indices_buf.ctx(),
                                                 r.indices_groups[i], v.it.data(), v.it.ctx(),
                                                 r.indices.data() + disp, r.indices.ctx(), EWOP{});
                disp += r.indices_groups[i];
            }
            _t.cost = (double)sizeof(T) * r.indices.size() * r.blocksize;
        }

        /// Return a counter, used by `send_receive`

        inline std::size_t &getSendReceiveCallNumer() {
            static std::size_t call_number = 0;
            return call_number;
        }

        /// Asynchronous sending and receiving
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param xpubuff1: context to hold the mpi receiver buffer
        /// \param v1: destination data
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <typename IndexType, typename XPUbuff0, typename XPUbuff1, std::size_t Nd0,
                  std::size_t Nd1, typename T, typename Q, typename XPU0, typename XPU1,
                  typename XPUr, typename EWOP>
        Request send_receive(const Order<Nd0> &o0, const std::vector<Proc_ranges<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, XPUbuff0 xpubuff0,
                             const Order<Nd1> &o1, const Proc_ranges<Nd1> &toReceive,
                             const Component<Nd1, Q, XPUr> &v1, XPUbuff1 xpubuff1, MpiComm comm,
                             EWOP, CoorOrder co, typename elem<T>::type alpha) {

            if (comm.nprocs <= 1) return [] {};

            // Annotate the calls so that the returned lambda can be paired with this call
            std::size_t call_number = ++getSendReceiveCallNumer();

            struct tag_type {}; // For hashing template arguments
            if (getDebugLevel() > 0) {
                check_consistency(std::make_tuple(std::string("send_receive"), call_number, o0, o1,
                                                  co, alpha, typeid(tag_type).hash_code()),
                                  comm);
            }

            tracker<Cpu> _t("packing", Cpu{});

            // Pack v0 and prepare for receiving data from other processes
            PackedValues<Q, XPUbuff0> v0ToSend =
                pack<IndexType, Q>(toSend, v0, o0, o1, comm, xpubuff0, co);
            UnpackedValues<IndexType, Q, XPUbuff1, XPUr> v1ToReceive =
                prepare_unpack<IndexType>(toReceive, v1, xpubuff1, comm, co, EWOP{});

            // Do a ton of checking
            static MPI_Datatype dtype = get_mpi_datatype();
            assert(v0ToSend.counts.size() == comm.nprocs);
            assert(v0ToSend.displ.size() == comm.nprocs);
            assert(v1ToReceive.counts.size() == comm.nprocs);
            assert(v1ToReceive.displ.size() == comm.nprocs);
            int dtype_size = 0;
            MPI_check(MPI_Type_size(dtype, &dtype_size));
            (void)dtype_size;
            assert((std::size_t)dtype_size == MpiTypeSize);
            assert((v0ToSend.displ.back() + v0ToSend.counts.back()) * MpiTypeSize <=
                   v0ToSend.buf.size() * sizeof(Q));
            assert((v1ToReceive.displ.back() + v1ToReceive.counts.back()) * MpiTypeSize <=
                   v1ToReceive.buf.size() * sizeof(Q));
            assert(v0ToSend.counts[comm.rank] == 0);
            assert(v1ToReceive.counts[comm.rank] == 0);
            assert(align(dtype_size, v0ToSend.buf.size() * sizeof(T), v0ToSend.buf.data(),
                         v0ToSend.buf.size() * sizeof(T)) == v0ToSend.buf.data());
            assert(align(dtype_size, v1ToReceive.buf.size() * sizeof(Q), v1ToReceive.buf.data(),
                         v1ToReceive.buf.size() * sizeof(Q)) == v1ToReceive.buf.data());
            if (getDebugLevel() > 0) {
                // Check that all processes agree in the amount of data to send/receive
                std::vector<int> send_counts(comm.rank == 0 ? comm.nprocs * comm.nprocs : 0);
                MPI_check(MPI_Gather(v0ToSend.counts.data(), comm.nprocs, MPI_INT,
                                     send_counts.data(), comm.nprocs, MPI_INT, 0, comm.comm));
                std::vector<int> recv_counts(comm.rank == 0 ? comm.nprocs * comm.nprocs : 0);
                MPI_check(MPI_Gather(v1ToReceive.counts.data(), comm.nprocs, MPI_INT,
                                     recv_counts.data(), comm.nprocs, MPI_INT, 0, comm.comm));
                if (comm.rank == 0)
                    for (unsigned int i = 0; i < comm.nprocs; ++i)
                        for (unsigned int j = 0; j < comm.nprocs; ++j)
                            if (send_counts[i * comm.nprocs + j] !=
                                recv_counts[j * comm.nprocs + i])
                                throw std::runtime_error(
                                    "send_receive: inconsistent communication pattern");
            }

            // Do the MPI communication
            std::vector<MPI_Request> r;
            const int tag = 0;
            const unsigned int T_num = dtype_size / sizeof(T);
            causalConnectTo(v1ToReceive.buf.ctx(), v0ToSend.buf.ctx());
            sync(v0ToSend.buf.ctx());
            if (deviceId(v0ToSend.buf.ctx()) != deviceId(v1ToReceive.buf.ctx()) ||
                getStream(v0ToSend.buf.ctx()) != getStream(v1ToReceive.buf.ctx()))
                sync(v1ToReceive.buf.ctx());
            _t.stop();
            if (getUseMPINonBlock()) {
                if (getUseAlltoall()) {
                    tracker<Cpu> _t("MPI ialltoall", Cpu{});
                    r.resize(1);
                    MPI_check(MPI_Ialltoallv(v0ToSend.buf.data(), v0ToSend.counts.data(),
                                             v0ToSend.displ.data(), dtype, v1ToReceive.buf.data(),
                                             v1ToReceive.counts.data(), v1ToReceive.displ.data(),
                                             dtype, comm.comm, &r.front()));
                } else {
                    tracker<Cpu> _t("MPI isend_recv", Cpu{});
                    r.reserve(comm.nprocs * 2);
                    for (unsigned int p = 0; p < comm.nprocs; ++p) {
                        if (v1ToReceive.counts[p] == 0) continue;
                        r.push_back(MPI_REQUEST_NULL);
                        MPI_check(MPI_Irecv(v1ToReceive.buf.data() + v1ToReceive.displ[p] * T_num,
                                            v1ToReceive.counts[p], dtype, p, tag, comm.comm,
                                            &r.back()));
                    }
                    for (unsigned int p = 0; p < comm.nprocs; ++p) {
                        if (v0ToSend.counts[p] == 0) continue;
                        r.push_back(MPI_REQUEST_NULL);
                        MPI_check(MPI_Isend(v0ToSend.buf.data() + v0ToSend.displ[p] * T_num,
                                            v0ToSend.counts[p], dtype, p, tag, comm.comm,
                                            &r.back()));
                    }
                }
            } else {
                if (getUseAlltoall()) {
                    tracker<Cpu> _t("MPI alltoall", Cpu{});
                    MPI_check(MPI_Alltoallv(v0ToSend.buf.data(), v0ToSend.counts.data(),
                                            v0ToSend.displ.data(), dtype, v1ToReceive.buf.data(),
                                            v1ToReceive.counts.data(), v1ToReceive.displ.data(),
                                            dtype, comm.comm));
                } else {
                    tracker<Cpu> _t("MPI send_recv", Cpu{});
                    for (unsigned int p = 0; p < comm.nprocs; ++p) {
                        if (v0ToSend.counts[p] == 0 && v1ToReceive.counts[p] == 0) continue;
                        MPI_check(MPI_Sendrecv(
                            v0ToSend.buf.data() + v0ToSend.displ[p] * T_num, v0ToSend.counts[p],
                            dtype, p, tag, v1ToReceive.buf.data() + v1ToReceive.displ[p] * T_num,
                            v1ToReceive.counts[p], dtype, p, tag, comm.comm, MPI_STATUS_IGNORE));
                    }
                }
                if (deviceId(v1ToReceive.buf.ctx()) >= 0) syncLegacyStream(v1ToReceive.buf.ctx());
                unpack(v1ToReceive, v1, EWOP{}, Q(alpha));
                return {};
            }

            // Do this later
            // NOTE: keep `v0ToSend` and `v1ToReceive` around until `MPI_Ialltoallv` is finished
            return [=]() mutable {
                // Make sure that all processes wait for the copy operations in the same order
                if (getDebugLevel() > 0) {
                    check_consistency(std::make_tuple(std::string("wait for send_receive"),
                                                      call_number, typeid(tag_type).hash_code()),
                                      comm);
                }

                // Wait for the MPI communication to finish
                {
                    tracker<Cpu> _t("MPI wait", Cpu{});
                    MPI_check(MPI_Waitall((int)r.size(), r.data(), MPI_STATUS_IGNORE));
                }

                // Clear origin buffer
                v0ToSend.buf.clear();

                // Copy back to v1
                if (deviceId(v1ToReceive.buf.ctx()) >= 0) syncLegacyStream(v1ToReceive.buf.ctx());
                unpack(v1ToReceive, v1, EWOP{}, Q(alpha));
            };
        }
#else

        inline void barrier(SelfComm) {}

#endif // SUPERBBLAS_USE_MPI

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <typename IndexType, typename XPUbuff0, typename XPUbuff1, std::size_t Nd0,
                  std::size_t Nd1, typename T, typename Q, typename XPU0, typename XPU1,
                  typename XPUr, typename EWOP>
        Request send_receive(const Order<Nd0> &o0, const std::vector<Proc_ranges<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, XPUbuff0 xpubuff0,
                             const Order<Nd1> &o1, const Proc_ranges<Nd1> &toReceive,
                             const Component<Nd1, Q, XPUr> &v1, XPUbuff1 xpubuff1, SelfComm comm,
                             EWOP, CoorOrder co, typename elem<T>::type alpha) {
            (void)o0;
            (void)toSend;
            (void)v0;
            (void)xpubuff0;
            (void)o1;
            (void)toReceive;
            (void)v1;
            (void)xpubuff1;
            (void)co;
            (void)alpha;
            if (comm.nprocs <= 1) return [] {};
            throw std::runtime_error("Unsupported SelfComm with nprocs > 1");
        }

        /// Return the volume of the largest range
        /// \param ranges: list of tensor ranges

        template <std::size_t Nd> std::size_t maximum_volume(const Proc_ranges<Nd> &ranges) {
            std::size_t vol = 0;
            for (auto const &r : ranges)
                for (auto const &it : r) vol = std::max(vol, volume(it[1]));
            return vol;
        }

        /// Return whether some MPI calls support GPU pointers

        inline bool test_support_for_mpi_gpu() {
#ifdef SUPERBBLAS_TEST_MPI_GPU
            static const bool test_mpi_gpu = [] {
#    ifdef SUPERBBLAS_USE_CUDA
                return (bool)MPIX_Query_cuda_support();
#    elif defined(SUPERBBLAS_USE_HIP)
                return (bool)MPIX_Query_rocm_support();
#    else
                return false;
#    endif
            }();
            return test_mpi_gpu;
#else
            return false;
#endif // SUPERBBLAS_TEST_MPI_GPU
        }

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors
        ///
        /// NOTE: choose size_t as the IndexType in case the local volume is too large

        template <std::size_t Nd0, typename XPUbuff0, typename XPUbuff1, std::size_t Nd1,
                  typename T, typename Q, typename XPU0, typename XPU1, typename XPUr,
                  typename Comm, typename EWOp>
        Request
        send_receive_choose_size(const Order<Nd0> &o0, const std::vector<Proc_ranges<Nd0>> &toSend,
                                 const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                                 XPUbuff0 xpubuff0, const Order<Nd1> &o1,
                                 const Proc_ranges<Nd1> &toReceive,
                                 const Component<Nd1, Q, XPUr> &v1, XPUbuff1 xpubuff1, Comm comm,
                                 EWOp, CoorOrder co, typename elem<T>::type alpha) {

            bool use_size_t = false;
            for (const auto &r : toSend)
                if (maximum_volume(r) >= (std::size_t)std::numeric_limits<IndexType>::max())
                    use_size_t = true;
            if (maximum_volume(toReceive) >= (std::size_t)std::numeric_limits<IndexType>::max())
                use_size_t = true;

            if (!use_size_t) {
                return send_receive<IndexType>(o0, toSend, v0, xpubuff0, o1, toReceive, v1,
                                               xpubuff1, comm, EWOp{}, co, alpha);
            } else {
                return send_receive<std::size_t>(o0, toSend, v0, xpubuff0, o1, toReceive, v1,
                                                 xpubuff1, comm, EWOp{}, co, alpha);
            }
        }

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1, typename XPUr, typename Comm, typename EWOp>
        Request send_receive(const Order<Nd0> &o0, const std::vector<Proc_ranges<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, const Proc_ranges<Nd1> &toReceive,
                             const Component<Nd1, Q, XPUr> &v1, Comm comm, EWOp, CoorOrder co,
                             typename elem<T>::type alpha) {

            // Whether to allow the use of gpu buffers for the sender/receiver buffers
            static const bool use_mpi_gpu = [] {
                if (getUseMPIGpu() == 0) return test_support_for_mpi_gpu();
                return getUseMPIGpu() > 0;
            }();

            // Use mpi send/receive buffers on cpu memory
            if (!use_mpi_gpu                                       // not use gpu-aware
                || (v0.first.size() == 0 && v0.second.size() == 0) // v0 is empty
                || v0.second.size() > 0                            // v0 has some cpu components
                || deviceId(v1.it.ctx()) == CPU_DEVICE_ID          // v1 is on cpu
            ) {
#ifdef SUPERBBLAS_USE_GPU
                // Make the sender/receiver buffers on host pinned memory to improve the transfer rates copying
                // data from/to the gpus
                if (v0.first.size() > 0) {
                    Gpu gpu0 = v0.first.front().it.ctx().toCpuPinned();
                    if (deviceId(v1.it.ctx()) >= 0) {
                        return send_receive_choose_size(o0, toSend, v0, gpu0, o1, toReceive, v1,
                                                        v1.it.ctx().toCpuPinned(), comm, EWOp{}, co,
                                                        alpha);
                    } else {
                        return send_receive_choose_size(o0, toSend, v0, gpu0, o1, toReceive, v1,
                                                        Cpu{}, comm, EWOp{}, co, alpha);
                    }
                } else if (deviceId(v1.it.ctx()) >= 0) {
                    return send_receive_choose_size(o0, toSend, v0, Cpu{}, o1, toReceive, v1,
                                                    v1.it.ctx().toCpuPinned(), comm, EWOp{}, co,
                                                    alpha);
                }
#endif // SUPERBBLAS_USE_GPU
                return send_receive_choose_size(o0, toSend, v0, Cpu{}, o1, toReceive, v1, Cpu{},
                                                comm, EWOp{}, co, alpha);
            }

            // Use mpi send/receive buffers on gpu memory
            // NOTE: both buffers should be on the same device
            if (v0.second.size() == 0) {
                return send_receive_choose_size(o0, toSend, v0, v0.first.front().it.ctx(), o1,
                                                toReceive, v1, v0.first.front().it.ctx(), comm,
                                                EWOp{}, co, alpha);
            } else {
                return send_receive_choose_size(o0, toSend, v0, v1.it.ctx(), o1, toReceive, v1,
                                                v1.it.ctx(), comm, EWOp{}, co, alpha);
            }
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
                IndexType fromr0 = 0, sizer0 = 0, fromr1 = 0, sizer1 = 0, fromr2 = 0, sizer2 = 0;

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
                    intersection(from0[i] + dim[i], size0[i], from1[i], size1[i], dim[i], fromr2,
                                 sizer2);
                }
                if (sizer0 > 0) {
                    grid[grid_n[i]][0][i] = fromr0;
                    grid[grid_n[i]++][1][i] = sizer0;
                }
                if (sizer1 > 0) {
                    grid[grid_n[i]][0][i] = fromr1;
                    grid[grid_n[i]++][1][i] = sizer1;
                }
                if (sizer2 > 0) {
                    grid[grid_n[i]][0][i] = fromr2;
                    grid[grid_n[i]++][1][i] = sizer2;
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
                fromr = Coor<Nd>{{}};
                sizer = Coor<Nd>{{}};
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
            IndexType vol = volume(p.second);
            if (vol == 0) {
                return {};
            } else if (vol == 1) {
                From_size_out<Nd> r(1, Cpu{});
                r[0] = p.first[0];
                return r;
            } else {
                From_size_out<Nd> r(vol, Cpu{});
                Coor<Nd> stride = get_strides<IndexType>(p.second, FastToSlow);
                for (IndexType i = 0; i < vol; ++i) {
                    Coor<Nd> c = index2coor(i, p.second, stride);
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
                Coor<Nd> stride = get_strides<IndexType>(p[i].second, FastToSlow);
                for (IndexType j = 0, j1 = volume(p[i].second); j < j1; ++j) {
                    Coor<Nd> c = index2coor(j, p[i].second, stride);
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

        /// Sort a list of ranges based on the first coordinate
        /// \param fs: vector of first coordinate and size of the ranges to order
        /// \param dim: dimensions of the tensor where the ranges belong
        /// \param stride: strides for those dimensions

        template <std::size_t Nd, typename SIdx>
        From_size_out<Nd> sort_ranges(const From_size_out<Nd> &fs, const Coor<Nd> &dim,
                                      const Coor<Nd, SIdx> &stride) {
            From_size_out<Nd> r(fs.size(), Cpu{});
            for (std::size_t i = 0; i < fs.size(); ++i) r[i] = fs[i];
            std::sort(r.begin(), r.end(),
                      [&](const From_size_item<Nd> &a, const From_size_item<Nd> &b) {
                          return coor2index(a[0], dim, stride) < coor2index(b[0], dim, stride);
                      });
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
            if (volume(sizer) == 0) sizer = Coor<Nd1>{{}};
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
            Coor<Nd1, std::size_t> stride1 = get_strides<std::size_t>(dim1, FastToSlow);
            unsigned int nprocs = p1.size() / ncomponents1;
            Proc_ranges<Nd0> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd1> &local_from1 = p1[i * ncomponents1 + componentId1][0];
                const Coor<Nd1> &local_size1 = p1[i * ncomponents1 + componentId1][1];
                r[i] =
                    shift_ranges(translate_range(sort_ranges(intersection<Nd1>(rfs1, local_from1,
                                                                               local_size1, dim1),
                                                             dim1, stride1),
                                                 from1, dim1, from0, dim0, perm1),
                                 local_from0, {{}}, dim0);
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
            Coor<Nd1, std::size_t> stride1 = get_strides<std::size_t>(dim1, FastToSlow);
            unsigned int nprocs = p0.size();
            Proc_ranges<Nd1> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd0> &local_from0 = p0[i][0];
                const Coor<Nd0> &local_size0 = p0[i][1];
                r[i] =
                    shift_ranges(sort_ranges(translate_range(intersection<Nd0>(rfs0, local_from0,
                                                                               local_size0, dim0),
                                                             from0, dim0, from1, dim1, perm0),
                                             dim1, stride1),
                                 local_from1, {{}}, dim1);
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
            vector<std::size_t, Cpu> get_mock_components(const Coor<Nd> &from, const Coor<Nd> &size,
                                                         const Coor<Nd> &dim, Cpu cpu, CoorOrder co,
                                                         MockFilling mf) {
                std::size_t vol = volume(size);
                vector<std::size_t, Cpu> r(vol, cpu);

                if (mf == FillWithIndices) {
                    Coor<Nd, std::size_t> local_stride = get_strides<std::size_t>(size, co);
                    Coor<Nd, std::size_t> stride = get_strides<std::size_t>(dim, co);
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
            vector<std::size_t, XPU> get_mock_components(const Coor<Nd> &from, const Coor<Nd> &size,
                                                         const Coor<Nd> &dim, XPU xpu, CoorOrder co,
                                                         MockFilling mf) {
                std::size_t vol = volume(size);
                vector<std::size_t, XPU> r(vol, xpu);
                vector<std::size_t, Cpu> r_host =
                    get_mock_components(from, size, dim, Cpu{}, co, mf);
                copy_n<std::size_t>(1, r_host.data(), r_host.ctx(), vol, r.data(), r.ctx(),
                                    EWOp::Copy{});
                return r;
            }

            template <typename T>
            using mockIndexType = typename std::conditional<std::is_const<T>::value,
                                                            const std::size_t, std::size_t>::type;

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
                    r.first.push_back(Component<Nd, std::size_t, XPU0>{
                        get_mock_components(p[c.componentId + comm.rank * ncomponents][0], c.dim,
                                            dim, c.it.ctx(), co, mf),
                        c.dim, c.componentId, c.mask_it});
                }
                for (const Component<Nd, T, XPU1> &c : v.second) {
                    r.second.push_back(Component<Nd, std::size_t, XPU1>{
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
                                 const Component<Nd1, std::size_t, XPU> &v,
                                 const Coor<Nd1> &local_from1, EWOP, CoorOrder co) {

                Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
                Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
                Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
                std::size_t vol = volume(v.dim);
                Coor<Nd1, std::size_t> local_stride1 = get_strides<std::size_t>(v.dim, co);
                Coor<Nd0, std::size_t> stride0 = get_strides<std::size_t>(dim0, co);
                vector<std::size_t, Cpu> v_host = makeSure(v.it, Cpu{});
                vector<MaskType, Cpu> m_host = makeSure(v.mask_it, Cpu{});

#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < vol; ++i) {
                    Coor<Nd1> c1 =
                        normalize_coor(index2coor(i, v.dim, local_stride1) + local_from1, dim1);
                    std::size_t true_val = 0;
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
                const Components_tmpl<Nd0, const std::size_t, XPU0, XPU1> v0_ =
                    get_mock_components(p0, dim0, v0, co, FillWithIndices, comm);
                const Components_tmpl<Nd1, std::size_t, XPU0, XPU1> v1_ =
                    get_mock_components(p1, dim1, v1, co, FillWithZeros, comm);

                // Copy the indices
                copy(1, p0, from0, size0, dim0, o0, v0_, p1, from1, dim1, o1, v1_, comm, EWOP{}, co,
                     false);

                // Check that the modified elements on v1_ are what they should be
                unsigned int ncomponents1 = v1.first.size() + v1.second.size();
                for (const Component<Nd1, std::size_t, XPU0> &c : v1_.first) {
                    test_copy_check<Nd0, Nd1>(p0, from0, size0, dim0, o0, from1, dim1, o1, c,
                                              p1[c.componentId + comm.rank * ncomponents1][0],
                                              EWOP{}, co);
                }
                for (const Component<Nd1, std::size_t, XPU1> &c : v1_.second) {
                    test_copy_check<Nd0, Nd1>(p0, from0, size0, dim0, o0, from1, dim1, o1, c,
                                              p1[c.componentId + comm.rank * ncomponents1][0],
                                              EWOP{}, co);
                }

                getTrackingTime() = trackingTime;
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

            // Simple heuristic: if there's no need for communications all elements from the local origin tensor
            // will land on the local destination tensor, and all local destination elements are receiving
            // the values from the local origin tensor
            unsigned int nprocs = p0.size();
            for (unsigned int i = 0; i < nprocs; ++i) {
                // Restrict (from0, size0) to the p0[i] range
                auto fs0 = intersection(from0, size0, p0[i][0], p0[i][1], dim0);

                // Translate the range to the destination tensor
                auto fs01 = translate_range(fs0, from0, dim0, from1, dim1, perm0);

                // Intersect the range with p1[i] range
                auto rfs01 = intersection(fs01, p1[i][0], p1[i][1], dim1);

                // Intersect the destination range with the destination range
                auto fs1 = intersection(from1, size1, p1[i][0], p1[i][1], dim1);

                // If it is not a complete map, it means that some elements in p0[i] range
                // will go to other processes, or some elements in p1[1] will come from other ranges
                std::size_t vol_fs0 = volume(fs0);
                if (vol_fs0 != volume(rfs01) || vol_fs0 != volume(fs1)) return true;
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
        std::array<Request, 2> copy_request_dest_component(
            typename elem<T>::type alpha, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
            const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
            const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
            unsigned int ncomponents1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
            const Order<Nd1> &o1, const Component<Nd1, Q, XPU> &v1, Comm comm, EWOP ewop,
            CoorOrder co) {

            // Find precomputed pieces on cache
            using Key = std::tuple<From_size<Nd0>, Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, From_size<Nd1>,
                                   Coor<Nd1>, Coor<Nd1>, PairPerms<Nd0, Nd1>, int, int, int>;
            struct Value {
                std::vector<Proc_ranges<Nd0>> toSend;
                Proc_ranges<Nd1> toReceive;
                bool need_comms;
            };
            struct cache_tag {};
            auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(p0.ctx());
            Key key{p0,           from0,          size0,    dim0,
                    p1,           from1,          dim1,     get_perms(o0, o1),
                    ncomponents1, v1.componentId, comm.rank};
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
                if (comm.nprocs > 1)
                    need_comms = may_need_communications(p0, from0, size0, dim0, o0, p1, from1,
                                                         dim1, o1, EWOP{});
                else
                    need_comms = false;

                // Save the results
                cache.insert(key, {toSend, toReceive, need_comms}, 0);
            } else {
                toSend = it->second.value.toSend;
                toReceive = it->second.value.toReceive;
                need_comms = it->second.value.need_comms;
            }

            // Do the sending and receiving
            Request mpi_req;
            if (need_comms)
                mpi_req = send_receive<Nd0, Nd1>(o0, toSend, v0, o1, toReceive, v1, comm, ewop, co,
                                                 alpha);

            // Do the local copies
            Request local_req = [=] {
                unsigned int ncomponents0 = v0.first.size() + v0.second.size();
                for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                    assert(toSend[c0.componentId][comm.rank].size() ==
                           toReceive[c0.componentId + comm.rank * ncomponents0].size());
                    for (unsigned int i = 0, i1 = toSend[c0.componentId][comm.rank].size(); i < i1;
                         ++i) {
                        assert(check_equivalence(
                            o0, toSend[c0.componentId][comm.rank][i][1], o1,
                            toReceive[c0.componentId + comm.rank * ncomponents0][i][1]));
                        local_copy<Nd0, Nd1, T, Q>(
                            alpha, o0, toSend[c0.componentId][comm.rank][i][0],
                            toSend[c0.componentId][comm.rank][i][1], c0.dim, c0.it, c0.mask_it, o1,
                            toReceive[c0.componentId + comm.rank * ncomponents0][i][0], v1.dim,
                            v1.it, v1.mask_it, ewop, co);
                    }
                }
                for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                    assert(toSend[c0.componentId][comm.rank].size() ==
                           toReceive[c0.componentId + comm.rank * ncomponents0].size());
                    for (unsigned int i = 0, i1 = toSend[c0.componentId][comm.rank].size(); i < i1;
                         ++i) {
                        assert(check_equivalence(
                            o0, toSend[c0.componentId][comm.rank][i][1], o1,
                            toReceive[c0.componentId + comm.rank * ncomponents0][i][1]));
                        local_copy<Nd0, Nd1, T, Q>(
                            alpha, o0, toSend[c0.componentId][comm.rank][i][0],
                            toSend[c0.componentId][comm.rank][i][1], c0.dim, c0.it, c0.mask_it, o1,
                            toReceive[c0.componentId + comm.rank * ncomponents0][i][0], v1.dim,
                            v1.it, v1.mask_it, ewop, co);
                    }
                }
            };

            return {local_req, mpi_req};
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Return the gpu components with a parallel context
        /// \param v: components

        template <std::size_t Nd, typename T>
        Components_tmpl<Nd, T, Gpu, Cpu>
        anabranch_begin(const Components_tmpl<Nd, T, Gpu, Cpu> &v) {
            // Trivial case: do nothing if v have zero or one gpu components
            if (v.first.size() <= 1) return v;

            // Recreate v but with new gpu contexts
            Components_tmpl<Nd, T, Gpu, Cpu> r;
            for (const auto &c : v.first)
                r.first.push_back(c.withNewContext(anabranch_begin(c.it.ctx())));
            r.second = v.second;

            // Return the new v
            return r;
        }
#endif // SUPERBBLAS_USE_GPU

        template <std::size_t Nd, typename T>
        Components_tmpl<Nd, T, Cpu, Cpu>
        anabranch_begin(const Components_tmpl<Nd, T, Cpu, Cpu> &v) {
            // Trivial case: do nothing if v doesn't have gpu contexts
            return v;
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Merge back all operations executed asynchronously on the new contexts
        /// \param v: components

        template <std::size_t Nd, typename T>
        void anabranch_end(const Components_tmpl<Nd, T, Gpu, Cpu> &v) {
            // Trivial case: do nothing if v have zero or one components
            if (v.first.size() <= 1) return;

            // Join back all gpu contexts on v
            for (const auto &c : v.first) anabranch_end(c.it.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        template <std::size_t Nd, typename T>
        void anabranch_end(const Components_tmpl<Nd, T, Cpu, Cpu> &) {
            // Trivial case: do nothing if v have no gpu components
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

        template <std::size_t Nd, typename T, typename Q, typename Comm, typename XPU0,
                  typename XPU1, typename EWOP>
        DECL_COPY_REQUEST_T_Q(Request copy_request(
            typename elem<T>::type alpha, const From_size<Nd> &p0, const Coor<Nd> &from0,
            const Coor<Nd> &size0, const Coor<Nd> &dim0, const Order<Nd> &o0,
            const Components_tmpl<Nd, const T, XPU0, XPU1> &v0, const From_size<Nd> &p1,
            const Coor<Nd> &from1, const Coor<Nd> &dim1, const Order<Nd> &o1,
            const Components_tmpl<Nd, Q, XPU0, XPU1> &v1, Comm comm, EWOP ewop, CoorOrder co,
            bool do_test))
        IMPL({
            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("copy_request"), alpha, p0, from0,
                                                  size0, dim0, o0, p1, from1, dim1, o1, co, do_test,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            if (getDebugLevel() >= 2 && do_test) {
                ns_copy_test::test_copy(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1, o1,
                                        v1, comm, EWOP{}, co);
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
            if (!check_isomorphic<Nd, Nd>(o0, size0, dim0, o1, dim1))
                throw std::runtime_error("Invalid copy operation");

            // Split the work for each receiving component
            std::vector<std::array<Request, 2>> reqs;
            for (unsigned int i = 0; i < ncomponents1; ++i) {
                for (const Component<Nd, Q, XPU0> &c : v1.first) {
                    if (c.componentId == i)
                        reqs.push_back(copy_request_dest_component<Nd, Nd, T, Q>(
                            alpha, p0, from0, size0, dim0, o0, v0, p1, ncomponents1, from1, dim1,
                            o1, c, comm, ewop, co));
                }
                for (const Component<Nd, Q, XPU1> &c : v1.second) {
                    if (c.componentId == i)
                        reqs.push_back(copy_request_dest_component<Nd, Nd, T, Q>(
                            alpha, p0, from0, size0, dim0, o0, v0, p1, ncomponents1, from1, dim1,
                            o1, c, comm, ewop, co));
                }
            }

            // Do the local part
            for (const auto &r : reqs) wait(r[0]);

            // Finish the rest later if there's something pending
            bool pending_request = false;
            for (const auto &r : reqs)
                if (r[1]) pending_request = true;
            if (pending_request)
                return [=] {
                    for (const auto &r : reqs) wait(r[1]);
                };
            return Request();
        })

        /// Return an empty mask, all levels are free to be used

        inline std::vector<bool> get_labels_mask() {
            return std::vector<bool>((int)std::numeric_limits<char>::max() -
                                     (int)std::numeric_limits<char>::min());
        }

        /// Mark the given labels as used
        /// \param o: labels
        /// \param m: mask

        template <std::size_t Nd> void update_label_mask(const Order<Nd> &o, std::vector<bool> &m) {
            for (char c : o) m[(int)c - (int)std::numeric_limits<char>::min()] = true;
        }

        /// Auxiliary struct used by `dummy_normalize_copy`

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        struct tensor_description {
            From_size<Nd> p;
            Coor<Nd> from, size, dim;
            Order<Nd> o;
            Components_tmpl<Nd, T, XPU0, XPU1> v;
        };

        /// Return an equivalent tensor but the given `Nd` dimensions
        /// \param p0: partitioning of the tensor in consecutive ranges
        /// \param o0: dimension labels for the tensor
        /// \param from0: first coordinate to copy from the tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the tensor

        template <std::size_t Nd, std::size_t Nd0, typename T, typename XPU0, typename XPU1,
                  typename std::enable_if<(Nd0 < Nd), bool>::type = true>
        tensor_description<Nd, T, XPU0, XPU1>
        dummy_normalize_copy(const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                             const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                             const Components_tmpl<Nd0, T, XPU0, XPU1> &v0, std::vector<bool> &m) {
            From_size_out<Nd> new_p(p0.size(), Cpu{});
            for (std::size_t i = 0; i < p0.size(); ++i) {
                std::copy_n(p0[i][0].begin(), Nd0, new_p[i][0].begin());
                std::copy_n(p0[i][1].begin(), Nd0, new_p[i][1].begin());
                for (std::size_t j = Nd0; j < Nd; ++j) new_p[i][0][j] = 0;
                for (std::size_t j = Nd0; j < Nd; ++j) new_p[i][1][j] = 1;
            }
            Coor<Nd> new_from, new_size, new_dim;
            std::copy_n(from0.begin(), Nd0, new_from.begin());
            std::copy_n(size0.begin(), Nd0, new_size.begin());
            std::copy_n(dim0.begin(), Nd0, new_dim.begin());
            for (std::size_t j = Nd0; j < Nd; ++j) new_from[j] = 0;
            for (std::size_t j = Nd0; j < Nd; ++j) new_size[j] = new_dim[j] = 1;
            Order<Nd> new_o;
            std::copy_n(o0.begin(), Nd0, new_o.begin());
            std::size_t j = Nd0;
            for (unsigned int c = (unsigned int)(-std::numeric_limits<char>::min()) + 1u;
                 c < m.size() && j < Nd; ++c) {
                if (!m[c]) {
                    new_o[j++] = (char)((int)c + (int)std::numeric_limits<char>::min());
                    m[c] = true;
                }
            }
            if (j != Nd) throw std::runtime_error("dummy_normalize_copy: run out of labels");

            Components_tmpl<Nd, T, XPU0, XPU1> new_v;
            for (const auto &c0 : v0.first) {
                Coor<Nd> new_dim;
                std::copy_n(c0.dim.begin(), Nd0, new_dim.begin());
                for (std::size_t j = Nd0; j < Nd; ++j) new_dim[j] = 1;
                new_v.first.push_back(
                    Component<Nd, T, XPU0>{c0.it, new_dim, c0.componentId, c0.mask_it});
            }
            for (const auto &c0 : v0.second) {
                Coor<Nd> new_dim;
                std::copy_n(c0.dim.begin(), Nd0, new_dim.begin());
                for (std::size_t j = Nd0; j < Nd; ++j) new_dim[j] = 1;
                new_v.second.push_back(
                    Component<Nd, T, XPU1>{c0.it, new_dim, c0.componentId, c0.mask_it});
            }

            return {new_p, new_from, new_size, new_dim, new_o, new_v};
        }

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        tensor_description<Nd, T, XPU0, XPU1>
        dummy_normalize_copy(const From_size<Nd> &p0, const Coor<Nd> &from0, const Coor<Nd> &size0,
                             const Coor<Nd> &dim0, const Order<Nd> &o0,
                             const Components_tmpl<Nd, T, XPU0, XPU1> &v0, std::vector<bool> &) {
            return {p0, from0, size0, dim0, o0, v0};
        };

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
        ///
        /// NOTE: this function makes the origin and the destination tensor of the same number of dimensions
        /// to reduce the compilation times

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1, typename EWOP>
        Request copy_request_normalized(typename elem<T>::type alpha, const From_size<Nd0> &p0,
                                        const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                                        const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                                        const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                                        const From_size<Nd1> &p1, const Coor<Nd1> &from1,
                                        const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                                        const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm,
                                        EWOP ewop, CoorOrder co, bool do_test = true) {
            auto m = get_labels_mask();
            update_label_mask(o0, m);
            update_label_mask(o1, m);
            constexpr std::size_t Nd = std::max(Nd0, Nd1);
            auto t0 = dummy_normalize_copy<Nd>(p0, from0, size0, dim0, o0, v0, m);
            auto t1 = dummy_normalize_copy<Nd>(p1, from1, Coor<Nd1>{{}}, dim1, o1, v1, m);
            return copy_request(alpha, t0.p, t0.from, t0.size, t0.dim, t0.o, t0.v, t1.p, t1.from,
                                t1.dim, t1.o, t1.v, comm, ewop, co, do_test);
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

            wait(copy_request_normalized(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1, o1,
                                         v1, comm, ewop, co, do_test));
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
        Request copy(typename elem<T>::type alpha, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
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

            Request r;
            switch (copyadd) {
            case Copy:
                r = copy_request_normalized(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1,
                                            o1, v1, comm, EWOp::Copy{}, co);
                break;
            case Add:
                r = copy_request_normalized(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1,
                                            o1, v1, comm, EWOp::Add{}, co);
                break;
            }

            if (getDebugLevel() >= 1) {
                for (const auto &i : v1.first) sync(i.it.ctx());
                for (const auto &i : v1.second) sync(i.it.ctx());
                barrier(comm);
            }

            return r;
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
                if (volume(pr[i][1]) == 0) pr[i][0] = pr[i][1] = Coor<Ndo>{{}};
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
        void zeroed_repeated_tensor(From_size<Nd0> p0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
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
                                           (vector<const T, XPU>)v, Mask<XPU>{}, o_r, fromr,
                                           pr[componentId][1], v, Mask<XPU>{}, EWOp::Copy{}, co);
            }
        }

        /// Return a new components based on a partition
        /// \param p: partitioning
        /// \param v: tensor components
        /// \param co: coordinate linearization order

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        Components_tmpl<N, T, XPU0, XPU1>
        like_this_components(const From_size<N> &p, const Components_tmpl<N, T, XPU0, XPU1> &v,
                             Comm comm, CacheAlloc cacheAlloc = dontCacheAlloc) {

            // Allocate the tensor
            unsigned int ncomponents = v.first.size() + v.second.size();
            Components_tmpl<N, T, XPU0, XPU1> v1;
            for (unsigned int i = 0; i < v.first.size(); ++i) {
                const unsigned int componentId = v.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<N> &dimi = p[pi][1];
                vector<T, XPU0> v1i(volume(dimi), v.first[i].it.ctx(), cacheAlloc);
                v1.first.push_back(Component<N, T, XPU0>{v1i, dimi, componentId, Mask<XPU0>{}});
            }
            for (unsigned int i = 0; i < v.second.size(); ++i) {
                const unsigned int componentId = v.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<N> &dimi = p[pi][1];
                vector<T, XPU1> v1i(volume(dimi), v.second[i].it.ctx(), cacheAlloc);
                v1.second.push_back(Component<N, T, XPU1>{v1i, dimi, componentId, Mask<XPU1>{}});
            }

            return v1;
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
            if (!force_copy && from0 == Coor<N>{{}} && o0 == o1 && p0 == p1) return v0;

            // Allocate the tensor
            auto v1 = like_this_components(p1, v0, comm);

            // Copy the content of v0 into v1
            copy<N, N, T>(T{1}, p0, from0, size0, dim0, o0, toConst(v0), p1, {{}}, dim1, o1, v1,
                          comm, EWOp::Copy{}, co);

            return v1;
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
        std::pair<Components_tmpl<N, T, XPU0, XPU1>, Request>
        reorder_tensor_request(const From_size<N> &p0, const Order<N> &o0, const Coor<N> &from0,
                               const Coor<N> &size0, const Coor<N> &dim0,
                               const Components_tmpl<N, T, XPU0, XPU1> &v0, const From_size<N> &p1,
                               const Coor<N> &dim1, const Order<N> &o1, Comm comm, CoorOrder co,
                               CacheAlloc cacheAlloc = dontCacheAlloc, bool force_copy = false) {

            // If the two orderings and partitions are equal, return the tensor
            if (!force_copy && from0 == Coor<N>{{}} && o0 == o1 && p0 == p1) return {v0, Request()};

            // Allocate the tensor
            auto v1 = like_this_components(p1, v0, comm, cacheAlloc);

            // Copy the content of v0 into v1
            return {v1, copy_request_normalized<N, N, T>(T{1}, p0, from0, size0, dim0, o0,
                                                         toConst(v0), p1, {{}}, dim1, o1, v1, comm,
                                                         EWOp::Copy{}, co, true /* do test */)};
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

        /// Return partitions for the input tensors that are compatible for contraction
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param sug_o0: suggested dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param sug_o1: suggested dimension labels for the second operator

        template <std::size_t Nd0, std::size_t Nd1>
        std::pair<From_size<Nd0>, From_size<Nd1>>
        get_input_partitions_for_contraction(const From_size<Nd0> &p0, const Coor<Nd0> &dim0,
                                             const Order<Nd0> &o0, const Order<Nd0> &sug_o0,
                                             const From_size<Nd1> &p1, const Coor<Nd1> &dim1,
                                             const Order<Nd1> &o1, const Order<Nd1> &sug_o1) {

            // Normalize the first tensor as the larger of the two in volume
            if (volume(dim0) < volume(dim1)) {
                auto p10 = get_input_partitions_for_contraction(p1, dim1, o1, sug_o1, p0, dim0, o0,
                                                                sug_o0);
                return {p10.second, p10.first};
            }

            // Reorder the first tensor if needed
            From_size_out<Nd0> p0r;
            if (o0 != sug_o0) {
                p0r = From_size_out<Nd0>(p0.size(), p0.ctx());
                Coor<Nd0> perm = find_permutation(o0, sug_o0);
                for (unsigned int i = 0; i < p0.size(); ++i) {
                    p0r[i][0] = reorder_coor(p0[i][0], perm);
                    p0r[i][1] = reorder_coor(p0[i][1], perm);
                }
            }

            // Change the second partition by using the same distribution as the first tensor
            // for the shared labels and replicated for the remaining labels
            From_size_out<Nd1> p1r(p0.size(), p0.ctx());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                p1r[i][0] = get_dimensions(o0, p0[i][0], o1, Coor<Nd1>{{}}, sug_o1, false);
                p1r[i][1] = get_dimensions(o0, p0[i][1], o1, dim1, sug_o1, false);

                // Avoid the contraction of empty tensors
                if (volume(p0[i][1]) == 0 || volume(p1r[i][1]) == 0) {
                    if (p0[i][0] != Coor<Nd0>{{}} || p0[i][1] != Coor<Nd0>{{}}) {
                        if (p0r.size() == 0) {
                            p0r = From_size_out<Nd0>(p0.size(), p0.ctx());
                            std::copy_n(p0.data(), p0.size(), p0r.data());
                        }
                        p0r[i][0] = p0r[i][1] = Coor<Nd0>{{}};
                    }
                    p1r[i][0] = p1r[i][1] = Coor<Nd1>{{}};
                }
            }
            bool changed1 = (o1 != sug_o1 || p1 != (From_size<Nd1>)p1r);

            // Return the change on the second tensor
            return {p0r.size() > 0 ? (From_size<Nd0>)p0r : p0, changed1 ? (From_size<Nd1>)p1r : p1};
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
                         const Components_tmpl<Nd0, T, XPU0, XPU1> &v0, const From_size<Nd1> &p1,
                         const Coor<Nd1> &dim1, const Order<Nd1> &o1, bool conj1,
                         const Components_tmpl<Nd1, T, XPU0, XPU1> &v1, T beta,
                         const From_size<Ndo> &pr, const Coor<Ndo> &dimr, const Order<Ndo> &o_r,
                         const Components_tmpl<Ndo, T, XPU0, XPU1> &vr, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vr.first) sync(i.it.ctx());
                for (const auto &i : vr.second) sync(i.it.ctx());
                barrier(comm);
            }

            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("contraction"), alpha, p0, dim0, o0,
                                                  conj0, p1, dim1, o1, conj1, beta, dimr, o_r, co,
                                                  typeid(tag_type).hash_code()),
                                  comm);
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
            Order<Nd0> sug_o0;
            Order<Nd1> sug_o1;
            Order<Ndo> sug_or;
            bool swap_operands;
            suggested_orders_for_contraction(o0, dim0, conj0, o1, dim1, conj1, o_r, dimr, sug_o0,
                                             sug_o1, sug_or, swap_operands, co);
            Coor<Nd0> sug_dim0 = reorder_coor(dim0, find_permutation(o0, sug_o0));
            Coor<Nd1> sug_dim1 = reorder_coor(dim1, find_permutation(o1, sug_o1));
            Coor<Ndo> sug_dimr = reorder_coor(dimr, find_permutation(o_r, sug_or));

            // Change the partition of the input tensors so that the local portions to contract
            // are local
            auto p01 =
                get_input_partitions_for_contraction(p0, dim0, o0, sug_o0, p1, dim1, o1, sug_o1);
            auto p0_ = p01.first;
            auto p1_ = p01.second;
            Components_tmpl<Nd0, T, XPU0, XPU1> v0_ =
                reorder_tensor(p0, o0, {{}}, dim0, dim0, v0, p0_, sug_dim0, sug_o0, comm, co);
            Components_tmpl<Nd1, T, XPU0, XPU1> v1_ =
                reorder_tensor(p1, o1, {{}}, dim1, dim1, v1, p1_, sug_dim1, sug_o1, comm, co);

            // Generate the partitioning and the storage for the output tensor
            unsigned int ncomponents = v0_.first.size() + v0_.second.size();
            From_size<Ndo> pr_ = get_output_partition<Nd0, Nd1, Ndo>(p0_, sug_dim0, sug_o0, p1_,
                                                                     sug_dim1, sug_o1, sug_or)
                                     .first;
            Components_tmpl<Ndo, const T, XPU0, XPU1> vr_;
            std::vector<vector<T, XPU0>> vr0(v0_.first.size());
            for (unsigned int i = 0; i < v0_.first.size(); ++i) {
                const unsigned int componentId = v0_.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Ndo> &dimi = pr_[pi][1];
                vr0[i] = vector<T, XPU0>(volume<Ndo>(dimi), v0_.first[i].it.ctx());
                vr_.first.push_back(
                    Component<Ndo, T, XPU0>{vr0[i], dimi, componentId, Mask<XPU0>{}});
                local_contraction<Nd0, Nd1, Ndo, T>(
                    alpha, sug_o0, p0_[pi][1], conj0, vector<const T, XPU0>(v0_.first[i].it),
                    sug_o1, p1_[pi][1], conj1, vector<const T, XPU0>(v1_.first[i].it), T{0.0},
                    sug_or, dimi, vr0[i], co);
                zeroed_repeated_tensor(p0_, sug_dim0, sug_o0, p1_, sug_dim1, sug_o1, pr_, sug_dimr,
                                       sug_or, pi, vr0[i], co);
            }
            std::vector<vector<T, XPU1>> vr1(v0_.second.size());
            for (unsigned int i = 0; i < v0_.second.size(); ++i) {
                const unsigned int componentId = v0_.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Ndo> &dimi = pr_[pi][1];
                vr1[i] = vector<T, XPU1>(volume<Ndo>(dimi), v0_.second[i].it.ctx());
                vr_.second.push_back(
                    Component<Ndo, T, XPU1>{vr1[i], dimi, componentId, Mask<XPU1>{}});
                local_contraction<Nd0, Nd1, Ndo, T>(
                    alpha, sug_o0, p0_[pi][1], conj0, vector<const T, XPU1>(v0_.second[i].it),
                    sug_o1, p1_[pi][1], conj1, vector<const T, XPU1>(v1_.second[i].it), T{0.0},
                    sug_or, dimi, vr1[i], co);
                zeroed_repeated_tensor(p0_, sug_dim0, sug_o0, p1_, sug_dim1, sug_o1, pr_, sug_dimr,
                                       sug_or, pi, vr1[i], co);
            }

            // Scale the output tensor by beta
            copy<Ndo, Ndo, T>(beta, pr, {{}}, dimr, dimr, o_r, toConst(vr), pr, {{}}, dimr, o_r, vr,
                              comm, EWOp::Copy{}, co);

            // Scale the output tensor by beta and reduce all the subtensors to the final tensor
            copy<Ndo, Ndo, T>(1, pr_, {{}}, sug_dimr, sug_dimr, sug_or, vr_, pr, {{}}, dimr, o_r,
                              vr, comm, EWOp::Add{}, co);

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

    namespace detail {
        /// Approximate factorization of a number with factors of 2 and 3.
        /// The returning value is largest than 0.75 times the original value.

        struct factors_2_3 {
            unsigned int two;   ///< powers of two
            unsigned int three; ///< powers of three
            unsigned int value; ///< 2^two * 3^three

            /// Empty construction; initialize to 1
            factors_2_3() : two(0), three(0), value(1) {}

            /// Constructor
            /// \param number: value to factorize

            factors_2_3(unsigned int number) {
                if (number == 0) throw std::runtime_error("unsupported value");

                // a) Try to exactly factorize the number with powers of two and three
                two = three = 0;
                value = 1;
                unsigned int remaining = number;
                for (; remaining % 2 == 0; ++two, remaining /= 2, value *= 2)
                    ;
                for (; remaining % 3 == 0; ++three, remaining /= 3, value *= 3)
                    ;

                // b) Find as many powers as possible of tree and then two
                for (; remaining >= 3; ++three, remaining /= 3, value *= 3)
                    ;
                if (remaining >= 2) ++two, remaining /= 2, value *= 2;

                // c) Try to exchange factors of 3 by 4
                for (; three > 0 && value * 4 / 3 <= number;
                     --three, two += 2, value = value * 4 / 3)
                    ;
            }

            /// Internal constructor
            factors_2_3(unsigned int two, unsigned int three, unsigned int value)
                : two(two), three(three), value(value) {}

            factors_2_3 operator*(const factors_2_3 &v) const {
                return {two + v.two, three + v.three, value * v.value};
            }
        };
    }

    /// Return the number of processes in each direction to partition the tensor
    /// \param order: dimension labels
    /// \param dim: dimension size for the tensor
    /// \param dist_labels: labels to distribute
    /// \param nprocs: number of precesses

    template <std::size_t Nd, typename std::enable_if<(Nd > 0), bool>::type = true>
    Coor<Nd> partitioning_distributed_procs(const char *order, const Coor<Nd> &dim,
                                            const char *dist_labels, unsigned int nprocs) {

        Coor<Nd> p; // returning value

        // The default is no distribution, which is one proc in each direction
        for (std::size_t i = 0; i < Nd; ++i) p[i] = 1;

        // Get the labels that are going to be distributed
        Order<Nd> order_ = detail::toArray<Nd>(order, "order");
        Coor<Nd> dist_perm;
        unsigned int dist_n = 0;
        for (unsigned int i = 0, n = std::strlen(dist_labels); i < n; ++i) {
            const auto &it = std::find(order_.begin(), order_.end(), dist_labels[i]);
            if (it != order_.end() && dim[it - order_.begin()] > 1)
                dist_perm[dist_n++] = it - order_.begin();
        }

        // Return the default distribution If no dimension is going to be distributed or the tensor is empty
        if (dist_n == 0 || detail::volume(dim) == 0 || nprocs <= 1) return p;

        std::array<detail::factors_2_3, Nd> p_f23;
        for (unsigned int i = 0; i < dist_n; ++i) p_f23[i] = detail::factors_2_3(1);
        detail::factors_2_3 vol_p(1);

        // Iteratively put factors 2 and 3 on the coordinates with largest size per process
        detail::factors_2_3 nprocs_f23(nprocs);
        std::array<detail::factors_2_3, 2> factors{3u, 2u};
        while (true) {
            // Sort the dimensions by local size from largest to smalles
            Coor<Nd> perm;
            for (unsigned int j = 0; j < dist_n; ++j) perm[j] = j;
            for (unsigned int j = 0; j < dist_n; ++j) {
                unsigned int large_i = j;
                std::size_t large_val = dim[dist_perm[perm[j]]] / p_f23[perm[j]].value;
                for (unsigned int i = j + 1; i < dist_n; ++i) {
                    std::size_t val = dim[dist_perm[perm[i]]] / p_f23[perm[i]].value;
                    if (large_val < val) large_i = i, large_val = val;
                }
                std::swap(perm[j], perm[large_i]);
            }

            // Try to put a factor of three or two in that direction
            bool factor_applied = false;
            for (unsigned int j = 0; j < dist_n; ++j) {
                for (const auto &factor : factors) {
                    if (nprocs_f23.value % (vol_p.value * factor.value) == 0) {
                        p_f23[perm[j]] = p_f23[perm[j]] * factor;
                        vol_p = vol_p * factor;
                        factor_applied = true;
                        break;
                    }
                }
                if (factor_applied) break;
            }
            if (factor_applied) continue;

            // Get out if we cannot put more factors
            break;
        }

        for (unsigned int i = 0; i < dist_n; ++i) p[dist_perm[i]] = p_f23[i].value;
        assert(detail::volume(p) <= nprocs && detail::volume(p) >= nprocs * 3 / 4);
        return p;
    }

    /// Return a partitioning for a tensor of `dim` dimension onto a grid of processes
    /// \param order: (can be null) dimension labels
    /// \param dim: dimension size for the tensor
    /// \param procs: number of processes in each direction
    /// \param dist_labels: (can be null) order use to assign the processes to each subtensor
    /// \param nprocs: (optional) number of precesses
    /// \param ncomponents: (optional) number of components

    template <std::size_t Nd>
    std::vector<PartitionItem<Nd>> basic_partitioning(const char *order, Coor<Nd> dim,
                                                      Coor<Nd> procs, const char *dist_labels,
                                                      int nprocs = -1, int ncomponents = 1) {

        // Check other arguments
        int vol_procs = (int)detail::volume<Nd>(procs);
        if (nprocs >= 0 && vol_procs > nprocs)
            std::runtime_error(
                "The total number of processes from `procs` is greater than `nprocs`");

        // Reorder the labels starting with dist_labels
        Coor<Nd> perm;
        if (order != nullptr && dist_labels != nullptr) {
            if (std::strlen(order) != Nd)
                throw std::runtime_error("basic_partitioning: invalid `order`, its length doesn't "
                                         "match the template parameter");
            const unsigned int n = std::strlen(dist_labels);
            unsigned int dist_n = 0;
            for (unsigned int i = 0; i < n; ++i) {
                const auto &it = std::find(order, order + Nd, dist_labels[i]);
                if (it != order + Nd) perm[dist_n++] = it - order;
            }
            for (unsigned int i = 0; i < Nd; ++i) {
                const auto &it = std::find(dist_labels, dist_labels + n, order[i]);
                if (it == dist_labels + n) perm[dist_n++] = i;
            }
        } else {
            for (unsigned int i = 0; i < Nd; ++i) perm[i] = i;
        }

        std::vector<PartitionItem<Nd>> fs((nprocs < 0 ? vol_procs : nprocs) * ncomponents);
        Coor<Nd> procs_perm = detail::reorder_coor(procs, perm);
        Coor<Nd> stride_perm = detail::get_strides<IndexType>(procs_perm, SlowToFast);
        for (int rank = 0; rank < vol_procs; ++rank) {
            Coor<Nd> cproc = detail::index2coor(rank, procs_perm, stride_perm);
            PartitionItem<Nd> fsi;
            for (std::size_t i = 0; i < Nd; ++i) {
                // Number of elements in process with rank 'cproc[i]' on dimension 'i'
                fsi[1][perm[i]] = dim[perm[i]] / procs_perm[i] +
                                  (dim[perm[i]] % procs_perm[i] > cproc[i] ? 1 : 0);

                // First coordinate in process with rank 'rank' on dimension 'i'
                fsi[0][perm[i]] = fsi[1][perm[i]] == dim[perm[i]]
                                      ? 0
                                      : dim[perm[i]] / procs_perm[i] * cproc[i] +
                                            std::min(cproc[i], dim[perm[i]] % procs_perm[i]);
            }

            // Normalize empty ranges
            if (detail::volume(fsi[1]) == 0) fsi[0] = fsi[1] = Coor<Nd>{{}};

            if (ncomponents == 1) {
                fs[rank] = fsi;
            } else {
                auto fsi_components = basic_partitioning(
                    order, fsi[1],
                    partitioning_distributed_procs(order, fsi[1], dist_labels, ncomponents),
                    dist_labels, ncomponents);
                for (int c = 0; c < ncomponents; ++c) {
                    using detail::operator+;
                    fs[rank * ncomponents + c] = {fsi_components[c][0] + fsi[0],
                                                  fsi_components[c][1]};
                    if (detail::volume(fs[rank * ncomponents + c][1]) == 0)
                        fs[rank * ncomponents + c][0] = fs[rank * ncomponents + c][1] =
                            Coor<Nd>{{}};
                }
            }
        }

        return fs;
    }

    /// Return a partitioning for a tensor of `dim` dimension onto a grid of processes
    /// \param dim1: dimension size for the tensor
    /// \param procs: number of processes on each dimension
    /// \param nprocs: (optional) total number of processes; if not given or it is less than the zero,
    ///                it will be the product of all elements in `procs`
    /// \param replicate: (optional) if true and the total processes of `procs` is one, then replicate
    ///                   the support of the tensor on every process
    /// \param ext_power: (optional) extend the support that many units in the positive and negative
    ///                   direction for each dimension

    template <std::size_t Nd>
    std::vector<PartitionItem<Nd>> basic_partitioning(Coor<Nd> dim, Coor<Nd> procs, int nprocs = -1,
                                                      bool replicate = false,
                                                      Coor<Nd> ext_power = {{}}) {
        int vol_procs = (int)detail::volume<Nd>(procs);
        if (nprocs >= 0 && vol_procs > nprocs)
            std::runtime_error(
                "The total number of processes from `procs` is greater than `nprocs`");
        for (std::size_t i = 0; i < Nd; ++i)
            if (ext_power[i] < 0) throw std::runtime_error("Unsupported value for `power`");

        std::vector<PartitionItem<Nd>> fs(nprocs < 0 ? vol_procs : nprocs);
        Coor<Nd> stride = detail::get_strides<IndexType>(procs, SlowToFast);
        for (int rank = 0; rank < vol_procs; ++rank) {
            Coor<Nd> cproc = detail::index2coor(rank, procs, stride);
            for (std::size_t i = 0; i < Nd; ++i) {
                // Number of elements in process with rank 'cproc[i]' on dimension 'i'
                fs[rank][1][i] = std::min(
                    dim[i] / procs[i] + (dim[i] % procs[i] > cproc[i] ? 1 : 0) + ext_power[i] * 2,
                    dim[i]);

                // First coordinate in process with rank 'rank' on dimension 'i'
                fs[rank][0][i] = fs[rank][1][i] == dim[i] ? 0
                                                          : (dim[i] / procs[i] * cproc[i] +
                                                             std::min(cproc[i], dim[i] % procs[i]) -
                                                             ext_power[i] + dim[i]) %
                                                                dim[i];
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
    /// \param request: (optional) return a callback to finish the operation later with `wait`
    /// \param session: (optional) concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
              const T **v0, const MaskType **mask0, const Context *ctx0,
              const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1,
              const Coor<Nd1> &from1, const Coor<Nd1> &dim1, Q **v1, const MaskType **mask1,
              const Context *ctx1, MPI_Comm mpicomm, CoorOrder co, CopyAdd copyadd,
              Request *request = nullptr, Session session = 0) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        Request r = detail::copy<Nd0, Nd1>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, mask0, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1, dim1,
            detail::toArray<Nd1>(o1, "o1"),
            detail::get_components<Nd1>(v1, mask1, ctx1, ncomponents1, p1, comm, session), comm,
            copyadd, co);

        if (request)
            *request = r;
        else
            wait(r);
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
    /// \param request: (optional) return a callback to finish the operation later with `wait`
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> from0, const Coor<Nd0> size0, const Coor<Nd0> dim0,
              const T **v0, const MaskType **mask0, const Context *ctx0,
              const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1, const Coor<Nd1> from1,
              const Coor<Nd1> dim1, Q **v1, const MaskType **mask1, const Context *ctx1,
              CoorOrder co, CopyAdd copyadd, Request *request = nullptr, Session session = 0) {

        detail::SelfComm comm = detail::get_comm();

        wait(detail::copy<Nd0, Nd1>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, mask0, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1, dim1,
            detail::toArray<Nd1>(o1, "o1"),
            detail::get_components<Nd1>(v1, mask1, ctx1, ncomponents1, p1, comm, session), comm,
            copyadd, co));
        if (request) *request = Request{};
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
            detail::get_components<Nd0>((T **)v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), dim1, o1_, conj1,
            detail::get_components<Nd1>((T **)v1, nullptr, ctx1, ncomponents1, p1, comm, session),
            beta, detail::get_from_size(pr, ncomponentsr * comm.nprocs, session), dimr, o_r_,
            detail::get_components<Ndo>(vr, nullptr, ctxr, ncomponentsr, pr, comm, session), comm,
            co);
    }

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<!detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T, const PartitionItem<Nd0> *, const Coor<Nd0> &, int, const char *, bool,
                     const T **, const Context *, const PartitionItem<Nd1> *, const Coor<Nd1> &,
                     int, const char *, bool, const T **, const Context *, T,
                     const PartitionItem<Ndo> *, const Coor<Ndo> &, int, const char, T **,
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
            detail::get_components<Nd0>((T **)v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), dim1, o1_, conj1,
            detail::get_components<Nd1>((T **)v1, nullptr, ctx1, ncomponents1, p1, comm, session),
            beta, detail::get_from_size(pr, ncomponentsr * comm.nprocs, session), dimr, o_r_,
            detail::get_components<Ndo>(vr, nullptr, ctxr, ncomponentsr, pr, comm, session), comm,
            co);
    }

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<!detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T, const PartitionItem<Nd0> *, const Coor<Nd0> &, int, const char *, bool,
                     const T **, const Context *, const PartitionItem<Nd1> *, const Coor<Nd1> &,
                     int, const char *, bool, const T **, const Context *, T,
                     const PartitionItem<Ndo> *, const Coor<Ndo> &, int, const char, T **,
                     const Context *, CoorOrder, Session = 0) {
        throw std::runtime_error("contraction: unsupported type");
    }

    namespace detail {
        /// Return the subranges resulting from subtracting a range, that is, making a hole
        /// \param from: first element of the range to subtract
        /// \param size: number of elements in each direction of the range to subtract
        /// \param dim: total number of elements in each direction

        template <std::size_t N>
        vector<std::array<Coor<N>, 2>, Cpu> make_hole(const Coor<N> &from, const Coor<N> &size,
                                                      const Coor<N> &dim) {
            /// Shortcut when N == 0
            if (N == 0) return {};

            /// Shortcut when subtracting an empty range
            if (detail::volume(size) == 0) {
                vector<std::array<Coor<N>, 2>, Cpu> r(1, Cpu{});
                r[0] = std::array<Coor<N>, 2>{Coor<N>{{}}, dim};
                return r;
            }

            // In the general case, return as many subranges as dimensions, each of the subranges
            // follows the pattern
            //  returned |  Coor 0  |  Coor 1  |  Coor 2  |
            //  subrange | subrange | subrange | subrange | ...
            //  --------------------------------------------
            //      0    | antihole |   full   |  full    | ...
            //      1    |   hole   | antihole |  full    | ...
            //      2    |   hole   |   hole   | antihole | ...
            //    ...

            vector<std::array<Coor<N>, 2>, Cpu> r(N, Cpu{}); // subranges to return
            for (std::size_t i = 0; i < N; ++i) {
                Coor<N> nfrom, nsize;
                // Fill with hole
                for (std::size_t j = 0; j < i; j++) {
                    nfrom[j] = from[j];
                    nsize[j] = size[j];
                }

                // Fill with the antihole
                nfrom[i] = detail::normalize_coor(from[i] + size[i], dim[i]);
                nsize[i] = dim[i] - size[i];

                // Fill with full
                for (std::size_t j = i + 1; j < N; j++) {
                    nfrom[j] = 0;
                    nsize[j] = dim[j];
                }

                r[i] = std::array<Coor<N>, 2>{nfrom, nsize};
            }

            return r;
        }
    }

    /// Return the subranges resulting from subtracting a range from another range, that is, making a hole
    /// \param from: first element of the range to subtract from
    /// \param size: number of elements in each direction of the range to subtract from
    /// \param hole_from: first element of the range to subtract
    /// \param hole_size: number of elements in each direction of the range to subtract
    /// \param dim: total number of elements in each direction

    template <std::size_t N>
    std::vector<std::array<Coor<N>, 2>> make_hole(const Coor<N> &from, const Coor<N> &size,
                                                  const Coor<N> &hole_from,
                                                  const Coor<N> &hole_size, const Coor<N> &dim) {
        /// Shortcut when N == 0
        if (N == 0) return {};

        /// Shortcut when subtracting an empty range
        if (detail::volume(hole_size) == 0)
            return std::vector<std::array<Coor<N>, 2>>(1, std::array<Coor<N>, 2>{from, size});

        // Make a hole on the whole tensor
        auto parts = detail::make_hole(hole_from, hole_size, dim);

        // Intersect the parts with the range
        auto final_parts = detail::intersection(parts, from, size, dim);

        // Filter out empty subregions
        std::vector<std::array<Coor<N>, 2>> r;
        r.reserve(final_parts.size());
        for (const auto &fs : final_parts)
            if (detail::volume(fs[1]) > 0) r.push_back(fs);
        return r;
    }
}

#endif //  __SUPERBBLAS_DIST__
