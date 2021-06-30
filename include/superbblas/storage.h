#ifndef __SUPERBBLAS_STORAGE__
#define __SUPERBBLAS_STORAGE__

#include "dist.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

/// Specification for simple, sparse, streamed tensor (S3T) format
/// magic_number <i32>: 314
/// version <i32>: version of S3T format (currently 0)
/// values_datatype <i32>: datatype used for the values; currently supported values:
///  - 0: float
///  - 1: double
///  - 2: complex float
///  - 3: complex double
/// dimensions <i32>: number of dimensions
/// metadata_size <i32>: length of the metadata in <char>s
/// metadata_content <char*metadata_size>: content of the metadata
/// padding <char*((8 - metadata_size % 8) % 8 + 4)>: zero
/// size <double*dimensions>: size of the tensor in each dimension (coordinates in SlowToFast)
/// num_chunks <double>: number of chunks that follows
/// chunk: repeat as many times as needed
///  -  number_of_blocks <double>: number of blocks
///  -  from_size <{from <double*dimensions>, size <double*dimensions>}*number_of_blocks>: the i-th
///     pair of coordinates indicates the first coordinate present in the i-th block and the size
///     of the block in each dimension;
///  -  values <type indicated by values_type>
///
/// NOTES:
/// - The restrictions in the metadata's length and the padding are to make all subsequent fields
///   properly aligned in case an implementation accesses the file by mapping it into memory.
/// - A slightly simpler implementation is to restrict each chunk to a single block; allowing
///   multiple blocks in a chunks gives a mechanism to increase the locality when reading all
///   "from_size" in the file.
/// - The type of the coordinates for from_size is double instead of the obvious better type i64
///   just because the latter type is not supported by MPI.

namespace superbblas {

    /// Type of the values
    enum values_datatype { FLOAT = 0, DOUBLE = 1, CFLOAT = 2, CDOUBLE = 3, CHAR = 4, INT = 5 };

    namespace detail {
        /// Magic number
        const int magic_number = 314;

        /// Open file modes
        enum Mode { CreateForReadWrite, ReadWrite };

        /// Return the values_datatype of a type
        template <typename T> values_datatype get_values_datatype();
        template <> inline values_datatype get_values_datatype<float>() { return FLOAT; }
        template <> inline values_datatype get_values_datatype<double>() { return DOUBLE; }
        template <> inline values_datatype get_values_datatype<std::complex<float>>() {
            return CFLOAT;
        }
        template <> inline values_datatype get_values_datatype<std::complex<double>>() {
            return CDOUBLE;
        }
        template <> inline values_datatype get_values_datatype<int>() { return INT; }
        template <> inline values_datatype get_values_datatype<char>() { return CHAR; }

        /// File descriptor
        template <typename Comm> struct File;

        /// Communicator
        enum CommType { SEQ, MPI };

#ifdef SUPERBBLAS_USE_MPI
        /// Return the MPI_Datatype of a type
        template <typename T> MPI_Datatype mpi_datatype_basic_from_type();
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<float>() { return MPI_FLOAT; }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<double>() {
            return MPI_DOUBLE;
        }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<std::complex<float>>() {
            return MPI_FLOAT;
        }
        template <> inline MPI_Datatype mpi_datatype_from_type<std::complex<double>>() {
            return MPI_DOUBLE;
        }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<int>() { return MPI_INT; }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<char>() { return MPI_CHAR; }

        /// Return how many items of the basic type are in T type
        template <typename T> unsigned int get_count_from_type();
        template <> inline unsigned int get_count_from_type<float>() { return 1; }
        template <> inline unsigned int get_count_from_type<double>() { return 1; }
        template <> inline unsigned int get_count_from_type<std::complex<float>>() { return 2; }
        template <> inline unsigned int get_count_from_type<std::complex<double>>() { return 2; }
        template <> inline unsigned int get_count_from_type<int>() { return 1; }
        template <> inline unsigned int get_count_from_type<char>() { return 1; }

        // File descriptor specialization for MpiComm
        template <> struct File<MpiComm> {
            using type = MPI_File;
            constexpr CommType value = MPI;
        };

        inline MPI_File file_open(MpiComm comm, const char *filename, Mode mode) {
            MPI_File fh;
            switch (mode) {
            case CreateForReadWrite:
                // Delete file if it exists
                MPI_File_delete(filename, MPI_INFO_NULL);
                MPI_check(MPI_File_open(comm.comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                        MPI_INFO_NULL, &fh));
                break;
            case ReadWrite:
                MPI_check(MPI_File_open(comm.comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh));
                break;
            }
            return fh;
        }

        inline void preallocate(MPI_File f, std::size_t n) {
            MPI_check(MPI_File_preallocate(f, n));
        }

        inline void seek(MPI_File f, std::size_t offset) {
            MPI_check(MPI_File_seek(f, offset, MPI_SEEK_SET));
        }

        template <typename T> void write(MPI_File f, const T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_write(f, v, n * get_count_from_type<T>(),
                                     mpi_datatype_basic_from_type<T>(), &status));
        }

        template <typename T>
        void write_at(MPI_File f, std::size_t offset, const T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_write_at(f, offset, v, n * get_count_from_type<T>(),
                                        mpi_datatype_basic_from_type<T>(), &status));
        }

        template <typename T> void read(MPI_File f, T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_read(f, v, n * get_count_from_type<T>(),
                                    mpi_datatype_basic_from_type<T>(), &status));
        }

        template <typename T> void read_at(MPI_File f, std::size_t offset, T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_read_at(f, offset, v, n * get_count_from_type<T>(),
                                       mpi_datatype_basic_from_type<T>(), &status));
        }

        inline void close(MPI_File f) { MPI_check(MPI_File_close(f)); }

#endif // SUPERBBLAS_USE_MPI

        // File descriptor specialization for SelfComm
        template <> struct File<SelfComm> {
            using type = std::FILE *;
            static constexpr CommType value = SEQ;
        };

        inline std::FILE *file_open(SelfComm, const char *filename, Mode mode) {
            std::FILE *f = nullptr;
            switch (mode) {
            case CreateForReadWrite: f = std::fopen(filename, "w+"); break;
            case ReadWrite: f = std::fopen(filename, "r+"); break;
            }
            if (f == nullptr) {
                std::stringstream ss;
                ss << "Error opening file `" << filename << "'";
                throw std::runtime_error(ss.str());
            }
            return f;
        }

        inline void seek(std::FILE *f, std::size_t offset) {
            if (fseeko(f, offset, SEEK_SET) != 0)
                throw std::runtime_error("Error setting file position");
        }

        template <typename T> void write(std::FILE *f, const T *v, std::size_t n) {
            if (fwrite(v, sizeof(T), n, f) != n)
                throw std::runtime_error("Error writing in a file");
        }

        template <typename T>
        void write_at(std::FILE *f, std::size_t offset, const T *v, std::size_t n) {
            seek(f, offset);
            write(f, v, n);
        }

        template <typename T> void read(std::FILE *f, T *v, std::size_t n) {
            if (fread(v, sizeof(T), n, f) != n)
                throw std::runtime_error("Error reading from a file");
        }

        template <typename T> void read_at(std::FILE *f, std::size_t offset, T *v, std::size_t n) {
            seek(f, offset);
            read(f, v, n);
        }

        inline void preallocate(std::FILE *f, std::size_t n) {
            off_t old_offset = 0, end_of_file;

            // Save the current position on the file
            if ((old_offset = ftello(f)) == -1)
                throw std::runtime_error("Error getting file position");

            // Get the current size of the file
            if (fseeko(f, -1, SEEK_END) != 0) throw std::runtime_error("Error setting file position");
            if ((end_of_file = ftello(f) + 1) == 0)
                throw std::runtime_error("Error getting file position");
            std::size_t curr_size = end_of_file;

            if (curr_size < n) {
                std::vector<char> v(std::min(n - curr_size, (std::size_t)256 * 1024 * 1024));

                if (fseeko(f, 0, SEEK_END) != 0)
                    throw std::runtime_error("Error setting file position");
                while (curr_size < n) {
                    std::size_t d = std::min(n - curr_size, v.size());
                    write(f, v.data(), d);
                    curr_size += d;
                }
            }

            // Restore position on the file
            if (fseeko(f, old_offset, SEEK_SET) != 0)
                throw std::runtime_error("Error setting file position");
        }

        inline void close(std::FILE *f) {
            if (fclose(f) != 0) throw std::runtime_error("Error closing file");
        }

        template <typename T> void change_endianness(T *v, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                char *c = (char *)&v[i];
                for (std::size_t j = 0; j < sizeof(T) / 2; ++j)
                    std::swap(c[i], c[sizeof(T) - 1 - j]);
            }
        }

        struct Storage_context_abstract {
            virtual std::size_t getNdim() { throw std::runtime_error("Not implemented"); }
            virtual CommType getCommType() { throw std::runtime_error("Not implemented"); }
            virtual ~Storage_context_abstract() {}
        };

        template <std::size_t N, typename Comm> struct Storage_context : Storage_context_abstract {
            values_datatype values_type;           ///< type of the nonzero values
            std::size_t header_size;               ///< number of bytes before the field num_chunks
            std::size_t disp;                      ///< number of bytes before the current chunk
            typename File<Comm>::type fh;          ///< file descriptor
            Coor<N> dim;                           ///< global tensor dimensions
            bool change_endianness;                ///< whether to change endianness
            std::size_t num_chunks;                ///< number of chunks written
            std::vector<From_size_item<N>> blocks; ///< list of blocks already written
            std::vector<std::size_t> disps;        ///< displacements of the chunks in bytes

            Storage_context(values_datatype values_type, std::size_t header_size,
                            typename File<Comm>::type fh, Coor<N> dim, bool change_endianness)
                : values_type(values_type),
                  header_size(header_size),
                  disp(header_size + sizeof(double)), // hop over num_chunks
                  fh(fh),
                  dim(dim),
                  change_endianness(change_endianness),
                  num_chunks(0) {}

            std::size_t getNdim() override { return N; }
            CommType getCommType() override { return File<Comm>::value; }
            ~Storage_context() override { close(fh); }
        };

        template <std::size_t N, typename T, typename Comm>
        Storage_context<N, Comm> *get_storage_context(Storage_context_abstract *stoh) {
            if (stoh == nullptr) throw std::runtime_error("The given storage handle is null");
            if (stoh->getNdim() != N || stoh->getCommType() != File<Comm>::value)
                throw std::runtime_error("Invalid storage handle");
            Storage_context<N, Comm> *sto_ctx = (Storage_context<N, Comm> *)stoh;
            if (sto_ctx->values_type != get_values_datatype<T>())
                throw std::runtime_error("Invalid storage handle");
            return sto_ctx;
        }

        /// Return the ranges involved in the source and destination tensors
        /// \param dim0: dimension size for the origin tensor
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param dim1: dimension size for the destination tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied

        template <std::size_t Nd0, std::size_t Nd1, typename From_size0, typename From_size1>
        std::tuple<std::vector<From_size_out<Nd0>>, std::vector<From_size_out<Nd1>>>
        get_ranges_to_send_receive(Coor<Nd0> dim0, const From_size0 &p0, const Order<Nd0> &o0,
                                   const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                                   const Coor<Nd1> dim1, const From_size1 &p1, const Order<Nd1> &o1,
                                   const Coor<Nd1> &from1) {

            Cpu cpu{0};
            tracker<Cpu> _t("comp. tensor overlaps on storage", cpu);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            std::vector<From_size_out<Nd0>> send_out;
            std::vector<From_size_out<Nd1>> receive_out;
            for (unsigned int j = 0; j < p1.size(); ++j) {
                From_size_out<Nd0> s(p0.size(), cpu); // ranges to send
                From_size_out<Nd1> r(p0.size(), cpu); // ranges to receive
                for (unsigned int i = 0; i < p0.size(); ++i) {
                    // Restrict the local range in v0 to the range from0, size0
                    Coor<Nd0> rlocal_from0, rlocal_size0;
                    intersection<Nd0>(from0, size0, p0[i][0], p0[i][1], dim0, rlocal_from0,
                                      rlocal_size0);

                    // Translate the restricted range to the destination lattice
                    Coor<Nd1> rfrom1, rsize1;
                    translate_range(rlocal_from0, rlocal_size0, from0, dim0, from1, dim1, perm0,
                                    rfrom1, rsize1);

                    // Compute the range to receive
                    intersection<Nd1>(rfrom1, rsize1, p1[j][0], p1[j][1], dim1, r[i][0], r[i][1]);

                    // Compute the range to send
                    translate_range(r[i][0], r[i][1], from1, dim1, from0, dim0, perm1, s[i][0],
                                    s[i][1]);
                    s[i][0] = normalize_coor(s[i][0] - p0[i][0], dim0);
                }
                send_out.push_back(s);
                receive_out.push_back(r);
            }

            return std::tuple<std::vector<From_size_out<Nd0>>, std::vector<From_size_out<Nd1>>>{
                send_out, receive_out};
        }

        /// Copy the content of tensor v0 into the storage
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param fh: MPI file handler
        /// \param disp: number of bytes from the beginning of the file before the coordinate zero of this block
        /// \param co: coordinate linearization order
        ///
        /// NOTE: the current file position should be at the beginning of the "values" section
        /// where v0 is going to be written.

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename FileT>
        void local_save(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, vector<const T, XPU0> v0,
                        Order<Nd1> o1, Coor<Nd1> from1, Coor<Nd1> dim1, FileT fh, std::size_t disp,
                        CoorOrder co, bool do_change_endianness) {

            tracker<XPU0> _t("local save", v0.ctx());

            // Shortcut for an empty range
            if (volume(size0) == 0) return;

            // Make agree in ordering source and destination
            if (co != SlowToFast) {
                o1 = reverse(o1);
                from1 = reverse(from1);
                dim1 = reverse(dim1);
            }

            // Get the permutation vectors
            Indices<XPU0> indices0;
            Indices<Cpu> indices1;
            IndexType disp0, disp1;
            Cpu cpu = v0.ctx().toCpu();
            get_permutation_origin_cache<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1,
                                                   v0.ctx(), indices0, disp0, co);
            get_permutation_destination_cache<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1,
                                                        cpu, indices1, disp1, co);

            // Write the values of v0 contiguously
            vector<Q, Cpu> v0_host(indices0.size(), cpu);
            copy_n<IndexType, T, Q>(alpha, v0.data() + disp0, indices0.begin(), v0.ctx(),
                                    indices0.size(), v0_host.data(), cpu, EWOp::Copy{});

            // Change endianness
            if (do_change_endianness) change_endianness(v0_host.data(), v0_host.size());

            // Do the copy
            for (std::size_t i = 0; i < indices1.size();) {
                std::size_t n = 1;
                for (; i + n < indices1.size() && indices1[i + n - 1] + 1 == indices1[i + n]; ++n)
                    ;
                write_at(fh, disp + (disp1 + indices1[i]) * sizeof(Q), v0_host.data() + i, n);
                i += n;
            }
        }

        /// Copy from a storage into the tensor v1
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param fh: file handler
        /// \param disp: number of bytes from the beginning of the file before the coordinate zero of this block
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU1,
                  typename EWOP, typename FileT>
        void local_load(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, FileT fh, std::size_t disp,
                        Order<Nd1> o1, Coor<Nd1> from1, Coor<Nd1> dim1, vector<Q, XPU1> v1, EWOP,
                        CoorOrder co, bool do_change_endianness) {

            tracker<XPU1> _t("local load", v1.ctx());

            // Shortcut for an empty range
            if (volume(size0) == 0) return;

            // Make agree in ordering source and destination
            if (co != SlowToFast) {
                o1 = reverse(o1);
                from1 = reverse(from1);
                dim1 = reverse(dim1);
                co = SlowToFast;
            }

            // Get the permutation vectors
            Indices<Cpu> indices0;
            Indices<XPU1> indices1;
            IndexType disp0, disp1;
            Cpu cpu = v1.ctx().toCpu();
            get_permutation_origin_cache<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1, cpu,
                                                   indices0, disp0, co);
            get_permutation_destination_cache<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1,
                                                        v1.ctx(), indices1, disp1, co);

            // Do the reading
            vector<T, Cpu> v0(indices0.size(), cpu);
            for (std::size_t i = 0; i < indices0.size();) {
                std::size_t n = 1;
                for (; i + n < indices0.size() && indices0[i + n - 1] + 1 == indices0[i + n]; ++n)
                    ;
                read_at(fh, disp + (disp0 + indices0[i]) * sizeof(T), v0.data() + i, n);
                i += n;
            }

            // Change endianness
            if (do_change_endianness) change_endianness(v0.data(), v0.size());

            // Write the values of v0 into v1
            copy_n<IndexType, T, Q>(alpha, v0.data(), v0.ctx(), indices0.size(), v1.data() + disp1,
                                    indices1.data(), v1.ctx(), EWOP{});
        }

        /// Copy the content of plural tensor v0 into a storage
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the storage
        /// \param sto: storage context
        /// \param comm: communicator context
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1>
        void save(typename elem<T>::type alpha, const From_size<Nd0> &p0, const Coor<Nd0> &from0,
                  const Coor<Nd0> &size0, const Order<Nd0> &o0,
                  const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, Order<Nd1> o1,
                  Storage_context<Nd1, Comm> &sto, Coor<Nd1> from1, Comm comm, CoorOrder co) {

            // Turn o1 and from1 into SlowToFast
            if (co == FastToSlow) {
                o1 = reverse(o1);
                from1 = reverse(from1);
            }

            // Generate the list of subranges to send from each component from v0 to v1
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            auto send_receive = get_ranges_to_send_receive(dim0, p0, o0, from0, size0, sto.dim,
                                                           sto.blocks, o1, from1);

            // Do the local file modifications
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            for (unsigned int j = 0; j < sto.blocks.size(); ++j) {
                const From_size<Nd0> &toSend = std::get<0>(send_receive)[j];
                const From_size<Nd1> &toReceive = std::get<1>(send_receive)[j];
                for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                    int i = c0.componentId + comm.rank * ncomponents0;
                    assert(check_equivalence(o0, toSend[i][1], o1, toReceive[i][1]));
                    local_save<Nd0, Nd1, T, Q>(alpha, o0, toSend[i][0], toSend[i][1], c0.dim, c0.it,
                                               o1, toReceive[i][0], sto.dim, sto.fh, sto.disps[j],
                                               co, sto.change_endianness);
                }
                for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                    int i = c0.componentId + comm.rank * ncomponents0;
                    assert(check_equivalence(o0, toSend[i][1], o1, toReceive[i][1]));
                    local_save<Nd0, Nd1, T, Q>(alpha, o0, toSend[i][0], toSend[i][1], c0.dim, c0.it,
                                               o1, toReceive[i][0], sto.dim, sto.fh, sto.disps[j],
                                               co, sto.change_endianness);
                }
            }
        }

        /// Copy a range from a storage into a plural tensor v0
        /// \param alpha: factor on the copy
        /// \param sto: storage context
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o0: dimension labels for the origin tensor
        /// \param p1: partitioning of the origin tensor in consecutive ranges
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param o1: dimension labels for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1, typename EWOP>
        void load(typename elem<T>::type alpha, Storage_context<Nd0, Comm> &sto, Coor<Nd0> from0,
                  Coor<Nd0> size0, Order<Nd0> o0, const From_size<Nd1> &p1, const Coor<Nd1> &from1,
                  const Order<Nd1> &o1, const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm,
                  EWOP, CoorOrder co) {

            // Turn o0, from0, and size0 into SlowToFast
            if (co == FastToSlow) {
                o0 = reverse(o0);
                from0 = reverse(from0);
                size0 = reverse(size0);
            }

            // Generate the list of subranges to send from each component from v0 to v1
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);
            auto send_receive = get_ranges_to_send_receive(sto.dim, sto.blocks, o0, from0, size0,
                                                           dim1, p1, o1, from1);

            // Do the local file modifications
            unsigned int ncomponents0 = v1.first.size() + v1.second.size();
            for (unsigned int j = 0; j < sto.blocks.size(); ++j) {
                const From_size<Nd0> &toSend = std::get<0>(send_receive)[j];
                const From_size<Nd1> &toReceive = std::get<1>(send_receive)[j];
                for (const Component<Nd1, Q, XPU0> &c0 : v1.first) {
                    int i = c0.componentId + comm.rank * ncomponents0;
                    assert(check_equivalence(o0, toSend[i][1], o1, toReceive[i][1]));
                    local_load<Nd0, Nd1, T, Q>(alpha, o0, toSend[i][0], toSend[i][1],
                                               sto.blocks[j][1], sto.fh, sto.disps[j], o1,
                                               toReceive[i][0], c0.dim, c0.it, EWOP{}, co,
                                               sto.change_endianness);
                }
                for (const Component<Nd1, Q, XPU1> &c0 : v1.second) {
                    int i = c0.componentId + comm.rank * ncomponents0;
                    assert(check_equivalence(o0, toSend[i][1], o1, toReceive[i][1]));
                    local_load<Nd0, Nd1, T, Q>(alpha, o0, toSend[i][0], toSend[i][1],
                                               sto.blocks[j][1], sto.fh, sto.disps[j], o1,
                                               toReceive[i][0], c0.dim, c0.it, EWOP{}, co,
                                               sto.change_endianness);
                }
            }
        }

        /// Create a file where to store a tensor
        /// \param dim: tensor dimensions
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
        /// \param filename: path and name of the file
        /// \param metadata: metadata content
        /// \param metadata_length: number of characters in the metadata
        /// \param stoh (out) handle to a tensor storage
        ///
        /// If the file exists, its content will be lost

        template <std::size_t Nd, typename T, typename Comm>
        Storage_context<Nd, Comm> create_storage(Coor<Nd> dim, CoorOrder co, const char *filename,
                                                 const char *metadata, int metadata_length,
                                                 Comm comm) {

            if (co == FastToSlow) dim = detail::reverse(dim);

            // Check that int has a size of 4
            if (sizeof(int) != 4) throw std::runtime_error("Expected int to have size 4");

            // Create file
            typename File<Comm>::type fh = file_open(comm, filename, CreateForReadWrite);

            // Root process writes down header
            std::size_t padding_size = (8 - metadata_length % 8) % 8 + 4;
            if (comm.rank == 0) {
                // Write magic_number
                int i32 = magic_number;
                write(fh, &i32, 1);

                // Write version
                i32 = 0;
                write(fh, &i32, 1);

                // Write values_datatype
                i32 = get_values_datatype<T>();
                write(fh, &i32, 1);

                // Write number of dimensions
                i32 = Nd;
                write(fh, &i32, 1);

                // Write metadata
                write(fh, &metadata_length, 1);
                write(fh, metadata, metadata_length);

                // Write padding
                std::vector<char> padding(padding_size);
                write(fh, padding.data(), padding.size());

                // Write tensor size
                std::array<double, Nd> dimd;
                std::copy_n(dim.begin(), Nd, dimd.begin());
                write(fh, &dimd[0], Nd);

                // Write num_chunks
                double d = 0;
                write(fh, &d, 1);
            }

            // Create the handler
            std::size_t header_size =
                sizeof(int) * 5 + metadata_length + padding_size + sizeof(double) * Nd;
            return Storage_context<Nd, Comm>{get_values_datatype<T>(), header_size, fh, dim, false};
        }

        /// Read fields in the header of a storage
        /// \param filename: path and name of the file
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
        /// \param values_dtype: (out) type of the values
        /// \param metadata: (out) metadata content
        /// \param size: (out)tensor dimensions
        /// \param header_size: (out) number of bytes before the first chunk
        /// \param fh: (out) file handler

        template <typename Comm>
        void open_storage(const char *filename, CoorOrder co, values_datatype &values_dtype,
                          std::vector<char> &metadata, std::vector<IndexType> &size,
                          std::size_t &header_size, bool &do_change_endianness, Comm comm,
                          typename File<Comm>::type &fh) {

            // Check that int has a size of 4
            if (sizeof(int) != 4) throw std::runtime_error("Expected int to have size 4");

            // Open the existing file for reading and writing
            fh = file_open(comm, filename, ReadWrite);

            // Read magic_number and check Endianness
            do_change_endianness = false;
            int i32;
            read(fh, &i32, 1);
            if (i32 != magic_number) {
                change_endianness(&i32, 1);
                if (i32 != magic_number) {
                    throw std::runtime_error("Unexpected value for the magic number; the file may "
                                             "not be a tensor storage");
                }
                do_change_endianness = true;
            }

            // Read version
            read(fh, &i32, 1);
            if (do_change_endianness) change_endianness(&i32, 1);
            if (i32 != 0)
                throw std::runtime_error(
                    "Unsupported version of the tensor format; try a newer version of supperbblas");

            // Read values_datatype
            read(fh, &i32, 1);
            if (do_change_endianness) change_endianness(&i32, 1);
            values_dtype = (values_datatype)i32;

            // Read the number of dimensions
            int Nd = 0;
            if (do_change_endianness) change_endianness(&Nd, 1);
            read(fh, &Nd, 1);

            // Read metadata
            int metadata_length = 0;
            read(fh, &metadata_length, 1);
            if (do_change_endianness) change_endianness(&metadata_length, 1);
            metadata.resize(metadata_length);
            read(fh, metadata.data(), metadata_length);

            // Read padding
            std::vector<char> padding((8 - metadata_length % 8) % 8 + 4);
            read(fh, padding.data(), padding.size());

            // Read tensor size
            std::vector<double> dimd(Nd);
            read(fh, &dimd[0], Nd);
            if (do_change_endianness) change_endianness(&dimd[0], Nd);
            size.resize(Nd);
            std::copy_n(dimd.begin(), Nd, size.begin());
            if (co == FastToSlow) std::reverse(size.begin(), size.end());

            // Compute total header size
            header_size = sizeof(int) * 5 + metadata_length + padding.size() + sizeof(double) * Nd;
        }

        /// Add blocks to storage
        /// \param p: partitioning of the origin tensor in consecutive ranges
        /// \param num_blocks: number of items in p
        /// \param stoh: handle to a tensor storage
        /// \param comm: communicator context
        /// \param co: coordinates order

        template <std::size_t Nd1, typename Q, typename Comm>
        void append_blocks(const PartitionItem<Nd1> *p, std::size_t num_blocks,
                           Storage_context<Nd1, Comm> &sto, Comm comm, CoorOrder co) {

            // Write the coordinates for all non-empty blocks
            sto.blocks.reserve(sto.blocks.size() + num_blocks);

            if (comm.rank == 0) seek(sto.fh, sto.disp + sizeof(double)); // skip num_blocks

            std::size_t num_values = 0;          ///< number of values in this chunk
            std::size_t num_nonempty_blocks = 0; ///< number of non-empty blocks
            for (std::size_t i = 0; i < num_blocks; ++i) {
                std::size_t vol = volume(p[i][1]);
                if (vol > 0) {
                    From_size_item<Nd1> fs =
                        (co == SlowToFast ? p[i]
                                          : From_size_item<Nd1>{detail::reverse(p[i][0]),
                                                                detail::reverse(p[i][1])});
                    num_nonempty_blocks++;
                    num_values += vol;
                    sto.blocks.push_back(fs);

                    // Root process writes the "from" and "size" for each block
                    if (comm.rank == 0) {
                        std::array<double, Nd1> from, size;
                        std::copy_n(fs[0].begin(), Nd1, from.begin());
                        std::copy_n(fs[1].begin(), Nd1, size.begin());
                        if (sto.change_endianness) change_endianness(&from[0], Nd1);
                        if (sto.change_endianness) change_endianness(&size[0], Nd1);
                        write(sto.fh, &from[0], Nd1);
                        write(sto.fh, &size[0], Nd1);
                    }
                }
            }

            // Annotate where the nonzero values start for the new blocks
            std::size_t values_start =
                sto.disp + sizeof(double) + num_nonempty_blocks * Nd1 * sizeof(double) * 2;
            sto.disps.reserve(sto.blocks.size());
            for (std::size_t i = 0; i < num_nonempty_blocks; ++i) sto.disps.push_back(values_start);

            // Set the beginning for a new block
            std::size_t new_disp = values_start + sizeof(Q) * num_values;

            // Write the number of blocks in this chunk and preallocate for the values
            if (comm.rank == 0) {
                double d = num_nonempty_blocks;
                if (sto.change_endianness) change_endianness(&d, 1);
                write_at(sto.fh, sto.disp, &d, 1);
                preallocate(sto.fh, new_disp);
            }

            // Update disp
            sto.disp = new_disp;

            // Update num_chunks
            sto.num_chunks++;
            if (comm.rank == 0) {
                double num_chunks = sto.num_chunks;
                if (sto.change_endianness) change_endianness(&num_chunks, 1);
                write_at(sto.fh, sto.header_size, &num_chunks, 1);
            }
        }

        /// Read all blocks from storage
        /// \param stoh: handle to a tensor storage

        template <std::size_t Nd1, typename Q, typename Comm>
        void read_all_blocks(Storage_context<Nd1, Comm> &sto) {

            // Read num_chunks
            double num_chunks = 0;
            std::size_t cur = sto.header_size;
            read_at(sto.fh, cur, &num_chunks, 1);
            cur += sizeof(double);
            if (sto.change_endianness) change_endianness(&num_chunks, 1);
            sto.num_chunks = num_chunks;

            // Read chunks
            sto.blocks.resize(0);
            for (std::size_t chunk = 0; chunk < sto.num_chunks; chunk++) {
                // Read the number of blocks in this chunk
                double d;
                read(sto.fh, &d, 1);
                if (sto.change_endianness) change_endianness(&d, 1);
                std::size_t num_blocks = d;

                // Write the coordinates for all non-empty blocks
                sto.blocks.reserve(sto.blocks.size() + num_blocks);

                std::size_t num_values = 0;          ///< number of values in this chunk
                for (std::size_t i = 0; i < num_blocks; ++i) {
                    // Read from and size
                    std::array<double, Nd1> fromd, sized;
                    read(sto.fh, &fromd[0], Nd1);
                    read(sto.fh, &sized[0], Nd1);
                    if (sto.change_endianness) change_endianness(&fromd[0], Nd1);
                    if (sto.change_endianness) change_endianness(&sized[0], Nd1);
                    Coor<Nd1> from, size;
                    std::copy_n(fromd.begin(), Nd1, from.begin());
                    std::copy_n(sized.begin(), Nd1, size.begin());
                    num_values += volume(size);
                    sto.blocks.push_back(From_size_item<Nd1>{from, size});
                }

                // Annotate where the nonzero values start for the block
                cur += sizeof(double) + num_blocks * Nd1 * sizeof(double) * 2;
                sto.disps.reserve(sto.blocks.size());
                for (std::size_t i = 0; i < num_blocks; ++i) sto.disps.push_back(cur);

                // Set the beginning for a new block
                cur += sizeof(Q) * num_values;
                seek(sto.fh, cur);
            }

            // Update disp
            sto.disp = cur;
        }

        /// Open a storage for reading and writing
        /// \param filename: path and name of the file
        /// \param comm: communicator
        ///
        /// NOTE: If the file does not exist, an exception will raise

        template <std::size_t Nd, typename T, typename Comm>
        Storage_context<Nd, Comm> *open_storage_template(const char *filename, Comm comm) {

            // Open storage and check template parameters
            typename File<Comm>::type fh;
            values_datatype values_dtype;
            std::vector<char> metadata;
            std::vector<IndexType> size;
            std::size_t header_size;
            bool do_change_endianness;
            open_storage(filename, SlowToFast, values_dtype, metadata, size, header_size,
                         do_change_endianness, comm, fh);

            if (values_dtype != get_values_datatype<T>())
                throw std::runtime_error(
                    "The template parameter T does not match with the datatype of the storage");
            if (Nd != size.size())
                throw std::runtime_error(
                    "The template parameter Nd does not match with the number of "
                    "dimensions of the storage");
            Coor<Nd> dim;
            std::copy_n(size.begin(), Nd, dim.begin());

            // Create the handler
            Storage_context<Nd, Comm> *sto = new Storage_context<Nd, Comm>{
                values_dtype, header_size, fh, dim, do_change_endianness};

            // Read the nonzero blocks
            read_all_blocks<Nd, T, Comm>(*sto);

            // Return handler
            return sto;
        }
    }

    using Storage_handle = detail::Storage_context_abstract *;

#ifdef SUPERBBLAS_USE_MPI
    /// Create a file where to store a tensor
    /// \param dim: tensor dimensions
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param filename: path and name of the file
    /// \param metadata: metadata content
    /// \param metadata_length: number of characters in the metadata
    /// \param stoh (out) handle to a tensor storage
    ///
    /// If the file exists, its content will be lost

    template <std::size_t Nd, typename T>
    void create_storage(const Coor<Nd> &dim, CoorOrder co, const char *filename,
                        const char *metadata, int metadata_length, MPI_Comm mpicomm,
                        Storage_handle *stoh) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        *stoh = new Storage_context{
            create_storage<Nd, T>(dim, co, filename, metadata, metadata_length, comm)};
    }

    /// Read fields in the header of a storage
    /// \param filename: path and name of the file
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param values_type: (out) type of the values, 0 is float, 1 is double, 2 is complex float,
    ///        3: complex double
    /// \param metadata: (out) metadata content
    /// \param dim: (out) tensor dimensions

    void read_storage_header(const char *filename, CoorOrder co, values_datatype &values_dtype,
                             std::vector<char> &metadata, std::vector<IndexType> &size,
                             MPI_Comm mpicomm) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        File fh;
        std::size_t header_size;
        bool do_change_endianness;
        detail::open_storage(filename, co, values_dtype, metadata, size, header_size,
                             do_change_endianness, comm, fh);
        close(fh);
    }

    /// Open a storage for reading and writing
    /// \param filename: path and name of the file
    /// \param stoh (out) handle to a tensor storage
    ///
    /// NOTE: If the file does not exist, an exception will raise

    template <std::size_t Nd, typename T>
    void open_storage(const char *filename, MPI_Comm mpicomm, Storage_handle *stoh) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        // Open storage and check template parameters
        *stoh = detail::open_storage_template<Nd, T>(filename, comm);
    }

    /// Add blocks to storage
    /// \param p: partitioning of the origin tensor in consecutive ranges
    /// \param num_blocks: number of items in p
    /// \param stoh: handle to a tensor storage
    /// \param mpicomm: MPI communicator context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd1> *p, int num_blocks, Storage_handle stoh,
                       MPI_Comm mpicomm, CoorOrder co) {

        Storage_context<Nd1> &sto = *get_storage_context<Nd1, Q>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::append_blocks<Nd1, Q>(p, num_blocks, sto, comm, co);
    }

    /// Copy the content of plural tensor v0 into a storage
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param mpicomm: MPI communicator context
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param o1: dimension labels for the storage
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param stoh: handle to a tensor storage
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void save(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const T **v0,
              const Context *ctx0, const char *o1, const Coor<Nd1> &from1, Storage_handle stoh,
              MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Storage_context<Nd1> *sto_ctx = get_storage_context<Nd1, Q>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::save<Nd0, Nd1, T, Q>(
            detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), from0, size0,
            detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0, comm, session),
            detail::toArray<Nd1>(o1, "o1"), *sto_ctx, from1, comm, co);
    }

    /// Copy from a storage into a plural tensor v1
    /// \param alpha: factor applied to v0
    /// \param stoh: handle to a tensor storage
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void load(typename elem<T>::type alpha, Storage_handle stoh, const char *o0,
              const Coor<Nd0> from0, const Coor<Nd0> size0, const PartitionItem<Nd1> *p1,
              int ncomponents1, const char *o1, const Coor<Nd1> from1, Q **v1, const Context *ctx1,
              MPI_Comm mpicomm, CoorOrder co, CopyAdd copyadd, Session session = 0) {

        Storage_context<Nd1> *sto_ctx = get_storage_context<Nd1, Q>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        if (copyadd == Copy)
            detail::load<Nd0, Nd1>(
                alpha, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1,
                detail::toArray<Nd1>(o1, "o1"),
                detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm, session), comm,
                EWOp::Copy{}, co);
        else
            detail::load<Nd0, Nd1>(
                alpha, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1,
                detail::toArray<Nd1>(o1, "o1"),
                detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm, session), comm,
                EWOp::Add{}, co);
    }

#endif // SUPERBBLAS_USE_MPI

    /// Create a file where to store a tensor
    /// \param dim: tensor dimensions
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param filename: path and name of the file
    /// \param metadata: metadata content
    /// \param metadata_length: number of characters in the metadata
    /// \param stoh (out) handle to a tensor storage
    ///
    /// If the file exists, its content will be lost

    template <std::size_t Nd, typename T>
    void create_storage(const Coor<Nd> &dim, CoorOrder co, const char *filename,
                        const char *metadata, int metadata_length, Storage_handle *stoh) {

        detail::SelfComm comm = detail::get_comm();

        *stoh = new detail::Storage_context<Nd, detail::SelfComm>{
            create_storage<Nd, T>(dim, co, filename, metadata, metadata_length, comm)};
    }

    /// Read fields in the header of a storage
    /// \param filename: path and name of the file
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param values_type: (out) type of the values, 0 is float, 1 is double, 2 is complex float,
    ///        3: complex double
    /// \param metadata: (out) metadata content
    /// \param dim: (out) tensor dimensions

    void read_storage_header(const char *filename, CoorOrder co, values_datatype &values_dtype,
                             std::vector<char> &metadata, std::vector<IndexType> &size) {

        detail::SelfComm comm = detail::get_comm();

        typename detail::File<detail::SelfComm>::type fh;
        std::size_t header_size;
        bool do_change_endianness;
        detail::open_storage(filename, co, values_dtype, metadata, size, header_size,
                             do_change_endianness, comm, fh);
        detail::close(fh);
    }

    /// Close a storage
    /// \param stoh:  handle to a tensor storage

    void close_storage(Storage_handle stoh) {
        // Destroy handle
        delete stoh;
    }

    /// Open a storage for reading and writing
    /// \param filename: path and name of the file
    /// \param stoh (out) handle to a tensor storage
    ///
    /// NOTE: If the file does not exist, an exception will raise

    template <std::size_t Nd, typename T>
    void open_storage(const char *filename, Storage_handle *stoh) {

        detail::SelfComm comm = detail::get_comm();

        // Open storage and check template parameters
        *stoh = detail::open_storage_template<Nd, T>(filename, comm);
    }

    /// Add blocks to storage
    /// \param p: partitioning of the origin tensor in consecutive ranges
    /// \param num_blocks: number of items in p
    /// \param stoh: handle to a tensor storage
    /// \param mpicomm: MPI communicator context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd1> *p, std::size_t num_blocks, Storage_handle stoh,
                       CoorOrder co) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::append_blocks<Nd1, Q>(p, num_blocks, sto, comm, co);
    }

    /// Copy the content of plural tensor v0 into a storage
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param mpicomm: MPI communicator context
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param o1: dimension labels for the storage
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param stoh: handle to a tensor storage
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void save(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const T **v0,
              const Context *ctx0, const char *o1, const Coor<Nd1> &from1, Storage_handle stoh,
              CoorOrder co, Session session = 0) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::save<Nd0, Nd1, T, Q>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, session), from0, size0,
            detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0, comm, session),
            detail::toArray<Nd1>(o1, "o1"), sto, from1, comm, co);
    }

    /// Copy from a storage into a plural tensor v1
    /// \param alpha: factor applied to v0
    /// \param stoh: handle to a tensor storage
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void load(typename elem<T>::type alpha, Storage_handle stoh, const char *o0,
              const Coor<Nd0> from0, const Coor<Nd0> size0, const PartitionItem<Nd1> *p1,
              int ncomponents1, const char *o1, const Coor<Nd1> from1, Q **v1, const Context *ctx1,
              CoorOrder co, CopyAdd copyadd, Session session = 0) {

        detail::Storage_context<Nd0, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd0, T, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        if (copyadd == Copy)
            detail::load<Nd0, Nd1, T, Q>(
                alpha, sto, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1,
                detail::toArray<Nd1>(o1, "o1"),
                detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm, session), comm,
                detail::EWOp::Copy{}, co);
        else
            detail::load<Nd0, Nd1, T, Q>(
                alpha, sto, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                detail::get_from_size(p1, ncomponents1 * comm.nprocs, session), from1,
                detail::toArray<Nd1>(o1, "o1"),
                detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1, comm, session), comm,
                detail::EWOp::Add{}, co);
    }
}

#endif // __SUPERBBLAS_STORAGE__
