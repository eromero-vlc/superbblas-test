#ifndef __SUPERBBLAS_STORAGE__
#define __SUPERBBLAS_STORAGE__

#include "dist.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <list>
#include <set>
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

        /// Call used for open the storage
        enum CommType {
            SEQ, ///< without MPI
            MPI  ///< with MPI
        };

        //
        // Low layer implementation without MPI
        //

        // File descriptor specialization for SelfComm
        template <> struct File<SelfComm> {
            using type = std::FILE *;
            static constexpr CommType value = SEQ;
        };

        template <typename Str> void gen_error(const Str &error_msg) {
            std::stringstream ss;
            ss << error_msg << ": " << strerror(errno);
            throw std::runtime_error(ss.str());
        }

        inline std::FILE *file_open(SelfComm, const char *filename, Mode mode) {
            std::FILE *f = nullptr;
            switch (mode) {
            case CreateForReadWrite: f = std::fopen(filename, "wb+"); break;
            case ReadWrite: f = std::fopen(filename, "rb+"); break;
            }
            if (f == nullptr) {
                std::stringstream ss;
                ss << "Error opening file `" << filename << "'";
                gen_error(ss.str());
            }
            return f;
        }

        inline void seek(std::FILE *f, std::size_t offset) {
            if (offset >= std::numeric_limits<long>::max())
                gen_error("Too small type to represent the displacement");
            if (std::fseek(f, offset, SEEK_SET) != 0) gen_error("Error setting file position");
        }

        template <typename T> void write(std::FILE *f, const T *v, std::size_t n) {
            if (std::fwrite(v, sizeof(T), n, f) != n) gen_error("Error writing in a file");
        }

        template <typename T> void read(std::FILE *f, T *v, std::size_t n) {
            if (std::fread(v, sizeof(T), n, f) != n) gen_error("Error reading from a file");
        }

        inline void preallocate(std::FILE *f, std::size_t n) {
            off_t old_offset = 0, end_of_file;
            if (n >= std::numeric_limits<long>::max())
                throw std::runtime_error("Too small type to represent the displacement");

            // Save the current position on the file
            if ((old_offset = std::ftell(f)) == -1) gen_error("Error getting file position");

            // Get the current size of the file
            if (std::fseek(f, -1, SEEK_END) != 0) gen_error("Error setting file position");
            if ((end_of_file = std::ftell(f) + 1) == 0) gen_error("Error getting file position");
            std::size_t curr_size = end_of_file;

            if (curr_size < n) {
                std::vector<char> v(std::min(n - curr_size, (std::size_t)256 * 1024 * 1024));

                if (std::fseek(f, 0, SEEK_END) != 0) gen_error("Error setting file position");
                while (curr_size < n) {
                    std::size_t d = std::min(n - curr_size, v.size());
                    write(f, v.data(), d);
                    curr_size += d;
                }
            }

            // Restore position on the file
            if (std::fseek(f, old_offset, SEEK_SET) != 0) gen_error("Error setting file position");
        }

        inline void flush(std::FILE *f) {
            if (std::fflush(f) != 0) gen_error("Error flushing file");
        }

        inline void close(std::FILE *f) {
            if (std::fclose(f) != 0) gen_error("Error closing file");
        }

        //
        // Low layer implementation with MPI
        //

#ifdef SUPERBBLAS_USE_MPI
#    ifdef SUPERBBLAS_USE_MPIIO
        /// Use MPI IO

        /// Return the MPI_Datatype of a type
        template <typename T> MPI_Datatype mpi_datatype_basic_from_type();
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<float>() { return MPI_FLOAT; }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<double>() {
            return MPI_DOUBLE;
        }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<std::complex<float>>() {
            return MPI_FLOAT;
        }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<std::complex<double>>() {
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

        struct AllocAbstract {
            MPI_Request req;
            AllocAbstract(MPI_Request req) : req{req} {}
            virtual ~AllocAbstract() {}
        };

        template <typename T> struct Alloc : public AllocAbstract {
            vector<T, Cpu> v;
            Alloc(MPI_Request req, vector<T, Cpu> v) : AllocAbstract{req}, v{v} {}
            ~Alloc() override {}
        };

        // MPI_File replacement
        struct File_Requests {
            MPI_File f;
            std::list<AllocAbstract *> reqs;
        };

        // File descriptor specialization for MpiComm
        template <> struct File<MpiComm> {
            using type = File_Requests;
            static constexpr CommType value = MPI;
        };

        inline File_Requests file_open(MpiComm comm, const char *filename, Mode mode) {
            MPI_File fh;
            barrier(comm);
            switch (mode) {
            case CreateForReadWrite:
                // Delete file if it exists
                MPI_File_delete(filename, MPI_INFO_NULL);
                MPI_check(MPI_File_open(comm.comm, filename, MPI_MODE_CREATE | MPI_MODE_RDWR,
                                        MPI_INFO_NULL, &fh));
                break;
            case ReadWrite:
                MPI_check(MPI_File_open(comm.comm, filename, MPI_MODE_RDWR, MPI_INFO_NULL, &fh));
                break;
            }
            return File_Requests{fh, {}};
        }

        inline void seek(File_Requests &f, std::size_t offset) {
            MPI_check(MPI_File_seek(f.f, offset, MPI_SEEK_SET));
        }

        template <typename T> void write(File_Requests &f, const T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_write(f.f, v, n * get_count_from_type<T>(),
                                     mpi_datatype_basic_from_type<T>(), &status));
        }

        template <typename T> void read(File_Requests &f, T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_read(f.f, v, n * get_count_from_type<T>(),
                                    mpi_datatype_basic_from_type<T>(), &status));
        }

        template <typename T>
        void iwrite(File_Requests &f, const T *v, std::size_t n, vector<T, Cpu> w) {
            MPI_Request req;
            MPI_check(MPI_File_iwrite(f.f, v, n * get_count_from_type<T>(),
                                      mpi_datatype_basic_from_type<T>(), &req));
            f.reqs.push_back(new Alloc<T>{req, w});
        }

        template <typename T> void iwrite(File_Requests &f, const T *v, std::size_t n) {
            vector<T, Cpu> w(n, Cpu{0});
            std::copy_n(v, n, w.data());
            iwrite(f, w.data(), n, w);
        }

        inline void flush(File_Requests &f) {
            for (AllocAbstract *r : f.reqs) {
                MPI_check(MPI_Wait(&r->req, MPI_STATUS_IGNORE));
                delete r;
            }
            f.reqs.clear();
            MPI_check(MPI_File_sync(f.f));
        }

        inline void check_pending_requests(File_Requests &f) {
            f.reqs.remove_if([](AllocAbstract *r) {
                int finished = 0;
                MPI_check(MPI_Test(&r->req, &finished, MPI_STATUS_IGNORE));
                if (finished) {
                    delete r;
                    return true;
                }
                return false;
            });
        }

        inline void preallocate(File_Requests &f, std::size_t n) {
            flush(f);
            MPI_check(MPI_File_preallocate(f.f, n));
        }

        inline void close(File_Requests &f) {
            flush(f);
            MPI_check(MPI_File_close(&f.f));
        }

#    else  // SUPERBBLAS_USE_MPIIO
        /// Don't use MPI IO

        // MPI_File replacement
        struct File_Comm {
            std::FILE *f;
            MpiComm comm;
        };

        // File descriptor specialization for MpiComm
        template <> struct File<MpiComm> {
            using type = File_Comm;
            static constexpr CommType value = MPI;
        };

        inline File_Comm file_open(MpiComm comm, const char *filename, Mode mode) {
            std::FILE *f = nullptr;
            // Avoid all processes to create the file at the same time; so root process create the file, and the rest open it
            if (comm.rank == 0) {
                f = file_open(detail::get_comm(), filename, mode);
                barrier(comm);
            } else {
                barrier(comm);
                f = file_open(detail::get_comm(), filename, ReadWrite);
            }
            return {f, comm};
        }

        inline void preallocate(File_Comm f, std::size_t n) {
            if (f.comm.rank == 0) preallocate(f.f, n);
            barrier(f.comm);
        }

        inline void seek(File_Comm f, std::size_t offset) { seek(f.f, offset); }

        template <typename T> void write(File_Comm f, const T *v, std::size_t n) {
            write(f.f, v, n);
        }

        template <typename T> void iwrite(File_Comm f, const T *v, std::size_t n) {
            write(f.f, v, n);
        }

        template <typename T> void iwrite(File_Comm f, const T *v, std::size_t n, vector<T, Cpu>) {
            write(f.f, v, n);
        }

        template <typename T> void read(File_Comm f, T *v, std::size_t n) { read(f.f, v, n); }

        inline void flush(File_Comm f) {
            flush(f.f);
            barrier(f.comm);
        }

        inline void check_pending_requests(File_Comm) {}

        inline void close(File_Comm &f) {
            close(f.f);
            barrier(f.comm);
        }
#    endif // SUPERBBLAS_USE_MPIIO

#endif // SUPERBBLAS_USE_MPI

        /// Range begin, end, and current state
        template <std::size_t N, typename GRID> struct Grid_range {
            // Current iterators
            std ::array<typename GRID::const_iterator, N> it;
            // List of N grids
            const GRID *const grid;
            /// Don't iterate on this dimension
            std::size_t const excepting;

            Grid_range(const GRID *grid, std::size_t excepting = N)
                : grid(grid), excepting(excepting) {
                for (std::size_t i = 0; i < N; ++i) {
                    if (i == excepting) continue;
                    it[i] = grid[i].begin();
                }
            }

            std::size_t volume() const {
                if (N == 0 || (N == 1 && excepting == 0)) return 0;
                std::size_t vol = 1;
                for (std::size_t i = 0; i < N; ++i) {
                    if (i == excepting) continue;
                    vol *= grid[i].size();
                }
                return vol;
            }

            Grid_range<N, GRID> &operator++() {
                for (std::size_t i = 0; i < N; ++i) {
                    if (i == excepting) continue;
                    ++it[i];
                    if (it[i] != grid[i].end()) return *this;
                    it[i] = grid[i].begin();
                }
                return *this;
            }

            Grid_range<N, GRID> &operator--() {
                for (std::size_t i = 0; i < N; ++i) {
                    if (i == excepting) continue;
                    if (it[i] != grid[i].begin()) {
                        --it[i];
                        return *this;
                    }
                    it[i] = --grid[i].end();
                }
                return *this;
            }
        };

        /// Data-structure to accelerate the intersection of sparse tensors
        template <std::size_t N, typename Key = void> struct GridHash {
            /// Index element in `blocks` and `values`
            using BlockIndex = std::size_t;
            /// Sparse tensor dimensions
            Coor<N> dim;
            /// Nonzero blocks of the sparse tensors
            std::vector<From_size_item<N>> blocks;
            /// Values associated to each block
            std::vector<Key> values;

            /// Ordered list
            template <typename T> using ordered_list = std::set<T>;
            /// Set of hyperplanes forming a non-regular grid
            using Grid = std::array<ordered_list<IndexType>, N>;
            /// Unordered hyperplanes
            using Unsorted_Grid = std::array<std::vector<IndexType>, N>;

            /// Set of hyperplanes containing the faces of each nonzero subtensor
            Grid grid;
            /// From grid coordinate index (SlowToFast) to `blocks` and `values` indices
            std::unordered_multimap<std::size_t, BlockIndex> gridToBlocks;

            GridHash(Coor<N> dim) : dim{dim}, grid{}, gridToBlocks{16} {
                assert(check_positive(dim));
            }

            void append_block(Coor<N> from, Coor<N> size, Key key) {
                // Shortcut for empty ranges
                std::size_t vol = volume(size);
                if (vol == 0) return;

                // Normalize from when being the whole dimension
                from = normalize_from(from, size);

                // Check if the block has overlaps with other blocks in this tensor
                std::size_t vol_overlaps = get_overlap_volume(from, size);
                if (vol == vol_overlaps) return;
                if (vol_overlaps != 0)
                    throw std::runtime_error(
                        "Ups! Unsupported the addition of blocks with partial support "
                        "on the sparse tensor");

                // Add the faces of the given range as hyperplanes on the grid
                for (std::size_t i = 0; i < N; ++i) {
                    add_hyperplane(from[i], i);
                    add_hyperplane(from[i] + size[i], i);
                }

                // Add the new block
                BlockIndex new_block_index = blocks.size();
                blocks.push_back({from, size});
                values.push_back(key);

                // Add new block on the grid
                Unsorted_Grid g = grid_intersection(from, size);
                Grid_range<N, std::vector<IndexType>> git(&g[0]);
                Coor<N> dim_strides = get_strides(dim, SlowToFast);
                for (std::size_t g_i = 0, g_vol = git.volume(); g_i < g_vol; ++g_i, ++git) {
                    Coor<N> g_coor;
                    for (std::size_t i = 0; i < N; ++i) g_coor[i] = *git.it[i];
                    std::size_t new_grid_index = coor2index(g_coor, dim, dim_strides);
                    gridToBlocks.insert(std::make_pair(new_grid_index, new_block_index));
                }
            }

            /// Return a list of the blocks, the intersection relative to the blocks, and the
            /// associated keys of blocks with non-empty overlap with the given range.
            std::vector<std::pair<std::array<From_size_item<N>, 2>, Key>>
            intersection(Coor<N> from, Coor<N> size) const {
                // Shortcut for empty ranges
                if (volume(size) == 0) return {};

                // Compute the intersections between the given range the grid
                Unsorted_Grid g = grid_intersection(from, size);

                // Compute the return
                std::vector<std::pair<std::array<From_size_item<N>, 2>, Key>> r;
                Grid_range<N, std::vector<IndexType>> git(&g[0]);
                Coor<N> dim_strides = get_strides(dim, SlowToFast);
                std::set<BlockIndex> visited;
                for (std::size_t g_i = 0, g_vol = git.volume(); g_i < g_vol; ++g_i, ++git) {
                    Coor<N> g_coor;
                    for (std::size_t i = 0; i < N; ++i) g_coor[i] = *git.it[i];
                    std::size_t grid_index = coor2index(g_coor, dim, dim_strides);
                    auto range = gridToBlocks.equal_range(grid_index);
                    for (auto it = range.first; it != range.second; ++it) {
                        BlockIndex bidx = it->second;

                        // Skip if already visited
                        if (visited.count(bidx) > 0) continue;

                        // Do intersection between the block and the given range
                        Coor<N> rfrom, rsize;
                        detail::intersection(blocks[bidx][0], blocks[bidx][1], from, size, dim,
                                             rfrom, rsize);
                        if (volume(rsize) == 0) continue;
                        rfrom = normalize_coor(rfrom - blocks[bidx][0], dim);
                        r.push_back({{blocks[bidx], {rfrom, rsize}}, values[bidx]});

                        // Note the visited block
                        visited.insert(bidx);
                    }
                }

                return r;
            }

            /// Return the overlap volume of the given range on this tensor
            std::size_t get_overlap_volume(Coor<N> from, Coor<N> size) {
                auto overlaps = intersection(from, size);
                std::size_t vol_overlaps = 0;
                for (const auto &i : overlaps) vol_overlaps += volume(i.first[1][1]);
                return vol_overlaps;
            }

        private:
            /// Return the first hyperplane whose slice contains the point
            ordered_list<IndexType>::iterator get_hyperslice(IndexType from, std::size_t n) const {
                if (grid[n].size() == 0) return grid[n].end();

                Grid_range<1, std::set<IndexType>> git(&grid[n]);
                git.it[0] = grid[n].lower_bound(from);
                if (git.it[0] == grid[n].end()) return --grid[n].end();
                if (*git.it[0] != from) --git;
                return git.it[0];
            }

            /// Return the subgrid with overlaps with the given range
            Unsorted_Grid grid_intersection(Coor<N> from, Coor<N> size) const {
                // Normalize from when being the whole dimension
                from = normalize_from(from, size);

                Unsorted_Grid r{};
                for (std::size_t i = 0; i < N; ++i) {
                    if (grid[i].size() == 0) continue;
                    if (grid[i].size() == 1) {
                        r[i].push_back(*grid[i].begin());
                        continue;
                    }

                    Grid_range<1, std::set<IndexType>> git(&grid[i]);
                    git.it[0] = get_hyperslice(from[i], i);
                    IndexType gFrom = *git.it[0];
                    ++git;
                    for (std::size_t j = 0, vol = git.volume();
                         j < vol && has_overlap(from[i], size[i], gFrom,
                                                normalize_coor(*git.it[0] - gFrom, dim[i]), dim[i]);
                         gFrom = *git.it[0], ++git, ++j)
                        r[i].push_back(gFrom);
                }
                return r;
            }

            void add_hyperplane(IndexType from, std::size_t n) {
                // Normalize
                from = detail::normalize_coor(from, dim[n]);

                // Skip if the hyperplane is already in the grid
                if (grid[n].count(from) > 0) return;

                if (grid[n].size() > 0) {
                    // Figure out the hyperplanes that contain `from`
                    IndexType gFrom = *get_hyperslice(from, n);

                    // Insert the new hyperplanes
                    Coor<N> dim_strides = get_strides(dim, SlowToFast);
                    Grid_range<N, std::set<IndexType>> git(&grid[0], n);
                    std::vector<BlockIndex> affected_blocks;
                    for (std::size_t i = 0, vol = git.volume(); i < vol; ++git, ++i) {
                        // Get the coordinates of an old cell
                        Coor<N> g_coor;
                        for (std::size_t j = 0; j < N; ++j)
                            if (j != n) g_coor[j] = *git.it[j];
                        g_coor[n] = gFrom;
                        std::size_t grid_index = coor2index(g_coor, dim, dim_strides);

                        // Get the coordinates of the new cell
                        g_coor[n] = from;
                        std::size_t new_grid_index = coor2index(g_coor, dim, dim_strides);

                        // Insert all subtensors on the old cell also on the new cell
                        auto range = gridToBlocks.equal_range(grid_index);
                        affected_blocks.resize(0);
                        for (auto it = range.first; it != range.second; ++it)
                            affected_blocks.push_back(it->second);
                        for (BlockIndex b : affected_blocks)
                            gridToBlocks.insert(std::make_pair(new_grid_index, b));
                    }
                }

                // Add hyperplanes
                grid[n].insert(from);
            }

            /// Return whether the given ranges overlap
            /// NOTE: the first range can refer to a periodic lattice, but not the second
            static bool has_overlap(IndexType from0, IndexType size0, IndexType from1,
                                    IndexType size1, IndexType dim) {

                if (size0 <= 0 || size1 <= 0) throw std::runtime_error("This shouldn't happen");
                if (from0 + size0 > from1 && from0 < from1 + size1) return true;
                from1 += dim;
                if (from0 + size0 > from1 && from0 < from1 + size1) return true;
                return false;
            }

            // Normalize from when being the whole dimension
            Coor<N> normalize_from(Coor<N> from, const Coor<N> &size) const {
                for (std::size_t i = 0; i < N; ++i)
                    if (size[i] == dim[i]) from[i] = 0;
                return normalize_coor(from, dim);
            }
        };

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
            virtual void flush() {}
            virtual void preallocate(std::size_t) {}
            virtual ~Storage_context_abstract() {}
        };

        template <std::size_t N, typename Comm> struct Storage_context : Storage_context_abstract {
            values_datatype values_type;  ///< type of the nonzero values
            std::size_t header_size;      ///< number of bytes before the field num_chunks
            std::size_t disp;             ///< number of bytes before the current chunk
            typename File<Comm>::type fh; ///< file descriptor
            Coor<N> dim;                  ///< global tensor dimensions
            bool change_endianness;       ///< whether to change endianness
            bool modified;                ///< whether the storage content changed
            std::size_t num_chunks;       ///< number of chunks written
            GridHash<N, std::size_t>
                blocks; ///< list of blocks already written and their displacements in bytes

            Storage_context(values_datatype values_type, std::size_t header_size,
                            typename File<Comm>::type fh, Coor<N> dim, bool change_endianness)
                : values_type(values_type),
                  header_size(header_size),
                  disp(header_size + sizeof(double)), // hop over num_chunks
                  fh(fh),
                  dim(dim),
                  change_endianness(change_endianness),
                  modified(false),
                  num_chunks(0),
                  blocks(dim) {}

            std::size_t getNdim() override { return N; }
            CommType getCommType() override { return File<Comm>::value; }
            void flush() override { detail::flush(fh); }
            void preallocate(std::size_t size) override { detail::preallocate(fh, size); }
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

        template <std::size_t Nd0, std::size_t Nd1> struct Op {
            From_size_item<Nd0> first_tensor;
            From_size_item<Nd0> first_subtensor;
            From_size_item<Nd1> second_tensor;
            From_size_item<Nd1> second_subtensor;
            std::size_t disp;
        };

        /// Return the ranges involved in the source and destination tensors
        /// \param dim0: dimension size for the origin tensor
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param grid: sparse tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied

        template <std::size_t Nd0, std::size_t Nd1, typename From_size0>
        std::vector<std::vector<Op<Nd0, Nd1>>>
        get_overlap_ranges(Coor<Nd0> dim0, const From_size0 &p0, const Order<Nd0> &o0,
                           const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                           const GridHash<Nd1, std::size_t> &grid, const Order<Nd1> &o1,
                           const Coor<Nd1> &from1) {

            Cpu cpu{0};
            tracker<Cpu> _t("comp. tensor overlaps on storage", cpu);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, grid.dim)));

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            std::vector<std::vector<Op<Nd0, Nd1>>> r(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                // Restrict the local range in v0 to the range from0, size0
                Coor<Nd0> rlocal_from0, rlocal_size0;
                intersection<Nd0>(from0, size0, p0[i][0], p0[i][1], dim0, rlocal_from0,
                                  rlocal_size0);

                // Translate the restricted range to the destination lattice
                Coor<Nd1> rfrom1, rsize1;
                translate_range(rlocal_from0, rlocal_size0, from0, dim0, from1, grid.dim, perm0,
                                rfrom1, rsize1);

                // Compute the range to receive
                auto overlaps = grid.intersection(rfrom1, rsize1);

                for (const auto &o : overlaps) {
                    // Compute the range to left tensor
                    Coor<Nd0> rfrom0, rsize0;
                    translate_range(o.first[0][0] + o.first[1][0], o.first[1][1], from1, grid.dim,
                                    from0, dim0, perm1, rfrom0, rsize0);
                    rfrom0 = normalize_coor(rfrom0 - p0[i][0], dim0);

                    r[i].push_back({p0[i], {rfrom0, rsize0}, o.first[0], o.first[1], o.second});
                }
            }

            return r;
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
                        Order<Nd1> o1, Coor<Nd1> from1, Coor<Nd1> dim1, FileT &fh, std::size_t disp,
                        CoorOrder co, bool do_change_endianness) {

            tracker<XPU0> _t("local save", v0.ctx());

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
                seek(fh, disp + (disp1 + indices1[i]) * sizeof(Q));
                iwrite(fh, v0_host.data() + i, n, v0_host);
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
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, FileT& fh, std::size_t disp,
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
                seek(fh, disp + (disp0 + indices0[i]) * sizeof(T));
                read(fh, v0.data() + i, n);
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

            tracker<XPU1> _t("save", p0.ctx());

            // Turn o1 and from1 into SlowToFast
            if (co == FastToSlow) {
                o1 = reverse(o1);
                from1 = reverse(from1);
            }

            // Generate the list of subranges to send from each component from v0 to v1
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            auto overlaps = get_overlap_ranges(
                dim0, to_vector(p0.data() + comm.rank * ncomponents0, ncomponents0, p0.ctx()), o0,
                from0, size0, sto.blocks, o1, from1);

            // Do the local file modifications
            for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                for (const auto &o : overlaps[c0.componentId]) {
                    assert(check_equivalence(o0, o.first_subtensor[1], o1, o.second_subtensor[1]));
                    local_save<Nd0, Nd1, T, Q>(alpha, o0, o.first_subtensor[0],
                                               o.first_subtensor[1], c0.dim, c0.it, o1,
                                               o.second_subtensor[0], o.second_tensor[1], sto.fh,
                                               o.disp, co, sto.change_endianness);
                }
            }
            for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                for (const auto &o : overlaps[c0.componentId]) {
                    assert(check_equivalence(o0, o.first_subtensor[1], o1, o.second_subtensor[1]));
                    local_save<Nd0, Nd1, T, Q>(alpha, o0, o.first_subtensor[0],
                                               o.first_subtensor[1], c0.dim, c0.it, o1,
                                               o.second_subtensor[0], o.second_tensor[1], sto.fh,
                                               o.disp, co, sto.change_endianness);
                }
            }

            // Mark the storage as modified
            sto.modified = true;

            // Release resources on finished requests
            check_pending_requests(sto.fh);
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

            tracker<XPU1> _t("load", p1.ctx());

            // Turn o0, from0, and size0 into SlowToFast
            if (co == FastToSlow) {
                o0 = reverse(o0);
                from0 = reverse(from0);
                size0 = reverse(size0);
            }

            // Generate the list of subranges to send from each component from v0 to v1
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);
            unsigned int ncomponents1 = v1.first.size() + v1.second.size();
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            auto overlaps = get_overlap_ranges(
                dim1, to_vector(p1.data() + comm.rank * ncomponents1, ncomponents1, p1.ctx()), o1,
                from1, size1, sto.blocks, o0, from0);

            // Synchronize the content of the storage before reading from it
            if (sto.modified) {
                flush(sto.fh);
                sto.modified = false;
            }

            // Do the local file modifications
            for (const Component<Nd1, Q, XPU0> &c1 : v1.first) {
                for (const auto &o : overlaps[c1.componentId]) {
                    assert(check_equivalence(o0, o.second_subtensor[1], o1, o.first_subtensor[1]));
                    local_load<Nd0, Nd1, T, Q>(alpha, o0, o.second_subtensor[0],
                                               o.second_subtensor[1], o.second_tensor[1], sto.fh,
                                               o.disp, o1, o.first_subtensor[0], c1.dim, c1.it,
                                               EWOP{}, co, sto.change_endianness);
                }
            }
            for (const Component<Nd1, Q, XPU1> &c1 : v1.second) {
                for (const auto &o : overlaps[c1.componentId]) {
                    local_load<Nd0, Nd1, T, Q>(alpha, o0, o.second_subtensor[0],
                                               o.second_subtensor[1], o.second_tensor[1], sto.fh,
                                               o.disp, o1, o.first_subtensor[0], c1.dim, c1.it,
                                               EWOP{}, co, sto.change_endianness);
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

        template <std::size_t Nd0, std::size_t Nd1, typename From_size0>
        std::vector<From_size_item<Nd1>>
        translate_ranges(Coor<Nd0> dim0, const From_size0 &p0, const Order<Nd0> &o0,
                         const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd1> dim1,
                         const Order<Nd1> &o1, const Coor<Nd1> &from1) {

            Cpu cpu{0};
            tracker<Cpu> _t("comp. tensor overlaps on storage", cpu);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            std::vector<From_size_item<Nd1>> r(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                // Restrict the local range in v0 to the range from0, size0
                Coor<Nd0> rlocal_from0, rlocal_size0;
                intersection<Nd0>(from0, size0, p0[i][0], p0[i][1], dim0, rlocal_from0,
                                  rlocal_size0);

                // Translate the restricted range to the destination lattice
                translate_range(rlocal_from0, rlocal_size0, from0, dim0, from1, dim1, perm0,
                                r[i][0], r[i][1]);
            }

            return r;
        }

        /// Add blocks to storage after restricted the range indicated by from0, size0, and from1
        /// \param p0: blocks to add
        /// \param num_blocks: number of items in p0
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate of the range to consider on p0
        /// \param size0: number of elements to consider in each dimension
        /// \param o1: dimension labels for the storage
        /// \param sto: storage context
        /// \param from1: first coordinate of the ranges to add
        /// \param comm: communicator context
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename Q, typename Comm>
        void append_blocks(const PartitionItem<Nd0> *p0, std::size_t num_blocks,
                           const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Order<Nd0> &o0,
                           Order<Nd1> o1, Storage_context<Nd1, Comm> &sto, Coor<Nd1> from1,
                           Comm comm, CoorOrder co) {

            tracker<Cpu> _t("append blocks", Cpu{0});

            // Generate the list of subranges to add
            Coor<Nd0> dim0 = get_dim<Nd0>(p0, num_blocks);
            auto p0_ = to_vector(p0, num_blocks, Cpu{0});
            if (co == FastToSlow) {
                o1 = reverse(o1);
                from1 = reverse(from1);
            }
            auto p = translate_ranges(dim0, p0_, o0, from0, size0, sto.dim, o1, from1);

            // Write the coordinates for all non-empty blocks
            std::vector<From_size_item<Nd1>> new_blocks;
            new_blocks.reserve(num_blocks);

            std::vector<std::size_t> num_values; ///< number of values for each block
            num_values.reserve(num_blocks);
            std::size_t num_nonempty_blocks = 0; ///< number of non-empty blocks
            std::vector<double> chunk_header(1); ///< header of the chunk
            chunk_header.reserve(1 + num_blocks * Nd1 * 2);
            for (std::size_t i = 0; i < num_blocks; ++i) {
                // Skip if the range is empty or fully included on the blocks
                std::size_t vol = volume(p[i][1]);
                if (vol == 0 || vol == sto.blocks.get_overlap_volume(p[i][0], p[i][1])) continue;

                const From_size_item<Nd1> &fs = p[i];
                num_nonempty_blocks++;
                num_values.push_back(vol);
                new_blocks.push_back(fs);

                // Root process writes the "from" and "size" for each block
                if (comm.rank == 0) {
                    chunk_header.insert(chunk_header.end(), fs[0].begin(), fs[0].end());
                    chunk_header.insert(chunk_header.end(), fs[1].begin(), fs[1].end());
                }
            }

            // If no new block, get out
            if (num_nonempty_blocks == 0) return;

            // Annotate where the nonzero values start for the new blocks
            std::size_t values_start =
                sto.disp + sizeof(double) + num_nonempty_blocks * Nd1 * sizeof(double) * 2;
            for (std::size_t i = 0; i < num_nonempty_blocks; ++i) {
                sto.blocks.append_block(new_blocks[i][0], new_blocks[i][1], values_start);
                values_start += num_values[i] * sizeof(Q);
            }

            // Write the number of blocks in this chunk and preallocate for the values
            if (comm.rank == 0) {
                chunk_header[0] = num_nonempty_blocks;
                if (sto.change_endianness)
                    change_endianness(chunk_header.data(), chunk_header.size());
                seek(sto.fh, sto.disp);
                iwrite(sto.fh, chunk_header.data(), chunk_header.size());
            }

            // Update disp
            sto.disp = values_start;

            // Update num_chunks
            sto.num_chunks++;
            if (comm.rank == 0) {
                double num_chunks = sto.num_chunks;
                if (sto.change_endianness) change_endianness(&num_chunks, 1);
                seek(sto.fh, sto.header_size);
                iwrite(sto.fh, &num_chunks, 1);
            }
        }

        /// Read all blocks from storage
        /// \param stoh: handle to a tensor storage

        template <std::size_t Nd1, typename Q, typename Comm>
        void read_all_blocks(Storage_context<Nd1, Comm> &sto) {

            // Read num_chunks
            double num_chunks = 0;
            std::size_t cur = sto.header_size;
            seek(sto.fh, cur);
            read(sto.fh, &num_chunks, 1);
            cur += sizeof(double);
            if (sto.change_endianness) change_endianness(&num_chunks, 1);
            sto.num_chunks = num_chunks;

            // Read chunks
            for (std::size_t chunk = 0; chunk < sto.num_chunks; chunk++) {
                // Read the number of blocks in this chunk
                double d;
                read(sto.fh, &d, 1);
                if (sto.change_endianness) change_endianness(&d, 1);
                std::size_t num_blocks = d;

                // Read blocks
                std::vector<std::size_t> num_values;     ///< number of values for each block
                std::vector<From_size_item<Nd1>> blocks; ///< ranges of the blocks
                num_values.reserve(num_blocks);
                blocks.reserve(num_blocks);
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
                    num_values.push_back(volume(size));
                    blocks.push_back(From_size_item<Nd1>{from, size});
                }

                // Annotate where the nonzero values start for the block
                cur += sizeof(double) + num_blocks * Nd1 * sizeof(double) * 2;
                for (std::size_t i = 0; i < num_blocks; ++i) {
                    sto.blocks.append_block(blocks[i][0], blocks[i][1], cur);
                    cur += num_values[i] * sizeof(Q);
                }

                // Set the beginning for a new block
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

        *stoh = new detail::Storage_context<Nd, detail::MpiComm>{
            detail::create_storage<Nd, T>(dim, co, filename, metadata, metadata_length, comm)};
    }

    /// Read fields in the header of a storage
    /// \param filename: path and name of the file
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param values_type: (out) type of the values, 0 is float, 1 is double, 2 is complex float,
    ///        3: complex double
    /// \param metadata: (out) metadata content
    /// \param dim: (out) tensor dimensions

    inline void read_storage_header(const char *filename, CoorOrder co,
                                    values_datatype &values_dtype, std::vector<char> &metadata,
                                    std::vector<IndexType> &size, MPI_Comm mpicomm) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        typename detail::File<detail::MpiComm>::type fh;
        std::size_t header_size;
        bool do_change_endianness;
        detail::open_storage(filename, co, values_dtype, metadata, size, header_size,
                             do_change_endianness, comm, fh);
        detail::close(fh);
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

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::append_blocks<Nd1, Nd1, Q>(p, num_blocks, {}, detail::get_dim(p, num_blocks),
                                           detail::trivial_order<Nd1>(),
                                           detail::trivial_order<Nd1>(), sto, {}, comm, co);
    }

    /// Add blocks to storage
    /// \param p0: blocks to add
    /// \param num_blocks: number of items in p0
    /// \param o0: dimension labels for the blocks to add
    /// \param from0: first coordinate of the range to consider on p0
    /// \param size0: number of elements to copy in each dimension
    /// \param o1: dimension labels for the storage
    /// \param from1: first coordinate of the ranges to add
    /// \param stoh: handle to a tensor storage
    /// \param mpicomm: MPI communicator context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd0> *p0, int num_blocks, const char *o0,
                       const Coor<Nd0> &from0, const Coor<Nd0> &size0, const char *o1,
                       const Coor<Nd1> &from1, Storage_handle stoh, MPI_Comm mpicomm,
                       CoorOrder co) {

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::append_blocks<Nd0, Nd1, Q>(p0, num_blocks, from0, size0,
                                           detail::toArray<Nd0>(o0, "o0"),
                                           detail::toArray<Nd1>(o1, "o1"), sto, from1, comm, co);
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

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

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
              MPI_Comm mpicomm, CoorOrder co, CopyAdd copyadd, Session session = 0) {

        detail::Storage_context<Nd0, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd0, T, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

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
            detail::create_storage<Nd, T>(dim, co, filename, metadata, metadata_length, comm)};
    }

    /// Read fields in the header of a storage
    /// \param filename: path and name of the file
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param values_type: (out) type of the values, 0 is float, 1 is double, 2 is complex float,
    ///        3: complex double
    /// \param metadata: (out) metadata content
    /// \param dim: (out) tensor dimensions

    inline void read_storage_header(const char *filename, CoorOrder co,
                                    values_datatype &values_dtype, std::vector<char> &metadata,
                                    std::vector<IndexType> &size) {

        detail::SelfComm comm = detail::get_comm();

        typename detail::File<detail::SelfComm>::type fh;
        std::size_t header_size;
        bool do_change_endianness;
        detail::open_storage(filename, co, values_dtype, metadata, size, header_size,
                             do_change_endianness, comm, fh);
        detail::close(fh);
    }

    /// Extend the size of the file
    /// \param stoh:  handle to a tensor storage
    /// \param size: expected file size in bytes

    inline void preallocate_storage(Storage_handle stoh, std::size_t size) {
        stoh->preallocate(size);
    }

    /// Force pending writing to make them visible to other processes
    /// \param stoh:  handle to a tensor storage

    inline void flush_storage(Storage_handle stoh) { stoh->flush(); }

    /// Close a storage
    /// \param stoh:  handle to a tensor storage

    inline void close_storage(Storage_handle stoh) {
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
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd1> *p, int num_blocks, Storage_handle stoh,
                       CoorOrder co) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::append_blocks<Nd1, Nd1, Q>(
            p, num_blocks, Coor<Nd1>{}, detail::get_dim(p, num_blocks),
            detail::trivial_order<Nd1>(), detail::trivial_order<Nd1>(), sto, Coor<Nd1>{}, comm, co);
    }

    /// Add blocks to storage
    /// \param p0: blocks to add
    /// \param num_blocks: number of items in p0
    /// \param o0: dimension labels for the blocks to add
    /// \param from0: first coordinate of the range to consider on p0
    /// \param size0: number of elements to copy in each dimension
    /// \param o1: dimension labels for the storage
    /// \param from1: first coordinate of the ranges to add
    /// \param stoh: handle to a tensor storage
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd0> *p0, int num_blocks, const char *o0,
                       const Coor<Nd0> &from0, const Coor<Nd0> &size0, const char *o1,
                       const Coor<Nd1> &from1, Storage_handle stoh, CoorOrder co) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::append_blocks<Nd0, Nd1, Q>(p0, num_blocks, from0, size0,
                                           detail::toArray<Nd0>(o0, "o0"),
                                           detail::toArray<Nd1>(o1, "o1"), sto, from1, comm, co);
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
