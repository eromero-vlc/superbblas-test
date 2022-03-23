#ifndef __SUPERBBLAS_BSR__
#define __SUPERBBLAS_BSR__

#include "dist.h"
#include "superbblas/blas.h"
#include <numeric>
#include <stdexcept>

namespace superbblas {

    namespace detail {
        enum num_type { float_t, cfloat_t, double_t, cdouble_t };
        template <typename T> struct num_type_v;
        template <> struct num_type_v<float> { static constexpr num_type value = float_t; };
        template <> struct num_type_v<double> { static constexpr num_type value = double_t; };
        template <> struct num_type_v<std::complex<float>> {
            static constexpr num_type value = cfloat_t;
        };
        template <> struct num_type_v<std::complex<double>> {
            static constexpr num_type value = cdouble_t;
        };
    }

    // /// Handle for a BSR operator
    struct BSR_handle {
        /// Return number of dimensions of the image space
        /// \param Nd: number of dimensions of the domain (columns) operator
        /// \param Ni: number of dimensions of the image (rows) operator
        /// \param type: nonzero values type
        /// \param ctx: contexts
        /// \param ncomponents: number of consecutive components in each MPI rank
        /// \param nprocs: number of processes supporting the matrix
        /// \param rank: current MPI rank
        /// \param co: coordinate linearization order

        virtual bool check(std::size_t, std::size_t, detail::num_type, const Context *, int,
                           unsigned int, unsigned int, CoorOrder) {
            return false;
        }

        BSR_handle() {}
        virtual ~BSR_handle() {}
    };

    namespace detail {

        template <typename XPU> struct CsrIndices {
            Indices<XPU> i; ///< Where the columns indices (j) for the ith row start
            Indices<XPU> j; ///< Column indices
        };

        /// Component of a BSR tensor
        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU> struct BSRComponent {
            Indices<XPU> i;          ///< number of nonzero blocks on each block image
            vector<Coor<Nd>, XPU> j; ///< domain coordinates of the nonzero blocks
            vector<T, XPU> it;       ///< nonzero values
            Coor<Nd> dimd;           ///< dimensions of the domain space
            Coor<Ni> dimi;           ///< dimensions of the image space
            Coor<Nd> blockd;         ///< dimensions of a block in the domain space
            Coor<Ni> blocki;         ///< dimensions of a block in the image space
            bool blockImFast; ///< whether the image indices are the fastest on the dense blocks
            CoorOrder co;     ///< Coordinate order of ii and jj
            unsigned int componentId; ///< Component Id

            template <typename Q = T, typename = typename std::enable_if<std::is_same<
                                          Q, typename std::remove_const<Q>::type>::value>::type>
            operator BSRComponent<Nd, Ni, const Q, XPU>() const {
                return {i, j, it, dimd, dimi, blockd, blocki, blockImFast, co, componentId};
            }
        };

        ///
        /// Implementation of operations for each platform
        ///

        /// Constrains on layout of the input and output dense tensors for a
        /// sparse-dense tensor contraction

        enum SpMMAllowedLayout {
            SameLayoutForXAndY, //< X and Y should have the same layout, either row-major or column-major
            ColumnMajorForY,  //< X can be either way but Y should be column-major
            AnyLayoutForXAndY //< X and Y can be either way
        };

        /// Matrix layout
        enum MatrixLayout { RowMajor, ColumnMajor };

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU> struct BSR;

#if defined(SUPERBBLAS_USE_MKL)
        inline void checkMKLSparse(sparse_status_t status) {
            static std::map<sparse_status_t, std::string> statuses = {
                {SPARSE_STATUS_NOT_INITIALIZED, "SPARSE_STATUS_NOT_INITIALIZED"},
                {SPARSE_STATUS_ALLOC_FAILED, "SPARSE_STATUS_ALLOC_FAILED"},
                {SPARSE_STATUS_INVALID_VALUE, "SPARSE_STATUS_INVALID_VALUE"},
                {SPARSE_STATUS_EXECUTION_FAILED, "SPARSE_STATUS_EXECUTION_FAILED"},
                {SPARSE_STATUS_INTERNAL_ERROR, "SPARSE_STATUS_INTERNAL_ERROR"},
                {SPARSE_STATUS_NOT_SUPPORTED, "SPARSE_STATUS_NOT_SUPPORTED"}};

            if (status != SPARSE_STATUS_SUCCESS) {
                std::stringstream ss;
                ss << "MKL sparse function returned error " << statuses[status];
                throw std::runtime_error(ss.str());
            }
        }

        template <std::size_t Nd, std::size_t Ni, typename T> struct BSR<Nd, Ni, T, Cpu> {
            BSRComponent<Nd, Ni, T, Cpu> v;     ///< BSR general information
            vector<MKL_INT, Cpu> ii, jj;        ///< BSR row and column nonzero indices
            std::shared_ptr<sparse_matrix_t> A; ///< MKL BSR descriptor

            static const SpMMAllowedLayout allowLayout = SameLayoutForXAndY;

            BSR(const BSRComponent<Nd, Ni, T, Cpu> &v) : v(v) {
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                if (volume(v.blocki) != volume(v.blockd))
                    throw std::runtime_error(" MKL Sparse does not support non-square blocks");
                IndexType block_size = volume(v.blocki);
                IndexType block_rows = volume(v.dimi) / block_size;
                IndexType block_cols = volume(v.dimd) / block_size;
                auto bsr = get_bsr_indices(v, true);
                ii = bsr.i;
                jj = bsr.j;
                A = std::shared_ptr<sparse_matrix_t>(new sparse_matrix_t, [=](sparse_matrix_t *A) {
                    checkMKLSparse(mkl_sparse_destroy(*A));
                    delete A;
                });
                checkMKLSparse(mkl_sparse_create_bsr(&*A, SPARSE_INDEX_BASE_ZERO,
                                                     v.blockImFast ? SPARSE_LAYOUT_COLUMN_MAJOR
                                                                   : SPARSE_LAYOUT_ROW_MAJOR,
                                                     block_rows, block_cols, block_size, ii.data(),
                                                     ii.data() + 1, jj.data(), v.it.data()));
                checkMKLSparse(mkl_sparse_set_mm_hint(
                    *A, SPARSE_OPERATION_NON_TRANSPOSE,
                    (struct matrix_descr){.type = SPARSE_MATRIX_TYPE_GENERAL,
                                          .mode = SPARSE_FILL_MODE_LOWER /* Not used */,
                                          .diag = SPARSE_DIAG_NON_UNIT},
                    SPARSE_LAYOUT_ROW_MAJOR, 100, 1000));
            }

            void operator()(bool conjA, const T *x, IndexType ldx, MatrixLayout lx, T *y,
                            IndexType ldy, MatrixLayout ly, IndexType ncols, T beta = T{0}) const {
                T one{1.0};
                if (lx != ly) throw std::runtime_error("Unsupported operation with MKL");
                checkMKLSparse(mkl_sparse_mm(
                    !conjA ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    one, *A,
                    (struct matrix_descr){.type = SPARSE_MATRIX_TYPE_GENERAL,
                                          .mode = SPARSE_FILL_MODE_LOWER /* Not used */,
                                          .diag = SPARSE_DIAG_NON_UNIT},
                    lx == ColumnMajor ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR, x,
                    ncols, ldx, beta, y, ldy));
            }

            ~BSR() {}
        };
#else

        template <std::size_t Nd, std::size_t Ni, typename T> struct BSR<Nd, Ni, T, Cpu> {
            BSRComponent<Nd, Ni, T, Cpu> v; ///< BSR general information
            vector<IndexType, Cpu> ii, jj;  ///< BSR row and column nonzero indices

            static const SpMMAllowedLayout allowLayout = AnyLayoutForXAndY;

            BSR(const BSRComponent<Nd, Ni, T, Cpu> &v) : v(v) {
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                auto bsr = get_bsr_indices(v);
                ii = bsr.i;
                jj = bsr.j;
            }

            void operator()(bool conjA, const T *x, IndexType ldx, MatrixLayout lx, T *y,
                            IndexType ldy, MatrixLayout ly, IndexType ncols, T beta = T{0}) const {
                if (conjA) throw std::runtime_error("Not implemented");
                IndexType bi = volume(v.blocki);
                IndexType bd = volume(v.blockd);
                IndexType block_rows = volume(v.dimi) / bi;
                xscal(volume(v.dimi) * ncols, beta, y, 1, Cpu{});
                T *nonzeros = v.it.data();
                const bool tx = lx == RowMajor;
                const bool tb = !v.blockImFast;
                const IndexType xs = lx == ColumnMajor ? 1 : ldx;
#    ifdef _OPENMP
#        pragma omp parallel for
#    endif
                for (IndexType i = 0; i < block_rows; ++i) {
                    for (IndexType j = ii[i], j1 = ii[i + 1]; j < j1; ++j) {
                        if (ly == ColumnMajor)
                            xgemm(tb ? 'T' : 'N', tx ? 'T' : 'N', bi, ncols, bd, T{1},
                                  nonzeros + j * bi * bd, bi, x + jj[j] * xs, ldx, T{1}, y + i * bi,
                                  ldy, Cpu{});
                        else
                            xgemm(!tx ? 'T' : 'N', !tb ? 'T' : 'N', ncols, bi, bd, T{1},
                                  x + jj[j] * xs, ldx, nonzeros + j * bi * bd, bi, T{1},
                                  y + i * bi * ldy, ldy, Cpu{});
                    }
                }
            }

            ~BSR() {}
        };
#endif

#ifdef SUPERBBLAS_USE_GPU
        template <std::size_t Nd, std::size_t Ni, typename T> struct BSR<Nd, Ni, T, Gpu> {
            BSRComponent<Nd, Ni, T, Gpu> v; ///< BSR general information
            vector<IndexType, Gpu> ii, jj;  ///< BSR row and column nonzero indices
#    ifdef SUPERBBLAS_USE_CUDA
            cusparseMatDescr_t descrA; ///< cuSparse descriptor
#    else
            hipsparseMatDescr_t descrA; ///< hipSparse descriptor
#    endif

            static const SpMMAllowedLayout allowLayout = ColumnMajorForY;

            BSR(BSRComponent<Nd, Ni, T, Gpu> v) : v(v) {
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                if (volume(v.blocki) != volume(v.blockd))
                    throw std::runtime_error(" MKL Sparse does not support non-square blocks");
                auto bsr = get_bsr_indices(v, true);
                ii = bsr.i;
                jj = bsr.j;
#    ifdef SUPERBBLAS_USE_CUDA
                cusparseCheck(cusparseCreateMatDescr(&descrA));
                cusparseCheck(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
                cusparseCheck(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
#    else
                hipsparseCheck(hipsparseCreateMatDescr(&descrA));
                hipsparseCheck(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO));
                hipsparseCheck(hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL));
#    endif
            }

            void operator()(bool conjA, const T *x, IndexType ldx, MatrixLayout lx, T *y,
                            IndexType ldy, MatrixLayout ly, IndexType ncols, T beta = T{0}) const {
                if (ly == RowMajor)
                    throw std::runtime_error(
                        "Unsupported row-major on Y with cu/hipSPARSE's bsrmm");
                IndexType block_size = volume(v.blocki);
                IndexType block_cols = volume(v.dimd) / block_size;
                IndexType block_rows = volume(v.dimi) / block_size;
                IndexType num_blocks = jj.size();
                T one{1};
#    ifdef SUPERBBLAS_USE_CUDA
                cusparseCheck(cusparseXbsrmm(
                    ii.ctx().cusparseHandle,
                    v.blockImFast ? CUSPARSE_DIRECTION_COLUMN : CUSPARSE_DIRECTION_ROW,
                    !conjA ? CUSPARSE_OPERATION_NON_TRANSPOSE
                           : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    lx == ColumnMajor ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                      : CUSPARSE_OPERATION_TRANSPOSE,
                    block_rows, ncols, block_cols, num_blocks, one, descrA, v.it.data(), ii.data(),
                    jj.data(), block_size, x, ldx, beta, y, ldy));
#    else
                hipsparseCheck(hipsparseXbsrmm(
                    ii.ctx().hipsparseHandle,
                    v.blockImFast ? HIPSPARSE_DIRECTION_COLUMN : HIPSPARSE_DIRECTION_ROW,
                    !conjA ? HIPSPARSE_OPERATION_NON_TRANSPOSE
                           : HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    lx == ColumnMajor ? HIPSPARSE_OPERATION_NON_TRANSPOSE
                                      : HIPSPARSE_OPERATION_TRANSPOSE,
                    block_rows, ncols, block_cols, num_blocks, one, descrA, v.it.data(), ii.data(),
                    jj.data(), block_size, x, ldx, beta, y, ldy));
#    endif
            }

            ~BSR() {}
        };
#endif // SUPERBBLAS_USE_GPU

        /// A BSR tensor composed of several components
        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU0, typename XPU1>
        struct BSRComponents_tmpl : BSR_handle {
            /// Partition of the domain space
            From_size<Nd> pd;
            /// Partition of the image space
            From_size<Ni> pi;
            /// Components of the BSR operator
            std::pair<std::vector<BSR<Nd, Ni, T, XPU0>>, std::vector<BSR<Nd, Ni, T, XPU1>>> c;
            Coor<Nd> blockd; ///< dimensions of a block in the domain space
            Coor<Ni> blocki; ///< dimensions of a block in the image space
            CoorOrder co;    ///< Coordinate order of ii and jj
            /// Reference to a BSR operator with extra rows to compute power without extra communications
            std::shared_ptr<BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1>> ext;
            /// Power of the `ext` operator
            unsigned int ext_power;

            bool check(std::size_t Nd_, std::size_t Ni_, detail::num_type type, const Context *ctx,
                       int ncomponents, unsigned int nprocs, unsigned int rank,
                       CoorOrder co) override {
                (void)rank;
                if (Nd_ != Nd || Ni_ != Ni || num_type_v<T>::value != type ||
                    nprocs * ncomponents != pd.size())
                    return false;
                for (const auto &i : c.first)
                    if (0 > i.v.componentId || i.v.componentId >= ncomponents * nprocs ||
                        i.v.co != co)
                        return false;
                /// TODO: check ctx[i].platform matches the components c
                (void)ctx;
                return true;
            }
        };

#ifdef SUPERBBLAS_USE_GPU
        /// A tensor composed of several CPU and GPU elements
        template <std::size_t Nd, std::size_t Ni, typename T>
        using BSRComponents = BSRComponents_tmpl<Nd, Ni, T, Gpu, Cpu>;
#else
        /// A tensor composed of only of CPU components
        template <std::size_t Nd, std::size_t Ni, typename T>
        using BSRComponents = BSRComponents_tmpl<Nd, Ni, T, Cpu, Cpu>;
#endif // SUPERBBLAS_USE_GPU

        template <std::size_t Nd, std::size_t Ni, typename T, typename Comm>
        BSRComponents<Nd, Ni, T>
        get_bsr_components(T **v, IndexType **ii, Coor<Nd> **jj, const Context *ctx,
                           unsigned int ncomponents, From_size_iterator<Ni> pi,
                           From_size_iterator<Nd> pd, Coor<Nd> blockd, Coor<Ni> blocki,
                           bool blockImFast, Comm comm, CoorOrder co, Session session) {
            // Get components on the local process
            From_size_iterator<Nd> fsd = pd + comm.rank * ncomponents;
            From_size_iterator<Ni> fsi = pi + comm.rank * ncomponents;

            BSRComponents<Nd, Ni, T> r{};
            r.pd = detail::get_from_size(pd, ncomponents * comm.nprocs, session);
            r.pi = detail::get_from_size(pi, ncomponents * comm.nprocs, session);
            r.blockd = blockd;
            r.blocki = blocki;
            r.co = co;
            for (unsigned int i = 0; i < ncomponents; ++i) {
                std::size_t nii = volume(fsi[i][1]) / volume(blocki);
                std::size_t njj =
                    ctx[i].plat == CPU ? sum(to_vector(ii[i], nii, ctx[i].toCpu(session))) :
#ifdef SUPERBBLAS_USE_GPU
                                       sum(to_vector(ii[i], nii, ctx[i].toGpu(session)))
#else
                                       0
#endif
                    ;
                std::size_t nvalues = njj * volume(blockd) * volume(blocki);
                switch (ctx[i].plat) {
#ifdef SUPERBBLAS_USE_GPU
                case CPU:
                    r.c.second.push_back(BSR<Nd, Ni, T, Cpu>{BSRComponent<Nd, Ni, T, Cpu>{
                        to_vector(ii[i], nii, ctx[i].toCpu(session)),
                        to_vector(jj[i], njj, ctx[i].toCpu(session)),
                        to_vector(v[i], nvalues, ctx[i].toCpu(session)), fsd[i][1], fsi[i][1],
                        blockd, blocki, blockImFast, co, i}});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
                case GPU:
                    r.c.first.push_back(BSR<Nd, Ni, T, Gpu>{BSRComponent<Nd, Ni, T, Gpu>{
                        to_vector(ii[i], nii, ctx[i].toGpu(session)),
                        to_vector(jj[i], njj, ctx[i].toGpu(session)),
                        to_vector(v[i], nvalues, ctx[i].toGpu(session)), fsd[i][1], fsi[i][1],
                        blockd, blocki, blockImFast, co, i}});
                    assert(!v[i] || getPtrDevice(v[i]) == ctx[i].device);
                    break;
#else // SUPERBBLAS_USE_GPU
                case CPU:
                    r.c.first.push_back(BSR<Nd, Ni, T, Cpu>{BSRComponent<Nd, Ni, T, Cpu>{
                        to_vector(ii[i], nii, ctx[i].toCpu(session)),
                        to_vector(jj[i], njj, ctx[i].toCpu(session)),
                        to_vector(v[i], nvalues, ctx[i].toCpu(session)), fsd[i][1], fsi[i][1],
                        blockd, blocki, blockImFast, co, i}});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
#endif
                default: throw std::runtime_error("Unsupported platform");
                }
            }
            return r;
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename Comm>
        BSRComponents<Nd, Ni, T> *
        get_bsr_components_from_handle(BSR_handle *bsrh, const Context *ctx, int ncomponents,
                                       Comm comm, CoorOrder co) {
            if (!bsrh->check(Nd, Ni, detail::num_type_v<T>::value, ctx, ncomponents, comm.nprocs,
                             comm.rank, co))
                throw std::runtime_error(
                    "Given BSR handle doesn't match the template parameters Nd, Ni, or T, does not "
                    "match contexts, or does not match MPI communicator");
            return static_cast<BSRComponents<Nd, Ni, T> *>(bsrh);
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU0, typename XPU1>
        std::vector<Coor<Ni>>
        get_bsr_dim_image_space(const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &bsr) {
            unsigned int ncomponents = bsr.c.first.size() + bsr.c.second.size();
            std::vector<Coor<Ni>> r(ncomponents);
            for (const auto &c : bsr.c.first) r[c.componentId] = c.dimi;
            for (const auto &c : bsr.c.second) r[c.componentId] = c.dimi;
            return r;
        }

        //
        // BSR implementations
        //

        template <typename T, std::size_t Na, std::size_t Nb>
        std::array<T, Na + Nb> concat(const std::array<T, Na> &a, const std::array<T, Nb> &b) {
            std::array<T, Na + Nb> r;
            std::copy_n(a.begin(), Na, r.begin());
            std::copy_n(b.begin(), Nb, r.begin() + Na);
            return r;
        }

        //
        // Auxiliary functions
        //

        /// Return BSR indices for a given tensor BSR
        ///

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU>
        CsrIndices<XPU> get_bsr_indices(const BSRComponent<Nd, Ni, T, XPU> &v,
                                        bool return_jj_blocked = false) {
            Indices<Cpu> ii(v.i.size() + 1, Cpu{}), jj(v.j.size(), Cpu{});
            Indices<Cpu> vi = makeSure(v.i, Cpu{});
            vector<Coor<Nd>, Cpu> vj = makeSure(v.j, Cpu{});

            // Compute index for the first nonzero on the ith row
            ii[0] = 0;
            for (std::size_t i = 0; i < vi.size(); ++i) ii[i + 1] = ii[i] + vi[i];

            // Transform the domain coordinates into indices
            Coor<Nd> strided = get_strides<Nd>(v.dimd, v.co);
            std::size_t block_nnz = v.j.size();
            std::size_t bd = return_jj_blocked ? volume(v.blockd) : 1;
#ifdef _OPENMP
#    pragma omp parallel for
#endif
            for (std::size_t i = 0; i < block_nnz; ++i) {
                jj[i] = coor2index<Nd>(vj[i], v.dimd, strided) / bd;
            }

            return {makeSure(ii, v.i.ctx()), makeSure(jj, v.j.ctx())};
        }

        /// Return splitting of dimension labels for the RSB operator - tensor multiplication
        /// \param dimi: dimension of the RSB operator image in consecutive ranges
        /// \param dimd: dimension of the RSB operator domain in consecutive ranges
        /// \param oi: dimension labels for the RSB operator image space
        /// \param od: dimension labels for the RSB operator domain space
        /// \param blocki: image dimensions of the block
        /// \param blockd: domain dimensions of the block
        /// \param dimx: dimension of the right tensor in consecutive ranges
        /// \param ox: dimension labels for the right operator
        /// \param dimy: dimension of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param co: coordinate linearization order
        ///
        /// For SlowToFast, it supports:
        ///   (I,D,i,d) x (C,D,d) -> (C,I,i)
        /// where
        /// - (D,d) are the domain RSB dimensions labels, od;
        /// - (I,i) are the image RSB dimensions labels, oi;
        /// - (C,D,d) are the dimensions labels of the right input tensor, ox
        /// - (C,I,i) are the dimensions labels of the output tensor, oy
        /// - D and I has dimensions labels that are not blocked
        /// - d and i has all blocked dimensions labels

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny>
        void local_bsr_krylov_check(const Coor<Ni> &dimi, const Coor<Nd> &dimd, const Order<Ni> &oi,
                                    const Order<Nd> &od, const Coor<Ni> &blocki,
                                    const Coor<Nd> &blockd, const Coor<Nx> &dimx,
                                    const Order<Nx> &ox, const Coor<Ny> &dimy, const Order<Ny> &oy,
                                    char okr, SpMMAllowedLayout xylayout, CoorOrder co,
                                    bool &transSp, MatrixLayout &lx, MatrixLayout &ly,
                                    std::size_t &volC) {

            if (co == FastToSlow) {
                local_bsr_krylov_check(reverse(dimi), reverse(dimd), reverse(oi), reverse(od),
                                       reverse(blocki), reverse(blockd), reverse(dimx), reverse(ox),
                                       reverse(dimy), reverse(oy), okr, xylayout, SlowToFast,
                                       transSp, lx, ly, volC);
                return;
            }

            // Find all common labels in od and oi
            for (char c : od)
                if (std::find(oi.begin(), oi.end(), c) != oi.end())
                    std::runtime_error(
                        "Common label between the domain and image of the sparse matrix");

            // Find all common labels in od+oi, ox, and oy
            Order<Nx> oT;
            std::size_t nT = 0;
            for (char c : ox)
                if ((std::find(od.begin(), od.end(), c) != od.end() ||
                     std::find(oi.begin(), oi.end(), c) != oi.end()) &&
                    std::find(oy.begin(), oy.end(), c) != oy.end())
                    oT[nT++] = c;
            if (nT > 0)
                std::runtime_error("Still not supported common dimensions between the sparse input "
                                   "matrix, the dense input matrix, and the dense output matrix");

            // Split the od into the non-blocked dimensions (Ds) and the blocked ones (ds)
            Order<Nd> oDs, ods;
            Coor<Nd> dimDs, dimds;
            std::size_t nDs = 0, nds = 0;
            for (std::size_t i = 0; i < Nd; ++i) {
                if (blockd[i] > 1) {
                    if (blockd[i] != dimd[i])
                        throw std::runtime_error(
                            "Still not supported partially blocking a dimension");
                    ods[nds] = od[i];
                    dimds[nds++] = dimd[i];
                } else {
                    oDs[nDs] = od[i];
                    dimDs[nDs++] = dimd[i];
                }
            }

            // Split the oi into the non-blocked dimensions (Is) and the blocked ones (is)
            Order<Ni> oIs, ois;
            Coor<Ni> dimIs, dimis;
            std::size_t nIs = 0, nis = 0;
            for (std::size_t i = 0; i < Ni; ++i) {
                if (blocki[i] > 1) {
                    if (blocki[i] != dimi[i])
                        throw std::runtime_error(
                            "Still not supported partially blocking a dimension");
                    ois[nis] = oi[i];
                    dimis[nis++] = dimi[i];
                } else {
                    oIs[nIs] = oi[i];
                    dimIs[nIs++] = dimi[i];
                }
            }

            // Find all common labels in ox to oy
            Order<Nx> oC;
            std::size_t nC = 0;
            volC = 1;
            enum { None, ContractWithDomain, ContractWithImage } kindx = None, kindy = None;
            int ix = 0;
            for (char c : ox) {
                if (std::find(oi.begin(), oi.end(), c) != oi.end()) {
                    if (kindx == ContractWithDomain)
                        throw std::runtime_error(
                            "Unsupported to contract dense input tensor with domain and image "
                            "dimensions of the sparse tensor");
                    else
                        kindx = ContractWithImage;
                } else if (std::find(od.begin(), od.end(), c) != od.end()) {
                    if (kindx == ContractWithImage)
                        throw std::runtime_error(
                            "Unsupported to contract dense input tensor with domain and image "
                            "dimensions of the sparse tensor");
                    else
                        kindx = ContractWithDomain;
                } else if (std::find(oy.begin(), oy.end(), c) != oy.end()) {
                    oC[nC++] = c;
                    volC *= dimx[ix];
                } else
                    throw std::runtime_error(
                        "Dimension label for the dense input vector doesn't match the "
                        "input sparse dimensions nor the dense output dimensions");

                ix++;
            }

            // Find all common labels in ox to oy
            bool powerFound = false;
            for (char c : oy) {
                if (std::find(oi.begin(), oi.end(), c) != oi.end()) {
                    if (kindy == ContractWithDomain)
                        throw std::runtime_error(
                            "Unsupported to an output tensor with dimensions both from the domain "
                            "and the image on the sparse matrix");
                    else
                        kindy = ContractWithImage;
                } else if (std::find(od.begin(), od.end(), c) != od.end()) {
                    if (kindy == ContractWithImage)
                        throw std::runtime_error(
                            "Unsupported to an output tensor with dimensions both from the domain "
                            "and the image on the sparse matrix");
                    else
                        kindy = ContractWithDomain;
                } else if (std::find(ox.begin(), ox.end(), c) != ox.end()) {
                    // Do nothing
                } else if (okr != 0 && c == okr) {
                    powerFound = true;
                } else {
                    throw std::runtime_error(
                        "Dimension label for the dense input vector doesn't match the "
                        "input sparse dimensions nor the dense output dimensions");
                }
            }

            // Check okr: either zero or a label on oy
            if (okr != 0 && !powerFound)
                throw std::runtime_error("The power dimension isn't on the output dense tensor");

            // Check that the power should be the slowest dimension in y
            if (okr != 0 && oy[0] != okr)
                throw std::runtime_error(
                    "The power dimension should be at the slowest dimension on "
                    "the output dense tensor");

            // If power, dimi should be equal to dimd
            if (okr != 0 && dimy[0] > 1) {
                if (std::search(dimd.begin(), dimd.end(), dimi.begin(), dimi.end()) == dimd.end() ||
                    std::search(blockd.begin(), blockd.end(), blocki.begin(), blocki.end()) ==
                        blockd.end())
                    throw std::runtime_error(
                        "When using powers the domain and the image of the sparse operator should "
                        "be the same, and the dense input and output tensors should be similar.");

                Coor<Nx> permxd = find_permutation(od, ox);
                Coor<Nx> permxi = find_permutation(oi, ox);
                Coor<Ny> permyd = find_permutation(od, oy);
                Coor<Ny> permyi = find_permutation(oi, oy);
                bool fail = false;
                if (kindx == ContractWithDomain) {
                    fail = std::search(permxd.begin(), permxd.end(), permyi.begin(),
                                       permyi.end()) == permxd.end();
                } else if (kindx == ContractWithImage) {
                    fail = std::search(permxi.begin(), permxi.end(), permyd.begin(),
                                       permyd.end()) == permxi.end();
                }
                if (fail)
                    throw std::runtime_error(
                        "When using powers the domain and the image of the sparse operator should "
                        "be the same, and the dense input and output tensors should be similar.");
            }

            // Check that kindx and kindy aren't None and they are distinct
            if (kindx == None)
                throw std::runtime_error(
                    "Unsupported to contract sparse and dense tensors without a common dimension");
            if (kindy == None)
                throw std::runtime_error(
                    "Unsupported to the resulting dense tensor of contracting sparse and dense "
                    "tensors has no common dimension with the sparse tensor");
            if (kindx == kindy) throw std::runtime_error("Invalid contraction");

            // Check that ox should one of (C,D,d) or (D,d,C) or (C,I,i) or (I,i,C)
            if (kindx == ContractWithDomain) {
                auto sCx = std::search(ox.begin(), ox.end(), oC.begin(), oC.begin() + nC);
                auto sDx = std::search(ox.begin(), ox.end(), oDs.begin(), oDs.begin() + nDs);
                auto sdx = std::search(ox.begin(), ox.end(), ods.begin(), ods.begin() + nds);
                if ((nC > 0 && sCx == ox.end()) || (nDs > 0 && sDx == ox.end()) ||
                    (nds > 0 && sdx == ox.end()) || (nDs > 0 && nds > 0 && sDx > sdx) ||
                    (nC > 0 && nDs > 0 && nds > 0 && sDx < sCx && sCx < sdx))
                    throw std::runtime_error(
                        "Unsupported dimensions order for the dense input tensor");
                lx = (nC == 0 || ((nDs == 0 || sCx < sDx) && (nds == 0 || sCx < sdx))) ? ColumnMajor
                                                                                       : RowMajor;

                auto sCy = std::search(oy.begin(), oy.end(), oC.begin(), oC.begin() + nC);
                auto sIy = std::search(oy.begin(), oy.end(), oIs.begin(), oIs.begin() + nIs);
                auto siy = std::search(oy.begin(), oy.end(), ois.begin(), ois.begin() + nis);
                if ((nC > 0 && sCy == oy.end()) || (nIs > 0 && sIy == oy.end()) ||
                    (nis > 0 && siy == oy.end()) || (nIs > 0 && nis > 0 && sIy > siy) ||
                    (nC > 0 && nIs > 0 && nis > 0 && sIy < sCy && sCy < siy))
                    throw std::runtime_error(
                        "Unsupported dimensions order for the dense input tensor");
                ly = (nC == 0 || ((nIs == 0 || sCy < sIy) && (nds == 0 || sCy < siy))) ? ColumnMajor
                                                                                       : RowMajor;
                transSp = false;
            } else {
                auto sCx = std::search(ox.begin(), ox.end(), oC.begin(), oC.begin() + nC);
                auto sIx = std::search(ox.begin(), ox.end(), oIs.begin(), oIs.begin() + nIs);
                auto six = std::search(ox.begin(), ox.end(), ois.begin(), ois.begin() + nis);
                if ((nC > 0 && sCx == ox.end()) || (nIs > 0 && sIx == ox.end()) ||
                    (nis > 0 && six == ox.end()) || (nIs > 0 && nis > 0 && sIx > six) ||
                    (nC > 0 && nIs > 0 && nis > 0 && sIx < sCx && sCx < six))
                    throw std::runtime_error(
                        "Unsupported dimensions order for the dense input tensor");
                lx = (nC == 0 || ((nIs == 0 || sCx < sIx) && (nis == 0 || sCx < six))) ? ColumnMajor
                                                                                       : RowMajor;

                auto sCy = std::search(oy.begin(), oy.end(), oC.begin(), oC.begin() + nC);
                auto sDy = std::search(oy.begin(), oy.end(), oDs.begin(), oDs.begin() + nDs);
                auto sdy = std::search(oy.begin(), oy.end(), ods.begin(), ods.begin() + nds);
                if ((nC > 0 && sCy == oy.end()) || (nDs > 0 && sDy == oy.end()) ||
                    (nds > 0 && sdy == oy.end()) || (nDs > 0 && nds > 0 && sDy > sdy) ||
                    (nC > 0 && nDs > 0 && nds > 0 && sDy < sCy && sCy < sdy))
                    throw std::runtime_error(
                        "Unsupported dimensions order for the dense input tensor");
                ly = (nC == 0 || ((nDs == 0 || sCy < sDy) && (nds == 0 || sCy < sdy))) ? ColumnMajor
                                                                                       : RowMajor;
                transSp = true;
            }

            if (lx != ly && xylayout == SameLayoutForXAndY)
                throw std::runtime_error(
                    "Unsupported layout for the dense input and output tensors");
            if (ly == RowMajor && xylayout == ColumnMajorForY)
                throw std::runtime_error("Unsupported layout for the output tensor");
        }

        /// RSB operator - tensor multiplication
        /// \param pim: partitioning of the RSB operator image in consecutive ranges
        /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
        /// \param oim: dimension labels for the RSB operator image space
        /// \param odm: dimension labels for the RSB operator domain space
        /// \param blockim: image dimensions of the block
        /// \param blockdm: domain dimensions of the block
        /// \param vm: BSR tensor components
        /// \param px: partitioning of the right tensor in consecutive ranges
        /// \param ox: dimension labels for the right operator
        /// \param vx: right input tensor components
        /// \param py: partitioning of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param vy: output tensor components
        /// \param co: coordinate linearization order
        /// \param copyadd: either copy or add the multiplication result into the output tensor y

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T,
                  typename XPU, typename EWOP>
        void local_bsr_krylov(const BSR<Nd, Ni, T, XPU> &bsr, const Order<Ni> &oim,
                              const Order<Nd> &odm, const Coor<Nx> &dimx, const Order<Nx> &ox,
                              vector<const T, XPU> vx, const Coor<Ny> &dimy, const Order<Ny> &oy,
                              char okr, vector<T, XPU> vy, EWOP) {

            tracker<XPU> _t("local BSR matvec", vx.ctx());

            // Check inputs and get the common dimensions
            bool transSp;
            MatrixLayout lx, ly;
            std::size_t volC;
            local_bsr_krylov_check(bsr.v.dimi, bsr.v.dimd, oim, odm, bsr.v.blocki, bsr.v.blockd,
                                   dimx, ox, dimy, oy, okr, bsr.allowLayout, bsr.v.co, transSp, lx,
                                   ly, volC);

            // Get the number of powers
            int powers = 1;
            std::size_t power_pos = 0;
            if (okr) {
                power_pos = std::find(oy.begin(), oy.end(), okr) - oy.begin();
                powers = dimy[power_pos];
            }
            if (powers == 0) return;
            if (powers > 1 && ((power_pos > 0 && bsr.v.co == SlowToFast) ||
                               (power_pos < Ny - 1 && bsr.v.co == FastToSlow)))
                throw std::runtime_error(
                    "Unsupported power position: it can be at the slowest index");

            // Set zero
            local_copy<Ny, Ny, T, T>(0, oy, {}, dimy, dimy, (vector<const T, XPU>)vy, oy, {}, dimy,
                                     vy, EWOp::Copy{}, bsr.v.co);

            std::size_t vold = volume(bsr.v.dimd), voli = volume(bsr.v.dimi);
            IndexType ldx = lx == ColumnMajor ? (!transSp ? vold : voli) : volC;
            IndexType ldy = ly == ColumnMajor ? (!transSp ? voli : vold) : volC;

            // Do first power
            bsr(transSp, vx.data(), ldx, lx, vy.data(), ldy, ly, volC);
            if (powers <= 1) return;

            // Do remaining powers
            for (int p = 1; p < powers; ++p)
                bsr(transSp, vy.data() + vold * volC * (p - 1), ldy, ly,
                    vy.data() + vold * volC * p, ldy, ly, volC);
        }

        /// Return a BSR operator extended for doing powers without extra communications
        /// \param bsr: BSR tensor components
        /// \param power: maximum power to compute
        /// \param comm: communications handle

        template <std::size_t Nd, std::size_t Ni, typename T, typename Comm, typename XPU0,
                  typename XPU1, typename std::enable_if<Nd != Ni, bool>::type = true>
        const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &
        bsr_krylov(const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &bsr, unsigned int power,
                   Comm comm) {

            // Cannot do powers with rectangular matrices
            if (power > 1)
                throw std::runtime_error(
                    "Cannot do powers on BSR operator with different domain and image spaces");
            return bsr;
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename Comm, typename XPU0,
                  typename XPU1, typename std::enable_if<Nd == Ni, bool>::type = true>
        const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &
        bsr_krylov(const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &bsr, unsigned int power,
                   Comm comm) {

            // Check if the extensions have already been computed
            if (power <= 1) return bsr;
            if (power <= bsr.ext_power) return *bsr.ext;

            while (power > bsr.ext_power) {
                // If the image and domain spaces coincide for all components, no extra communications are required to build powers
                const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &bsr0 = (bsr.ext ? *bsr.ext : bsr);
                if (bsr0.pd == bsr0.pi) return bsr0;

                Coor<Nd> dimd = get_dim(bsr0.pd);
                Coor<Ni> dimi = get_dim(bsr0.pi);
            }
        }

        /// Get the partitions for the dense input and output tensors
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny>
        std::pair<From_size<Nx>, From_size<Ny>>
        get_output_partition(From_size<Nd> pd, const Order<Nd> &od, From_size<Ni> pi,
                             const Order<Ni> &oi, From_size<Nx> px, const Order<Nx> &ox,
                             const Order<Ny> &oy, char okr, int power) {
            assert(pd.size() == pi.size() && pi.size() == px.size());

            // Find partition on cache
            Order<Nd + Ni> om = concat(od, oi);
            using Key =
                std::tuple<From_size<Nd>, From_size<Ni>, From_size<Nx>, PairPerms<Nd + Ni, Nx>,
                           PairPerms<Nx, Ny>, PairPerms<Nd + Ni, Ny>>;
            struct cache_tag {};
            auto cache =
                getCache<Key, std::pair<From_size<Nx>, From_size<Ny>>, TupleHash<Key>, cache_tag>(
                    pd.ctx());
            Key key{pd, pi, px, get_perms(om, ox), get_perms(ox, oy), get_perms(om, oy)};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Find position of the power label
            int power_pos = 0;
            if (okr != 0) {
                auto it_oy = std::find(oy.begin(), oy.end(), okr);
                if (it_oy == oy.end())
                    throw std::runtime_error("The dimension label `okr` wasn't found on `oy`");
                power_pos = it_oy - oy.begin();
            }

            // Create partition
            From_size_out<Nx> pxr(px.size(), px.ctx());
            From_size_out<Ny> pyr(px.size(), px.ctx());
            for (unsigned int i = 0; i < px.size(); ++i) {
                pxr[i][0] = get_dimensions(om, concat(pd[i][0], pi[i][0]), ox, px[i][0], ox, false);
                pxr[i][1] = get_dimensions(om, concat(pd[i][1], pi[i][1]), ox, px[i][1], ox, false);
                pyr[i][0] = get_dimensions(om, concat(pd[i][0], pi[i][0]), ox, px[i][0], oy, false);
                pyr[i][1] = get_dimensions(om, concat(pd[i][1], pi[i][1]), ox, px[i][1], oy, false);
                if (okr != 0) {
                    pyr[i][0][power_pos] = 0;
                    pyr[i][1][power_pos] = power;
                }
            }
            cache.insert(key, {pxr, pyr}, storageSize(pxr) + storageSize(pyr));

            return {pxr, pyr};
        }

        /// RSB operator - tensor multiplication
        /// \param bsr: BSR tensor components
        /// \param oim: dimension labels for the RSB operator image space
        /// \param odm: dimension labels for the RSB operator domain space
        /// \param px: partitioning of the right tensor in consecutive ranges
        /// \param ox: dimension labels for the right operator
        /// \param vx: right input tensor components
        /// \param py: partitioning of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param vy: output tensor components
        /// \param co: coordinate linearization order
        /// \param copyadd: either copy or add the multiplication result into the output tensor y

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T,
                  typename Comm, typename XPU0, typename XPU1, typename EWOP>
        void bsr_krylov(const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &bsr, const Order<Ni> &oim,
                        const Order<Nd> &odm, const From_size<Nx> &px, const Order<Nx> &ox,
                        const Components_tmpl<Nx, const T, XPU0, XPU1> &vx, const From_size<Ny> &py,
                        const Order<Ny> &oy, char okr, const Components_tmpl<Ny, T, XPU0, XPU1> &vy,
                        Comm comm, EWOP, CoorOrder co) {

            tracker<Cpu> _t("distributed BSR matvec", Cpu{0});

            // Check the compatibility of the tensors
            //Coor<Nd> dimd = get_dim(bsr.pd);
            //Coor<Ni> dimi = get_dim(bsr.pi);
            Coor<Nx> dimx = get_dim(px);
            Coor<Ny> dimy = get_dim(py);
            //std::size_t volA, volB, volC;
            //local_bar_krylov_check<Nd, Ni, Nx, Ny>(dimi, dimd, oim, odm, bsr.c.first[0].blocki,bsr.c.first[0].blockm  );

            // Check that vm and vx have the same components and on the same device
            if (bsr.c.first.size() != vx.first.size() || bsr.c.second.size() != vx.second.size())
                throw std::runtime_error(
                    "the two input tensors should have the same number of components");
            bool unmatch_dev = false;
            for (unsigned int i = 0; i < bsr.c.first.size(); ++i)
                if (deviceId(bsr.c.first[i].v.it.ctx()) != deviceId(vx.first[i].it.ctx()))
                    unmatch_dev = true;
            for (unsigned int i = 0; i < bsr.c.second.size(); ++i)
                if (deviceId(bsr.c.second[i].v.it.ctx()) != deviceId(vx.second[i].it.ctx()))
                    unmatch_dev = true;
            if (unmatch_dev)
                throw std::runtime_error(
                    "Each component of the input tensors should be on the same device");

            /// Get power and bring pieces of BSR operator to do powers without extra communications
            unsigned int power = 1;
            if (okr != 0) {
                auto it_oy = std::find(oy.begin(), oy.end(), okr);
                if (it_oy == oy.end())
                    throw std::runtime_error("The dimension label `okr` wasn't found on `oy`");
                power = dimy[it_oy - oy.begin()];
            }
            //const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &bsr0 = get_bsr_power(bsr, power, comm);

            // Generate the partitioning and the storage for the dense matrix input and output tensor

            auto pxy_ = get_output_partition(bsr.pd, odm, bsr.pi, oim, px, ox, oy, okr, power);
            unsigned int ncomponents = vx.first.size() + vx.second.size();

            // Copy the input dense tensor to a compatible layout to the sparse tensor
            From_size<Nx> px_ = pxy_.first;
            Components_tmpl<Nx, const T, XPU0, XPU1> vx_;
            if (px_ != px) {
                Components_tmpl<Nx, T, XPU0, XPU1> vx0_;
                for (unsigned int i = 0; i < bsr.c.first.size(); ++i) {
                    const unsigned int componentId = bsr.c.first[i].v.componentId;
                    const unsigned int pi = comm.rank * ncomponents + componentId;
                    const Coor<Nx> &dimx = px_[pi][1];
                    vector<T, XPU0> vxi(volume(dimx), bsr.c.first[i].v.it.ctx());
                    vx0_.first.push_back(Component<Nx, T, XPU0>{vxi, dimx, componentId});
                }
                for (unsigned int i = 0; i < bsr.c.second.size(); ++i) {
                    const unsigned int componentId = bsr.c.second[i].v.componentId;
                    const unsigned int pi = comm.rank * ncomponents + componentId;
                    const Coor<Nx> &dimx = px_[pi][1];
                    vector<T, XPU1> vxi(volume(dimx), bsr.c.second[i].v.it.ctx());
                    vx0_.second.push_back(Component<Nx, T, XPU1>{vxi, dimx, componentId});
                }
                copy<Nx, Nx, T>(T{1}, px, {}, dimx, ox, vx, px_, {}, ox, vx0_, comm, EWOp::Copy{},
                                co);
                vx_ = toConst(vx0_);
            } else {
                vx_ = vx;
            }

            // Allocate the output tensor and do the contraction
            From_size<Ny> py_ = pxy_.second;
            Components_tmpl<Ny, T, XPU0, XPU1> vy_;
            for (unsigned int i = 0; i < bsr.c.first.size(); ++i) {
                const unsigned int componentId = bsr.c.first[i].v.componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Nx> &dimx = px_[pi][1];
                const Coor<Ny> &dimy = py_[pi][1];
                vector<T, XPU0> vyi(volume(dimy), bsr.c.first[i].v.it.ctx());
                vy_.first.push_back(Component<Ny, T, XPU0>{vyi, dimy, componentId});
                local_bsr_krylov<Nd, Ni, Nx, Ny, T>(bsr.c.first[i], oim, odm, dimx, ox,
                                                    vx_.first[i].it, dimy, oy, okr, vy_.first[i].it,
                                                    EWOP{});
            }
            for (unsigned int i = 0; i < bsr.c.second.size(); ++i) {
                const unsigned int componentId = bsr.c.second[i].v.componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Nx> &dimx = px_[pi][1];
                const Coor<Ny> &dimy = py_[pi][1];
                vector<T, XPU1> vyi(volume(dimy), bsr.c.second[i].v.it.ctx());
                vy_.second.push_back(Component<Ny, T, XPU1>{vyi, dimy, componentId});
                local_bsr_krylov<Nd, Ni, Nx, Ny, T>(bsr.c.second[i], oim, odm, dimx, ox,
                                                    vx_.second[i].it, dimy, oy, okr, vy_.second[i].it,
                                                    EWOP{});
            }

            // Scale the output tensor by beta
            copy<Ny, Ny, T>(T{0}, py, {}, dimy, oy, toConst(vy), py, {}, oy, vy, comm, EWOp::Copy{},
                            co);

            // Reduce all the subtensors to the final tensor
            copy<Ny, Ny, T>(1.0, py_, {}, dimy, oy, toConst(vy_), py, {}, oy, vy, comm, EWOp::Add{},
                            co);
        }

        /// RSB operator - tensor multiplication
        /// \param oim: dimension labels for the RSB operator image space
        /// \param odm: dimension labels for the RSB operator domain space
        /// \param px: partitioning of the right tensor in consecutive ranges
        /// \param ox: dimension labels for the right operator
        /// \param vx: right input tensor components
        /// \param py: partitioning of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param vy: output tensor components
        /// \param co: coordinate linearization order
        /// \param copyadd: either copy or add the multiplication result into the output tensor y

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T,
                  typename Comm, typename XPU0, typename XPU1>
        void bsr_krylov(const BSRComponents<Nd, Ni, T> &bsr, const Order<Ni> &oim,
                        const Order<Nd> &odm, const From_size<Nx> &px, const Order<Nx> &ox,
                        const Components_tmpl<Nx, const T, XPU0, XPU1> &vx, const From_size<Ny> &py,
                        const Order<Ny> &oy, char okr, const Components_tmpl<Ny, T, XPU0, XPU1> &vy,
                        Comm comm, CoorOrder co, CopyAdd copyadd) {

            if (getDebugLevel() >= 1) {
                barrier(comm);
                for (const auto &i : bsr.c.first) sync(i.v.it.ctx());
                for (const auto &i : bsr.c.second) sync(i.v.it.ctx());
            }

            switch (copyadd) {
            case Copy:
                bsr_krylov(bsr, oim, odm, px, ox, vx, py, oy, okr, vy, comm, EWOp::Copy{}, co);
                break;
            case Add:
                bsr_krylov(bsr, oim, odm, px, ox, vx, py, oy, okr, vy, comm, EWOp::Add{}, co);
                break;
            }

            if (getDebugLevel() >= 1) {
                for (const auto &i : bsr.c.first) sync(i.v.it.ctx());
                for (const auto &i : bsr.c.second) sync(i.v.it.ctx());
                barrier(comm);
            }
        }
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Create BSR sparse operator
    /// \param pim: partitioning of the RSB operator image in consecutive ranges
    /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param blockim: image dimensions of the block
    /// \param blockdm: domain dimensions of the block
    /// \param blockImFast: whether the blocks are stored with the image indices the fastest
    /// \param ii: ii[i] is the index of the first nonzero block on the i-th blocked image operator element
    /// \param jj: domain coordinates of the nonzero blocks of RSB operator
    /// \param v: nonzero values
    /// \param ctx: context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param bsrh (out) handle to BSR nonzero pattern
    ///
    /// NOTE: keep allocated the space pointed out by ii, jj, and v until calling `destroy_bsr`.

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_bsr(const PartitionItem<Ni> *pim, const PartitionItem<Nd> *pdm, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, Coor<Nd> **jj, const T **v, const Context *ctx,
                    MPI_Comm mpicomm, CoorOrder co, BSR_handle **bsrh, Session session = 0) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ii, jj, ctx, ncomponents, pim, pdm, blockdm, blockim, blockImFast, comm,
                co, session)};
        *bsrh = r;
    }

    /// BSR sparse operator - tensor multiplication
    /// \param bsrh: BSR handle
    /// \param oim: dimension labels for the RSB operator image space
    /// \param odm: dimension labels for the RSB operator domain space
    /// \param px: partitioning of the right tensor in consecutive ranges
    /// \param ox: dimension labels for the right operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the resulting tensor in consecutive ranges
    /// \param oy: dimension labels for the output tensor
    /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
    /// \param vy: data for the output tensor
    /// \param ctxy: context for each data pointer in vy

    template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T>
    void bsr_krylov(BSR_handle *bsrh, const char *oim, const char *odm, const PartitionItem<Nx> *px,
                    int ncomponents, const char *ox, const T **vx, const PartitionItem<Ny> *py,
                    const char *oy, char okr, T **vy, const Context *ctx, MPI_Comm mpicomm,
                    CoorOrder co, CopyAdd copyadd, Session session = 0) {

        Order<Ni> oim_ = detail::toArray<Ni>(oim, "oim");
        Order<Nd> odm_ = detail::toArray<Nd>(odm, "odm");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        detail::bsr_krylov<Nd, Ni, Nx, Ny, T>(
            *bsr, oim_, odm_, detail::get_from_size(px, ncomponents * comm.nprocs, session), ox_,
            detail::get_components<Nx>(vx, ctx, ncomponents, px, comm, session),
            detail::get_from_size(py, ncomponents * comm.nprocs, session), oy_, okr,
            detail::get_components<Ny>(vy, ctx, ncomponents, py, comm, session), comm, co, copyadd);
    }
#endif // SUPERBBLAS_USE_MPI

    /// Create BSR sparse operator
    /// \param pim: partitioning of the RSB operator image in consecutive ranges
    /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param blockim: image dimensions of the block
    /// \param blockdm: domain dimensions of the block
    /// \param blockImFast: whether the blocks are stored with the image indices the fastest
    /// \param ii: ii[i] is the index of the first nonzero block on the i-th blocked image operator element
    /// \param jj: domain coordinates of the nonzero blocks of RSB operator
    /// \param v: nonzero values
    /// \param ctx: context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param bsrh (out) handle to BSR nonzero pattern
    ///
    /// NOTE: keep allocated the space pointed out by ii, jj, and v until calling `destroy_bsr`.

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_bsr(const PartitionItem<Ni> *pim, const PartitionItem<Nd> *pdm, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, Coor<Nd> **jj, const T **v, const Context *ctx, CoorOrder co,
                    BSR_handle **bsrh, Session session = 0) {

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ii, jj, ctx, ncomponents, pim, pdm, blockdm, blockim, blockImFast, comm,
                co, session)};
        *bsrh = r;
    }

    /// Destroy RSB sparse operator
    /// \param bsrh: origin RSB handle to copy from

    void destroy_bsr(BSR_handle *bsrh) { delete bsrh; }

    /// BSR sparse operator - tensor multiplication
    /// \param bsrh: BSR handle
    /// \param oim: dimension labels for the RSB operator image space
    /// \param odm: dimension labels for the RSB operator domain space
    /// \param px: partitioning of the right tensor in consecutive ranges
    /// \param ox: dimension labels for the right operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the resulting tensor in consecutive ranges
    /// \param oy: dimension labels for the output tensor
    /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
    /// \param vy: data for the output tensor
    /// \param ctxy: context for each data pointer in vy

    template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T>
    void bsr_krylov(BSR_handle *bsrh, const char *oim, const char *odm, const PartitionItem<Nx> *px,
                    int ncomponents, const char *ox, const T **vx, const PartitionItem<Ny> *py,
                    const char *oy, char okr, T **vy, const Context *ctx, CoorOrder co,
                    CopyAdd copyadd, Session session = 0) {

        Order<Ni> oim_ = detail::toArray<Ni>(oim, "oim");
        Order<Nd> odm_ = detail::toArray<Nd>(odm, "odm");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        detail::bsr_krylov<Nd, Ni, Nx, Ny, T>(
            *bsr, oim_, odm_, detail::get_from_size(px, ncomponents * comm.nprocs, session), ox_,
            detail::get_components<Nx>(vx, ctx, ncomponents, px, comm, session),
            detail::get_from_size(py, ncomponents * comm.nprocs, session), oy_, okr,
            detail::get_components<Ny>(vy, ctx, ncomponents, py, comm, session), comm, co, copyadd);
    }
}
#endif // __SUPERBBLAS_BSR__