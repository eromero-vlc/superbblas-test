#ifndef __SUPERBBLAS_BSR__
#define __SUPERBBLAS_BSR__

#include "dist.h"
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
            Indices<XPU> i;              ///< Where the columns indices (j) for the ith row start
            Indices<XPU> j;              ///< Column indices
            bool j_has_negative_indices; ///< whether j has -1 to skip these nonzeros
            int num_nnz_per_row;         ///< either the number of block nnz per row or
                                         ///< -1 if not all rows has the same number of nonzeros
            IndexType nnz;               ///< total number of nonzero blocks
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
            static const MatrixLayout preferredLayout = RowMajor;
            static std::string implementation() { return "MKL"; }

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
                if (bsr.j_has_negative_indices)
                    throw std::runtime_error("bsr: unsupported -1 column indices when using MKL");
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

            double getCostPerMatvec() const {
                double block_size = (double)volume(v.blocki);
                return block_size * block_size * jj.size();
            }

            void operator()(T alpha, bool conjA, const T *x, IndexType ldx, MatrixLayout lx, T *y,
                            IndexType ldy, MatrixLayout ly, IndexType ncols, T beta = T{0}) const {
                if (lx != ly) throw std::runtime_error("Unsupported operation with MKL");
                checkMKLSparse(mkl_sparse_mm(
                    !conjA ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    alpha, *A,
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
            static std::string implementation() { return "builtin_cpu"; }

            static const SpMMAllowedLayout allowLayout = AnyLayoutForXAndY;
            static const MatrixLayout preferredLayout = RowMajor;

            BSR(const BSRComponent<Nd, Ni, T, Cpu> &v) : v(v) {
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                auto bsr = get_bsr_indices(v);
                ii = bsr.i;
                jj = bsr.j;
            }

            double getCostPerMatvec() const {
                double bi = (double)volume(v.blocki), bd = (double)volume(v.blockd);
                return bi * bd * jj.size();
            }

            void operator()(T alpha, bool conjA, const T *x, IndexType ldx, MatrixLayout lx, T *y,
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
#        pragma omp parallel for schedule(static)
#    endif
                for (IndexType i = 0; i < block_rows; ++i) {
                    for (IndexType j = ii[i], j1 = ii[i + 1]; j < j1; ++j) {
                        if (jj[j] == -1) continue;
                        if (ly == ColumnMajor)
                            xgemm(tb ? 'T' : 'N', tx ? 'T' : 'N', bi, ncols, bd, alpha,
                                  nonzeros + j * bi * bd, bi, x + jj[j] * xs, ldx, T{1}, y + i * bi,
                                  ldy, Cpu{});
                        else
                            xgemm(!tx ? 'T' : 'N', !tb ? 'T' : 'N', ncols, bi, bd, alpha,
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
            std::shared_ptr<cusparseMatDescr_t>
                descrA_bsr; ///< cuSparse descriptor for BSR matrices
            std::shared_ptr<cusparseSpMatDescr_t>
                descrA_other; ///< cuSparse descriptor for CSR and ELL matrices
            enum SparseFormat{FORMAT_BSR, FORMAT_CSR, FORMAT_ELL} spFormat; ///< the sparse format
            mutable vector<T, Gpu> buffer; ///< Auxiliary memory used by cusparseSpMM
#    else
            std::shared_ptr<hipsparseMatDescr_t> descrA_bsr; ///< hipSparse descriptor
#    endif

            SpMMAllowedLayout allowLayout;
            static const MatrixLayout preferredLayout = ColumnMajor;
            std::string implementation_;
            const std::string &implementation() const { return implementation_; }

            BSR(BSRComponent<Nd, Ni, T, Gpu> v) : v(v) {
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                if (volume(v.blocki) != volume(v.blockd))
                    throw std::runtime_error("cuSPARSE does not support non-square blocks");
                auto bsr = get_bsr_indices(v, true);
                ii = bsr.i;
                jj = bsr.j;

#    ifdef SUPERBBLAS_USE_CUDA
                IndexType block_size = volume(v.blocki);
                cudaDeviceProp prop;
                cudaCheck(cudaGetDeviceProperties(&prop, deviceId(v.i.ctx())));
                /// TODO: ELL format is disable, it isn't correct currently
                if (false && bsr.num_nnz_per_row >= 0 && !is_complex<T>::value && prop.major >= 8) {
                    spFormat = FORMAT_ELL;
                } else if (!bsr.j_has_negative_indices) {
                    spFormat = block_size == 1 ? FORMAT_CSR : FORMAT_BSR;
                } else {
                    throw std::runtime_error("bsr: unsupported -1 column indices when using "
                                             "cuSPARSE and not using ELL");
                }

                if (spFormat == FORMAT_BSR) {
                    implementation_ = "cusparse_bsr";
                    allowLayout = ColumnMajorForY;
                    descrA_bsr = std::shared_ptr<cusparseMatDescr_t>(
                        new cusparseMatDescr_t, [](cusparseMatDescr_t *p) {
                            cusparseDestroyMatDescr(*p);
                            delete p;
                        });
                    cusparseCheck(cusparseCreateMatDescr(&*descrA_bsr));
                    cusparseCheck(cusparseSetMatIndexBase(*descrA_bsr, CUSPARSE_INDEX_BASE_ZERO));
                    cusparseCheck(cusparseSetMatType(*descrA_bsr, CUSPARSE_MATRIX_TYPE_GENERAL));
                } else {
                    static_assert(sizeof(IndexType) == 4);
                    IndexType num_cols = volume(v.dimd);
                    IndexType num_rows = volume(v.dimi);
                    descrA_other = std::shared_ptr<cusparseSpMatDescr_t>(
                        new cusparseSpMatDescr_t, [](cusparseSpMatDescr_t *p) {
                            cusparseDestroySpMat(*p);
                            delete p;
                        });
                    allowLayout = SameLayoutForXAndY;
                    if (spFormat == FORMAT_CSR) {
                        implementation_ = "cusparse_csr";
                        cusparseCheck(cusparseCreateCsr(
                            &*descrA_other, num_rows, num_cols, bsr.nnz, ii.data(), jj.data(),
                            v.it.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, toCudaDataType<T>()));
                    } else {
                        implementation_ = "cusparse_ell";
                        cusparseCheck(cusparseCreateBlockedEll(
                            &*descrA_other, num_rows, num_cols, block_size,
                            block_size * bsr.num_nnz_per_row, jj.data(), v.it.data(),
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, toCudaDataType<T>()));
                    }
                }
#    else
                if (bsr.j_has_negative_indices)
                    throw std::runtime_error("bsr: unsupported -1 column indices when using "
                                             "hipSPARSE");

                implementation_ = "hipsparse_bsr";
                allowLayout = ColumnMajorForY;
                descrA_bsr = std::shared_ptr<hipsparseMatDescr_t>(new hipsparseMatDescr_t,
                                                                  [](hipsparseMatDescr_t *p) {
                                                                      hipsparseDestroyMatDescr(*p);
                                                                      delete p;
                                                                  });
                hipsparseCheck(hipsparseCreateMatDescr(&*descrA_bsr));
                hipsparseCheck(hipsparseSetMatIndexBase(*descrA_bsr, HIPSPARSE_INDEX_BASE_ZERO));
                hipsparseCheck(hipsparseSetMatType(*descrA_bsr, HIPSPARSE_MATRIX_TYPE_GENERAL));
#    endif
            }

            double getCostPerMatvec() const {
                double block_size = (double)volume(v.blocki);
                return block_size * block_size * jj.size();
            }

            void operator()(T alpha, bool conjA, const T *x, IndexType ldx, MatrixLayout lx, T *y,
                            IndexType ldy, MatrixLayout ly, IndexType ncols, T beta = T{0}) const {
                // Check layout
                if ((allowLayout == SameLayoutForXAndY && lx != ly) ||
                    (allowLayout == ColumnMajorForY && ly == RowMajor))
                    throw std::runtime_error("BSR operator(): Unexpected layout");

                IndexType block_size = volume(v.blocki);
                IndexType num_cols = volume(v.dimd);
                IndexType num_rows = volume(v.dimi);
                IndexType block_cols = num_cols / block_size;
                IndexType block_rows = num_rows / block_size;
                IndexType num_blocks = jj.size();
#    ifdef SUPERBBLAS_USE_CUDA
                if (spFormat == FORMAT_BSR) {
                    cusparseCheck(cusparseXbsrmm(
                        ii.ctx().cusparseHandle,
                        v.blockImFast ? CUSPARSE_DIRECTION_COLUMN : CUSPARSE_DIRECTION_ROW,
                        !conjA ? CUSPARSE_OPERATION_NON_TRANSPOSE
                               : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                        lx == ColumnMajor ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                          : CUSPARSE_OPERATION_TRANSPOSE,
                        block_rows, ncols, block_cols, num_blocks, alpha, *descrA_bsr, v.it.data(),
                        ii.data(), jj.data(), block_size, x, ldx, beta, y, ldy));
                } else {
                    cusparseDnMatDescr_t matx, maty;
                    cudaDataType cudaType = toCudaDataType<T>();
                    cusparseCheck(cusparseCreateDnMat(
                        &matx, !conjA ? num_cols : num_rows, ncols, ldx, (void *)x, cudaType,
                        lx == ColumnMajor ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));
                    cusparseCheck(cusparseCreateDnMat(
                        &maty, !conjA ? num_rows : num_cols, ncols, ldy, (void *)y, cudaType,
                        ly == ColumnMajor ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));
                    std::size_t bufferSize;
                    cusparseCheck(cusparseSpMM_bufferSize(
                        ii.ctx().cusparseHandle,
                        !conjA ? CUSPARSE_OPERATION_NON_TRANSPOSE
                               : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *descrA_other, matx, &beta, maty,
                        cudaType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
                    if (bufferSize > buffer.size() * sizeof(T))
                        buffer = vector<T, Gpu>((bufferSize + sizeof(T) - 1) / sizeof(T), ii.ctx());
                    cusparseCheck(cusparseSpMM(ii.ctx().cusparseHandle,
                                               !conjA ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                                      : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                               *descrA_other, matx, &beta, maty, cudaType,
                                               CUSPARSE_SPMM_ALG_DEFAULT, buffer.data()));
                    cusparseDestroyDnMat(matx);
                    cusparseDestroyDnMat(maty);
                }
#    else
                hipsparseCheck(hipsparseXbsrmm(
                    ii.ctx().hipsparseHandle,
                    v.blockImFast ? HIPSPARSE_DIRECTION_COLUMN : HIPSPARSE_DIRECTION_ROW,
                    !conjA ? HIPSPARSE_OPERATION_NON_TRANSPOSE
                           : HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    lx == ColumnMajor ? HIPSPARSE_OPERATION_NON_TRANSPOSE
                                      : HIPSPARSE_OPERATION_TRANSPOSE,
                    block_rows, ncols, block_cols, num_blocks, alpha, *descrA_bsr, v.it.data(),
                    ii.data(), jj.data(), block_size, x, ldx, beta, y, ldy));
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
            /// Dimensions of the domain space
            Coor<Nd> dimd;
            /// Partition of the image space
            From_size<Ni> pi;
            // Dimensiosn of the image space
            Coor<Ni> dimi;
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
                    if (i.v.componentId >= ncomponents * nprocs || i.v.co != co) return false;
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
                           const Coor<Ni> &dimi, From_size_iterator<Nd> pd, const Coor<Nd> &dimd,
                           Coor<Nd> blockd, Coor<Ni> blocki, bool blockImFast, Comm comm,
                           CoorOrder co, Session session) {
            // Get components on the local process
            From_size_iterator<Nd> fsd = pd + comm.rank * ncomponents;
            From_size_iterator<Ni> fsi = pi + comm.rank * ncomponents;

            BSRComponents<Nd, Ni, T> r{};
            r.dimd = dimd;
            r.dimi = dimi;
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

        /// Concatenate p (if it isn't zero), a, b, and c

        template <std::size_t N, typename T, std::size_t Na, std::size_t Nb, std::size_t Nc>
        std::array<T, N> concat(char p, const std::array<T, Na> &a, std::size_t na,
                                const std::array<T, Nb> &b, std::size_t nb,
                                const std::array<T, Nc> &c, std::size_t nc) {
            std::array<T, N> r;
            int np = p == 0 ? 0 : 1;
            if (N != np + na + nb + nc) throw std::runtime_error("concat: invalid arguments");
            if (p != 0) r[0] = p;
            std::copy_n(a.begin(), na, r.begin() + np);
            std::copy_n(b.begin(), nb, r.begin() + np + na);
            std::copy_n(c.begin(), nc, r.begin() + np + na + nb);
            return r;
        }

        //
        // Auxiliary functions
        //

        /// Return BSR indices for a given tensor BSR and whether the nonzero
        /// pattern is compatible with blocked ELL, all rows has the same number
        /// of nonzeros, and unused nonzeros may be reported with a -1 on the
        /// the first domain coordinate.

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU,
                  typename std::enable_if<(Nd > 0 && Ni > 0), bool>::type = true>
        CsrIndices<XPU> get_bsr_indices(const BSRComponent<Nd, Ni, T, XPU> &v,
                                        bool return_jj_blocked = false) {
            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= volume(v.dimd))
                throw std::runtime_error("Ups! IndexType isn't big enough");

            Indices<Cpu> ii(v.i.size() + 1, Cpu{}), jj(v.j.size(), Cpu{});
            Indices<Cpu> vi = makeSure(v.i, Cpu{});
            vector<Coor<Nd>, Cpu> vj = makeSure(v.j, Cpu{});

            // Check if all rows has the same number of nonzeros
            bool same_nnz_per_row = true;
            for (std::size_t i = 1; i < vi.size(); ++i)
                if (vi[i] != vi[0]) same_nnz_per_row = false;
            int num_nnz_per_row = vi.size() > 0 && same_nnz_per_row ? vi[0] : -1;

            // Compute index for the first nonzero on the ith row
            ii[0] = 0;
            for (std::size_t i = 0; i < vi.size(); ++i) ii[i + 1] = ii[i] + vi[i];

            // Transform the domain coordinates into indices
            Coor<Nd> strided = get_strides<IndexType>(v.dimd, v.co);
            IndexType block_nnz = v.j.size();
            IndexType bd = return_jj_blocked ? volume(v.blockd) : 1;
            bool there_are_minus_ones_in_columns = false;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
            for (IndexType i = 0; i < block_nnz; ++i) {
                if (vj[i][0] == -1) there_are_minus_ones_in_columns = true;
                jj[i] = (vj[i][0] == -1 ? -1 : coor2index(vj[i], v.dimd, strided) / bd);
            }

            // Unsupported -1 in the domain coordinate unless using the ELL format, which
            // requires all rows to have the same number of nonzeros
            if (there_are_minus_ones_in_columns && !same_nnz_per_row)
                throw std::runtime_error(
                    "bsr: unsupported nonzero pattern specification, some domain coordinates have "
                    "-1 but not all block rows have the same number of nonzero blocks");

            return {makeSure(ii, v.i.ctx()), makeSure(jj, v.j.ctx()),
                    there_are_minus_ones_in_columns, num_nnz_per_row, ii[vi.size()]};
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU,
                  typename std::enable_if<(Nd == 0 || Ni == 0), bool>::type = true>
        std::pair<CsrIndices<XPU>, int> get_bsr_indices(const BSRComponent<Nd, Ni, T, XPU> &v,
                                                        bool return_jj_blocked = false) {
            (void)return_jj_blocked;
            return {Indices<XPU>(0, v.i.ctx()), Indices<XPU>(0, v.j.ctx()), false, -1, 0};
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
        /// \param xylayout: possible layouts for x and y
        /// \param preferred_layout: preferred layout for x and y
        /// \param co: coordinate linearization order
        /// \param transSp: (output) whether to contract with the sparse operator image space
        /// \param lx: (output) layout for the x tensor
        /// \param ly: (output) layout for the y tensor
        /// \param volC: (output) number of columns to contract
        /// \param sug_ox: (output) suggested order for the x
        /// \param sug_oy: (output) suggested order for the y
        /// \param sug_oy_trans: (output) suggested order for copying y into x
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
                                    char okr, SpMMAllowedLayout xylayout,
                                    MatrixLayout preferred_layout, CoorOrder co, bool &transSp,
                                    MatrixLayout &lx, MatrixLayout &ly, std::size_t &volC,
                                    Order<Nx> &sug_ox, Order<Ny> &sug_oy, Order<Ny> &sug_oy_trans) {

            if (co == FastToSlow) {
                Order<Nx> sug_ox0;
                Order<Ny> sug_oy0;
                Order<Ny> sug_oy_trans0;
                local_bsr_krylov_check(reverse(dimi), reverse(dimd), reverse(oi), reverse(od),
                                       reverse(blocki), reverse(blockd), reverse(dimx), reverse(ox),
                                       reverse(dimy), reverse(oy), okr, xylayout, preferred_layout,
                                       SlowToFast, transSp, lx, ly, volC, sug_ox0, sug_oy0,
                                       sug_oy_trans0);
                sug_ox = reverse(sug_ox0);
                sug_oy = reverse(sug_oy0);
                sug_oy_trans = reverse(sug_oy_trans0);
                return;
            }

            // Find all common labels in od and oi
            for (char c : od)
                if (std::find(oi.begin(), oi.end(), c) != oi.end())
                    std::runtime_error(
                        "Common label between the domain and image of the sparse matrix");

            // Check the dimensions
            bool failMatch = false;
            for (unsigned int i = 0; i < Nx; ++i) {
                if (ox[i] == okr) continue;
                auto sd = std::find(od.begin(), od.end(), ox[i]);
                if (sd != od.end()) {
                    failMatch |= (dimd[sd - od.begin()] != dimx[i]);
                    continue;
                }
                auto si = std::find(oi.begin(), oi.end(), ox[i]);
                if (si != oi.end()) failMatch |= (dimi[si - oi.begin()] != dimx[i]);
            }
            if (failMatch)
                throw std::runtime_error("bsr_krylov: dimensions of the dense input tensor "
                                         "doesn't match the sparse tensor");
            for (unsigned int i = 0; i < Ny; ++i) {
                if (ox[i] == okr) continue;
                auto sd = std::find(od.begin(), od.end(), oy[i]);
                if (sd != od.end()) {
                    failMatch |= (dimd[sd - od.begin()] != dimy[i]);
                    continue;
                }
                auto si = std::find(oi.begin(), oi.end(), oy[i]);
                if (si != oi.end()) {
                    failMatch |= (dimi[si - oi.begin()] != dimy[i]);
                    continue;
                }
                auto sx = std::find(ox.begin(), ox.end(), oy[i]);
                if (sx != ox.end()) {
                    failMatch |= (dimx[sx - ox.begin()] != dimy[i]);
                    continue;
                }
            }
            if (failMatch)
                throw std::runtime_error(
                    "bsr_krylov: dimensions of the dense output tensor "
                    "doesn't match the sparse tensor or the dense input tensor");

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
            bool powerFoundOnx = false;
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
                } else if (okr != 0 && c == okr) {
                    powerFoundOnx = true;
                    if (dimx[ix] > 1)
                        throw std::runtime_error(
                            "The power dimension on the input vector has a size larger than one");
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
            bool powerFoundOny = false;
            int power = 1;
            int iy = 0;
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
                } else if (okr != 0 && c == okr) {
                    powerFoundOny = true;
                    power = dimy[iy];
                } else if (std::find(ox.begin(), ox.end(), c) != ox.end()) {
                    // Do nothing
                } else {
                    throw std::runtime_error(
                        "Dimension label for the dense input vector doesn't match the "
                        "input sparse dimensions nor the dense output dimensions");
                }
                iy++;
            }

            // Check okr: either zero or a label on oy
            if (okr != 0 && (!powerFoundOnx || !powerFoundOny))
                throw std::runtime_error(
                    "The power dimension isn't on the input or output dense tensor");

            // If power, dimi should be equal to dimd
            if (okr != 0) {
                if (Nd != Ni)
                    throw std::runtime_error(
                        "Unsupported power for operators that have different number of dimensions "
                        "for the domain and image spaces");
                if (power > 1)
                    for (unsigned int i = 0, i1 = std::min(Nd, Ni); i < i1; ++i)
                        if (dimd[i] != dimi[i])
                            throw std::runtime_error("When using powers the domain and the image "
                                                     "of the sparse operator should be the same");
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

            // Check that ox should one of (okr,C,D,d) or (okr,D,d,C) or (okr,C,I,i) or (okr,I,i,C)
            if (kindx == ContractWithDomain) {
                auto sCx = std::search(ox.begin(), ox.end(), oC.begin(), oC.begin() + nC);
                auto sDx = std::search(ox.begin(), ox.end(), oDs.begin(), oDs.begin() + nDs);
                auto sdx = std::search(ox.begin(), ox.end(), ods.begin(), ods.begin() + nds);
                if ((okr != 0 && ox[0] != okr) || (nC > 0 && sCx == ox.end()) ||
                    (nDs > 0 && sDx == ox.end()) || (nds > 0 && sdx == ox.end()) ||
                    (nDs > 0 && nds > 0 && sDx > sdx) ||
                    (nC > 0 && nDs > 0 && nds > 0 && sDx < sCx && sCx < sdx)) {
                    lx = preferred_layout;
                    sug_ox = (lx == ColumnMajor ? concat<Nx>(okr, oC, nC, oDs, nDs, ods, nds)
                                                : concat<Nx>(okr, oDs, nDs, ods, nds, oC, nC));
                } else {
                    lx = (nC == 0 || ((nDs == 0 || sCx < sDx) && (nds == 0 || sCx < sdx)))
                             ? ColumnMajor
                             : RowMajor;
                    sug_ox = ox;
                }

                auto sCy = std::search(oy.begin(), oy.end(), oC.begin(), oC.begin() + nC);
                auto sIy = std::search(oy.begin(), oy.end(), oIs.begin(), oIs.begin() + nIs);
                auto siy = std::search(oy.begin(), oy.end(), ois.begin(), ois.begin() + nis);
                ly = (nC == 0 || ((nIs == 0 || sCy < sIy) && (nds == 0 || sCy < siy))) ? ColumnMajor
                                                                                       : RowMajor;
                if ((okr != 0 && oy[0] != okr) || (nC > 0 && sCy == oy.end()) ||
                    (nIs > 0 && sIy == oy.end()) || (nis > 0 && siy == oy.end()) ||
                    (nIs > 0 && nis > 0 && sIy > siy) ||
                    (nC > 0 && nIs > 0 && nis > 0 && sIy < sCy && sCy < siy) ||
                    (lx != ly && xylayout == SameLayoutForXAndY) ||
                    (ly == RowMajor && xylayout == ColumnMajorForY)) {
                    ly = (xylayout == SameLayoutForXAndY
                              ? lx
                              : (xylayout == ColumnMajorForY ? ColumnMajor : preferred_layout));
                    sug_oy = (ly == ColumnMajor ? concat<Ny>(okr, oC, nC, oIs, nIs, ois, nis)
                                                : concat<Ny>(okr, oIs, nIs, ois, nis, oC, nC));
                } else {
                    sug_oy = oy;
                }

                if (okr != 0 && power > 1) {
                    sug_oy_trans = sug_oy;
                    for (unsigned int i = 0; i < Ny; ++i) {
                        auto s = std::find(oi.begin(), oi.end(), sug_oy_trans[i]);
                        if (s != oi.end()) sug_oy_trans[i] = od[s - oi.begin()];
                    }
                }

                transSp = false;
            } else {
                auto sCx = std::search(ox.begin(), ox.end(), oC.begin(), oC.begin() + nC);
                auto sIx = std::search(ox.begin(), ox.end(), oIs.begin(), oIs.begin() + nIs);
                auto six = std::search(ox.begin(), ox.end(), ois.begin(), ois.begin() + nis);
                if ((okr != 0 && ox[0] != okr) || (nC > 0 && sCx == ox.end()) ||
                    (nIs > 0 && sIx == ox.end()) || (nis > 0 && six == ox.end()) ||
                    (nIs > 0 && nis > 0 && sIx > six) ||
                    (nC > 0 && nIs > 0 && nis > 0 && sIx < sCx && sCx < six)) {
                    lx = preferred_layout;
                    sug_ox = (lx == ColumnMajor ? concat<Nx>(okr, oC, nC, oIs, nIs, ois, nis)
                                                : concat<Nx>(okr, oIs, nIs, ois, nis, oC, nC));
                } else {
                    lx = (nC == 0 || ((nIs == 0 || sCx < sIx) && (nis == 0 || sCx < six)))
                             ? ColumnMajor
                             : RowMajor;
                    sug_ox = ox;
                }

                auto sCy = std::search(oy.begin(), oy.end(), oC.begin(), oC.begin() + nC);
                auto sDy = std::search(oy.begin(), oy.end(), oDs.begin(), oDs.begin() + nDs);
                auto sdy = std::search(oy.begin(), oy.end(), ods.begin(), ods.begin() + nds);
                ly = (nC == 0 || ((nDs == 0 || sCy < sDy) && (nds == 0 || sCy < sdy))) ? ColumnMajor
                                                                                       : RowMajor;
                if ((okr != 0 && oy[0] != okr) || (nC > 0 && sCy == oy.end()) ||
                    (nDs > 0 && sDy == oy.end()) || (nds > 0 && sdy == oy.end()) ||
                    (nDs > 0 && nds > 0 && sDy > sdy) ||
                    (nC > 0 && nDs > 0 && nds > 0 && sDy < sCy && sCy < sdy) ||
                    (lx != ly && xylayout == SameLayoutForXAndY) ||
                    (ly == RowMajor && xylayout == ColumnMajorForY)) {

                } else {
                    ly = (xylayout == SameLayoutForXAndY
                              ? lx
                              : (xylayout == ColumnMajorForY ? ColumnMajor : preferred_layout));
                    sug_oy = (ly == ColumnMajor ? concat<Ny>(okr, oC, nC, oDs, nDs, ods, nds)
                                                : concat<Ny>(okr, oDs, nDs, ods, nds, oC, nC));
                }

                if (okr != 0 && power > 1) {
                    sug_oy_trans = sug_oy;
                    for (unsigned int i = 0; i < Ny; ++i) {
                        auto s = std::find(od.begin(), od.end(), sug_oy_trans[i]);
                        if (s != od.end()) sug_oy_trans[i] = oi[s - od.begin()];
                    }
                }

                transSp = true;
            }
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
                  typename XPU>
        void local_bsr_krylov(T alpha, const BSR<Nd, Ni, T, XPU> &bsr, const Order<Ni> &oim,
                              const Order<Nd> &odm, const Coor<Nx> &dimx, const Order<Nx> &ox,
                              vector<T, XPU> vx, const Coor<Ny> &dimy, const Order<Ny> &oy,
                              char okr, vector<T, XPU> vy) {

            tracker<XPU> _t(std::string("local BSR matvec (") + bsr.implementation() +
                                std::string(")"),
                            vx.ctx());

            // Quick exit
            if (volume(dimx) == 0 && volume(dimy) == 0) return;

            // Check inputs and get the common dimensions
            bool transSp;
            MatrixLayout lx, ly;
            std::size_t volC;
            Order<Nx> sug_ox;
            Order<Ny> sug_oy;
            Order<Ny> sug_oy_trans;
            local_bsr_krylov_check(bsr.v.dimi, bsr.v.dimd, oim, odm, bsr.v.blocki, bsr.v.blockd,
                                   dimx, ox, dimy, oy, okr, bsr.allowLayout, bsr.preferredLayout,
                                   bsr.v.co, transSp, lx, ly, volC, sug_ox, sug_oy, sug_oy_trans);
            if (sug_ox != ox || sug_oy != oy)
                throw std::runtime_error(
                    "Unsupported layout for the input and output dense tensors");

            // Set zero
            local_copy<Ny, Ny, T, T>(0, oy, {{}}, dimy, dimy, (vector<const T, XPU>)vy, Mask<XPU>{},
                                     oy, {{}}, dimy, vy, Mask<XPU>{}, EWOp::Copy{}, bsr.v.co);

            std::size_t vold = volume(bsr.v.dimd), voli = volume(bsr.v.dimi);
            if (vold == 0 || voli == 0) return;
            IndexType ldx = lx == ColumnMajor ? (!transSp ? vold : voli) : volC;
            IndexType ldy = ly == ColumnMajor ? (!transSp ? voli : vold) : volC;

            // Do the contraction
            _t.cost = bsr.getCostPerMatvec() * volC * multiplication_cost<T>::value;
            bsr(alpha, transSp, vx.data(), ldx, lx, vy.data(), ldy, ly, volC);
        }

        /// Get the partitions for the dense input and output tensors
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param sizex: number of elements to operate in each dimension
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny>
        std::pair<From_size<Nx>, From_size<Ny>>
        get_output_partition(From_size<Nd> pd, const Order<Nd> &od, From_size<Ni> pi,
                             const Order<Ni> &oi, From_size<Nx> px, const Order<Nx> &ox,
                             const Order<Nx> &sug_ox, const Coor<Nx> &sizex, const Order<Ny> &oy,
                             const Order<Ny> &sug_oy, char okr) {
            assert(pd.size() == pi.size() && pi.size() == px.size());

            // Find partition on cache
            Order<Nd + Ni> om = concat(od, oi);
            using Key = std::tuple<From_size<Nd>, From_size<Ni>, From_size<Nx>, Coor<Nx>,
                                   PairPerms<Nd + Ni, Nx>, PairPerms<Nx, Ny>,
                                   PairPerms<Nd + Ni, Ny>, PairPerms<Nx, Nx>, PairPerms<Ny, Ny>>;
            struct cache_tag {};
            auto cache =
                getCache<Key, std::pair<From_size<Nx>, From_size<Ny>>, TupleHash<Key>, cache_tag>(
                    pd.ctx());
            Key key{pd,
                    pi,
                    px,
                    sizex,
                    get_perms(om, ox),
                    get_perms(ox, oy),
                    get_perms(om, oy),
                    get_perms(ox, sug_ox),
                    get_perms(oy, sug_oy)};
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
                pxr[i][0] = get_dimensions(om, concat(pd[i][0], pi[i][0]), ox, {{}}, sug_ox, false);
                pxr[i][1] =
                    get_dimensions(om, concat(pd[i][1], pi[i][1]), ox, sizex, sug_ox, false);
                pyr[i][0] = get_dimensions(om, concat(pd[i][0], pi[i][0]), ox, {{}}, sug_oy, false);
                pyr[i][1] =
                    get_dimensions(om, concat(pd[i][1], pi[i][1]), ox, sizex, sug_oy, false);
                if (okr != 0) {
                    pyr[i][0][power_pos] = 0;
                    pyr[i][1][power_pos] = 1;
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
                  typename Comm, typename XPU0, typename XPU1>
        Request bsr_krylov(T alpha, const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> bsr,
                           const Order<Ni> oim, const Order<Nd> odm, const From_size<Nx> &px,
                           const Order<Nx> &ox, const Coor<Nx> &fromx, const Coor<Nx> &sizex,
                           const Coor<Nx> &dimx, const Components_tmpl<Nx, T, XPU0, XPU1> &vx,
                           const From_size<Ny> py, const Order<Ny> oy, const Coor<Ny> fromy,
                           const Coor<Ny> sizey, const Coor<Ny> dimy, char okr,
                           const Components_tmpl<Ny, T, XPU0, XPU1> vy, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                barrier(comm);
                for (const auto &i : bsr.c.first) sync(i.v.it.ctx());
                for (const auto &i : bsr.c.second) sync(i.v.it.ctx());
            }

            tracker<Cpu> _t("distributed BSR matvec", Cpu{0});

            // Check that co is that same as BSR
            if (co != bsr.co)
                throw std::runtime_error("Unsupported to use a different coordinate ordering that "
                                         "one used to create the matrix");

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
            unsigned int power_pos = 0;
            if (okr != 0) {
                auto it_oy = std::find(oy.begin(), oy.end(), okr);
                if (it_oy == oy.end())
                    throw std::runtime_error("The dimension label `okr` wasn't found on `oy`");
                power_pos = it_oy - oy.begin();
                power = sizey[power_pos];
            }

            // Generate the partitioning and the storage for the dense matrix input and output tensor
            Order<Nx> sug_ox = ox;
            Order<Ny> sug_oy = oy;
            Order<Ny> sug_oy_trans;
            if (bsr.c.first.size() > 0) {
                bool transSp;
                MatrixLayout lx, ly;
                std::size_t volC;
                local_bsr_krylov_check(bsr.dimi, bsr.dimd, oim, odm, bsr.blocki, bsr.blockd, sizex,
                                       ox, sizey, oy, okr, bsr.c.first[0].allowLayout,
                                       bsr.c.first[0].preferredLayout, co, transSp, lx, ly, volC,
                                       sug_ox, sug_oy, sug_oy_trans);
            } else if (bsr.c.second.size() > 0) {
                bool transSp;
                MatrixLayout lx, ly;
                std::size_t volC;
                local_bsr_krylov_check(bsr.dimi, bsr.dimd, oim, odm, bsr.blocki, bsr.blockd, sizex,
                                       ox, sizey, oy, okr, bsr.c.second[0].allowLayout,
                                       bsr.c.second[0].preferredLayout, co, transSp, lx, ly, volC,
                                       sug_ox, sug_oy, sug_oy_trans);
            }
            Coor<Nx> sug_dimx = reorder_coor(dimx, find_permutation(ox, sug_ox));
            Coor<Ny> sizey0 = sizey;
            if (power > 1) sizey0[power_pos] = 1;
            Coor<Ny> sug_sizey = reorder_coor(sizey0, find_permutation(oy, sug_oy));

            auto pxy_ = get_output_partition(bsr.pd, odm, bsr.pi, oim, px, ox, sug_ox, sizex, oy,
                                             sug_oy, okr);
            unsigned int ncomponents = vx.first.size() + vx.second.size();

            // Copy the input dense tensor to a compatible layout to the sparse tensor
            From_size<Nx> px_ = pxy_.first;
            auto vx_and_req =
                reorder_tensor_request(px, ox, fromx, sizex, dimx, vx, px_, sug_dimx, sug_ox, comm,
                                       co, power > 1 /* force copy when power > 1 */);
            Components_tmpl<Nx, T, XPU0, XPU1> vx_ = vx_and_req.first;

            Request bsr_req =
                [=] {
                    // Wait for the data to be ready
                    wait(vx_and_req.second);

                    // Allocate the output tensor
                    From_size<Ny> py_ = pxy_.second;
                    Components_tmpl<Ny, T, XPU0, XPU1> vy_ = like_this_components(py_, vx_, comm);

                    // Do contraction
                    for (unsigned int p = 0; p < power; ++p) {
                        for (unsigned int i = 0; i < bsr.c.first.size(); ++i) {
                            const unsigned int componentId = bsr.c.first[i].v.componentId;
                            const unsigned int pi = comm.rank * ncomponents + componentId;
                            local_bsr_krylov<Nd, Ni, Nx, Ny, T>(
                                p == 0 ? alpha : T{1}, bsr.c.first[i], oim, odm, px_[pi][1], sug_ox,
                                vx_.first[i].it, py_[pi][1], sug_oy, okr, vy_.first[i].it);
                        }
                        for (unsigned int i = 0; i < bsr.c.second.size(); ++i) {
                            const unsigned int componentId = bsr.c.second[i].v.componentId;
                            const unsigned int pi = comm.rank * ncomponents + componentId;
                            local_bsr_krylov<Nd, Ni, Nx, Ny, T>(
                                p == 0 ? alpha : T{1}, bsr.c.second[i], oim, odm, px_[pi][1],
                                sug_ox, vx_.second[i].it, py_[pi][1], sug_oy, okr,
                                vy_.second[i].it);
                        }

                        // Copy the result to final tensor
                        Coor<Ny> fromyi = fromy;
                        if (p > 0) fromyi[power_pos] += p;
                        copy<Ny, Ny, T>(1.0, py_, {{}}, sug_sizey, sug_sizey, sug_oy, toConst(vy_),
                                        py, fromyi, dimy, oy, vy, comm, EWOp::Copy{}, co);

                        // Copy the result into x for doing the next power
                        if (p == power - 1) break;
                        copy<Nx, Nx, T>(T{0}, px_, {{}}, sug_dimx, sug_dimx, sug_ox, toConst(vx_),
                                        px_, {{}}, sug_dimx, sug_ox, vx_, comm, EWOp::Copy{}, co);
                        copy<Ny, Nx, T>(T{1}, py_, {{}}, sug_sizey, sug_sizey, sug_oy_trans,
                                        toConst(vy_), px_, {{}}, sug_dimx, sug_ox, vx_, comm,
                                        EWOp::Copy{}, co);
                    }
                };

            // Do the contraction now if we have all the data ready; otherwise, postpone
            Request r;
            if (vx_and_req.second)
                r = bsr_req;
            else
                wait(bsr_req);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : bsr.c.first) sync(i.v.it.ctx());
                for (const auto &i : bsr.c.second) sync(i.v.it.ctx());
                barrier(comm);
            }

            return r;
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
    void create_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                    const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, Coor<Nd> **jj, const T **v, const Context *ctx,
                    MPI_Comm mpicomm, CoorOrder co, BSR_handle **bsrh, Session session = 0) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ii, jj, ctx, ncomponents, pim, dimi, pdm, dimd, blockdm, blockim,
                blockImFast, comm, co, session)};
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
    void bsr_krylov(T alpha, BSR_handle *bsrh, const char *oim, const char *odm,
                    const PartitionItem<Nx> *px, int ncomponents, const char *ox,
                    const Coor<Nx> &fromx, const Coor<Nx> &sizex, const Coor<Nx> &dimx,
                    const T **vx, const PartitionItem<Ny> *py, const char *oy,
                    const Coor<Ny> &fromy, const Coor<Ny> &sizey, const Coor<Ny> &dimy, char okr,
                    T **vy, const Context *ctx, MPI_Comm mpicomm, CoorOrder co,
                    Request *request = nullptr, Session session = 0) {

        Order<Ni> oim_ = detail::toArray<Ni>(oim, "oim");
        Order<Nd> odm_ = detail::toArray<Nd>(odm, "odm");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        Request r = detail::bsr_krylov<Nd, Ni, Nx, Ny, T>(
            alpha, *bsr, oim_, odm_, detail::get_from_size(px, ncomponents * comm.nprocs, session),
            ox_, fromx, sizex, dimx,
            detail::get_components<Nx>((T **)vx, nullptr, ctx, ncomponents, px, comm, session),
            detail::get_from_size(py, ncomponents * comm.nprocs, session), oy_, fromy, sizey, dimy,
            okr, detail::get_components<Ny>(vy, nullptr, ctx, ncomponents, py, comm, session), comm,
            co);

        if (request)
            *request = r;
        else
            wait(r);
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
    void create_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                    const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, Coor<Nd> **jj, const T **v, const Context *ctx, CoorOrder co,
                    BSR_handle **bsrh, Session session = 0) {

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ii, jj, ctx, ncomponents, pim, dimi, pdm, dimd, blockdm, blockim,
                blockImFast, comm, co, session)};
        *bsrh = r;
    }

    /// Destroy RSB sparse operator
    /// \param bsrh: origin RSB handle to copy from

    inline void destroy_bsr(BSR_handle *bsrh) { delete bsrh; }

    /// BSR sparse operator - tensor multiplication
    /// \param bsrh: BSR handle
    /// \param oim: dimension labels for the RSB operator image space
    /// \param odm: dimension labels for the RSB operator domain space
    /// \param px: partitioning of the right tensor in consecutive ranges
    /// \param ox: dimension labels for the right operator
    /// \param fromx: first coordinate to operate from the origin tensor
    /// \param sizex: number of elements to operate in each dimension
    /// \param dimx: dimension size for the origin tensor
    /// \param vx: data for the second operator
    /// \param py: partitioning of the resulting tensor in consecutive ranges
    /// \param oy: dimension labels for the output tensor
    /// \param fromy: first coordinate to copy the result from the destination tensor
    /// \param sizey: number of elements to copy in each dimension
    /// \param dimy: dimension size for the destination tensor
    /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
    /// \param vy: data for the output tensor
    /// \param ctxy: context for each data pointer in vy

    template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T>
    void bsr_krylov(T alpha, BSR_handle *bsrh, const char *oim, const char *odm,
                    const PartitionItem<Nx> *px, int ncomponents, const char *ox,
                    const Coor<Nx> &fromx, const Coor<Nx> &sizex, const Coor<Nx> &dimx,
                    const T **vx, const PartitionItem<Ny> *py, const char *oy,
                    const Coor<Ny> &fromy, const Coor<Ny> &sizey, const Coor<Ny> &dimy, char okr,
                    T **vy, const Context *ctx, CoorOrder co, Request *request = nullptr,
                    Session session = 0) {

        Order<Ni> oim_ = detail::toArray<Ni>(oim, "oim");
        Order<Nd> odm_ = detail::toArray<Nd>(odm, "odm");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        wait(detail::bsr_krylov<Nd, Ni, Nx, Ny, T>(
            alpha, *bsr, oim_, odm_, detail::get_from_size(px, ncomponents * comm.nprocs, session),
            ox_, fromx, sizex, dimx,
            detail::get_components<Nx>((T **)vx, nullptr, ctx, ncomponents, px, comm, session),
            detail::get_from_size(py, ncomponents * comm.nprocs, session), oy_, fromy, sizey, dimy,
            okr, detail::get_components<Ny>(vy, nullptr, ctx, ncomponents, py, comm, session), comm,
            co));
        if (request) *request = Request{};
    }
}
#endif // __SUPERBBLAS_BSR__
