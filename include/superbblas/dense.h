#ifndef __SUPERBBLAS_DENSE__
#define __SUPERBBLAS_DENSE__

#include "dist.h"

namespace superbblas {

    namespace detail {

        /// Return an order concatenating the three given strings (or in reverse order if `co` is SlowToFast
        /// \param a,b,c: string to concatenate
        /// \return: the ordering

        template <std::size_t N, typename Va, typename Vb, typename Vc>
        Order<N> concat(const Va &a, const Vb &b, const Vc &c, CoorOrder co) {
            if (a.size() + b.size() + c.size() != N)
                throw std::runtime_error("concat: wrong string size to concat");
            if (co == FastToSlow) {
                Order<N> r;
                std::copy_n(a.begin(), a.size(), r.begin());
                std::copy_n(b.begin(), b.size(), r.begin() + a.size());
                std::copy_n(c.begin(), c.size(), r.begin() + a.size() + b.size());
                return r;
            } else {
                return concat<N>(c, b, a, FastToSlow);
            }
        }

        inline void throw_or_exit(const std::string &err_msg, bool terminate = false) {
            if (terminate) {
                std::cerr << err_msg << std::endl;
                std::exit(-1);
            } else {
                throw std::runtime_error(err_msg);
            }
        }

        inline void checkLapack(int info, bool terminate = false) {
            if (info == 0) return;
            if (info < 0)
                throw_or_exit(
                    std::string("Error in a lapack routine: wrong argument at position ") +
                        std::to_string(-info),
                    terminate);
            if (info > 0)
                throw_or_exit(std::string("Error in lapack routine: ") + std::to_string(info),
                              terminate);
        }

        template <typename T> void local_cholesky(std::size_t n, std::size_t k, vector<T, Cpu> v) {

            tracker<Cpu> _t("local cholesky (Cpu)", v.ctx());
            _t.cost = (double)n * n * n / 3 * k * multiplication_cost<T>::value;

#ifdef _OPENMP
            int num_threads = omp_get_max_threads();
#else
            int num_threads = 1;
#endif

            T *p = v.data();
            std::vector<int> info(num_threads, 0);

#ifdef _OPENMP
#    pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int id = omp_get_thread_num();
#else
                int id = 0;
#endif

#ifdef _OPENMP
#    pragma omp for schedule(static)
#endif
                for (std::size_t i = 0; i < k; ++i)
                    if (info[id] == 0) info[id] = xpotrf('U', n, p + n * n * i, n, Cpu{});
            }

            for (int i : info) checkLapack(i);
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_cholesky(std::size_t n, std::size_t k, const vector<T, Gpu> &v) {

            if (n == 0 || k == 0) return;
            if (deviceId(v.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_cholesky: unsupported allocation device");

            tracker<Gpu> _t("local cholesky (GPU)", v.ctx());
            _t.cost = (double)n * n * n / 3 * k * multiplication_cost<T>::value;

            auto xpu_host = v.ctx().toCpuPinned();
            vector<T *, Gpu> v_ps_cpu(k, xpu_host, doCacheAlloc);
            auto v_ps_cpu_ptr = v_ps_cpu.data();
            auto v_ptr = v.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i) v_ps_cpu_ptr[i] = v_ptr + n * n * i;
                },
                xpu_host);
            vector<T *, Gpu> v_ps_gpu = makeSure(v_ps_cpu, v.ctx(), doCacheAlloc);
            vector<int, Gpu> info(k, v.ctx());
            gpuSolverCheck(SUPERBBLAS_GPUSOLVER_SYMBOL(XpotrfBatched)(
                getGpuSolverHandle(v.ctx()),
                SUPERBBLAS_GPU_SELECT(xxx, CUBLAS_FILL_MODE_UPPER, HIPSOLVER_FILL_MODE_UPPER), n,
                v_ps_gpu.data(), n, info.data(), k));
            vector<int, Gpu> info_cpu = makeSure(info, xpu_host, doCacheAlloc);
            auto info_cpu_ptr = info_cpu.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i)
                        checkLapack(info_cpu_ptr[i], true /* terminate */);
                },
                xpu_host);
        }
#endif // SUPERBBLAS_USE_GPU

        /// If left_side, perform a\x -> x; and x/a -> x otherwise
        /// \param left_side: whether the inverse go to the left
        /// \param n: size of the matrix
        /// \param k: number of matrices to invert
        /// \param m: number of columns (if left_side) or rows (if !left_side) that x and y have

        template <typename T>
        void local_trsm(bool left_side, std::size_t n, std::size_t k, std::size_t m, T alpha,
                        vector<T, Cpu> a, vector<T, Cpu> x) {

            if (n == 0 || k == 0 || m == 0) return;

            tracker<Cpu> _t("local trsm (Cpu)", a.ctx());
            _t.cost = (double)n * n / 2 * m * k * multiplication_cost<T>::value;

            const T *ap = a.data();
            T *xp = x.data();

#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
            for (std::size_t i = 0; i < k; ++i)
                xtrsm(left_side ? 'L' : 'R', 'U', 'N', 'N', left_side ? n : m, left_side ? m : n,
                      alpha, ap + n * n * i, n, xp + n * m * i, left_side ? n : m, a.ctx());
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_trsm(bool left_side, std::size_t n, std::size_t k, std::size_t m, T alpha,
                        const vector<T, Gpu> &a, const vector<T, Gpu> &x) {

            if (n == 0 || k == 0 || m == 0) return;
            if (deviceId(a.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_trsm: unsupported allocation device");
            check_same_device(a.ctx(), x.ctx());
            causalConnectTo(x.ctx(), a.ctx());

            tracker<Gpu> _t("local trsm (GPU)", a.ctx());
            _t.cost = (double)n * n / 2 * m * k * multiplication_cost<T>::value;

#    ifdef SUPERBBLAS_USE_CUDA
            // NOTE: cublasXtrsmBatched presents an undocumented limitation: it fails when
            // one of the dimensions of the input matrices is too large
            auto xpu_host = a.ctx().toCpuPinned();
            const std::size_t max_m = 1u << 18; // = 2^18
            for (int step = 0; step < 2; ++step) {
                std::size_t k0, m0, nk;
                if (step == 0) {
                    k0 = 0;
                    nk = m / max_m;
                    m0 = max_m;
                } else {
                    k0 = m / max_m;
                    m0 = m % max_m;
                    nk = (m0 > 0u ? 1 : 0);
                }
                if (nk == 0) continue;
                vector<T *, Gpu> a_ps(k * nk, xpu_host, doCacheAlloc);
                vector<T *, Gpu> x_ps(k * nk, xpu_host, doCacheAlloc);
                auto a_ps_ptr = a_ps.data();
                auto x_ps_ptr = x_ps.data();
                auto a_ptr = a.data();
                auto x_ptr = x.data();
                launchHostKernel(
                    [=] {
                        for (std::size_t i = 0; i < k; ++i) {
                            for (std::size_t ki = k0, kii = 0; kii < nk; ++ki, ++kii) {
                                a_ps_ptr[i * nk + kii] = a_ptr + n * n * i;
                                x_ps_ptr[i * nk + kii] =
                                    x_ptr + n * m * i + (left_side ? n : 1u) * max_m * ki;
                            }
                        }
                    },
                    xpu_host);
                vector<T *, Gpu> a_ps_gpu = makeSure(a_ps, a.ctx(), doCacheAlloc),
                                 x_ps_gpu = makeSure(x_ps, a.ctx(), doCacheAlloc);
                gpuBlasCheck(cublasXtrsmBatched(
                    getGpuBlasHandle(a.ctx()), left_side ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, left_side ? n : m0,
                    left_side ? m0 : n, alpha, a_ps_gpu.data(), n, x_ps_gpu.data(),
                    left_side ? n : m, k * nk));
            }
#    else
            gpuBlasCheck(hipblasXtrsmStridedBatched(
                getGpuBlasHandle(a.ctx()), left_side ? HIPBLAS_SIDE_LEFT : HIPBLAS_SIDE_RIGHT,
                HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, left_side ? n : m,
                left_side ? m : n, alpha, a.data(), n, n * n, x.data(), left_side ? n : m, n * m,
                k));
#    endif
            causalConnectTo(a.ctx(), x.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        /// Perform a\x -> x
        /// \param trans: either 'n', 't', or 'c'
        /// \param n: size of the matrix
        /// \param k: number of matrices to invert
        /// \param m: number of columns (if left_side) or rows (if !left_side) that x and y have

        template <typename T>
        void local_gesm(char trans, std::size_t n, std::size_t k, std::size_t m, vector<T, Cpu> a,
                        vector<T, Cpu> x) {

            tracker<Cpu> _t("local gesm (Cpu)", a.ctx());
            // Cost approximated as the cost of LU plus multiplying two triangular matrices
            _t.cost = (double)n * n * n * 2 / 3 * k +
                      (double)n * n * m * k * multiplication_cost<T>::value;

            using BLASINT = std::int64_t;
            T *ap = a.data(), *xp = x.data();

#ifdef _OPENMP
            int num_threads = omp_get_max_threads();
#else
            int num_threads = 1;
#endif
            BLASINT *ipivs = new BLASINT[n * num_threads];
            std::vector<int> info(num_threads, 0);

#ifdef _OPENMP
#    pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int id = omp_get_thread_num();
#else
                int id = 0;
#endif
                BLASINT *ipiv = ipivs + n * id;
#ifdef _OPENMP
#    pragma omp for schedule(static)
#endif
                for (std::size_t i = 0; i < k; ++i) {
                    if (info[id] == 0) info[id] = xgetrf(n, n, ap + n * n * i, n, ipiv, Cpu{});
                    if (info[id] == 0)
                        info[id] =
                            xgetrs(trans, n, m, ap + n * n * i, n, ipiv, xp + n * m * i, n, Cpu{});
                }
            }
            for (int i : info) checkLapack(i);

            delete[] ipivs;
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_gesm(char trans, std::size_t n, std::size_t k, std::size_t m,
                        const vector<T, Gpu> &a, const vector<T, Gpu> &x) {

            if (n == 0 || k == 0 || m == 0) return;
            if (deviceId(a.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_gesm: unsupported allocation device");
            check_same_device(a.ctx(), x.ctx());
            causalConnectTo(x.ctx(), a.ctx());

            tracker<Gpu> _t("local gesm (GPU)", a.ctx());
            // Cost approximated as the cost of LU plus multiplying two triangular matrices
            _t.cost = (double)n * n * n * 2 / 3 * k +
                      (double)n * n * m * k * multiplication_cost<T>::value;

            vector<int, Gpu> ipivs(k * n, a.ctx(), doCacheAlloc), info(k, a.ctx(), doCacheAlloc);
            auto xpu_host = a.ctx().toCpuPinned();
#    ifdef SUPERBBLAS_USE_CUDA
            vector<T *, Gpu> a_ps(k, xpu_host, doCacheAlloc), x_ps(k, xpu_host, doCacheAlloc);
            auto a_ps_ptr = a_ps.data();
            auto x_ps_ptr = x_ps.data();
            auto a_ptr = a.data();
            auto x_ptr = x.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i) a_ps_ptr[i] = a_ptr + n * n * i;
                    for (std::size_t i = 0; i < k; ++i) x_ps_ptr[i] = x_ptr + n * m * i;
                },
                xpu_host);
            vector<T *, Gpu> a_ps_gpu = makeSure(a_ps, a.ctx(), doCacheAlloc),
                             x_ps_gpu = makeSure(x_ps, a.ctx(), doCacheAlloc);
            gpuBlasCheck(cublasXgetrfBatched(getGpuBlasHandle(a.ctx()), n, a_ps_gpu.data(), n,
                                             ipivs.data(), info.data(), k));
#    else
            gpuBlasCheck(hipblasXgetrfStridedBatched(getGpuBlasHandle(a.ctx()), n, a.data(), n,
                                                     n * n, ipivs.data(), n, info.data(), k));
#    endif
            vector<int, Gpu> info_cpu = makeSure(info, xpu_host, doCacheAlloc);
            auto info_cpu_ptr = info_cpu.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i)
                        checkLapack(info_cpu_ptr[i], true /* terminate */);
                },
                xpu_host);
            int info_getrs;
#    ifdef SUPERBBLAS_USE_CUDA
            gpuBlasCheck(cublasXgetrsBatched(getGpuBlasHandle(a.ctx()), toCublasTrans(trans), n, m,
                                             a_ps_gpu.data(), n, ipivs.data(), x_ps_gpu.data(), n,
                                             &info_getrs, k));
#    else
            gpuBlasCheck(hipblasXgetrsStridedBatched(
                getGpuBlasHandle(a.ctx()), toCublasTrans(trans), n, m, a.data(), n, n * n,
                ipivs.data(), n, x.data(), n, n * m, &info_getrs, k));
#    endif
            checkLapack(info_getrs);
            causalConnectTo(a.ctx(), x.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        /// Get the output partition
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t N>
        From_size<N> get_dense_output_partition(From_size<N> p0, const Coor<N> &dim,
                                                const Order<N> &o0, const Order<N> &o_r,
                                                unsigned int num_mat_dims, CoorOrder co) {
            // Find partition on cache
            using Key = std::tuple<From_size<N>, Coor<N>, PairPerms<N, N>, unsigned int>;
            struct cache_tag {};
            auto cache = getCache<Key, From_size<N>, TupleHash<Key>, cache_tag>(p0.ctx());
            Key key{p0, dim, get_perms(o0, o_r), num_mat_dims};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Create partition
            Coor<N> perm0 = find_permutation<N, N>(o0, o_r);
            Coor<N> dimr = reorder_coor<N, N>(dim, perm0, 1);
            From_size<N> pr(p0.size(), p0.ctx());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                if (volume(p0[i][1]) == 0) {
                    pr[i][0] = pr[i][1] = Coor<N>{};
                } else {
                    pr[i][0] = reorder_coor<N, N>(p0[i][0], perm0);
                    pr[i][1] = reorder_coor<N, N>(p0[i][1], perm0, 1);
                    if (co == FastToSlow) {
                        for (unsigned int j = 0; j < num_mat_dims; ++j) pr[i][0][j] = 0;
                        for (unsigned int j = 0; j < num_mat_dims; ++j) pr[i][1][j] = dimr[j];
                    } else {
                        for (unsigned int j = 0, j0 = N - 1; j < num_mat_dims; ++j, --j0)
                            pr[i][0][j0] = 0;
                        for (unsigned int j = 0, j0 = N - 1; j < num_mat_dims; ++j, --j0)
                            pr[i][1][j0] = dimr[j0];
                    }
                }
            }

            cache.insert(key, pr, storageSize(pr));

            return pr;
        }

        /// Return the tensor rearranged in the right ordering and distribution for doing Cholesky
        /// factorization/application
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param v0: data for the first operator
        /// \param orows: labels on the rows
        /// \param ocols: labels on the columns
        /// \param co: coordinate linearization order
        /// \param force_copy: whether to NOT avoid copy if the partition is the same
        /// \return tuple{pw,ow,vw,n}
        /// \param pw: (out) partitioning of the output tensor in consecutive ranges
        /// \param ow: (out) ordering of the output tensor
        /// \param vw: (out) data for the output operator
        /// \param n: (out) the number of rows/columns

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        std::tuple<From_size<N>, Coor<N>, Order<N>, Components_tmpl<N, T, XPU0, XPU1>, std::size_t>
        prepare_for_cholesky(const From_size<N> &p, const Coor<N> &dim, const Order<N> &o,
                             const Components_tmpl<N, T, XPU0, XPU1> &v, const char *orows,
                             const char *ocols, Comm comm, CoorOrder co, bool force_copy = false) {

            // Check the orderings

            const std::string orows_(orows), ocols_(ocols);
            for (char c : orows_) {
                if (std::find(ocols_.begin(), ocols_.end(), c) != ocols_.end())
                    throw std::runtime_error("Invalid `orows' and `ocols': they share labels");
                if (std::find(o.begin(), o.end(), c) == o.end())
                    throw std::runtime_error("Invalid `orows': invalid labels");
            }
            for (char c : ocols_)
                if (std::find(o.begin(), o.end(), c) == o.end())
                    throw std::runtime_error("Invalid `ocols': invalid labels");

            // Generate the working ordering

            std::size_t const nrows = orows_.size();
            std::size_t const ncols = ocols_.size();
            std::vector<char> ot;
            for (char c : o)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            std::size_t const nk = ot.size();
            Order<N> ow = concat<N>(orows_, ocols_, ot, co);

            // Check that number of rows and columns is the same
            Coor<N> perm0 = find_permutation<N, N>(o, ow);
            Coor<N> dimw = reorder_coor<N, N>(dim, perm0, 1);
            std::size_t n =
                (co == FastToSlow ? volume<N>(dimw.begin(), dimw.begin() + nrows)
                                  : volume<N>(dimw.begin() + nk, dimw.begin() + nk + ncols));
            std::size_t m =
                (co == FastToSlow ? volume<N>(dimw.begin() + nrows, dimw.begin() + nrows + ncols)
                                  : volume<N>(dimw.begin() + nk + ncols, dimw.end()));
            if (m != n) std::runtime_error("cholesky: the matrices to factorize should be square");

            // Generate the working partition
            From_size<N> pw = get_dense_output_partition(p, dim, o, ow, nrows + ncols, co);
            Components_tmpl<N, T, XPU0, XPU1> vw = reorder_tensor(
                p, o, {{}}, dim, dim, v, pw, dimw, ow, comm, co, force_copy, doCacheAlloc);

            return {pw, dimw, ow, vw, n};
        }

        /// Compute the Cholesky factorization of several matrices
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param ncomponents0: number of consecutive components in each MPI rank
        /// \param o0: dimension labels for the first operator
        /// \param v0: data for the first operator
        /// \param orows: labels on the rows
        /// \param ocols: labels on the columns
        /// \param ctx0: context for each data pointer in v0
        /// \param session: concurrent calls should have different session

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        void cholesky(const From_size<N> &p, const Coor<N> &dim, const Order<N> &o,
                      const Components_tmpl<N, T, XPU0, XPU1> &v, const char *orows,
                      const char *ocols, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed cholesky", p.ctx());

            // Reorder the tensor so can be processed by cholesky
            auto t = prepare_for_cholesky(p, dim, o, v, orows, ocols, comm, co);
            From_size<N> &pw = std::get<0>(t);
            const Coor<N> &dimw = std::get<1>(t);
            Order<N> &ow = std::get<2>(t);
            Components_tmpl<N, T, XPU0, XPU1> &vw = std::get<3>(t);
            std::size_t n = std::get<4>(t);

            // Do cholesky on the local pieces
            unsigned int ncomponents = vw.first.size() + vw.second.size();
            for (unsigned int i = 0; i < vw.first.size(); ++i) {
                const unsigned int componentId = vw.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pw[pi][1]) / n / n;
                local_cholesky(n, ki, vw.first[i].it);
            }
            for (unsigned int i = 0; i < vw.second.size(); ++i) {
                const unsigned int componentId = vw.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pw[pi][1]) / n / n;
                local_cholesky(n, ki, vw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<N, N, T>(T{1}, pw, {{}}, dimw, dimw, ow, toConst(vw), p, {{}}, dim, o, v, comm,
                          EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void trsm(T alpha, const From_size<Nc> &pc, const Coor<Nc> &dimc, const Order<Nc> &oc,
                  const Components_tmpl<Nc, T, XPU0, XPU1> &vc, const char *orows,
                  const char *ocols, const From_size<Nx> &px, const Coor<Nx> &dimx,
                  const Order<Nx> &ox, const Components_tmpl<Nx, T, XPU0, XPU1> &vx,
                  const From_size<Ny> &py, const Coor<Ny> &dimy, const Order<Ny> &oy,
                  const Components_tmpl<Ny, T, XPU0, XPU1> &vy, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vc.first) sync(i.it.ctx());
                for (const auto &i : vc.second) sync(i.it.ctx());
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed trsm", pc.ctx());

            // Check the compatibility of the tensors
            if (!check_dimensions(oc, dimc, ox, dimx, oy, dimy))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (!check_components_compatibility(vc, vx) || !check_components_compatibility(vx, vy))
                throw std::runtime_error(
                    "trsm: the given tensors don't have the same number of components "
                    "or they don't follow the same order on the devices");

            // Figure out whether x contracts with the rows or the columns of the cholesky factor

            const std::string orows_(orows), ocols_(ocols);
            bool contract_rows = false, contract_rows_set = false;
            bool fail = false;
            for (char c : ox) {
                if (std::find(ocols_.begin(), ocols_.end(), c) != ocols_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = false;
                        contract_rows_set = true;
                    } else if (contract_rows != false) {
                        fail = true;
                        break;
                    }
                }
                if (std::find(orows_.begin(), orows_.end(), c) != orows_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = true;
                        contract_rows_set = true;
                    } else if (contract_rows != true) {
                        fail = true;
                        break;
                    }
                }
            }
            if (fail || !contract_rows_set)
                throw std::runtime_error("trsm: cannot contract a mix of rows and column labels");

            // Check that all rows and column labels are in the x and y orderings

            for (char c : orows_) {
                if (contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (!contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            for (char c : ocols_) {
                if (!contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            if (fail) throw std::runtime_error("trsm: missing labels to contract");

            // Reorder the tensor so can be processed by cholesky
            auto t = prepare_for_cholesky(pc, dimc, oc, toNonConst(vc), orows, ocols, comm, co);
            From_size<Nc> &pcw = std::get<0>(t);
            Coor<Nc> &dimcw = std::get<1>(t);
            Order<Nc> &ocw = std::get<2>(t);
            Components_tmpl<Nc, T, XPU0, XPU1> &vcw = std::get<3>(t);
            std::size_t r = std::get<4>(t); // number of rows/columns

            // Find the labels

            std::vector<char> ot, on;
            for (char c : oc)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            for (char c : ox)
                if (std::find(oc.begin(), oc.end(), c) == oc.end()) on.push_back(c);

            // Compute the ordering for tensors x and y
            // If contracting rows, then X/C -> Y => (n,r,t) x (r,c,t) -> (n,c,t).
            // Otherwise C\X -> Y => (r,c,t) x (c,n,t) -> (r,n,t)

            Order<Nx> oxw =
                contract_rows ? concat<Nx>(on, orows_, ot, co) : concat<Nx>(ocols_, on, ot, co);
            Order<Ny> oyw =
                contract_rows ? concat<Ny>(on, ocols_, ot, co) : concat<Ny>(orows_, on, ot, co);

            // Generate the working tensors

            auto tx_ = get_output_partition(pcw, dimcw, ocw, px, dimx, ox, oxw, false);
            From_size<Nx> &pxw = tx_.first;
            const Coor<Nx> &dimxw = tx_.second;
            Components_tmpl<Nx, T, XPU0, XPU1> vxw =
                reorder_tensor(px, ox, {{}}, dimx, dimx, toNonConst(vx), pxw, dimxw, oxw, comm, co,
                               true /* Force copy */, doCacheAlloc);
            auto ty_ = get_output_partition(pcw, dimcw, ocw, pxw, dimxw, oxw, oyw);
            From_size<Ny> &pyw = ty_.first;
            const Coor<Ny> &dimyw = ty_.second;

            // Do the contraction of the local pieces

            unsigned int ncomponents = vcw.first.size() + vcw.second.size();
            for (unsigned int i = 0; i < vcw.first.size(); ++i) {
                const unsigned int componentId = vcw.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_trsm(!contract_rows, r, ki, ni, alpha, vcw.first[i].it, vxw.first[i].it);
            }
            for (unsigned int i = 0; i < vcw.second.size(); ++i) {
                const unsigned int componentId = vcw.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_trsm(!contract_rows, r, ki, ni, alpha, vcw.second[i].it, vxw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<Ny, Ny, T>(T{1}, pyw, {{}}, dimyw, dimyw, oyw, toConst(vxw), py, {{}}, dimy, oy,
                            vy, comm, EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void gesm(T alpha, const From_size<Nc> &pc, const Coor<Nc> dimc, const Order<Nc> &oc,
                  const Components_tmpl<Nc, T, XPU0, XPU1> &vc, const char *orows,
                  const char *ocols, const From_size<Nx> &px, const Coor<Nx> &dimx,
                  const Order<Nx> &ox, const Components_tmpl<Nx, T, XPU0, XPU1> &vx,
                  const From_size<Ny> &py, const Coor<Ny> &dimy, const Order<Ny> &oy,
                  const Components_tmpl<Ny, T, XPU0, XPU1> &vy, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vc.first) sync(i.it.ctx());
                for (const auto &i : vc.second) sync(i.it.ctx());
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed gesm", pc.ctx());

            // Check the compatibility of the tensors
            if (!check_dimensions(oc, dimc, ox, dimx, oy, dimy))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (!check_components_compatibility(vc, vx) || !check_components_compatibility(vx, vy))
                throw std::runtime_error(
                    "gesm: the given tensors don't have the same number of components "
                    "or they don't follow the same order on the devices");

            // Figure out whether x contracts with the rows or the columns of the matrix to invert

            const std::string orows_(orows), ocols_(ocols);
            bool contract_rows = false, contract_rows_set = false;
            bool fail = false;
            for (char c : ox) {
                if (std::find(ocols_.begin(), ocols_.end(), c) != ocols_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = false;
                        contract_rows_set = true;
                    } else if (contract_rows != false) {
                        fail = true;
                        break;
                    }
                }
                if (std::find(orows_.begin(), orows_.end(), c) != orows_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = true;
                        contract_rows_set = true;
                    } else if (contract_rows != true) {
                        fail = true;
                        break;
                    }
                }
            }
            if (fail || !contract_rows_set)
                throw std::runtime_error("gesm: cannot contract a mix of rows and column labels");

            // For now, only supported to contract with columns

            if (contract_rows)
                throw std::runtime_error("gesm: unsupported to contract with row labels");

            // Check that all rows and column labels are in the x and y orderings

            for (char c : orows_) {
                if (contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (!contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            for (char c : ocols_) {
                if (!contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            if (fail) throw std::runtime_error("gesm: missing labels to contract");

            // Reorder the tensor so can be processed by cholesky
            auto t = prepare_for_cholesky(pc, dimc, oc, toNonConst(vc), orows, ocols, comm, co,
                                          true /* Force copy */);
            From_size<Nc> &pcw = std::get<0>(t);
            Coor<Nc> &dimcw = std::get<1>(t);
            Order<Nc> &ocw = std::get<2>(t);
            Components_tmpl<Nc, T, XPU0, XPU1> &vcw = std::get<3>(t);
            std::size_t r = std::get<4>(t); // number of rows/columns

            // Find the labels

            std::vector<char> ot, on;
            for (char c : oc)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            for (char c : ox)
                if (std::find(oc.begin(), oc.end(), c) == oc.end()) on.push_back(c);

            // Compute the ordering for tensors x and y
            // If contracting rows, then X/C -> Y => (n,r,t) x (r,c,t) -> (n,c,t).
            // Otherwise C\X -> Y => (r,c,t) x (c,n,t) -> (r,n,t)

            Order<Nx> oxw =
                contract_rows ? concat<Nx>(on, orows_, ot, co) : concat<Nx>(ocols_, on, ot, co);
            Order<Ny> oyw =
                contract_rows ? concat<Ny>(on, ocols_, ot, co) : concat<Ny>(orows_, on, ot, co);

            // Generate the working tensors

            auto tx_ = get_output_partition(pcw, dimcw, ocw, px, dimx, ox, oxw, false);
            From_size<Nx> &pxw = tx_.first;
            const Coor<Nx> &dimxw = tx_.second;
            Components_tmpl<Nx, T, XPU0, XPU1> vxw =
                reorder_tensor(px, ox, {{}}, dimx, dimx, toNonConst(vx), pxw, dimxw, oxw, comm, co,
                               true /* Force copy */, doCacheAlloc);
            auto ty_ = get_output_partition(pcw, dimcw, ocw, pxw, dimxw, oxw, oyw);
            From_size<Ny> &pyw = ty_.first;
            const Coor<Ny> &dimyw = ty_.second;

            // Do the contraction of the local pieces

            unsigned int ncomponents = vcw.first.size() + vcw.second.size();
            for (unsigned int i = 0; i < vcw.first.size(); ++i) {
                const unsigned int componentId = vcw.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_gesm('N', r, ki, ni, vcw.first[i].it, vxw.first[i].it);
            }
            for (unsigned int i = 0; i < vcw.second.size(); ++i) {
                const unsigned int componentId = vcw.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_gesm('N', r, ki, ni, vcw.second[i].it, vxw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<Ny, Ny, T>(alpha, pyw, {{}}, dimyw, dimyw, oyw, toConst(vxw), py, {{}}, dimy, oy,
                            vy, comm, EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                barrier(comm);
            }
        }
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Compute the Cholesky factorization of several matrices, returning the upper triangular matrix
    /// \param p: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param o: dimension labels for the first operator
    /// \param v: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctx: context for each data pointer in v0
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t N, typename T>
    void cholesky(const PartitionItem<N> *p, const Coor<N> &dim, int ncomponents, const char *o,
                  T **v, const char *orows, const char *ocols, const Context *ctx, MPI_Comm mpicomm,
                  CoorOrder co, Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::cholesky<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, session), dim, o_,
            detail::get_components<N>(v, nullptr, ctx, ncomponents, p, comm, session), orows, ocols,
            comm, co);
    }

    /// Solve several upper triangular linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void trsm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::trsm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), dimx, ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }

    /// Solve several linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void gesm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::gesm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), dimx, ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }
#endif // SUPERBBLAS_USE_MPI

    /// Compute the Cholesky factorization of several matrices
    /// \param p: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param o: dimension labels for the first operator
    /// \param v: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctx: context for each data pointer in v0
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t N, typename T>
    void cholesky(const PartitionItem<N> *p, const Coor<N> &dim, int ncomponents, const char *o,
                  T **v, const char *orows, const char *ocols, const Context *ctx, CoorOrder co,
                  Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::SelfComm comm = detail::get_comm();

        detail::cholesky<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, session), dim, o_,
            detail::get_components<N>(v, nullptr, ctx, ncomponents, p, comm, session), orows, ocols,
            comm, co);
    }

    /// Solve several upper triangular linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void trsm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::trsm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), dimx, ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }

    /// Solve several linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void gesm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::gesm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), dimx, ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }
}

#endif // __SUPERBBLAS_DENSE__
