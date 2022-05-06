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

        inline void checkLapack(int info) {
            if (info == 0) return;
            if (info < 0)
                throw std::runtime_error(
                    std::string("Error in a lapack routine: wrong argument at position ") +
                    std::to_string(-info));
            if (info > 0)
                throw std::runtime_error(std::string("Error in lapack routine: ") +
                                         std::to_string(info));
        }

        template <typename T> void local_cholesky(std::size_t n, std::size_t k, vector<T, Cpu> v) {

            tracker<Cpu> _t("local cholesky (Cpu)", v.ctx());

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
                int id = omp_get_thread_num();

#ifdef _OPENMP
#    pragma omp for
#endif
                for (std::size_t i = 0; i < k; ++i)
                    if (info[id] == 0) info[id] = xpotrf('U', n, p + n * n * i, n, Cpu{});
            }

            for (int i : info) checkLapack(i);
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T> void local_cholesky(std::size_t n, std::size_t k, vector<T, Gpu> v) {

            tracker<Gpu> _t("local cholesky (GPU)", v.ctx());

            // TODO: use cudaSolverDN<t>potrfBatched

            vector<T, Cpu> v_cpu = makeSure(v, Cpu{0});
            local_cholesky(n, k, v_cpu);
            copy_n(v_cpu.data(), v_cpu.ctx(), v_cpu.size(), v.data(), v.ctx());
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

            tracker<Cpu> _t("local trsm (Cpu)", a.ctx());

            const T *ap = a.data();
            T *xp = x.data();

#ifdef _OPENMP
#    pragma omp for
#endif
            for (std::size_t i = 0; i < k; ++i)
                xtrsm(left_side ? 'L' : 'R', 'U', 'N', 'N', left_side ? n : m, left_side ? m : n,
                      alpha, ap + n * n * i, n, xp + n * m * i, left_side ? n : m, a.ctx());
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_trsm(bool left_side, std::size_t n, std::size_t k, std::size_t m, T alpha,
                        vector<T, Gpu> a, vector<T, Gpu> x) {

            tracker<Gpu> _t("local trsm (GPU)", a.ctx());

            // TODO: use cudaSolverDN<t>trsmBatched

            vector<T, Cpu> a_cpu = makeSure(a, Cpu{0}), x_cpu = makeSure(x, Cpu{0});
            local_trsm(left_side, n, k, m, alpha, a_cpu, x_cpu);
            copy_n(x_cpu.data(), x_cpu.ctx(), x_cpu.size(), x.data(), x.ctx());
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
                int id = omp_get_thread_num();
                BLASINT *ipiv = ipivs + n * id;
#ifdef _OPENMP
#    pragma omp for
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
        void local_gesm(char trans, std::size_t n, std::size_t k, std::size_t m, vector<T, Gpu> a,
                        vector<T, Gpu> x) {

            tracker<Gpu> _t("local gesm (GPU)", a.ctx());

            // TODO: use cudaSolverDN<t>trsmBatched

            vector<T, Cpu> a_cpu = makeSure(a, Cpu{0}), x_cpu = makeSure(x, Cpu{0});
            local_gesm(trans, n, k, m, a_cpu, x_cpu);
            copy_n(x_cpu.data(), x_cpu.ctx(), x_cpu.size(), x.data(), x.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        /// Get the output partition
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t N>
        From_size<N> get_dense_output_partition(From_size<N> p0, const Order<N> &o0,
                                                const Order<N> &o_r, unsigned int num_mat_dims,
                                                CoorOrder co) {
            // Find partition on cache
            using Key = std::tuple<From_size<N>, PairPerms<N, N>>;
            struct cache_tag {};
            auto cache = getCache<Key, From_size<N>, TupleHash<Key>, cache_tag>(p0.ctx());
            Key key{p0, get_perms(o0, o_r)};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Create partition
            Coor<N> dim = get_dim<N>(p0);
            Coor<N> perm0 = find_permutation<N, N>(o0, o_r);
            Coor<N> dimr = reorder_coor<N, N>(dim, perm0, 1);
            From_size_out<N> pr(p0.size(), p0.ctx());
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
        std::tuple<From_size<N>, Order<N>, Components_tmpl<N, T, XPU0, XPU1>, std::size_t>
        prepare_for_cholesky(const From_size<N> &p, const Order<N> &o,
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
            Coor<N> dim = get_dim<N>(p);
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
            From_size<N> pw = get_dense_output_partition(p, o, ow, nrows + ncols, co);
            Components_tmpl<N, T, XPU0, XPU1> vw =
                reorder_tensor(p, o, v, pw, ow, comm, co, force_copy);

            return {pw, ow, vw, n};
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
        void cholesky(const From_size<N> &p, const Order<N> &o,
                      const Components_tmpl<N, T, XPU0, XPU1> &v, const char *orows,
                      const char *ocols, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed cholesky", p.ctx());

            // Reorder the tensor so can be processed by cholesky
            auto t = prepare_for_cholesky(p, o, v, orows, ocols, comm, co);
            From_size<N> &pw = std::get<0>(t);
            Order<N> &ow = std::get<1>(t);
            Components_tmpl<N, T, XPU0, XPU1> &vw = std::get<2>(t);
            std::size_t n = std::get<3>(t);

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
            Coor<N> dimw = get_dim<N>(pw);
            copy<N, N, T>(T{1}, pw, {}, dimw, ow, toConst(vw), p, {}, o, v, comm, EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void trsm(T alpha, const From_size<Nc> &pc, const Order<Nc> &oc,
                  const Components_tmpl<Nc, T, XPU0, XPU1> &vc, const char *orows,
                  const char *ocols, const From_size<Nx> &px, const Order<Nx> &ox,
                  const Components_tmpl<Nx, T, XPU0, XPU1> &vx, const From_size<Ny> &py,
                  const Order<Ny> &oy, const Components_tmpl<Ny, T, XPU0, XPU1> &vy, Comm comm,
                  CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vc.first) sync(i.it.ctx());
                for (const auto &i : vc.second) sync(i.it.ctx());
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed trsm", pc.ctx());

            // Check the compatibility of the tensors
            Coor<Nc> dimc = get_dim(pc);
            Coor<Nx> dimx = get_dim(px);
            Coor<Ny> dimy = get_dim(py);
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
            auto t = prepare_for_cholesky(pc, oc, toNonConst(vc), orows, ocols, comm, co);
            From_size<Nc> &pcw = std::get<0>(t);
            Order<Nc> &ocw = std::get<1>(t);
            Components_tmpl<Nc, T, XPU0, XPU1> &vcw = std::get<2>(t);
            std::size_t r = std::get<3>(t); // number of rows/columns

            // Find the labels

            std::vector<char> ot, on;
            for (char c : oc)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            for (char c : ox)
                if (std::find(oc.begin(), oc.end(), c) == oc.end()) on.push_back(c);

            // Compute the ordering for tensors x and y
            // If contracting rows, then C\X -> Y => (c,r,t) x (r,n,t) -> (c,n,t).
            // Otherwise X/C -> Y => (n,c,t) x (c,r,t) -> (n,r,t)

            Order<Nx> oxw =
                contract_rows ? concat<Nx>(orows_, on, ot, co) : concat<Nx>(on, ocols_, ot, co);
            Order<Ny> oyw =
                contract_rows ? concat<Ny>(ocols_, on, ot, co) : concat<Ny>(on, orows_, ot, co);

            // Generate the working tensors

            From_size<Nx> pxw = get_output_partition(pcw, ocw, px, ox, oxw, false);
            Components_tmpl<Nx, T, XPU0, XPU1> vxw =
                reorder_tensor(px, ox, toNonConst(vx), pxw, oxw, comm, co, true /* Force copy */);
            std::size_t x_num_mat_dims = contract_rows ? orows_.size() : ocols_.size();
            std::size_t y_num_mat_dims = contract_rows ? ocols_.size() : orows_.size();
            Coor<Ny> permy2yw = find_permutation(oy, oyw);
            Coor<Ny> dimyw = reorder_coor(dimy, permy2yw);
            From_size_out<Ny> pyw(pxw.size(), Cpu{});
            From_size_item<Ny> fsy{Coor<Ny>{{}}, dimyw};
            for (std::size_t i = 0; i < pxw.size(); ++i) pyw[i] = fsy;
            if (co == FastToSlow) {
                for (std::size_t i = 0; i < pxw.size(); ++i)
                    for (std::size_t j = 0; j < 2; ++j)
                        std::copy_n(pxw[i][j].begin() + x_num_mat_dims, Nx - x_num_mat_dims,
                                    pyw[i][j].begin() + y_num_mat_dims);
            } else {
                for (std::size_t i = 0; i < pxw.size(); ++i)
                    for (std::size_t j = 0; j < 2; ++j)
                        std::copy_n(pxw[i][j].begin(), Nx - x_num_mat_dims, pyw[i][j].begin());
            }

            // Do the contraction of the local pieces

            unsigned int ncomponents = vcw.first.size() + vcw.second.size();
            for (unsigned int i = 0; i < vcw.first.size(); ++i) {
                const unsigned int componentId = vcw.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_trsm(contract_rows, r, ki, ni, alpha, vcw.first[i].it, vxw.first[i].it);
            }
            for (unsigned int i = 0; i < vcw.second.size(); ++i) {
                const unsigned int componentId = vcw.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_trsm(contract_rows, r, ki, ni, alpha, vcw.second[i].it, vxw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<Ny, Ny, T>(T{1}, pyw, {}, dimyw, oyw, toConst(vxw), py, {}, oy, vy, comm,
                            EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void gesm(T alpha, const From_size<Nc> &pc, const Order<Nc> &oc,
                  const Components_tmpl<Nc, T, XPU0, XPU1> &vc, const char *orows,
                  const char *ocols, const From_size<Nx> &px, const Order<Nx> &ox,
                  const Components_tmpl<Nx, T, XPU0, XPU1> &vx, const From_size<Ny> &py,
                  const Order<Ny> &oy, const Components_tmpl<Ny, T, XPU0, XPU1> &vy, Comm comm,
                  CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vc.first) sync(i.it.ctx());
                for (const auto &i : vc.second) sync(i.it.ctx());
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed gesm", pc.ctx());

            // Check the compatibility of the tensors
            Coor<Nc> dimc = get_dim(pc);
            Coor<Nx> dimx = get_dim(px);
            Coor<Ny> dimy = get_dim(py);
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
            auto t = prepare_for_cholesky(pc, oc, toNonConst(vc), orows, ocols, comm, co,
                                          true /* Force copy */);
            From_size<Nc> &pcw = std::get<0>(t);
            Order<Nc> &ocw = std::get<1>(t);
            Components_tmpl<Nc, T, XPU0, XPU1> &vcw = std::get<2>(t);
            std::size_t r = std::get<3>(t); // number of rows/columns

            // Find the labels

            std::vector<char> ot, on;
            for (char c : oc)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            for (char c : ox)
                if (std::find(oc.begin(), oc.end(), c) == oc.end()) on.push_back(c);

            // Compute the ordering for tensors x and y
            // If contracting rows, then C\X -> Y => (c,r,t) x (r,n,t) -> (c,n,t).
            // Otherwise X/C -> Y => (n,c,t) x (c,r,t) -> (n,r,t)

            Order<Nx> oxw = concat<Nx>(on, ocols_, ot, co);
            Order<Ny> oyw = concat<Ny>(on, orows_, ot, co);

            // Generate the working tensors

            From_size<Nx> pxw = get_output_partition(pcw, ocw, px, ox, oxw, false);
            Components_tmpl<Nx, T, XPU0, XPU1> vxw =
                reorder_tensor(px, ox, toNonConst(vx), pxw, oxw, comm, co, true /* Force copy */);
            std::size_t x_num_mat_dims = contract_rows ? orows_.size() : ocols_.size();
            std::size_t y_num_mat_dims = contract_rows ? ocols_.size() : orows_.size();
            Coor<Ny> permy2yw = find_permutation(oy, oyw);
            Coor<Ny> dimyw = reorder_coor(dimy, permy2yw);
            From_size_out<Ny> pyw(pxw.size(), Cpu{});
            From_size_item<Ny> fsy{Coor<Ny>{{}}, dimyw};
            for (std::size_t i = 0; i < pxw.size(); ++i) pyw[i] = fsy;
            if (co == FastToSlow) {
                for (std::size_t i = 0; i < pxw.size(); ++i)
                    for (std::size_t j = 0; j < 2; ++j)
                        std::copy_n(pxw[i][j].begin() + x_num_mat_dims, Nx - x_num_mat_dims,
                                    pyw[i][j].begin() + y_num_mat_dims);
            } else {
                for (std::size_t i = 0; i < pxw.size(); ++i)
                    for (std::size_t j = 0; j < 2; ++j)
                        std::copy_n(pxw[i][j].begin(), Nx - x_num_mat_dims, pyw[i][j].begin());
            }

            // Do the contraction of the local pieces

            unsigned int ncomponents = vcw.first.size() + vcw.second.size();
            for (unsigned int i = 0; i < vcw.first.size(); ++i) {
                const unsigned int componentId = vcw.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_gesm('N', r, ki, ni, vcw.first[i].it, vxw.first[i].it);
            }
            for (unsigned int i = 0; i < vcw.second.size(); ++i) {
                const unsigned int componentId = vcw.second[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                std::size_t ki = volume(pcw[pi][1]) / r / r;
                std::size_t ni = volume(pxw[pi][1]) / r / ki; // rows/columns of x and y
                local_gesm('N', r, ki, ni, vcw.second[i].it, vxw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<Ny, Ny, T>(alpha, pyw, {}, dimyw, oyw, toConst(vxw), py, {}, oy, vy, comm,
                            EWOp::Copy{}, co);

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
    void cholesky(const PartitionItem<N> *p, int ncomponents, const char *o, T **v,
                  const char *orows, const char *ocols, const Context *ctx, MPI_Comm mpicomm,
                  CoorOrder co, Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::cholesky<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, session), o_,
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
    void trsm(T alpha, const PartitionItem<Nc> *pc, int ncomponentsc, const char *oc, const T **vc,
              const char *orows, const char *ocols, const Context *ctxc,
              const PartitionItem<Nx> *px, int ncomponentsx, const char *ox, const T **vx,
              const Context *ctxx, const PartitionItem<Ny> *py, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::trsm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), oy_,
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
    void gesm(T alpha, const PartitionItem<Nc> *pc, int ncomponentsc, const char *oc, const T **vc,
              const char *orows, const char *ocols, const Context *ctxc,
              const PartitionItem<Nx> *px, int ncomponentsx, const char *ox, const T **vx,
              const Context *ctxx, const PartitionItem<Ny> *py, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::gesm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), oy_,
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
    void cholesky(const PartitionItem<N> *p, int ncomponents, const char *o, T **v,
                  const char *orows, const char *ocols, const Context *ctx, CoorOrder co,
                  Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::SelfComm comm = detail::get_comm();

        detail::cholesky<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, session), o_,
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
    void trsm(T alpha, const PartitionItem<Nc> *pc, int ncomponentsc, const char *oc, const T **vc,
              const char *orows, const char *ocols, const Context *ctxc,
              const PartitionItem<Nx> *px, int ncomponentsx, const char *ox, const T **vx,
              const Context *ctxx, const PartitionItem<Ny> *py, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::trsm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), oy_,
            detail::get_components<Ny>(vy, ctxy, nullptr, ncomponentsx, py, comm, session), comm,
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
    void gesm(T alpha, const PartitionItem<Nc> *pc, int ncomponentsc, const char *oc, const T **vc,
              const char *orows, const char *ocols, const Context *ctxc,
              const PartitionItem<Nx> *px, int ncomponentsx, const char *ox, const T **vx,
              const Context *ctxx, const PartitionItem<Ny> *py, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::gesm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, session), oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, session), ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, session), oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }
}

#endif // __SUPERBBLAS_DENSE__