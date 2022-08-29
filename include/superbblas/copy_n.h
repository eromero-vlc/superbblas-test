#ifndef __SUPERBBLAS_COPY_N__
#define __SUPERBBLAS_COPY_N__

#include "blas.h"

#ifdef SUPERBBLAS_CREATING_LIB
/// Generate template instantiations for copy_n functions with template parameters IndexType, T and Q

#    define DECL_COPY_T_Q_EWOP(...)                                                                \
        EMIT REPLACE1(copy_n, superbblas::detail::copy_n<IndexType, T, Q, XPU_GPU, EWOP>)          \
            REPLACE_IndexType REPLACE_T_Q REPLACE_XPU REPLACE_EWOP template __VA_ARGS__;

/// Generate template instantiations for copy_n functions with template parameters IndexType, T and Q

#    define DECL_COPY_BLOCKING_T_Q_EWOP(...)                                                       \
        EMIT REPLACE1(copy_n_blocking, superbblas::detail::copy_n_blocking<IndexType, T, Q, EWOP>) \
            REPLACE_IndexType REPLACE_T_Q REPLACE_EWOP template __VA_ARGS__;

/// Generate template instantiations for zero_n functions with template parameters IndexType and T

#    define DECL_ZERO_T(...)                                                                       \
        EMIT REPLACE1(zero_n, superbblas::detail::zero_n<IndexType, T, XPU_GPU>)                   \
            REPLACE_IndexType REPLACE_T REPLACE_XPU template __VA_ARGS__;

#else
#    define DECL_COPY_T_Q_EWOP(...) __VA_ARGS__
#    define DECL_COPY_BLOCKING_T_Q_EWOP(...) __VA_ARGS__
#    define DECL_ZERO_T(...) __VA_ARGS__
#endif

namespace superbblas {

    namespace detail {

        ///
        /// Non-blocking copy on CPU
        ///

        /// Copy n values, w[i] = v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w, Cpu,
                    EWOp::Copy) {
            assert((n == 0 || (void *)v != (void *)w || std::is_same<T, Q>::value));
            if (alpha == typename elem<T>::type{1}) {
                if (std::is_same<T, Q>::value && (void *)v == (void *)w) return;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] = v[i];
            } else if (std::norm(alpha) == 0) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] = T{0};
            } else {
                if (std::is_same<T, Q>::value && (void *)v == (void *)w) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] *= alpha;
                } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] = alpha * v[i];
                }
            }
        }

        /// Copy n values, w[i] += v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w, Cpu,
                    EWOp::Add) {
            assert((n == 0 || (void *)v != (void *)w || std::is_same<T, Q>::value));
            if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] += v[i];
            } else if (std::norm(alpha) == 0) {
                // Do nothing
            } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < n; ++i) w[i] += alpha * v[i];
            }
        }

        /// Copy n values, w[i] = v[indices[i]]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indices, Cpu,
                    std::size_t n, Q *w, Cpu, EWOp::Copy) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Copy{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] = v[indices[i]];
                } else if (std::norm(alpha) == 0) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] = T{0};
                } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] = alpha * v[indices[i]];
                }
            }
        }

        /// Copy n values, w[i] += v[indices[i]]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indices, Cpu,
                    std::size_t n, Q *w, Cpu, EWOp::Add) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Add{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] += v[indices[i]];
                } else if (alpha == typename elem<T>::type{1}) {
                    // Do nothing
                } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[i] += alpha * v[indices[i]];
                }
            }
        }

        /// Copy n values, w[indices[i]] = v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w,
                    const IndexType *indices, Cpu, EWOp::Copy) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Copy{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] = v[i];
                } else if (std::norm(alpha) == 0) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] = T{0};
                } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] = alpha * v[i];
                }
            }
        }

        /// Copy n values, w[indices[i]] += v[i]

        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, Cpu, std::size_t n, Q *w,
                    const IndexType *indices, Cpu, EWOp::Add) {
            if (indices == nullptr) {
                copy_n<IndexType>(alpha, v, Cpu{}, n, w, Cpu{}, EWOp::Add{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] += v[i];
                } else if (std::norm(alpha) == 0) {
                    // Do nothing
                } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indices[i]] += alpha * v[i];
                }
            }
        }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indicesv, Cpu,
                    std::size_t n, Q *w, const IndexType *indicesw, Cpu, EWOp::Copy) {
            if (indicesv == nullptr) {
                copy_n(alpha, v, Cpu{}, n, w, indicesw, Cpu{}, EWOp::Copy{});
            } else if (indicesw == nullptr) {
                copy_n(alpha, v, indicesv, Cpu{}, n, w, Cpu{}, EWOp::Copy{});
            } else {
                //assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = v[indicesv[i]];
                } else if (std::norm(alpha) == 0) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = T{0};
                } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] = alpha * v[indicesv[i]];
                }
            }
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indicesv, Cpu,
                    std::size_t n, Q *w, const IndexType *indicesw, Cpu, EWOp::Add) {
            if (indicesv == nullptr) {
                copy_n(alpha, v, Cpu{}, n, w, indicesw, Cpu{}, EWOp::Add{});
            } else if (indicesw == nullptr) {
                copy_n(alpha, v, indicesv, Cpu{}, n, w, Cpu{}, EWOp::Add{});
            } else {
                assert(n == 0 || (void *)v != (void *)w);
                if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] += v[indicesv[i]];
                } else if (std::norm(alpha) == 0) {
                    // Do nothing
                } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < n; ++i) w[indicesw[i]] += alpha * v[indicesv[i]];
                }
            }
        }

        ///
        /// Non-blocking copy on GPU
        ///

#ifdef SUPERBBLAS_USE_THRUST
        /// Addition of two values with different types
        template <typename T, typename Q> struct plus {
            typedef T first_argument_type;

            typedef Q second_argument_type;

            typedef Q result_type;

            __host__ __device__ result_type operator()(const T &lhs, const Q &rhs) const {
                return lhs + rhs;
            }
        };

        // Scala of a number
        template <typename T>
        struct scale : public thrust::unary_function<typename cuda_complex<T>::type,
                                                     typename cuda_complex<T>::type> {
            using cuda_T = typename cuda_complex<T>::type;
            using scalar_type = typename elem<cuda_T>::type;
            const scalar_type a;
            scale(scalar_type a) : a(a) {}
            __host__ __device__ cuda_T operator()(const cuda_T &i) const { return a * i; }
        };

        template <typename T, typename Q, typename IteratorV, typename IteratorW>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, const IteratorW &itw,
                                    EWOp::Copy) {
            thrust::copy_n(itv, n, itw);
        }

        template <typename T, typename Q, typename IteratorV, typename IteratorW>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, const IteratorW &itw,
                                    EWOp::Add) {
            thrust::transform(
                itv, itv + n, itw, itw,
                plus<typename cuda_complex<T>::type, typename cuda_complex<Q>::type>());
        }

        template <typename IndexType, typename T, typename Q, typename IteratorV, typename EWOP>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, Q *w,
                                    const IndexType *indicesw, EWOP) {
            if (indicesw == nullptr) {
                copy_n_same_dev_thrust<T, Q>(itv, n, encapsulate_pointer(w), EWOP{});
            } else {
                auto itw = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                             encapsulate_pointer(indicesw));
                copy_n_same_dev_thrust<T, Q>(itv, n, itw, EWOP{});
            }
        }

        template <typename IndexType, typename T, typename Q, typename XPU, typename EWOP>
        void copy_n_same_dev_thrust(typename elem<T>::type alpha, const T *v,
                                    const IndexType *indicesv, std::size_t n, Q *w,
                                    const IndexType *indicesw, XPU xpu, EWOP) {
            setDevice(xpu);
            if (indicesv == nullptr) {
                auto itv = encapsulate_pointer(v);
                if (alpha == typename elem<T>::type{1}) {
                    copy_n_same_dev_thrust<IndexType, T, Q>(itv, n, w, indicesw, EWOP{});
                } else {
                    copy_n_same_dev_thrust<IndexType, T, Q>(
                        thrust::make_transform_iterator(itv, scale<T>(alpha)), n, w, indicesw,
                        EWOP{});
                }
            } else {
                auto itv = thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                             encapsulate_pointer(indicesv));
                if (alpha == typename elem<T>::type{1}) {
                    copy_n_same_dev_thrust<IndexType, T, Q>(itv, n, w, indicesw, EWOP{});
                } else {
                    copy_n_same_dev_thrust<IndexType, T, Q>(
                        thrust::make_transform_iterator(itv, scale<T>(alpha)), n, w, indicesw,
                        EWOP{});
                }
            }
        }

        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param indices: indices of the elements to set
        /// \param n: number of elements to set
        /// \param xpu: device context

        template <typename IndexType, typename T, typename XPU>
        void zero_n_thrust(T *v, const IndexType *indices, std::size_t n, XPU xpu) {
            if (indices == nullptr) {
                zero_n(v, n, xpu);
            } else {
                setDevice(xpu);
                auto itv = thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                             encapsulate_pointer(indices));
                thrust::fill_n(itv, n, T{0});
            }
        }

#endif // SUPERBBLAS_USE_THRUST

#ifdef SUPERBBLAS_USE_GPU

        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param indices: indices of the elements to set
        /// \param n: number of elements to set
        /// \param xpu: device context

        template <typename IndexType, typename T, typename XPU>
        DECL_ZERO_T(void zero_n(T *v, const IndexType *indices, std::size_t n, XPU xpu))
        IMPL({ zero_n_thrust<IndexType, T, XPU>(v, indices, n, xpu); })

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] when v and w are on device

        template <typename IndexType, typename T, typename Q, typename XPU, typename EWOP>
        DECL_COPY_T_Q_EWOP(void copy_n(typename elem<T>::type alpha, const T *v,
                                       const IndexType *indicesv, XPU xpuv, std::size_t n, Q *w,
                                       const IndexType *indicesw, XPU xpuw, EWOP))
        IMPL({
            assert((n == 0 || (void *)v != (void *)w || std::is_same<T, Q>::value));
            if (n == 0) return;

            // Treat zero case
            if (std::norm(alpha) == 0) {
                if (std::is_same<EWOP, EWOp::Copy>::value)
                    zero_n<IndexType, Q, XPU>(w, indicesw, n, xpuw);
            }

            // Actions when the v and w are on the same device
            else if (deviceId(xpuv) == deviceId(xpuw)) {
                if (indicesv == nullptr && indicesw == nullptr &&
                    alpha == typename elem<T>::type{1} && std::is_same<T, Q>::value &&
                    std::is_same<EWOP, EWOp::Copy>::value) {
                    copy_n(v, xpuv, n, (T *)w, xpuw);
                } else {
                    copy_n_same_dev_thrust(alpha, v, indicesv, n, w, indicesw, xpuw, EWOP{});
                }
            }

            // Simple case when the v and w are NOT on the same device and no permutation is involved
            else if (indicesv == nullptr && indicesw == nullptr &&
                     alpha == typename elem<T>::type{1} && std::is_same<T, Q>::value &&
                     std::is_same<EWOP, EWOp::Copy>::value && deviceId(xpuv) != deviceId(xpuw)) {
                copy_n(v, xpuv, n, (T *)w, xpuw);
            }

            // If v is permuted, copy v[indices[i]] in a contiguous chunk, and then copy
            else if (indicesv != nullptr) {
                vector<Q, XPU> v0(n, xpuv);
                copy_n<IndexType>(alpha, v, indicesv, xpuv, n, v0.data(), nullptr, xpuv,
                                  EWOp::Copy{});
                copy_n<IndexType>(Q{1}, v0.data(), nullptr, xpuv, n, w, indicesw, xpuw, EWOP{});
            }

            // Otherwise copy v to xpuw, and then copy it to the w[indices[i]]
            else {
                vector<T, XPU> v1(n, xpuw);
                copy_n<IndexType>(T{1}, v, indicesv, xpuv, n, v1.data(), nullptr, xpuw,
                                  EWOp::Copy{});
                copy_n<IndexType>(T{alpha}, v1.data(), nullptr, xpuw, n, w, indicesw, xpuw, EWOP{});
            }
        })

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] from device to host and vice versa

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPU0, XPU1>::value, bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indicesv, XPU0 xpu0,
                    std::size_t n, Q *w, const IndexType *indicesw, XPU1 xpu1, EWOP) {
            if (n == 0) return;

            // Treat zero case
            if (std::norm(alpha) == 0) {
                if (std::is_same<EWOP, EWOp::Copy>::value) zero_n(w, indicesw, n, xpu1);
            }

            // Base case
            else if (std::is_same<T, Q>::value && std::is_same<EWOP, EWOp::Copy>::value &&
                     indicesv == nullptr && indicesw == nullptr) {
                copy_n(v, xpu0, n, (T *)w, xpu1);
                // Scale by alpha
                copy_n<IndexType>((Q)alpha, w, nullptr, xpu1, n, w, nullptr, xpu1, EWOp::Copy{});
            }

            // If v is permuted, copy v[indices[i]] in a contiguous chunk, and then copy
            else if (indicesv != nullptr) {
                vector<Q, XPU0> v0(n, xpu0);
                copy_n<IndexType>(alpha, v, indicesv, xpu0, n, v0.data(), nullptr, xpu0,
                                  EWOp::Copy{});
                copy_n<IndexType>(Q{1}, v0.data(), nullptr, xpu0, n, w, indicesw, xpu1, EWOP{});
            }

            // Otherwise copy v to xpu1, and then copy it to the w[indices[i]]
            else {
                vector<T, XPU1> v1(n, xpu1);
                copy_n<IndexType>(T{1}, v, indicesv, xpu0, n, v1.data(), nullptr, xpu1,
                                  EWOp::Copy{});
                copy_n<IndexType>(T{alpha}, v1.data(), nullptr, xpu1, n, w, indicesw, xpu1, EWOP{});
            }
        }

        /// Copy n values, w[i] (+)= v[i] when v or w is on device

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPU0, Cpu>::value |
                                              !std::is_same<XPU1, Cpu>::value,
                                          bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, XPU0 xpu0, std::size_t n, Q *w,
                    XPU1 xpu1, EWOP) {
            copy_n<IndexType>(alpha, v, nullptr, xpu0, n, w, nullptr, xpu1, EWOP{});
        }

        /// Copy n values, w[i] (+)= v[indices[i]] when v or w is on device

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPU0, Cpu>::value |
                                              !std::is_same<XPU1, Cpu>::value,
                                          bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, const IndexType *indices, XPU0 xpu0,
                    std::size_t n, Q *w, XPU1 xpu1, EWOP) {
            copy_n<IndexType>(alpha, v, indices, xpu0, n, w, nullptr, xpu1, EWOP{});
        }

        /// Copy n values, w[indices[i]] (+)= v[i] when v or w is on device

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPU0, Cpu>::value |
                                              !std::is_same<XPU1, Cpu>::value,
                                          bool>::type = true>
        void copy_n(typename elem<T>::type alpha, const T *v, XPU0 xpu0, std::size_t n, Q *w,
                    const IndexType *indices, XPU1 xpu1, EWOP) {
            copy_n<IndexType>(alpha, v, nullptr, xpu0, n, w, indices, xpu1, EWOP{});
        }

#endif // SUPERBBLAS_USE_GPU

        ///
        /// Blocking copy on CPU
        ///

#define COPY_N_BLOCKING_FOR(S)                                                                     \
    for (std::size_t i = 0; i < n; ++i) {                                                          \
        for (std::size_t j = 0; j < blocking; ++j) {                                               \
            std::size_t vj = indicesv[i] + j, wj = indicesw[i] + j;                                \
            S;                                                                                     \
        }                                                                                          \
    }

#define COPY_N_BLOCKING_FOR_NEW(S)                                                                 \
    for (std::size_t i = 0; i < n * blocking; ++i) {                                               \
        std::size_t vj = indicesv ? indicesv[i / blocking] + i % blocking : i,                     \
                    wj = indicesw ? indicesw[i / blocking] + i % blocking : i;                     \
        S;                                                                                         \
    }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n_blocking(typename elem<T>::type alpha, const T *SB_RESTRICT v,
                             std::size_t blocking, const IndexType *SB_RESTRICT indicesv, Cpu,
                             std::size_t n, Q *SB_RESTRICT w, const IndexType *SB_RESTRICT indicesw,
                             Cpu, EWOp::Copy) {

            if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                COPY_N_BLOCKING_FOR(w[wj] = v[vj]);
            } else if (std::norm(alpha) == 0) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                COPY_N_BLOCKING_FOR({
                    w[wj] = T{0};
                    (void)vj;
                });
            } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                COPY_N_BLOCKING_FOR(w[wj] = alpha * v[vj]);
            }
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n_blocking(typename elem<T>::type alpha, const T *SB_RESTRICT v,
                             std::size_t blocking, const IndexType *SB_RESTRICT indicesv, Cpu,
                             std::size_t n, Q *SB_RESTRICT w, const IndexType *SB_RESTRICT indicesw,
                             Cpu, EWOp::Add) {
            if (alpha == typename elem<T>::type{1}) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                COPY_N_BLOCKING_FOR(w[wj] += v[vj]);
            } else if (std::norm(alpha) == 0) {
                // Do nothing
            } else {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                COPY_N_BLOCKING_FOR(w[wj] += alpha * v[vj]);
            }
        }

#undef COPY_N_BLOCKING_FOR

        ///
        /// Blocking copy on GPU
        ///

#ifdef SUPERBBLAS_USE_THRUST
        template <typename IndexType, typename T, typename Q, typename EWOP>
        struct copy_n_blocking_elem;

        template <typename IndexType, typename T, typename Q>
        struct copy_n_blocking_elem<IndexType, T, Q, EWOp::Copy>
            : public thrust::unary_function<IndexType, void> {
            const T alpha;
            const T *const SB_RESTRICT v;
            const IndexType blocking;
            const IndexType *const SB_RESTRICT indicesv;
            Q *const SB_RESTRICT w;
            const IndexType *const SB_RESTRICT indicesw;
            copy_n_blocking_elem(T alpha, const T *v, IndexType blocking, const IndexType *indicesv,
                                 Q *w, const IndexType *indicesw)
                : alpha(alpha),
                  v(v),
                  blocking(blocking),
                  indicesv(indicesv),
                  w(w),
                  indicesw(indicesw) {}

            __HOST__ __DEVICE__ void operator()(IndexType i) {
                IndexType d = i / blocking, r = i % blocking;
                w[indicesw[d] + r] = alpha * v[indicesv[d] + r];
            }
        };

        template <typename IndexType, typename T, typename Q>
        struct copy_n_blocking_elem<IndexType, T, Q, EWOp::Add>
            : public thrust::unary_function<IndexType, void> {
            const T alpha;
            const T *const SB_RESTRICT v;
            const IndexType blocking;
            const IndexType *const SB_RESTRICT indicesv;
            Q *const SB_RESTRICT w;
            const IndexType *const SB_RESTRICT indicesw;
            copy_n_blocking_elem(T alpha, const T *v, IndexType blocking, const IndexType *indicesv,
                                 Q *w, const IndexType *indicesw)
                : alpha(alpha),
                  v(v),
                  blocking(blocking),
                  indicesv(indicesv),
                  w(w),
                  indicesw(indicesw) {}

            __HOST__ __DEVICE__ void operator()(IndexType i) {
                IndexType d = i / blocking, r = i % blocking;
                w[indicesw[d] + r] += alpha * v[indicesv[d] + r];
            }
        };

        template <typename IndexType, typename T, typename Q, typename XPU, typename EWOP>
        void copy_n_blocking_same_dev_thrust(typename elem<T>::type alpha, const T *v,
                                             IndexType blocking, const IndexType *indicesv,
                                             std::size_t n, Q *w, const IndexType *indicesw,
                                             XPU xpu, EWOP) {
            setDevice(xpu);
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator(IndexType(0)),
                               blocking * n,
                               copy_n_blocking_elem<IndexType, typename cuda_complex<T>::type,
                                                    typename cuda_complex<Q>::type, EWOP>(
                                   alpha, (typename cuda_complex<T>::type *)v, blocking, indicesv,
                                   (typename cuda_complex<Q>::type *)w, indicesw));
        }

#endif // SUPERBBLAS_USE_THRUST

#ifdef SUPERBBLAS_USE_GPU

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] when v and w are on device

        template <typename IndexType, typename T, typename Q, typename EWOP>
        DECL_COPY_BLOCKING_T_Q_EWOP(void copy_n_blocking(
            typename elem<T>::type alpha, const T *v, IndexType blocking, const IndexType *indicesv,
            Gpu xpuv, std::size_t n, Q *w, const IndexType *indicesw, Gpu xpuw, EWOP))
        IMPL({
            if (n == 0) return;

            // Actions when the v and w are on the same device
            if (deviceId(xpuv) == deviceId(xpuw)) {
                copy_n_blocking_same_dev_thrust(alpha, v, blocking, indicesv, n, w, indicesw, xpuw,
                                                EWOP{});
            }

            else {
                throw std::runtime_error("Not implemented");
            }
        })

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] from device to host or vice versa

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPU0, XPU1>::value, bool>::type = true>
        void copy_n_blocking(typename elem<T>::type alpha, const T *v, IndexType blocking,
                             const IndexType *indicesv, XPU0 xpu0, std::size_t n, Q *w,
                             const IndexType *indicesw, XPU1 xpu1, EWOP) {
            if (n == 0) return;

            if (indicesv || !std::is_same<T, Q>::value) {
                vector<Q, XPU0> v0(n * blocking, xpu0);
                copy_n_blocking<IndexType>(alpha, v, blocking, indicesv, xpu0, n, v0.data(),
                                           nullptr, xpu0, EWOp::Copy{});
                copy_n_blocking<IndexType, Q, Q>(Q{1}, v0.data(), blocking, nullptr, xpu0, n, w,
                                                 indicesw, xpu1, EWOP{});
            } else if (indicesw) {
                vector<Q, XPU1> w0(n * blocking, xpu1);
                copy_n<IndexType>(alpha, v, xpu0, blocking * n, w0.data(), xpu1, EWOp::Copy{});
                copy_n_blocking<IndexType>(Q{1}, w0.data(), blocking, (IndexType *)nullptr, xpu1, n,
                                           w, indicesw, xpu1, EWOP{});
            } else {
                copy_n<IndexType, T, Q>(alpha, v, xpu0, blocking * n, w, xpu1, EWOP{});
            }
        }
#endif // SUPERBBLAS_USE_GPU
    }
}
#endif // __SUPERBBLAS_COPY_N__
