#ifndef __SUPERBBLAS_TENSOR__
#define __SUPERBBLAS_TENSOR__

#include "blas.h"
#include "cache.h"
#include <algorithm>
#include <array>
#include <assert.h>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

//////////////////////
// NOTE:
// Functions in this file that uses `thrust` should be instrumented to remove the dependency from
// `thrust` when the superbblas library is used not as header-only. Use the macro `IMPL` to hide
// the definition of functions using `thrust` and use DECL_... macros to generate template
// instantiations to be included in the library.

#ifdef SUPERBBLAS_USE_THRUST
#    include <thrust/device_ptr.h>
#    include <thrust/execution_policy.h>
#    include <thrust/transform.h>
#endif

#ifdef SUPERBBLAS_CREATING_LIB

#    define COOR_DIMS 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
#    define REPLACE_Nd0_Nd1 REPLACE(Nd0, COOR_DIMS) REPLACE(Nd1, COOR_DIMS)

/// Generate template instantiations for get_permutation_origin and get_permutation_destination functions
/// with template parameters Nd0 and Nd1

#    define DECL_ORIG_PERM(...)                                                                    \
        EMIT REPLACE1(get_permutation_origin,                                                      \
                      superbblas::detail::get_permutation_origin<Nd0, Nd1>)                        \
            REPLACE_Nd0_Nd1 template __VA_ARGS__;

#    define DECL_DEST_PERM(...)                                                                    \
        EMIT REPLACE1(get_permutation_destination,                                                 \
                      superbblas::detail::get_permutation_destination<Nd0, Nd1>)                   \
            REPLACE_Nd0_Nd1 template __VA_ARGS__;

#else
#    define DECL_ORIG_PERM(...) __VA_ARGS__
#    define DECL_DEST_PERM(...) __VA_ARGS__
#endif

namespace superbblas {

    /// Coordinate Index type
    using IndexType = int;
    /// Coordinate type
    template <std::size_t Nd> using Coor = std::array<IndexType, Nd>;
    /// Vector of dimension labels
    template <std::size_t Nd> using Order = std::array<char, Nd>;

    /// How the coordinates are translates into positions in the tensor
    enum CoorOrder {
        SlowToFast, ///< The first coordinate runs the slowest and the last runs the fastest
        FastToSlow  ///< The first coordinate runs the fastest and the first runs the slowest
    };

    /// Action on the destination elements
    enum CopyAdd {
        Copy, ///< Copy the origin values into the destination tensor
        Add   ///< Add the origin values into the destination tensor
    };

    namespace detail {

#ifdef SUPERBBLAS_USE_THRUST

        /// Thrust does not support std::array container; here we implement a quick-and-dirty array container based on tuples

        template <typename T, std::size_t N> struct tarray;
        template <typename T, std::size_t N> struct tarray {
            static const std::size_t size_left = (N + 1) / 2;
            static const std::size_t size_right = N - size_left;
            tarray<T, size_left> left;
            tarray<T, size_right> right;
        };
        template <typename T> struct tarray<T, 0ul> {};
        template <typename T> struct tarray<T, 1ul> { T leaf; };

        /// Return the I-th element on a tarray
        /// \tparam I: index of the element to return
        /// \param t: input array

        template <std::size_t I, typename T, std::size_t N,
                  typename std::enable_if<(I > 0 && I < N), bool>::type = true>
        inline __HOST__ __DEVICE__ T &tget(tarray<T, N> &t) {
            return (I < t.size_left ? tget<I>(t.left) : tget<I - t.size_left>(t.right));
        }

        template <std::size_t I, typename T, std::size_t N,
                  typename std::enable_if<(I == 0 && N == 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T &tget(tarray<T, N> &t) {
            return t.leaf;
        }

        /// Return the i-th element on a tarray
        /// \param i: index of the element to return
        /// \param t: input array

        template <typename T, typename Indx, std::size_t N,
                  typename std::enable_if<(N > 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T tget(Indx i, const tarray<T, N> &t) {
            return (i < t.size_left ? tget(i, t.left) : tget(i - (Indx)t.size_left, t.right));
        }

        template <typename T, typename Indx, std::size_t N,
                  typename std::enable_if<(N == 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T tget(Indx i, const tarray<T, N> &t) {
            return (i == 0 ? t.leaf : T{0});
        }

        /// Coordinate based on tarray
        /// \tparam Nd: number of dimensions

        template <std::size_t Nd> using TCoor = tarray<IndexType, Nd>;
#endif

        /// Vector of `IndexType`
        template <typename XPU> using Indices = vector<IndexType, XPU>;

        //
        // Auxiliary functions
        //

        template <typename T, std::size_t Na, std::size_t Nb,
                  typename std::enable_if<Na != Nb, bool>::type = true>
        bool operator==(const std::array<T, Na> &, const std::array<T, Nb> &) {
            return false;
        }

        template <typename T, std::size_t N>
        std::array<T, N> operator-(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = a[i] - b[i];
            return r;
        }

        template <typename T, std::size_t N>
        bool all_less_or_equal(const std::array<T, N> &a, const std::array<T, N> &b) {
            for (std::size_t i = 0; i < N; i++)
                if (a[i] > b[i]) return false;
            return true;
        }

        template <typename T, std::size_t N>
        std::array<T, N> min_each(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = std::min(a[i], b[i]);
            return r;
        }

        template <typename T, std::size_t N>
        std::array<T, N> max_each(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = std::max(a[i], b[i]);
            return r;
        }

        template <typename T, std::size_t N> std::array<T, N> reverse(const std::array<T, N> v) {
            std::array<T, N> r = v;
            std::reverse(r.begin(), r.end());
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        struct ns_plus_aux {
            template <std::size_t Nd, typename std::enable_if<(Nd > 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd> plus_aux(const TCoor<Nd> &a,
                                                                 const TCoor<Nd> &b) {
                return {plus_aux(a.left, b.left), plus_aux(a.right, b.right)};
            }

            template <std::size_t Nd, typename std::enable_if<(Nd == 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd> plus_aux(const TCoor<Nd> &a,
                                                                 const TCoor<Nd> &b) {
                return {a.leaf + b.leaf};
            }
        };

        /// Add two arrays
        /// \param a: first array to add
        /// \param b: second array to add

        template <std::size_t Nd>
        __HOST__ __DEVICE__ inline TCoor<Nd> tplus(TCoor<Nd> a, TCoor<Nd> b) {
            return ns_plus_aux::plus_aux(a, b);
        }

        struct ns_toTCoor_aux {
            template <std::size_t I, std::size_t Nr, std::size_t N,
                      typename std::enable_if<(I < N && 1 < Nr), bool>::type = true>
            static inline TCoor<Nr> toTCoor_aux(const Coor<N> &a) {
                const auto sl = TCoor<Nr>::size_left;
                const auto sr = TCoor<Nr>::size_right;
                return {toTCoor_aux<I, sl>(a), toTCoor_aux<I + sl, sr>(a)};
            }

            template <std::size_t I, std::size_t Nr, std::size_t N,
                      typename std::enable_if<(I < N && 1 == Nr), bool>::type = true>
            static inline TCoor<Nr> toTCoor_aux(const Coor<N> &a) {
                return {a[I]};
            }
        };

        /// Convert from Coor to TCoor
        /// \param a: input coordinate

        template <std::size_t Nd> inline TCoor<Nd> toTCoor(const Coor<Nd> &a) {
            return ns_toTCoor_aux::toTCoor_aux<0, Nd>(a);
        }
#endif

        /// Return whether the point is in the interval
        /// \param from: first coordinate in the interval
        /// \param size: number of consecutive elements in the interval in each direction
        /// \param dim: tensor dimensions
        /// \param coor: coordinate to evaluate whether it is in the interval

        template <std::size_t N>
        bool is_in_interval(const Coor<N> &from, const Coor<N> &size, const Coor<N> &dim,
                            const Coor<N> &coor) {
            for (std::size_t i = 0; i < N; i++)
                if (!((from[i] <= coor[i] && coor[i] < from[i] + size[i]) ||
                      (from[i] <= coor[i] + dim[i] && coor[i] + dim[i] < from[i] + size[i])))
                    return false;
            return true;
        }

        /// Return an array from a string
        /// \param v: input string
        /// \param name: name of the variable

        template <std::size_t Nd, typename T>
        std::array<T, Nd> toArray(const T *v, const char *name) {
            if ((v == nullptr && Nd > 0) || std::strlen(v) != Nd) {
                std::stringstream ss;
                ss << "The length of the order should match the template argument; argument `"
                   << name << "` should have length " << Nd;
                throw std::runtime_error(ss.str());
            }
            std::array<T, Nd> r;
            std::copy_n(v, Nd, r.begin());
            return r;
        }

        /// Return the jumps to the next consecutive element in each dimension
        /// \param dim: lattice dimension
        /// \param co: coordinate linearization order

        template <std::size_t Nd> Coor<Nd> get_strides(const Coor<Nd> dim, CoorOrder co) {
            Coor<Nd> p;
            if (Nd > 0) {
                if (co == SlowToFast) {
                    // p(i) = prod(dim(end:-1:i))
                    p.back() = 1;
                    for (std::size_t i = p.size() - 1; i >= 1; i--) p[i - 1] = p[i] * dim[i];
                } else {
                    // p(i) = prod(dim(1:i))
                    p[0] = 1;
                    for (std::size_t i = 1; i < Nd; ++i) p[i] = p[i - 1] * dim[i - 1];
                }
            }
            return p;
        }

        /// Return the index associated to a coordinate
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions
        /// \param stride: jump to get to the next coordinate in each dimension

        template <std::size_t Nd>
        IndexType coor2index(const Coor<Nd> &coor, const Coor<Nd> &dim, const Coor<Nd> &stride) {
            IndexType r = 0;
            for (std::size_t j = 0; j < Nd; j++) r += (coor[j] % dim[j]) * stride[j];
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        template <std::size_t Nd, typename std::enable_if<(Nd > 1), bool>::type = true>
        __HOST__ __DEVICE__ IndexType coor2index(const TCoor<Nd> &coor, const TCoor<Nd> &dim,
                                                 const TCoor<Nd> &stride) {
            return coor2index(coor.left, dim.left, stride.left) +
                   coor2index(coor.right, dim.right, stride.right);
        }

        template <std::size_t Nd, typename std::enable_if<(Nd == 1), bool>::type = true>
        __HOST__ __DEVICE__ IndexType coor2index(const TCoor<Nd> &coor, const TCoor<Nd> &dim,
                                                 const TCoor<Nd> &stride) {
            return (coor.leaf % dim.leaf) * stride.leaf;
        }
#endif

        /// Return the coordinate associated to an index
        /// \param index: input vertex index
        /// \param dim: lattice dimensions
        /// \param stride: jump to get to the next coordinate in each dimension

        template <std::size_t Nd>
        inline Coor<Nd> index2coor(const IndexType &index, const Coor<Nd> &dim,
                                   const Coor<Nd> &stride) {
            Coor<Nd> r;
            for (std::size_t j = 0; j < Nd; j++) r[j] = (index / stride[j]) % dim[j];
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        struct ns_index2coor_aux {
            template <std::size_t Nd, typename std::enable_if<(Nd > 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd>
            index2coor(IndexType index, const TCoor<Nd> &dim, const TCoor<Nd> &stride) {
                return {index2coor(index, dim.left, stride.left),
                        index2coor(index, dim.right, stride.right)};
            }

            template <std::size_t Nd, typename std::enable_if<(Nd == 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd>
            index2coor(IndexType index, const TCoor<Nd> &dim, const TCoor<Nd> &stride) {
                return {(index / stride.leaf) % dim.leaf};
            }
        };

        template <std::size_t Nd>
        __HOST__ __DEVICE__ inline TCoor<Nd> index2coor(IndexType index, const TCoor<Nd> &dim,
                                                        const TCoor<Nd> &stride) {
            return ns_index2coor_aux::index2coor(index, dim, stride);
        }
#endif

        /// Check all dimension labels are distinct
        /// \param order: dimension labels
        ///
        /// Return whether all label dimension are distinct

        template <typename Vector> bool check_order(const Vector &order) {
            for (std::size_t i = 0; i < order.size(); ++i)
                if (std::find(order.begin() + i + 1, order.end(), order[i]) != order.end())
                    return false;
            return true;
        }

        /// Return the number of vertices in a lattice
        /// \param dim: lattice dimensions

        template <std::size_t Nd> std::size_t volume(const Coor<Nd> &dim) {
            if (dim.size() <= 0) return 0;

            std::size_t vol = dim[0];
            for (std::size_t i = 1; i < dim.size(); ++i) vol *= dim[i];
            return vol;
        }

        /// Return the number of vertices in a sublattice
        /// \param order: dimension labels
        /// \param dim: lattice dimensions
        /// \param starts_with: the first label of the sublattice
        /// \param size: number of consecutive dimension of the sublattice

        template <std::size_t Nd>
        std::size_t volume(typename Coor<Nd>::const_iterator begin,
                           typename Coor<Nd>::const_iterator end) {
            if (begin == end) return 0;

            std::size_t vol = 1;
            while (begin != end) {
                vol *= *begin;
                ++begin;
            }

            return vol;
        }

        /// Return a new array {coor[perm[0]], coor[perm[1]], ...}
        /// \param coor: input array
        /// \param perm: permutation
        /// \param black: value to set when perm[i] < 0
        ///
        /// NOTE: the output array will have zero on negative elements of `perm`.

        template <std::size_t Nd0, std::size_t Nd1>
        Coor<Nd1> reorder_coor(const Coor<Nd0> &coor, const Coor<Nd1> &perm, IndexType blanck = 0) {
            Coor<Nd1> r;
            for (std::size_t i = 0; i < Nd1; ++i) r[i] = perm[i] >= 0 ? coor[perm[i]] : blanck;
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        struct ns_reorder_coor_aux {
            template <std::size_t Nd0, std::size_t Nd1,
                      typename std::enable_if<(Nd1 > 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd1>
            reorder_coor(const TCoor<Nd0> &coor, const TCoor<Nd1> &perm, IndexType blanck) {
                return {reorder_coor(coor, perm.left, blanck),
                        reorder_coor(coor, perm.right, blanck)};
            }

            template <std::size_t Nd0, std::size_t Nd1,
                      typename std::enable_if<(Nd1 == 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd1>
            reorder_coor(const TCoor<Nd0> &coor, const TCoor<Nd1> &perm, IndexType blanck) {
                return {(perm.leaf >= 0 ? tget(perm.leaf, coor) : blanck)};
            }
        };

        template <std::size_t Nd0, std::size_t Nd1>
        __HOST__ __DEVICE__ inline TCoor<Nd1>
        reorder_coor(const TCoor<Nd0> &coor, const TCoor<Nd1> &perm, IndexType blanck = 0) {
            return ns_reorder_coor_aux::reorder_coor(coor, perm, blanck);
        }
#endif

        /// Check that there exists a permutation from the first tensor to the second
        /// \param o0: dimension labels
        /// \param dim0: dimension size for o0
        /// \param o1: dimension labels
        ///
        /// Return whether all labels with dimension size greater than one in o0 are also in o1 and
        /// and the dimension of the first is smaller or equal than the second

        template <std::size_t Nd0, std::size_t Nd1>
        bool is_a_subset_of(Order<Nd0> o0, Coor<Nd0> dim0, Order<Nd1> o1) {
            for (std::size_t i = 0; i < o0.size(); ++i)
                if (dim0[i] > 1 && std::find(o1.begin(), o1.end(), o0[i]) == o1.end()) return false;
            return true;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: source dimension labels
        /// \param o1: destination dimension labels
        ///
        /// NOTE: the permutation can be used in function `reorder_coor`.

        template <std::size_t Nd0, std::size_t Nd1>
        Coor<Nd1> find_permutation(const Order<Nd0> &o0, const Order<Nd1> &o1) {
            Coor<Nd1> r;
            for (std::size_t i = 0; i < Nd1; ++i) {
                const auto j = std::find(o0.begin(), o0.end(), o1[i]);
                r[i] = (j != o0.end() ? j - o0.begin() : -1);
            }
            return r;
        }

        /// Check that all values are positive
        /// \param from: coordinates to check

        template <std::size_t Nd> bool check_positive(const Coor<Nd> &from) {
            return all_less_or_equal({}, from);
        }

        /// Check that the copy operation is possible
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: first coordinate not to copy from the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor

        template <std::size_t Nd0, std::size_t Nd1>
        bool check_isomorphic(const Order<Nd0> &o0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                              const Order<Nd1> &o1, const Coor<Nd1> dim1) {

            if (!(check_order(o0) && check_order(o1) && check_positive<Nd0>(size0) &&
                  all_less_or_equal(size0, dim0) && is_a_subset_of<Nd0, Nd1>(o0, size0, o1)))
                return false;
            if (volume(size0) == 0) return true;

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            return all_less_or_equal(size1, dim1);
        }

        /// Return the permutation on the origin to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1>
        Indices<Cpu> get_permutation_origin(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                            const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                            const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                            const Coor<Nd1> &dim1, Cpu cpu, CoorOrder co) {
            (void)from1;
            (void)dim1;

            tracker<Cpu> _t("comp. permutations", cpu);

            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Quick exit
            if (volume<Nd0>(size0) == 0) { return Indices<Cpu>(); }

            // Compute the indices
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            std::size_t vol0 = volume<Nd0>(dim0);
            std::size_t vol = volume<Nd0>(size0);

            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= vol0)
                throw std::runtime_error("Ups! IndexType isn't big enough");

            Indices<Cpu> indices0(vol, cpu);
            Coor<Nd0> stride0 = get_strides<Nd0>(dim0, co);
            Coor<Nd1> new_stride1 = get_strides<Nd1>(size1, co);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
#ifdef _OPENMP
#    pragma omp parallel for
#endif
            for (std::size_t i = 0; i < vol; ++i) {
                Coor<Nd1> c1 = index2coor<Nd1>(i, size1, new_stride1);
                indices0[i] =
                    coor2index<Nd0>(reorder_coor<Nd1, Nd0>(c1, perm1) + from0, dim0, stride0);
                assert(0 <= indices0[i] && indices0[i] < (IndexType)vol0);
                (void)vol0;
            }

            return indices0;
        }

        /// Return the permutation on the destination to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1>
        Indices<Cpu> get_permutation_destination(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                                 const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                                 const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                                 const Coor<Nd1> &dim1, Cpu cpu, CoorOrder co) {
            (void)from0;
            (void)dim0;

            tracker<Cpu> _t("comp. permutations", cpu);

            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Quick exit
            if (volume<Nd0>(size0) == 0) { return Indices<Cpu>(); }

            // Compute the indices
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            std::size_t vol1 = volume<Nd1>(dim1);
            std::size_t vol = volume<Nd0>(size0);

            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= vol1)
                throw std::runtime_error("Ups! IndexType isn't big enough");

            Indices<Cpu> indices1(vol, cpu);
            Coor<Nd1> stride1 = get_strides<Nd1>(dim1, co);
            Coor<Nd1> new_stride1 = get_strides<Nd1>(size1, co);
#ifdef _OPENMP
#    pragma omp parallel for
#endif
            for (std::size_t i = 0; i < vol; ++i) {
                Coor<Nd1> c1 = index2coor<Nd1>(i, size1, new_stride1);
                indices1[i] = coor2index<Nd1>(c1 + from1, dim1, stride1);
                assert(0 <= indices1[i] && indices1[i] < (IndexType)vol1);
                (void)vol1;
            }

            return indices1;
        }

#ifdef SUPERBBLAS_USE_GPU

#    ifdef SUPERBBLAS_USE_THRUST
        /// Class that compute the origin permutation

        template <std::size_t Nd0, std::size_t Nd1>
        struct perm_orig_elem : public thrust::unary_function<IndexType, IndexType> {
            const TCoor<Nd0> from0, dim0, stride0;
            const TCoor<Nd1> new_stride1, size1;
            const TCoor<Nd0> perm1;
            perm_orig_elem(TCoor<Nd0> from0, TCoor<Nd0> dim0, TCoor<Nd0> stride0,
                           TCoor<Nd1> new_stride1, TCoor<Nd1> size1, TCoor<Nd0> perm1)
                : from0(from0),
                  dim0(dim0),
                  stride0(stride0),
                  new_stride1(new_stride1),
                  size1(size1),
                  perm1(perm1) {}

            __HOST__ __DEVICE__ IndexType operator()(IndexType i) {
                return coor2index(
                    tplus(reorder_coor(index2coor(i, size1, new_stride1), perm1), from0), dim0,
                    stride0);
            }
        };

        /// Class that compute the destination permutation

        template <std::size_t Nd1>
        struct perm_dest_elem : public thrust::unary_function<IndexType, IndexType> {
            const TCoor<Nd1> from1, dim1, stride1, new_stride1, size1;
            perm_dest_elem(TCoor<Nd1> from1, TCoor<Nd1> dim1, TCoor<Nd1> stride1,
                           TCoor<Nd1> new_stride1, TCoor<Nd1> size1)
                : from1(from1),
                  dim1(dim1),
                  stride1(stride1),
                  new_stride1(new_stride1),
                  size1(size1) {}

            __HOST__ __DEVICE__ IndexType operator()(IndexType i) {
                return coor2index(tplus(index2coor(i, size1, new_stride1), from1), dim1, stride1);
            }
        };
#    endif

        /// Return the permutation on the origin to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1>
        DECL_ORIG_PERM(Indices<Gpu> get_permutation_origin(
            const Order<Nd0> &o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
            const Coor<Nd0> &dim0, const Order<Nd1> &o1, const Coor<Nd1> &from1,
            const Coor<Nd1> &dim1, Gpu gpu, CoorOrder co))
        IMPL({
            (void)from1;
            (void)dim1;

            tracker<Gpu> _t("comp. permutations gpu", gpu);

            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Quick exit
            if (volume<Nd0>(size0) == 0) { return Indices<Gpu>(0, gpu); }

            // Compute the indices
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            std::size_t vol = volume<Nd0>(size0);

            Indices<Gpu> indices0(vol, gpu);
            Coor<Nd0> stride0 = get_strides<Nd0>(dim0, co);
            Coor<Nd1> new_stride1 = get_strides<Nd1>(size1, co);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);

            thrust::transform(thrust::device, thrust::make_counting_iterator(IndexType(0)),
                              thrust::make_counting_iterator(IndexType(vol)),
                              encapsulate_pointer(indices0.data()),
                              perm_orig_elem<Nd0, Nd1>(toTCoor(from0), toTCoor(dim0),
                                                       toTCoor(stride0), toTCoor(new_stride1),
                                                       toTCoor(size1), toTCoor(perm1)));

            return indices0;
        })

        /// Return the permutation on the destination to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1>
        DECL_DEST_PERM(Indices<Gpu> get_permutation_destination(
            const Order<Nd0> &o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
            const Coor<Nd0> &dim0, const Order<Nd1> &o1, const Coor<Nd1> &from1,
            const Coor<Nd1> &dim1, Gpu gpu, CoorOrder co))
        IMPL({
            (void)from0;
            (void)dim0;

            tracker<Gpu> _t("comp. permutations gpu", gpu);

            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Quick exit
            if (volume<Nd0>(size0) == 0) { return Indices<Gpu>(); }

            // Compute the indices
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            std::size_t vol = volume<Nd0>(size0);

            Indices<Gpu> indices1(vol, gpu);
            Coor<Nd1> stride1 = get_strides<Nd1>(dim1, co);
            Coor<Nd1> new_stride1 = get_strides<Nd1>(size1, co);

            thrust::transform(thrust::device, thrust::make_counting_iterator(IndexType(0)),
                              thrust::make_counting_iterator(IndexType(vol)),
                              encapsulate_pointer(indices1.data()),
                              perm_dest_elem<Nd1>(toTCoor(from1), toTCoor(dim1), toTCoor(stride1),
                                                  toTCoor(new_stride1), toTCoor(size1)));

            return indices1;
        })
#endif // SUPERBBLAS_USE_GPU

        //
        // Hash for tuples and arrays
        //

        template <typename T> struct Hash {
            template <typename U = T,
                      typename std::enable_if<!std::is_enum<U>::value, bool>::type = true>
            static std::size_t hash(U const &t) noexcept {
                return std::hash<T>{}(t);
            }
            template <typename U = T,
                      typename std::enable_if<std::is_enum<U>::value, bool>::type = true>
            static std::size_t hash(T const &t) noexcept {
                return std::size_t(t);
            }
        };

        template <typename T> struct Hash<const T> {
            static std::size_t hash(T const &t) noexcept { return Hash<T>::hash(t); }
        };

        /// Extend hash to std::array
        template <typename T, std::size_t N> struct Hash<std::array<T, N>> {
            static std::size_t hash(std::array<T, N> const &t) noexcept {
                std::size_t r = 12345;
                for (std::size_t i = 0; i < N; ++i) r = r ^ Hash<T>::hash(t[i]);
                return r;
            }
        };

        template <class Tuple> struct TupleHash;

        /// Extend Hash for std::tuple

        template <typename... Ts> struct Hash<std::tuple<Ts...>> {
            static std::size_t hash(std::tuple<Ts...> const &t) noexcept {
                return TupleHash<std::tuple<Ts...>>{}(t);
            }
        };

        /// Extend Hash for vector<T, Cpu>

        template <typename T> struct Hash<vector<T, Cpu>> {
            static std::size_t hash(vector<T, Cpu> const &t) noexcept {
                std::size_t r = 12345;
                for (std::size_t i = 0; i < t.size(); ++i) r = r ^ Hash<T>::hash(t[i]);
                return r;
            }
        };

        template <class Tuple, std::size_t N> struct TupleHashHelp {
            static std::size_t hash(Tuple const &t) noexcept {
                return Hash<typename std::tuple_element<N, Tuple>::type>::hash(std::get<N>(t)) ^
                       TupleHashHelp<Tuple, N - 1>::hash(t);
            }
        };

        template <class Tuple> struct TupleHashHelp<Tuple, 0> {
            static std::size_t hash(Tuple const &t) noexcept {
                return Hash<typename std::tuple_element<0, Tuple>::type>::hash(std::get<0>(t));
            }
        };

        /// Hash for tuples

        template <class T> struct TupleHash;

        template <class... TupleItems> struct TupleHash<typename std::tuple<TupleItems...>> {
            using Tuple = typename std::tuple<TupleItems...>;
            std::size_t operator()(Tuple const &t) const noexcept {
                return TupleHashHelp<Tuple, std::tuple_size<Tuple>::value - 1>::hash(t);
            }
        };

        template <typename T> struct TupleHash<vector<T, Cpu>> {
            using type = vector<T, Cpu>;
            std::size_t operator()(type const &t) const noexcept { return Hash<type>::hash(t); }
        };

        /// Return the memory footprint of an object
        /// \param v: input object

        template <typename T, typename XPU> std::size_t storageSize(const vector<T, XPU> &v) {
            return sizeof(T) * v.size();
        }

        /// Return the permutation on the destination to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector
        /// \param indices_out: returned permutation
        /// \param disp: returned displacement
        /// \param co: coordinate linearization order
        ///
        /// The ith element of the permutation is:
        ///   indices_out[i] + disp

        template <std::size_t Nd0, std::size_t Nd1, typename XPU>
        void get_permutation_destination_cache(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                               const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                               const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                               const Coor<Nd1> &dim1, XPU xpu,
                                               Indices<XPU> &indices_out, IndexType &disp,
                                               CoorOrder co) {
            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Quick exit
            if (volume<Nd0>(size0) == 0) {
                indices_out = Indices<XPU>(0, xpu);
                disp = 0;
                return;
            }

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);

            // Check in the storage
            using size_dim = std::tuple<Coor<Nd1>, Coor<Nd1>, int, CoorOrder>;
            using from_size_dim = std::tuple<Coor<Nd1>, Coor<Nd1>, Coor<Nd1>, int, CoorOrder>;
            struct size_dim_map_tag {};
            auto size_dim_map =
                getCache<size_dim, Indices<XPU>, TupleHash<size_dim>, size_dim_map_tag>(xpu);
            struct from_size_dim_map_tag {};
            auto from_size_dim_map = getCache<from_size_dim, Indices<XPU>, TupleHash<from_size_dim>,
                                              from_size_dim_map_tag>(xpu);
            {
                auto it =
                    from_size_dim_map.find(from_size_dim{from1, size1, dim1, deviceId(xpu), co});
                if (it != from_size_dim_map.end()) {
                    indices_out = it->second.value;
                    disp = 0;
                    return;
                }
            }
            if (all_less_or_equal(from1 + size1, dim1)) {
                auto it = size_dim_map.find(size_dim{size1, dim1, deviceId(xpu), co});
                if (it != size_dim_map.end()) {
                    indices_out = it->second.value;
                    Coor<Nd1> stride1 = get_strides<Nd1>(dim1, co);
                    disp = coor2index<Nd1>(from1, dim1, stride1);
                    return;
                }
            }

            // Get the permutation independent of 'from1' and store it in cache
            if (all_less_or_equal(from1 + size1, dim1)) {
                Indices<XPU> indices1_sd = get_permutation_destination<Nd0, Nd1>(
                    o0, {}, size0, dim0, o1, {}, dim1, xpu, co);
                size_dim_map.insert(size_dim{size1, dim1, deviceId(xpu), co}, indices1_sd,
                                    storageSize(indices1_sd));
                Coor<Nd1> stride1 = get_strides<Nd1>(dim1, co);
                disp = coor2index<Nd1>(from1, dim1, stride1);
                indices_out = indices1_sd;
                return;
            }

            // Get the permutation and store it in cache
            Indices<XPU> indices1 = get_permutation_destination<Nd0, Nd1>(o0, from0, size0, dim0,
                                                                          o1, from1, dim1, xpu, co);
            from_size_dim_map.insert(from_size_dim{from1, size1, dim1, deviceId(xpu), co}, indices1,
                                     storageSize(indices1));

            // Return the permutation
            indices_out = indices1;
            disp = 0;
        }

        /// Return the permutation on the origin to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector
        /// \param indices_out: returned permutation
        /// \param disp: returned displacement
        /// \param co: coordinate linearization order
        ///
        /// The ith element of the permutation is:
        ///   indices_out[i] + disp

        template <std::size_t Nd0, std::size_t Nd1, typename XPU>
        void get_permutation_origin_cache(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                          const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                          const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                          const Coor<Nd1> &dim1, XPU xpu, Indices<XPU> &indices_out,
                                          IndexType &disp, CoorOrder co) {
            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Quick exit
            if (volume<Nd0>(size0) == 0) {
                indices_out = Indices<XPU>(0, xpu);
                disp = 0;
                return;
            }

            // Check in the storage
            using perm_size_dim = std::tuple<Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, int, CoorOrder>;
            using perm_from_size_dim =
                std::tuple<Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, int, CoorOrder>;
            struct size_dim_map_tag {};
            auto size_dim_map =
                getCache<perm_size_dim, Indices<XPU>, TupleHash<perm_size_dim>, size_dim_map_tag>(
                    xpu);
            struct from_size_dim_map_tag {};
            auto from_size_dim_map =
                getCache<perm_from_size_dim, Indices<XPU>, TupleHash<perm_from_size_dim>,
                         from_size_dim_map_tag>(xpu);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            {
                auto it = from_size_dim_map.find(
                    perm_from_size_dim{perm1, from0, size0, dim0, deviceId(xpu), co});
                if (it != from_size_dim_map.end()) {
                    indices_out = it->second.value;
                    disp = 0;
                    return;
                }
            }
            if (all_less_or_equal(from0 + size0, dim0)) {
                auto it = size_dim_map.find(perm_size_dim{perm1, size0, dim0, deviceId(xpu), co});
                if (it != size_dim_map.end()) {
                    indices_out = it->second.value;
                    Coor<Nd0> stride0 = get_strides<Nd0>(dim0, co);
                    disp = coor2index<Nd0>(from0, dim0, stride0);
                    return;
                }
            }

            // Get the permutation independent of 'from1' and store it in cache
            if (all_less_or_equal(from0 + size0, dim0)) {
                Indices<XPU> indices0_sd =
                    get_permutation_origin<Nd0, Nd1>(o0, {}, size0, dim0, o1, {}, dim1, xpu, co);
                size_dim_map.insert(perm_size_dim{perm1, size0, dim0, deviceId(xpu), co},
                                    indices0_sd, storageSize(indices0_sd));
                Coor<Nd0> stride0 = get_strides<Nd0>(dim0, co);
                disp = coor2index<Nd0>(from0, dim0, stride0);
                indices_out = indices0_sd;
                return;
            }

            // Get the permutation and store it in cache
            Indices<XPU> indices0 =
                get_permutation_origin<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1, xpu, co);
            from_size_dim_map.insert(
                perm_from_size_dim{perm1, from0, size0, dim0, deviceId(xpu), co}, indices0,
                storageSize(indices0));

            // Return the permutation
            indices_out = indices0;
            disp = 0;
        }

        /// Check that all dimensions with the same label has the same size
        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        bool check_dimensions(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                              const Coor<Nd1> &dim1, const Order<Ndo> &o_r, const Coor<Ndo> &dimr) {
            std::map<char, IndexType> m;
            for (std::size_t i = 0; i < Nd0; ++i) m[o0[i]] = dim0[i];
            for (std::size_t i = 0; i < Nd1; ++i) {
                auto it = m.find(o1[i]);
                if (it != m.end()) {
                    if (it->second != dim1[i]) return false;
                } else {
                    m[o1[i]] = dim1[i];
                }
            }
            for (std::size_t i = 0; i < Ndo; ++i) {
                auto it = m.find(o_r[i]);
                if (it != m.end()) {
                    if (it->second != dimr[i]) return false;
                } else {
                    m[o_r[i]] = dimr[i];
                }
            }
            return true;
        }

        /// Copy the content of tensor v0 into v1
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
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1, typename EWOP>
        void local_copy(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, vector<const T, XPU0> v0,
                        const Order<Nd1> &o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                        vector<Q, XPU1> v1, EWOP ewop, CoorOrder co) {

            tracker<XPU1> _t("local copy", v1.ctx());

            // Shortcut to scale or zero out a tensor
            if (std::is_same<T, Q>::value && (void *)v0.data() == (void *)v1.data() && o0 == o1 &&
                from0 == Coor<Nd0>{} && from1 == Coor<Nd1>{} && size0 == dim0 && dim0 == dim1 &&
                std::is_same<EWOP, detail::EWOp::Copy>::value) {
                copy_n<IndexType, T, Q>(alpha, v0.data(), v0.ctx(), volume(size0), v1.data(),
                                        v1.ctx(), ewop);
                return;
            }

            // Get the permutation vectors
            Indices<XPU0> indices0;
            Indices<XPU1> indices1;
            IndexType disp0, disp1;
            get_permutation_origin_cache<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1,
                                                   v0.ctx(), indices0, disp0, co);
            get_permutation_destination_cache<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1,
                                                        v1.ctx(), indices1, disp1, co);

            // Do the copy
            copy_n<IndexType, T, Q>(alpha, v0.data() + disp0, indices0.begin(), v0.ctx(),
                                    indices0.size(), v1.data() + disp1, indices1.begin(), v1.ctx(),
                                    ewop);
        }

        /// Recommended orderings for contracting two tensors
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param sug_o0: (out) suggested dimension labels for the first operator
        /// \param sug_o1: (out) suggested dimension labels for the second operator
        /// \param sug_or: (out) suggested dimension labels for the output operator
        /// \param swap_operands: (out) suggest to swap the first and the second operator
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        void suggested_orders_for_contraction(
            const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0, const Order<Nd1> &o1,
            const Coor<Nd1> &dim1, bool conj1, const Order<Ndo> &o_r, const Coor<Ndo> &dimr,
            Order<Nd0> &sug_o0, Order<Nd1> &sug_o1, Order<Ndo> &sug_or, bool &swap_operands,
            unsigned int &nT, unsigned int &posT0, unsigned int &posT1, unsigned int &posTr,
            unsigned int &nA, unsigned int &posA0, unsigned int &posA1, unsigned int &nB,
            unsigned int &posB0, unsigned int &posBr, unsigned int &nC, unsigned int &posC1,
            unsigned int &posCr, CoorOrder co) {

            // TODO: not consider dimensions with a single element
            (void)dim0;
            (void)dim1;
            (void)dimr;

            // The rest of the code is for SlowToFast; so reverse if that is the case
            if (co == FastToSlow) {
                suggested_orders_for_contraction<Nd0, Nd1, Ndo>(
                    reverse(o0), reverse(dim0), conj0, reverse(o1), reverse(dim1), conj1,
                    reverse(o_r), reverse(dimr), sug_o0, sug_o1, sug_or, swap_operands, nT, posT0,
                    posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1, posCr, SlowToFast);
                sug_o0 = reverse(sug_o0);
                sug_o1 = reverse(sug_o1);
                sug_or = reverse(sug_or);
                posT0 = o0.size() - (posT0 + nT);
                posT1 = o1.size() - (posT1 + nT); 
                posTr = o_r.size() - (posTr + nT); 
                posA0 = o0.size() - (posA0 + nA); 
                posA1 = o1.size() - (posA1 + nA); 
                posB0 = o0.size() - (posB0 + nB); 
                posBr = o_r.size() - (posBr + nB); 
                posC1 = o1.size() - (posC1 + nC); 
                posCr = o_r.size() - (posCr + nC); 
                return;
            }

            // Find all common labels in o0, o1, and o_r
            Order<Nd0> oT;
            nT = 0;
            for (char c : o0)
                if (std::find(o1.begin(), o1.end(), c) != o1.end() &&
                    std::find(o_r.begin(), o_r.end(), c) != o_r.end())
                    oT[nT++] = c;

            // Find all common labels in o0 and o1 but not in oT
            Order<Nd0> oA;
            nA = 0;
            for (char c : o0)
                if (std::find(o1.begin(), o1.end(), c) != o1.end() &&
                    std::find(oT.begin(), oT.begin() + nT, c) == oT.begin() + nT)
                    oA[nA++] = c;

            // Find all common labels in o0 and o_r but not in oT
            Order<Nd0> oB;
            nB = 0;
            for (char c : o0)
                if (std::find(o_r.begin(), o_r.end(), c) != o_r.end() &&
                    std::find(oT.begin(), oT.begin() + nT, c) == oT.begin() + nT)
                    oB[nB++] = c;

            // Find all common labels in o1 and o_r but not in oT
            Order<Nd1> oC;
            nC = 0;
            for (char c : o1)
                if (std::find(o_r.begin(), o_r.end(), c) != o_r.end() &&
                    std::find(oT.begin(), oT.begin() + nT, c) == oT.begin() + nT)
                    oC[nC++] = c;

            // Check that o0 is made of the pieces T, A and B
            if (o0.size() != nT + nA + nB) throw std::runtime_error("o0 has unmatched dimensions");
            // Check that o1 is made of the pieces T, C and A
            if (o1.size() != nT + nA + nC) throw std::runtime_error("o1 has unmatched directions");
            // Check that o_r is made of the pieces T, C and B
            if (o_r.size() != nT + nB + nC)
                throw std::runtime_error("o_r has unmatched dimensions");

            // If oT, oA, or oB aren't found as either oT+oA+oB or oA+oT+oB or oT+oB+oA or oB+oT+oA for !conj,
            // and oT+oB+oA or oB+oT+oA for conj, then reorder the labels appropriately
            auto sT0 = std::search(o0.begin(), o0.end(), oT.begin(), oT.begin() + nT);
            auto sA0 = std::search(o0.begin(), o0.end(), oA.begin(), oA.begin() + nA);
            auto sB0 = std::search(o0.begin(), o0.end(), oB.begin(), oB.begin() + nB);
            if (sT0 == o0.end() || sA0 == o0.end() || sB0 == o0.end() ||
                (!conj0 && nT > 0 && nA > 0 && nB > 0 && sA0 < sT0 && sB0 < sT0) ||
                (conj0 && nA > 0 && ((nT > 0 && sA0 < sT0) || (nB > 0 && sA0 < sB0)))) {
                std::copy_n(oT.begin(), nT, sug_o0.begin());
                std::copy_n(oA.begin(), nA, sug_o0.begin() + nT + (!conj0 ? 0 : nB));
                std::copy_n(oB.begin(), nB, sug_o0.begin() + nT + (!conj0 ? nA : 0));
            } else
                sug_o0 = o0;

            // If oT, oA, or oC aren't found as either oT+oC+oA or oC+oT+oA or oT+oA+oC or oA+oT+oC for !conj,
            // and oT+oA+oC or oA+oT+oC for conj, then reorder the labels appropriately
            auto sT1 = std::search(o1.begin(), o1.end(), oT.begin(), oT.begin() + nT);
            auto sA1 = std::search(o1.begin(), o1.end(), oA.begin(), oA.begin() + nA);
            auto sC1 = std::search(o1.begin(), o1.end(), oC.begin(), oC.begin() + nC);
            if (sT1 == o1.end() || sA1 == o1.end() || sC1 == o1.end() ||
                (!conj1 && nT > 0 && nA > 0 && nC > 0 && sA1 < sT1 && sC1 < sT1) ||
                (conj1 && nC > 0 && ((nT > 0 && sC1 < sT1) || (nC > 0 && sC1 < sA1)))) {
                std::copy_n(oT.begin(), nT, sug_o1.begin());
                std::copy_n(oC.begin(), nC, sug_o1.begin() + nT + (!conj1 ? 0 : nA));
                std::copy_n(oA.begin(), nA, sug_o1.begin() + nT + (!conj1 ? nC : 0));
            } else
                sug_o1 = o1;

            // If oT, oB, or oC aren't found as either oT+oC+oB or oC+oT+oB, then reorder the labels appropriately
            auto sTr = std::search(o_r.begin(), o_r.end(), oT.begin(), oT.begin() + nT);
            auto sBr = std::search(o_r.begin(), o_r.end(), oB.begin(), oB.begin() + nB);
            auto sCr = std::search(o_r.begin(), o_r.end(), oC.begin(), oC.begin() + nC);
            swap_operands = false;
            if (sTr == o_r.end() || sBr == o_r.end() || sCr == o_r.end() ||
                (nT > 0 && nB > 0 && sBr < sTr) || (nB > 0 && nC > 0 && sBr < sCr)) {
                swap_operands = (nB > 0 && nC > 0 && sBr < sCr);
                std::copy_n(oT.begin(), nT, sug_or.begin());
                std::copy_n(oC.begin(), nC, sug_or.begin() + nT);
                std::copy_n(oB.begin(), nB, sug_or.begin() + nT + nC);
            } else
                sug_or = o_r;

            // Return positions
            posT0 = sT0 - o0.begin();
            posT1 = sT1 - o1.begin();
            posTr = sTr - o_r.begin();
            posA0 = sA0 - o0.begin();
            posA1 = sA1 - o1.begin();
            posB0 = sB0 - o0.begin();
            posBr = sBr - o_r.begin();
            posC1 = sC1 - o1.begin();
            posCr = sCr - o_r.begin();
        }

        /// Recommended orderings for contracting two tensors
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param sug_o0: (out) suggested dimension labels for the first operator
        /// \param sug_o1: (out) suggested dimension labels for the second operator
        /// \param sug_or: (out) suggested dimension labels for the output operator
        /// \param swap_operands: (out) suggest to swap the first and the second operator
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        void
        suggested_orders_for_contraction(const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0,
                                         const Order<Nd1> &o1, const Coor<Nd1> &dim1, bool conj1,
                                         const Order<Ndo> &o_r, const Coor<Ndo> &dimr,
                                         Order<Nd0> &sug_o0, Order<Nd1> &sug_o1, Order<Ndo> &sug_or,
                                         bool &swap_operands, CoorOrder co) {

            unsigned int nT, posT0, posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1,
                posCr;
            suggested_orders_for_contraction(
                o0, dim0, conj0, o1, dim1, conj1, o_r, dimr, sug_o0, sug_o1, sug_or, swap_operands,
                nT, posT0, posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1, posCr, co);
        }

        /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
        /// \param alpha: factor on the contraction
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param v0: data for the first operator
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param v1: data for the second operator
        /// \param beta: factor on the destination tensor
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param vr: data for the second operator
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T, typename XPU>
        void local_contraction(T alpha, const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0,
                               vector<const T, XPU> v0, const Order<Nd1> &o1, const Coor<Nd1> &dim1,
                               bool conj1, vector<const T, XPU> v1, T beta, const Order<Ndo> &o_r,
                               const Coor<Ndo> &dimr, vector<T, XPU> vr, CoorOrder co) {

            tracker<XPU> _t("local contraction", vr.ctx());

            if (deviceId(v0.ctx()) != deviceId(v1.ctx()) ||
                deviceId(v1.ctx()) != deviceId(vr.ctx()))
                throw std::runtime_error("all arrays should be on the same device");

            // Check orders
            if (!check_order(o0)) throw std::runtime_error("o0 has repeated labels");
            if (!check_order(o1)) throw std::runtime_error("o1 has repeated labels");
            if (!check_order(o_r)) throw std::runtime_error("o_r has repeated labels");

            // Check dimensions
            if (!check_dimensions<Nd0, Nd1, Ndo>(o0, dim0, o1, dim1, o_r, dimr))
                throw std::runtime_error("some dimension does not match");

            // The rest of the code is for SlowToFast; so reverse if that is the case
            if (co == FastToSlow) {
                local_contraction<Nd0, Nd1, Ndo, T, XPU>(
                    alpha, reverse(o0), reverse(dim0), conj0, v0, reverse(o1), reverse(dim1), conj1,
                    v1, beta, reverse(o_r), reverse(dimr), vr, SlowToFast);
                return;
            }

            // If o0, o1, and o_r aren't appropriate, permute the input operands and the output
            unsigned int nT, posT0, posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1,
                posCr;
            {
                Order<Nd0> sug_o0;
                Order<Nd1> sug_o1;
                Order<Ndo> sug_or;
                bool swap_operands;
                suggested_orders_for_contraction(o0, dim0, conj0, o1, dim1, conj1, o_r, dimr,
                                                 sug_o0, sug_o1, sug_or, swap_operands, nT, posT0,
                                                 posT1, posTr, nA, posA0, posA1, nB, posB0, posBr,
                                                 nC, posC1, posCr, co);
                if (swap_operands) {
                    local_contraction<Nd1, Nd0, Ndo, T, XPU>(alpha, o1, dim1, conj1, v1, o0, dim0,
                                                             conj0, v0, beta, o_r, dimr, vr,
                                                             SlowToFast);
                    return;
                }
                if (sug_o0 != o0 || sug_o1 != o1 || sug_or != o_r) {
                    Coor<Nd0> sug_dim0 = dim0;
                    vector<const T, XPU> sug_v0 = v0;
                    if (sug_o0 != o0) {
                        sug_dim0 = reorder_coor(dim0, find_permutation(o0, sug_o0));
                        vector<T, XPU> sug_v0_(v0.size(), v0.ctx());
                        local_copy<Nd0, Nd0>(T{1}, o0, {}, dim0, dim0, v0, sug_o0, {}, sug_dim0,
                                             sug_v0_, EWOp::Copy{}, SlowToFast);
                        sug_v0 = sug_v0_;
                    }

                    Coor<Nd1> sug_dim1 = dim1;
                    vector<const T, XPU> sug_v1 = v1;
                    if (sug_o1 != o1) {
                        sug_dim1 = reorder_coor(dim1, find_permutation(o1, sug_o1));
                        vector<T, XPU> sug_v1_(v1.size(), v1.ctx());
                        local_copy<Nd1, Nd1>(T{1}, o1, {}, dim1, dim1, v1, sug_o1, {}, sug_dim1,
                                             sug_v1_, EWOp::Copy{}, SlowToFast);
                        sug_v1 = sug_v1_;
                    }

                    Coor<Ndo> sug_dimr = dimr;
                    vector<T, XPU> sug_vr = vr;
                    if (sug_or != o_r) {
                        sug_dimr = reorder_coor(dimr, find_permutation(o_r, sug_or));
                        sug_vr = vector<T, XPU>(vr.size(), vr.ctx());
                        if (std::fabs(beta) != 0)
                            local_copy<Ndo, Ndo, T, T>(T{1}, o_r, {}, dimr, dimr,
                                                       vector<const T, XPU>(vr), sug_or, {},
                                                       sug_dimr, sug_vr, EWOp::Copy{}, SlowToFast);
                    }

                    local_contraction<Nd0, Nd1, Ndo, T, XPU>(alpha, sug_o0, sug_dim0, conj0, sug_v0,
                                                             sug_o1, sug_dim1, conj1, sug_v1, beta,
                                                             sug_or, sug_dimr, sug_vr, SlowToFast);

                    if (sug_or != o_r)
                        local_copy<Ndo, Ndo>(T{1}, sug_or, {}, sug_dimr, sug_dimr,
                                             vector<const T, XPU>(sug_vr), o_r, {}, dimr, vr,
                                             EWOp::Copy{}, SlowToFast);
                    return;
                }
            }

            // Compute the volume for each piece
            int volT = volume<Nd0>(dim0.begin() + posT0, dim0.begin() + posT0 + nT);
            int volA = volume<Nd0>(dim0.begin() + posA0, dim0.begin() + posA0 + nA);
            int volB = volume<Nd0>(dim0.begin() + posB0, dim0.begin() + posB0 + nB);
            int volC = volume<Nd1>(dim1.begin() + posC1, dim1.begin() + posC1 + nC);
            int vol0 = volume<Nd0>(dim0);
            int vol1 = volume<Nd1>(dim1);
            int volr = volume<Ndo>(dimr);

            // Deal with zero dimensions and implicit dimensions
            if (volr == 0) return;
            if (volT == 0) volT = 1;
            if ((vol0 > 0 || vol1 > 0) && volA == 0) volA = 1;
            if ((vol0 > 0 || volr > 0) && volB == 0) volB = 1;
            if ((vol1 > 0 || volr > 0) && volC == 0) volC = 1;
            assert(volT * volA * volB == vol0);
            assert(volT * volA * volC == vol1);
            assert(volT * volB * volC == volr);

            // Avoid issues with uninitialized memory by zeroing out
            if (std::fabs(beta) == 0.0) zero_n<T>(vr.data(), volume<Ndo>(dimr), vr.ctx());

            // Quick exit
            if (volA == 0) return;

            // Check that no order ends with T
            if (nT > 0 && posT0 + nT == o0.size() && volA > 1 && volB > 1)
                throw std::runtime_error(
                    "Unsupported contraction: the common dimensions to the input and "
                    "output tensors cannot be packed at the end of the first tensor");
            if (nT > 0 && posT1 + nT == o1.size() && volA > 1 && volC > 1)
                throw std::runtime_error(
                    "Unsupported contraction: the common dimensions to the input and "
                    "output tensors cannot be packed at the end of the second tensor");
            if (nT > 0 && posTr + nT == o_r.size() && volB > 1 && volC > 1)
                throw std::runtime_error(
                    "Unsupported contraction: the common dimensions to the input and "
                    "output tensors cannot be packed at the end of the output tensor");

            // We don't support empty tensors
            if ((o0.size() == 0) xor (o1.size() == 0))
                throw std::runtime_error("Unsupported contraction: one of the input tensors is "
                                         "empty and the other is not");
            if (o_r.size() == 0 && o0.size() + o1.size() > 0)
                throw std::runtime_error("Unsupported contraction: the output tensor is empty but "
                                         "some of the input tensors is not");

            // Check whether each order starts with T
            bool o0_starts_with_T = (volT <= 1);
            for (unsigned int i = 0; i < Nd0; ++i) {
                if (i == posT0) o0_starts_with_T = true;
                if (dim0[i] > 1) break;
            }
            bool o1_starts_with_T = (volT <= 1);
            for (unsigned int i = 0; i < Nd1; ++i) {
                if (i == posT1) o1_starts_with_T = true;
                if (dim1[i] > 1) break;
            }
            bool or_starts_with_T = (volT <= 1);
            for (unsigned int i = 0; i < Ndo; ++i) {
                if (i == posTr) or_starts_with_T = true;
                if (dimr[i] > 1) break;
            }

            // Check if o0 and o1 need transpose TAB, TCA, TCB -> BA, AC, BC
            bool o0_trans = (volA > 1 && volB > 1 && posB0 < posA0) |              // BA
                            (volA == 1 && volB > 1 && volT > 1 && posB0 < posT0) | // BT
                            (conj0 && ((o0_starts_with_T && (volA == 1 || volB == 1)) ||
                                       (!o0_starts_with_T && volA == 1 && volB == 1)));
            bool o1_trans = (volC > 1 && volA > 1 && posA1 < posC1) |              // AC
                            (volC == 1 && volA > 1 && volT > 1 && posA1 < posT1) | // AT
                            (conj1 && ((o1_starts_with_T && (volC == 1 || volA == 1)) ||
                                       (!o1_starts_with_T && volC == 1 && volA == 1)));
            bool or_trans = (volC > 1 && volB > 1 && posBr < posCr) |             // BC
                            (volC == 1 && volB > 1 && volT > 1 && posBr < posTr); // BT
            if (!o0_trans && conj0)
                throw std::runtime_error("Unsupported contraction: reorder the labels on the first "
                                         "tensor to use conjugation");
            if (!o1_trans && conj1)
                throw std::runtime_error("Unsupported contraction: reorder the labels on the "
                                         "second tensor to use conjugation");
            if (or_trans)
                throw std::runtime_error("Unsupported contraction: on the output labels, put "
                                         "the labels from the second "
                                         "tensor before the labels from the first tensor.");

            // Let's do (A, B) x (C, A) -> (C, B)
            char transab = (o0_trans ? (conj0 ? 'C' : 'T') : 'N');
            char transca = (o1_trans ? (conj1 ? 'C' : 'T') : 'N');
            int ldab = (o0_starts_with_T ? 1 : volT) * (!o0_trans ? volB : volA);
            int strideab = (o0_starts_with_T ? volA * volB : (!o0_trans ? volB : volA));
            int ldca = (o1_starts_with_T ? 1 : volT) * (!o1_trans ? volA : volC);
            int strideca = (o1_starts_with_T ? volA * volC : (!o1_trans ? volA : volC));
            int ldcb = (or_starts_with_T ? 1 : volT) * volB;
            int stridecb = (or_starts_with_T ? volB * volC : volB);
            xgemm_batch_strided(transab, transca, volB, volC, volA, alpha, v0.data(), ldab,
                                strideab, v1.data(), ldca, strideca, beta, vr.data(), ldcb,
                                stridecb, volT, vr.ctx());
        }

        /// Copy the content of tensor o0 into o1
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
        /// \param copyadd: either to copy or to add the origin values into the destination tensor
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1>
        void local_copy(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, vector<const T, XPU0> v0,
                        const Order<Nd1> &o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                        vector<Q, XPU1> v1, CopyAdd copyadd, CoorOrder co) {
            switch (copyadd) {
            case Copy:
                local_copy<Nd0, Nd1>(alpha, o0, from0, size0, dim0, v0, o1, from1, dim1, v1,
                                     EWOp::Copy{}, co);
                break;
            case Add:
                local_copy<Nd0, Nd1>(alpha, o0, from0, size0, dim0, v0, o1, from1, dim1, v1,
                                     EWOp::Add{}, co);
                break;
            }
        }
    }

    /// Copy the content of tensor o0 into o1
    /// \param alpha: factor on the copy
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of coordinates to copy in each direction
    /// \param dim0: dimension size for the origin tensor
    /// \param v0: data for the origin tensor
    /// \param ctx0: device context for v0
    /// \param o1: dimension labels for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param dim1: dimension size for the destination tensor
    /// \param v1: data for the destination tensor
    /// \param ctx1: device context for v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param copyadd: either copy or add the origin value to the destination values
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void local_copy(typename elem<T>::type alpha, const char *o0, const Coor<Nd0> &from0,
                    const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const T *v0, const Context ctx0,
                    const char *o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1, Q *v1,
                    const Context ctx1, CoorOrder co, CopyAdd copyadd, Session session = 0) {

        const Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        const Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");

        // Check the validity of the operation
        if (!detail::check_positive<Nd0>(from0))
            throw std::runtime_error("All values in `from0` should be non-negative");

        if (!detail::check_positive<Nd0>(size0))
            throw std::runtime_error("All values in `size0` should be non-negative");

        if (!detail::check_positive<Nd1>(from1))
            throw std::runtime_error("All values in `from1` should be non-negative");

        if (!detail::check_isomorphic<Nd0, Nd1>(o0_, size0, dim0, o1_, dim1))
            throw std::runtime_error("The orders and dimensions of the origin tensor are not "
                                     "compatible with the destination tensor");

        // Do the operation
        if (ctx0.plat == CPU && ctx1.plat == CPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0,
                detail::to_vector(v0, detail::volume(dim0), ctx0.toCpu(session)), o1_, from1, dim1,
                detail::to_vector(v1, detail::volume(dim1), ctx1.toCpu(session)), copyadd, co);
        }
#ifdef SUPERBBLAS_USE_GPU
        else if (ctx0.plat == CPU && ctx1.plat == GPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0,
                detail::to_vector(v0, detail::volume(dim0), ctx0.toCpu(session)), o1_, from1, dim1,
                detail::to_vector(v1, detail::volume(dim1), ctx1.toGpu(session)), copyadd, co);
        } else if (ctx0.plat == GPU && ctx1.plat == CPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0,
                detail::to_vector(v0, detail::volume(dim0), ctx0.toGpu(session)), o1_, from1, dim1,
                detail::to_vector(v1, detail::volume(dim1), ctx1.toCpu(session)), copyadd, co);
        } else if (ctx0.plat == GPU && ctx1.plat == GPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0,
                detail::to_vector(v0, detail::volume(dim0), ctx0.toGpu(session)), o1_, from1, dim1,
                detail::to_vector(v1, detail::volume(dim1), ctx1.toGpu(session)), copyadd, co);
        }
#endif
        else {
            throw std::runtime_error("Unsupported platform");
        }
    }

    /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
    /// \param alpha: factor on the contraction
    /// \param o0: dimension labels for the first operator
    /// \param dim0: dimension size for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param o1: dimension labels for the second operator
    /// \param dim1: dimension size for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param beta: factor on the destination tensor
    /// \param o_r: dimension labels for the output operator
    /// \param dimr: dimension size for the output operator
    /// \param vr: data for the second operator
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T>
    void local_contraction(T alpha, const char *o0, const Coor<Nd0> &dim0, bool conj0, const T *v0,
                           const char *o1, const Coor<Nd1> &dim1, bool conj1, const T *v1, T beta,
                           const char *o_r, const Coor<Ndo> &dimr, T *vr, const Context ctx,
                           CoorOrder co, Session session = 0) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        switch (ctx.plat) {
        case CPU:
            detail::local_contraction<Nd0, Nd1, Ndo, T>(
                alpha, o0_, dim0, conj0,
                detail::to_vector(v0, detail::volume(dim0), ctx.toCpu(session)), o1_, dim1, conj1,
                detail::to_vector(v1, detail::volume(dim1), ctx.toCpu(session)), beta, o_r_, dimr,
                detail::to_vector(vr, detail::volume(dimr), ctx.toCpu(session)), co);
            break;
#ifdef SUPERBBLAS_USE_GPU
        case GPU:
            detail::local_contraction<Nd0, Nd1, Ndo, T>(
                alpha, o0_, dim0, conj0,
                detail::to_vector(v0, detail::volume(dim0), ctx.toGpu(session)), o1_, dim1, conj1,
                detail::to_vector(v1, detail::volume(dim1), ctx.toGpu(session)), beta, o_r_, dimr,
                detail::to_vector(vr, detail::volume(dimr), ctx.toGpu(session)), co);
            break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }
}
#endif // __SUPERBBLAS_TENSOR__
