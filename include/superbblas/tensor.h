#ifndef __SUPERBBLAS_TENSOR__
#define __SUPERBBLAS_TENSOR__

#include "blas.h"
#include <algorithm>
#include <array>
#include <assert.h>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace superbblas {

    /// Coordinate Index type
    using IndexType = int;
    /// Coordinate type
    template <unsigned int Nd> using Coor = std::array<IndexType, Nd>;
    /// Vector of dimension labels
    template <unsigned int Nd> using Order = std::array<char, Nd>;

    namespace detail {

        /// Vector of `IndexType`
        template <typename XPU> using Indices = vector<IndexType, XPU>;

        //
        // Auxiliary functions
        //

        template <typename T, unsigned long N>
        std::array<T, N> operator+(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (unsigned int i = 0; i < N; i++) r[i] = a[i] + b[i];
            return r;
        }

        template <typename T, unsigned long N>
        std::array<T, N> operator-(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (unsigned int i = 0; i < N; i++) r[i] = a[i] - b[i];
            return r;
        }

        template <typename T, unsigned long N>
        bool all_less_or_equal(const std::array<T, N> &a, const std::array<T, N> &b) {
            for (unsigned int i = 0; i < N; i++)
                if (a[i] > b[i]) return false;
            return true;
        }

        template <typename T, unsigned long N>
        std::array<T, N> min_each(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (unsigned int i = 0; i < N; i++) r[i] = std::min(a[i], b[i]);
            return r;
        }

        template <typename T, unsigned long N>
        std::array<T, N> max_each(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (unsigned int i = 0; i < N; i++) r[i] = std::max(a[i], b[i]);
            return r;
        }

        /// Return an array with all elements set to a given value
        /// \param v: input value

        template <unsigned int Nd, typename T> std::array<T, Nd> fill_coor(T v = 0) {
            std::array<T, Nd> r;
            r.fill(v);
            return r;
        }

        /// Return an array from a string
        /// \param v: input string

        template <unsigned int Nd, typename T> std::array<T, Nd> toArray(const T *v) {
            std::array<T, Nd> r;
            std::copy_n(v, Nd, r.begin());
            return r;
        }

        /// Return an order with values 0, 1, 2, ..., N-1

        template <unsigned long N>
        Order<N> trivial_order() {
            Order<N> r;
            for (unsigned int i = 0; i < N; i++) r[i] = (char)i;
            return r;
        }

        /// Return coor[i] % dim[i]
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions

        template <unsigned int Nd>
        Coor<Nd> normalize_coor(const Coor<Nd> &coor, const Coor<Nd> &dim) {
            Coor<Nd> r;
            for (unsigned int j = 0; j < Nd; j++) r[j] = coor[j] % dim[j];
            return r;
        }

        /// Return whether a coordinate is in a range
        /// \param from: first coordinate of the range
        /// \param size: size of the range
        /// \param dim: lattice dimensions
        /// \param coor: coordinate to prove

        template <unsigned int Nd>
        bool coor_in_range(const Coor<Nd> &from, const Coor<Nd> &size, const Coor<Nd> &dim,
                           const Coor<Nd> &coor) {
            for (unsigned int j = 0; j < Nd; j++) {
                if ((coor[j] < from[j] || from[j] + size[j] <= coor[j]) &&
                    (coor[j] + dim[j] < from[j] || from[j] + size[j] <= coor[j] + dim[j]))
                    return false;
            }
            return true;
        }

        /// Return the closest coordinate to a given one in a range
        /// NOTE: the lattice is NOT considered toroidal
        /// \param from: first coordinate of the range
        /// \param size: size of the range
        /// \param coor: coordinate to prove

        template <unsigned int Nd>
        Coor<Nd> closest(const Coor<Nd> &from, const Coor<Nd> &size, const Coor<Nd> &coor) {
            Coor<Nd> r;
            for (unsigned int j = 0; j < Nd; j++) {
                r[j] = from[j] + std::min(std::max(coor[j] - from[j], 0), size[j]);
            }
            return r;
        }


        /// Return the intersection between two ranges
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param fromr: first coordinate of the resulting range
        /// \param sizer: size of the resulting range

        template <unsigned int Nd>
        void intersection(const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &from1,
                          const Coor<Nd> &size1, Coor<Nd> &fromr, Coor<Nd> &sizer) {
            fromr = closest<Nd>(from0, size0, from1);
            sizer = closest<Nd>(from0, size0, from1 + size1) - fromr;
        }

        /// Return the jumps to the next consecutive element in that dimension
        /// \param dim: lattice dimension
        ///
        /// NOTE: we used anti-natural order, the last coordinate moves the fastest

        template <unsigned int Nd> Coor<Nd> get_strides(const Coor<Nd> dim) {
            // p(i) = prod(dim(end:-1:i))
            Coor<Nd> p;
            p.back() = 1;
            for (int i = p.size() - 1; i >= 1; i--) p[i - 1] = p[i] * dim[i];
            return p;
        }

        /// Return the index associated to a coordinate
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions
        /// \param stride: jump to get to the next coordinate in each dimension

        template <unsigned int Nd>
        IndexType coor2index(const Coor<Nd> &coor, const Coor<Nd> &dim, const Coor<Nd> &stride) {
            IndexType r = 0;
            for (unsigned int j = 0; j < Nd; j++) r += (coor[j] % dim[j]) * stride[j];
            return r;
        }

        /// Return the coordinate associated to an index
        /// \param index: input vertex index
        /// \param dim: lattice dimensions
        /// \param stride: jump to get to the next coordinate in each dimension

        template <unsigned int Nd>
        inline Coor<Nd> index2coor(const IndexType &index, const Coor<Nd> &dim,
                                   const Coor<Nd> &stride) {
            Coor<Nd> r;
            for (unsigned int j = 0; j < Nd; j++) r[j] = (index / stride[j]) % dim[j];
            return r;
        }

        /// Check all dimension labels are distinct
        /// \param order: dimension labels
        ///
        /// Return whether all label dimension are distinct

        template <typename Vector> bool check_order(const Vector &order) {
            for (unsigned int i = 0; i < order.size(); ++i)
                if (std::find(order.begin() + i + 1, order.end(), order[i]) != order.end())
                    return false;
            return true;
        }

        /// Return the number of vertices in a lattice
        /// \param dim: lattice dimensions

        template <unsigned int Nd> std::size_t volume(const Coor<Nd> &dim) {
            if (dim.size() <= 0) return 0;

            std::size_t vol = dim[0];
            for (unsigned int i = 1; i < dim.size(); ++i) vol *= dim[i];
            return vol;
        }

        /// Return the number of vertices in a sublattice
        /// \param order: dimension labels
        /// \param dim: lattice dimensions
        /// \param starts_with: the first label of the sublattice
        /// \param size: number of consecutive dimension of the sublattice

        template <unsigned int Nd>
        std::size_t volume(const Order<Nd> &order, const Coor<Nd> &dim, char starts_with,
                           unsigned int size) {
            assert(size <= order.size());

            if (size <= 0) return 0;

            std::size_t vol = 1;
            for (unsigned int n = 0, i = std::find(order.begin(), order.end(), starts_with) -
                                         order.begin();
                 n < size; ++n, ++i)
                vol *= dim[i];

            return vol;
        }

        /// Return a new array {coor[perm[0]], coor[perm[1]], ...}
        /// \param coor: input array
        /// \param perm: permutation
        /// \param black: value to set when perm[i] < 0
        ///
        /// NOTE: the output array will have zero on negative elements of `perm`.

        template <unsigned int Nd0, unsigned int Nd1>
        Coor<Nd1> reorder_coor(const Coor<Nd0> &coor, const Coor<Nd1> &perm, IndexType blanck = 0) {
            Coor<Nd1> r;
            for (unsigned int i = 0; i < Nd1; ++i) r[i] = perm[i] >= 0 ? coor[perm[i]] : blanck;
            return r;
        }

        /// Check that there exists a permutation from the first tensor to the second
        /// \param o0: dimension labels
        /// \param dim0: dimension size for o0
        /// \param o1: dimension labels
        /// \param dim1: dimension size for o0
        ///
        /// Return whether all labels with dimension size greater than one in o0 are also in o1 and
        /// and the dimension of the first is smaller or equal than the second

        template <unsigned int Nd0, unsigned int Nd1>
        bool is_a_subset_of(Order<Nd0> o0, Coor<Nd0> dim0, Order<Nd1> o1) {
            for (unsigned int i = 0; i < o0.size(); ++i)
                if (dim0[i] > 0 && std::find(o1.begin(), o1.end(), o0[i]) == o1.end()) return false;
            return true;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: source dimension labels
        /// \param o1: destination dimension labels
        ///
        /// NOTE: the permutation can be used in function `reorder_coor`.

        template <unsigned int Nd0, unsigned int Nd1>
        Coor<Nd1> find_permutation(const Order<Nd0> &o0, const Order<Nd1> &o1) {
            Coor<Nd1> r;
            for (unsigned int i = 0; i < Nd1; ++i) {
                const auto j = std::find(o0.begin(), o0.end(), o1[i]);
                r[i] = (j != o0.end() ? j - o0.begin() : -1);
            }
            return r;
        }

        /// Check that all values are positive
        /// \param from: coordinates to check

        template <unsigned int Nd>
        bool check_positive(const Coor<Nd> &from) {
            Coor<Nd> zeros = {0};
            return all_less_or_equal(zeros, from);
        }

        /// Check that the copy operation is possible
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: first coordinate not to copy from the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor

        template <unsigned int Nd0, unsigned int Nd1>
        bool check_isomorphic(const Order<Nd0> &o0, const Coor<Nd0> &size0,
                             const Coor<Nd0> &dim0, const Order<Nd1> &o1, const Coor<Nd1> dim1) {

            if (!(check_order(o0) && check_order(o1) && check_positive<Nd0>(size0) &&
                  all_less_or_equal(size0, dim0) && is_a_subset_of<Nd0, Nd1>(o0, size0, o1)))
                return false;

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

        template <unsigned int Nd0, unsigned int Nd1>
        Indices<Cpu> get_permutation_origin(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                            const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                            const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                            const Coor<Nd1> &dim1, Cpu cpu) {
            (void)from1;
            (void)dim1;
            (void)cpu;

            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Compute the indices
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            std::size_t vol0 = volume<Nd0>(dim0);
            std::size_t vol = volume<Nd0>(size0);

            Indices<Cpu> indices0(vol);
            Coor<Nd0> stride0 = get_strides<Nd0>(dim0);
            Coor<Nd1> new_stride1 = get_strides<Nd1>(size1);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            for (std::size_t i = 0; i < vol; ++i) {
                Coor<Nd1> c1 = index2coor<Nd1>(i, size1, new_stride1);
                indices0[i] =
                    coor2index<Nd0>(reorder_coor<Nd1, Nd0>(c1, perm1) + from0, dim0, stride0);
                assert(0 <= indices0[i] && indices0[i] < (IndexType)vol0);
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

        template <unsigned int Nd0, unsigned int Nd1>
        Indices<Cpu> get_permutation_destination(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                                 const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                                 const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                                 const Coor<Nd1> &dim1, Cpu cpu) {
            (void)from0;
            (void)cpu;

            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Compute the indices
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            std::size_t vol1 = volume<Nd1>(dim1);
            std::size_t vol = volume<Nd0>(size0);

            Indices<Cpu> indices1(vol);
            Coor<Nd1> stride1 = get_strides<Nd1>(dim1);
            Coor<Nd1> new_stride1 = get_strides<Nd1>(size1);
            for (std::size_t i = 0; i < vol; ++i) {
                Coor<Nd1> c1 = index2coor<Nd1>(i, size1, new_stride1);
                indices1[i] = coor2index<Nd1>(c1 + from1, dim1, stride1);
                assert(0 <= indices1[i] && indices1[i] < (IndexType)vol1);
            }

            return indices1;
        }

#ifdef SUPERBBLAS_USE_CUDA
        /// Return the permutation on the origin to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector

        template <unsigned int Nd0, unsigned int Nd1>
        Indices<Cuda> get_permutation_origin(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                            const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                            const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                            const Coor<Nd1> &dim1, Cuda cuda) {

            (void)cuda;
            Indices<Cpu> indices_host =
                get_permutation_origin<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1, Cpu{});
            Indices<Cuda> indices = indices_host;
            return indices;
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

        template <unsigned int Nd0, unsigned int Nd1>
        Indices<Cuda> get_permutation_destination(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                                  const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                                  const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                                  const Coor<Nd1> &dim1, Cuda cuda) {
            (void)cuda;
            Indices<Cpu> indices_host = get_permutation_destination<Nd0, Nd1>(
                o0, from0, size0, dim0, o1, from1, dim1, Cpu{});
            Indices<Cuda> indices = indices_host;
            return indices;
        }
#endif // SUPERBBLAS_USE_CUDA

        /// Find common largest substring
        /// \param o0: dimension labels
        /// \param o1: dimension labels
        /// \param starts_with: (out) the first label of the common substring
        /// \param size: the number of common labels
        ///
        /// Return the largest common substring in o0 and o1 assuming that each
        /// dimension has different labels on each vector

        template <typename Vector0, typename Vector1, typename ConstIterator,
                  typename value_type = typename Vector0::value_type>
        void largest_common_substring_order(const Vector0 &o0, const Vector1 &o1,
                                            ConstIterator avoid, unsigned int nAvoid,
                                            value_type &starts_with, unsigned int &size) {
            size = 0;
            for (unsigned int i = 0; i < o0.size(); ++i) {
                if (nAvoid > 0 && std::find(avoid, avoid + nAvoid, o0[i]) != avoid + nAvoid)
                    continue;
                auto j = std::find(o1.begin(), o1.end(), o0[i]);
                if (j == o1.end()) continue;
                starts_with = o0[i];
                for (unsigned int i0 = i; i0 < o0.size() && j != o1.end() && o0[i0] == *j &&
                                          (nAvoid == 0 || *j != *avoid);
                     ++i0, ++j, ++size)
                    ;
                break;
            }
        }

        /// Find common largest substring
        /// \param o0: dimension labels
        /// \param o1: dimension labels
        /// \param o2: dimension labels
        /// \param starts_with: (out) the first label of the common substring
        /// \param size: the number of common labels
        ///
        /// Return the largest common substring in o0, o1 and o2 assuming that each dimension has
        /// different labels on each vector.

        template <typename Vector0, typename Vector1, typename Vector2,
                  typename value_type = typename Vector0::value_type>
        void largest_common_substring_order(const Vector0 &o0, const Vector1 &o1, const Vector2 &o2,
                                            value_type &starts_with, unsigned int &size) {
            size = 0;
            for (unsigned int i = 0; i < o0.size(); ++i) {
                auto j = std::find(o1.begin(), o1.end(), o0[i]);
                if (j == o1.end()) continue;
                auto k = std::find(o2.begin(), o2.end(), o0[i]);
                if (k == o2.end()) continue;
                starts_with = o0[i];
                for (unsigned int i0 = i; i0 < o0.size() && j != o1.end() && k != o2.end() &&
                                          o0[i0] == *j && o0[i0] == *k;
                     ++i0, ++j, ++k, ++size)
                    ;
                break;
            }
        }

        /// Contract two tensors
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param v0: data for the first operator
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param v1: data for the second operator
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param vr: data for the second operator
        ///
        /// The order of the labels should be as following:
        ///
        /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
        /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
        /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
        /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

        template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo, typename ConstIterator,
                  typename Iterator, typename XPU>
        void local_contraction(const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0,
                               ConstIterator v0, const Order<Nd1> &o1, const Coor<Nd1> &dim1,
                               bool conj1, ConstIterator v1, const Order<Ndo> &o_r,
                               const Coor<Ndo> &dimr, Iterator vr, XPU xpu) {

            static_assert(std::is_same<typename std::iterator_traits<ConstIterator>::value_type,
                                       typename std::iterator_traits<Iterator>::value_type>::value,
                          "v0 and v1 should have the same type");

            // Check orders
            assert(check_order(o0));
            assert(check_order(o1));
            assert(check_order(o_r));

            // Find T, the common labels between o0, o1, and o_r
            unsigned int nT = 0; // size of the piece T
            char sT = 0;         // starting letter of the piece T
            char eT = 0;         // ending letter of the piece T
            largest_common_substring_order(o0, o1, o_r, sT, nT);
            auto strT = o0.begin();
            if (nT > 0) {
                strT = std::find(o0.begin(), o0.end(), sT);
                eT = strT[nT - 1];
            }

            // Find A, the common labels between o0 and o1
            unsigned int nA = 0; // size of the piece A
            char sA = 0;         // starting letter of the piece A
            largest_common_substring_order(o0, o1, strT, nT, sA, nA);

            // Find B, the common labels between o0 and o_r
            unsigned int nB = 0; // size of the piece B
            char sB = 0;         // starting letter of the piece B
            largest_common_substring_order(o0, o_r, strT, nT, sB, nB);

            // Find C, the common labels between o1 and o_r
            unsigned int nC = 0; // size of the piece C
            char sC = 0;         // starting letter of the piece C
            largest_common_substring_order(o1, o_r, strT, nT, sC, nC);

            // Check that o0 is made of the pieces T, A and B
            assert(o0.size() == nT + nA + nB);
            // Check that o1 is made of the pieces T, C and A
            assert(o1.size() == nT + nA + nC);
            // Check that o_r is made of the pieces T, C and B
            assert(o_r.size() == nT + nB + nC);

            // Check that no order ends with T
            assert(nT == 0 || o0.size() == 0 || o0.back() != eT);
            assert(nT == 0 || o1.size() == 0 || o1.back() != eT);
            assert(nT == 0 || nB + nC == 0 || o_r.size() == 0 || o_r.back() != eT);

            // Check whether each order starts with T
            bool o0_starts_with_T = (nT == 0 || o0.size() == 0 || o0[0] == sT);
            bool o1_starts_with_T = (nT == 0 || o1.size() == 0 || o1[0] == sT);
            bool or_starts_with_T = (nT == 0 || o_r.size() == 0 || o_r[0] == sT);

            // Check if o0 and o1 need transpose
            bool o0_trans = (o0.size() > nT && o0[o0_starts_with_T ? nT : 0] == sB);
            bool o1_trans = (o1.size() > nT && o1[o1_starts_with_T ? nT : 0] == sA);
            bool or_trans = (o_r.size() > nT && o_r[o0_starts_with_T ? nT : 0] == sB);
            assert(!or_trans);          // Not supported this case for now
            assert(o0_trans || !conj0); // Not supported this case for now
            assert(o1_trans || !conj1); // Not supported this case for now

            // Compute the volume for each piece
            int volT = nT == 0 ? 1 : volume<Nd0>(o0, dim0, sT, nT);
            int volA = volume<Nd0>(o0, dim0, sA, nA);
            int volA_nonzero = volA > 0 ? 1 : 0;
            int volB = nB == 0 ? volA_nonzero : volume<Nd0>(o0, dim0, sB, nB);
            int volC = nC == 0 ? volA_nonzero : volume<Nd1>(o1, dim1, sC, nC);

            // Avoid issues with uninitialized memory by zeroing out
            fill_n(vr, volume<Ndo>(dimr), 0.0, xpu);

            // Let's do (A, B) x (C, A) -> (C, B)
            char transab = o0_trans ? (conj0 ? 'C' : 'T') : 'N';
            char transca = o1_trans ? (conj1 ? 'C' : 'T') : 'N';
            int ldab = (o0_starts_with_T ? 1 : volT) * (!o0_trans ? volB : volA);
            int strideab =
                (o0_starts_with_T ? volume<Nd0>(dim0) / volT : (!o0_trans ? volB : volA));
            int ldca = (o1_starts_with_T ? 1 : volT) * (!o1_trans ? volA : volC);
            int strideca =
                (o1_starts_with_T ? volume<Nd1>(dim1) / volT : (!o1_trans ? volA : volC));
            int ldcb = (or_starts_with_T ? 1 : volT) * (!o0_trans ? volB : volC);
            int stridecb =
                (or_starts_with_T ? volume<Ndo>(dimr) / volT : (!o0_trans ? volC : volB));
            using value_type = typename std::iterator_traits<ConstIterator>::value_type;
            value_type one = 1.0, zero = 0.0;
            xgemm_batch_strided<value_type>(transab, transca, (int)volB, (int)volC, (int)volA, one,
                                            const_raw_pointer(v0), ldab, strideab,
                                            const_raw_pointer(v1), ldca, strideca, zero,
                                            raw_pointer(vr), ldcb, stridecb, volT, xpu);
        }

        /// Copy the content of tensor o0 into o1
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param xpu0: device context for v0
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param xpu1: device context for v1

        template <unsigned int Nd0, unsigned int Nd1, typename T, typename XPU0, typename XPU1>
        void local_copy(const Order<Nd0> &o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                        const Coor<Nd0> &dim0, data<const T, XPU0> v0, XPU0 xpu0,
                        const Order<Nd1> &o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                        data<T, XPU1> v1, XPU1 xpu1) {

            // Get the permutation vectors
            Indices<XPU0> indices0 =
                get_permutation_origin<Nd0, Nd1>(o0, from0, size0, dim0, o1, from1, dim1, xpu0);
            Indices<XPU1> indices1 = get_permutation_destination<Nd0, Nd1>(o0, from0, size0, dim0,
                                                                           o1, from1, dim1, xpu1);

            // Do the copy
            copy_n<IndexType, T>(v0, indices0.begin(), xpu0, indices0.size(), v1, indices1.begin(),
                                 xpu1);
        }
    }

    /// Copy the content of tensor o0 into o1
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

    template <unsigned int Nd0, unsigned int Nd1, typename T>
    void local_copy(const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                    const Coor<Nd0> &dim0, const T *v0, Context ctx0, const char *o1,
                    const Coor<Nd1> &from1, const Coor<Nd1> &dim1, T *v1, Context ctx1) {
        if (std::strlen(o0) != Nd0)
            throw std::runtime_error("The length of `o0` does not match the template argument");
        if (std::strlen(o1) != Nd1)
            throw std::runtime_error("The length of `o1` does not match the template argument");
        const Order<Nd0> o0_ = detail::toArray<Nd0>(o0);
        const Order<Nd1> o1_ = detail::toArray<Nd1>(o1);
        local_copy<Nd0, Nd1>(o0_, from0, size0, dim0, v0, ctx0, o1_, from1, dim1, v1, ctx1);
    }

    /// Copy the content of tensor o0 into o1
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

    template <unsigned int Nd0, unsigned int Nd1, typename T>
    void local_copy(const Order<Nd0> &o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                    const Coor<Nd0> &dim0, const T *v0, Context ctx0, const Order<Nd1> &o1,
                    const Coor<Nd1> &from1, const Coor<Nd1> &dim1, T *v1, Context ctx1) {

        // Check the validity of the operation

        if (!detail::check_positive<Nd0>(from0))
            throw std::runtime_error("All values in `from0` should be non-negative");

        if (!detail::check_positive<Nd0>(size0))
            throw std::runtime_error("All values in `size0` should be non-negative");

        if (!detail::check_positive<Nd1>(from1))
            throw std::runtime_error("All values in `from1` should be non-negative");

        if (!detail::check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1))
            throw std::runtime_error("The orders and dimensions of the origin tensor are not "
                                     "compatible with the destination tensor");

        // Do the operation
        if (ctx0.plat == CPU && ctx1.plat == CPU) {
            detail::local_copy<Nd0, Nd1, T>(o0, from0, size0, dim0, v0, ctx0.toCpu(), o1, from1,
                                            dim1, v1, ctx1.toCpu());
        }
#ifdef SUPERBBLAS_USE_CUDA
        else if (ctx0.plat == CPU && ctx1.plat == CUDA) {
            detail::local_copy<Nd0, Nd1, T>(o0, from0, size0, dim0, v0, ctx0.toCpu(), o1, from1,
                                            dim1, detail::encapsulate_pointer(v1), ctx1.toCuda());
        } else if (ctx0.plat == CUDA && ctx1.plat == CPU) {
            detail::local_copy<Nd0, Nd1, T>(o0, from0, size0, dim0, detail::encapsulate_pointer(v0),
                                            ctx0.toCuda(), o1, from1, dim1, v1, ctx1.toCpu());
        } else if (ctx0.plat == CUDA && ctx1.plat == CUDA) {
            detail::local_copy<Nd0, Nd1, T>(o0, from0, size0, dim0, detail::encapsulate_pointer(v0),
                                            ctx0.toCuda(), o1, from1, dim1,
                                            detail::encapsulate_pointer(v1), ctx1.toCuda());
        }
#endif
        else {
            throw std::runtime_error("Unsupported platform");
        }
    }

    /// Contract two tensors
    /// \param o0: dimension labels for the first operator
    /// \param dim0: dimension size for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param o1: dimension labels for the second operator
    /// \param dim1: dimension size for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param o_r: dimension labels for the output operator
    /// \param dimr: dimension size for the output operator
    /// \param vr: data for the second operator
    ///
    /// The order of the labels should be as following:
    ///
    /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
    /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
    /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
    /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

    template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo, typename T>
    void local_contraction(const char *o0, const Coor<Nd0> &dim0, bool conj0, const T *v0,
                           const char *o1, const Coor<Nd1> &dim1, bool conj1, const T *v1,
                           const char *o_r, const Coor<Ndo> &dimr, T *vr, Context ctx) {
        assert(std::strlen(o0) == Nd0);
        assert(std::strlen(o1) == Nd1);
        assert(std::strlen(o_r) == Ndo);
        Order<Nd0> o0_ = detail::toArray<Nd0>(o0);
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1);
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r);
        local_contraction<Nd0, Nd1, Ndo>(o0_, dim0, conj0, v0, o1_, dim1, conj1, v1, o_r_, dimr, vr,
                                         ctx);
    }

    template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo, typename T>
    void local_contraction(const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0, const T *v0,
                           const Order<Nd1> &o1, const Coor<Nd1> &dim1, bool conj1, const T *v1,
                           const Order<Ndo> &o_r, const Coor<Ndo> &dimr, T *vr, Context ctx) {

        switch (ctx.plat) {
        case CPU:
            detail::local_contraction<Nd0, Nd1, Ndo>(o0, dim0, conj0, v0, o1, dim1, conj1, v1, o_r,
                                                     dimr, vr, ctx.toCpu());
            break;
#ifdef SUPERBBLAS_USE_CUDA
        case CUDA:
            detail::local_contraction<Nd0, Nd1, Ndo>(
                o0, dim0, conj0, detail::encapsulate_pointer(v0), o1, dim1, conj1,
                detail::encapsulate_pointer(v1), o_r, dimr, detail::encapsulate_pointer(vr),
                ctx.toCuda());
            break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }
}
#endif // __SUPERBBLAS_TENSOR__
