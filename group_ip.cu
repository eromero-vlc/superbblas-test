#include "cublas_v2.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cuda_runtime.h>
#include <ostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <type_traits>
#include <utility>
#include <vector>
#include <omp.h>

#ifdef USE_MKL
#    include "mkl.h"
#endif // USE_MKL

#ifdef __CUDA_ARCH__
#    define __HOST__ __host__
#    define __DEVICE__ __device__
#else
#    define __HOST__
#    define __DEVICE__
#endif // __CUDA_ARCH__

// template <class Ostream> Ostream &operator<<(Ostream &ostream, const SpinColor &sc) {
//     ostream << "{";
//     for (const auto &i : sc) ostream << " " << std::real(i) << " + i * " << std::imag(i);
//     ostream << "}";
//     return ostream;
// }

template <typename Vector> void print(const Vector &v, const char *name) {
    std::cout << name << ":";
    for (const auto &i : v) std::cout << " " << i;
    std::cout << std::endl;
}


namespace Test {

    struct CPU {};
    struct GPU {};

    template <typename V>
    using xPU = typename std::conditional<
        std::is_same<V, thrust::device_vector<typename V::value_type>>::value, GPU, CPU>::type;

    template <typename T, typename CPUGPU>
    void xgemm_batch_strided(char transa, char transb, int m, int n, int k, T alpha, const T *a,
                             int lda, int stridea, const T *b, int ldb, int strideb, T beta, T *c,
                             int ldc, int stridec, int batch_size);

#ifdef USE_MKL
    CBLAS_TRANSPOSE toCblasTrans(char trans) {
        switch (trans) {
        case 'n':
        case 'N': return CblasNoTrans;
        case 't':
        case 'T': return CblaTrans;
        case 'c':
        case 'C': return CblaConjTrans;
        default: throw std::runtime_error("Not valid value of trans");
        }
    }

    template <>
    void xgemm_batch_strided<std::complex<double>, CPU>(char transa, char transb, int m, int n,
                                                        int k, T alpha, const T *a, int lda,
                                                        int stridea, const T *b, int ldb,
                                                        int strideb, T beta, T *c, int ldc,
                                                        int stridec, int batch_size) {

        cblas_zgemm_batch_strided(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m, n,
                                  k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc,
                                  stridec, batch_size);
    }

#else
    extern "C" void zgemm_(const char *transa, const char *transb, const int *m, const int *n,
                           const int *k, const void *alpha, const void *a, const int *lda,
                           const void *b, const int *ldb, const void *beta, void *c,
                           const int *ldc);

    template <>
    void xgemm_batch_strided<std::complex<double>, CPU>(
        char transa, char transb, int m, int n, int k, std::complex<double> alpha,
        const std::complex<double> *a, int lda, int stridea, const std::complex<double> *b, int ldb,
        int strideb, std::complex<double> beta, std::complex<double> *c, int ldc, int stridec,
        int batch_size) {

#    ifdef _OPENMP
#        pragma omp for
#    endif
        for (int i = 0; i < batch_size; ++i) {
            zgemm_(&transa, &transb, &m, &n, &k, &alpha, a + stridea * i, &lda, b + strideb * i,
                   &ldb, &beta, c + stridec * i, &ldc);
        }
    }

#endif // USE_MKL

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


    // Coordinate type
    using IndexType = int;
    template <unsigned int Nd> using Coor = std::array<IndexType, Nd>;
    // Vector of indices
    using Indices = std::vector<IndexType>;
    // Vector of coordinates
    template <unsigned int Nd> using Coors = std::vector<Coor<Nd>>;
    // Vector of dimension labels
    template <unsigned int Nd> using Order = std::array<char, Nd>;

    // Return the jumps to the next consecutive element in that dimension
    // \param dim: lattice dimension
    //
    // NOTE: we used anti-natural order, the last coordinate moves the fastest

    template <unsigned int Nd> Coor<Nd> get_strides(const Coor<Nd> dim) {
        // p(i) = prod(dim(end:-1:i))
        Coor<Nd> p;
        p.back() = 1;
        for (int i = p.size() - 1; i >= 1; i--) p[i - 1] = p[i] * dim[i];
        return p;
    }

    // Quick exit
    // Return the indices associated to each coordinate
    // \param coors: input coordinates
    // \param dim: lattice dimension
    //
    // Return a vector with indices of the passed coordinates in
    // the same order.
    //
    // NOTE: we used anti-natural order, the last coordinate moves the fastest

    template <unsigned int Nd> Indices coor2index(const Coors<Nd> &coors, const Coor<Nd> dim) {
        // Quick exit
        if (dim.size() <= 0) return Indices();

        // Output array
        Indices indices(coors.size());

        // p(i) = prod(dim(end:-1:i))
        Coor<Nd> p = get_strides(dim);

        // indices(i) = inner product of coor(i) and p
        // NOTE: every coordinate value is normalize to modulus the lattice dimension
        for (unsigned int i = 0; i < coors.size(); i++) {
            IndexType r = 0;
            for (unsigned int j = 0; j < dim.size(); j++) r += (coors[i][j] % dim[j]) * p[j];
            indices[i] = r;
        }

        return indices;
    }

    template <unsigned int Nd> IndexType coor2index(const Coor<Nd> &coor, const Coor<Nd> dim) {
        return coor2index(Coors<Nd>(1, coor), dim)[0];
    }

    template <unsigned int Nd>
    IndexType coor2index(const Coor<Nd> &coor, const Coor<Nd> &dim, const Coor<Nd> &stride) {
        IndexType r = 0;
        for (unsigned int j = 0; j < Nd; j++) r += (coor[j] % dim[j]) * stride[j];
        return r;
    }

    // Return the coordinates associated to each index
    // \param indices: input vertex indices
    // \param dim: lattice dimension
    //
    // Return a vector with the coordinates of the passed indices in
    // the same order.
    //
    // NOTE: we used anti-natural order, the last coordinate moves the fastest

    template <unsigned int Nd> Coors<Nd> index2coor(const Indices &indices, const Coor<Nd> dim) {
        // Quick exit
        if (dim.size() <= 0) return Coors<Nd>();

        // Output array
        Coors<Nd> coors(indices.size());

        // p(i) = prod(dim(end:-1:i))
        Coor<Nd> p = get_strides(dim);

        // coors(i,j) = indices(i) / p(i)
        // NOTE: every coordinate value is normalize to modulus the lattice dimension
        for (unsigned int i = 0; i < indices.size(); i++)
            for (unsigned int j = 0; j < dim.size(); j++)
                coors[i][j] = (indices[i] / p[j]) % dim[j];

        return coors;
    }

    template <unsigned int Nd>
    inline Coor<Nd> index2coor(const IndexType &index, const Coor<Nd> &dim) {
        return index2coor(Indices(1, index), dim)[0];
    }

    template <unsigned int Nd>
    inline Coor<Nd> index2coor(const IndexType &index, const Coor<Nd> &dim,
                               const Coor<Nd> &stride) {
        Coor<Nd> r;
        for (unsigned int j = 0; j < Nd; j++) r[j] = (index / stride[j]) % dim[j];
        return r;
    }

    // Check all dimension labels are distinct
    // \param order: dimension labels
    //
    // Return whether all label dimension are distinct

    template <typename Vector> bool check_order(const Vector &order) {
        for (unsigned int i = 0; i < order.size(); ++i)
            if (std::find(order.begin() + i + 1, order.end(), order[i]) != order.end())
                return false;
        return true;
    }

    // Return the number of vertices in a lattice
    // \param dim: lattice dimensions

    template <unsigned int Nd> std::size_t volume(const Coor<Nd> &dim) {
        if (dim.size() <= 0) return 0;

        std::size_t vol = dim[0];
        for (unsigned int i = 1; i < dim.size(); ++i) vol *= dim[i];
        return vol;
    }

    // Return the number of vertices in a sublattice
    // \param order: dimension labels
    // \param dim: lattice dimensions
    // \param starts_with: the first label of the sublattice
    // \param size: number of consecutive dimension of the sublattice

    template <unsigned int Nd>
    std::size_t volume(const Order<Nd> &order, const Coor<Nd> &dim, char starts_with,
                       unsigned int size) {
        assert(size <= order.size());

        if (size <= 0) return 0;

        std::size_t vol = 1;
        for (unsigned int n = 0,
                          i = std::find(order.begin(), order.end(), starts_with) - order.begin();
             n < size; ++n, ++i)
            vol *= dim[i];

        return vol;
    }

    template <unsigned int Nd> Coor<Nd> fill_coor(const IndexType &v) {
        Coor<Nd> r;
        r.fill(v);
        return r;
    }

    template <unsigned int Nd0, unsigned int Nd1>
    Coor<Nd1> reorder_coor(const Coor<Nd0> &coor, const Coor<Nd1> &perm) {
        Coor<Nd1> r;
        for (unsigned int i = 0; i < Nd1; ++i) r[i] = perm[i] >= 0 ? coor[perm[i]] : 0;
        return r;
    }

    // Check that there exists a permutation from one labels order to the other
    // \param o0: dimension labels
    // \param o1: dimension labels
    //
    // Return whether all labels in o0 are also in o1

    template <unsigned int Nd0, unsigned int Nd1>
    bool isomorphic_tensor(const Order<Nd0> &o0, const Order<Nd1> &o1) {
        for (const auto &i : o0)
            if (std::find(o1.begin(), o1.end(), i) == o1.end()) return false;
        for (const auto &i : o1)
            if (std::find(o0.begin(), o0.end(), i) == o0.end()) return false;
        return true;
    }

    // Check that there exists a permutation from one labels order to the other for all dimensions
    // with size larger than one
    // \param o0: dimension labels
    // \param dim0: dimension size for o0
    // \param o1: dimension labels
    // \param dim1: dimension size for o0
    //
    // Return whether all labels with dimension size greater than one in o0 are also in o1 and
    // vice versa

    template <unsigned int Nd0, unsigned int Nd1>
    bool isomorphic_tensor(Order<Nd0> o0, Coor<Nd0> dim0, Order<Nd1> o1, Coor<Nd1> dim1) {

        for (unsigned int i = 0; i < o0.size(); ++i)
            if (dim0[i] > 0 && std::find(o1.begin(), o1.end(), o0[i]) == o1.end()) return false;
        for (unsigned int i = 0; i < o1.size(); ++i)
            if (dim1[i] > 0 && std::find(o0.begin(), o0.end(), o1[i]) == o0.end()) return false;
        return true;
    }

    template <unsigned int Nd0, unsigned int Nd1>
    Coor<Nd1> find_permutation(const Order<Nd0> &o0, const Order<Nd1> &o1) {
        assert((isomorphic_tensor<Nd0, Nd1>(o0, o1)));
        Coor<Nd1> r;
        for (unsigned int i = 0; i < Nd1; ++i) {
            const auto j = std::find(o0.begin(), o0.end(), o1[i]);
            r[i] = (j != o0.end() ? j - o0.begin() : -1);
        }
        return r;
    }

    template <unsigned int Nd0, unsigned int Nd1>
    void get_permutation(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Coor<Nd0> &from0,
                         const Coor<Nd0> &to0, const Order<Nd1> &o1, const Coor<Nd1> &dim1,
                         const Coor<Nd1> &from1, thrust::host_vector<IndexType> &indices0,
                         thrust::host_vector<IndexType> &indices1) {
        // Check orders
        assert(check_order(o0));
        assert(check_order(o1));
        assert(all_less_or_equal(from0, to0) && all_less_or_equal(to0, dim0));

        // Check the compatibility of the tensors
        assert((isomorphic_tensor<Nd0, Nd1>(o0, to0 - from0, o1, dim1)));

        // Compute the indices
        Coor<Nd0> perm = find_permutation<Nd1, Nd0>(o1, o0);
        Coor<Nd0> new_dim0 = to0 - from0;
        Coor<Nd1> new_dim1 = reorder_coor<Nd0, Nd1>(new_dim0, perm);
        assert(all_less_or_equal(new_dim1 + from1, dim1));
        assert((reorder_coor<Nd1, Nd0>(new_dim1, perm) == new_dim0));
        std::size_t vol0 = volume<Nd0>(dim0);
        std::size_t vol1 = volume<Nd1>(dim1);
        std::size_t vol = volume<Nd0>(new_dim0);
        indices0.resize(vol);
        indices1.resize(vol);

        Coor<Nd0> stride0 = get_strides<Nd0>(dim0);
        Coor<Nd1> stride1 = get_strides<Nd1>(dim1);
        Coor<Nd1> new_stride1 = get_strides<Nd1>(new_dim1);
        for (std::size_t i = 0; i < vol; ++i) {
            Coor<Nd1> c1 = index2coor<Nd1>(i, new_dim1, new_stride1) + from1;
            indices0[i] = coor2index<Nd0>(reorder_coor<Nd1, Nd0>(c1, perm), dim0, stride0);
            indices1[i] = coor2index<Nd1>(c1, dim1, stride1);
            assert(0 <= indices0[i] && indices0[i] < vol0);
            assert(0 <= indices1[i] && indices1[i] < vol1);
        }

        //print(indices0, "indices0");
        //print(indices1, "indices1");
    }

    template <unsigned int Nd0, unsigned int Nd1>
    void get_permutation(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Coor<Nd0> &from0,
                         const Coor<Nd0> &to0, const Order<Nd1> &o1, const Coor<Nd1> &dim1,
                         const Coor<Nd1> &from1, thrust::device_vector<IndexType> &indices0,
                         thrust::device_vector<IndexType> &indices1) {
        thrust::host_vector<IndexType> indices0_host, indices1_host;
        get_permutation<Nd0, Nd1>(o0, dim0, from0, to0, o1, dim1, from1, indices0_host,
                                  indices1_host);
        indices0 = indices0_host;
        indices1 = indices1_host;
    }

    // Find common largest substring
    // \param o0: dimension labels
    // \param o1: dimension labels
    // \param starts_with: (out) the first label of the common substring
    // \param size: the number of common labels
    //
    // Return the largest common substring in o0 and o1 assuming that each
    // dimension has different labels on each vector

    template <typename Vector0, typename Vector1,
              typename value_type = typename Vector0::value_type>
    void largest_common_substring_order(const Vector0 &o0, const Vector1 &o1,
                                        value_type &starts_with, unsigned int &size) {
        size = 0;
        for (unsigned int i = 0; i < o0.size(); ++i) {
            auto j = std::find(o1.begin(), o1.end(), o0[i]);
            if (j == o1.end()) continue;
            starts_with = o0[i];
            for (unsigned int i0 = i; i0 < o0.size() && j != o1.end() && o0[i0] == *j;
                 ++i0, ++j, ++size)
                ;
            break;
        }
    }

    // Find common largest substring
    // \param o0: dimension labels
    // \param o1: dimension labels
    // \param o2: dimension labels
    // \param starts_with: (out) the first label of the common substring
    // \param size: the number of common labels
    //
    // Return the largest common substring in o0, o1 and o2 assuming that each dimension has
    // different labels on each vector.

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
            for (unsigned int i0 = i;
                 i0 < o0.size() && j != o1.end() && k != o2.end() && o0[i0] == *j && o0[i0] == *k;
                 ++i0, ++j, ++k, ++size)
                ;
            break;
        }
    }

    template <typename Vector> const typename Vector::value_type *get_const_pointer(const Vector &v) {
        return v.data();
    }

    template <typename T> const T *get_const_pointer(const thrust::device_vector<T> &v) {
        return v.data().get();
    }

    template <typename Vector> typename Vector::value_type *get_pointer(Vector &v) {
        return v.data();
    }

    template <typename T> T *get_pointer(thrust::device_vector<T> &v) {
        return v.data().get();
    }


    // Contract two tensors
    // \param o0: dimension labels for the first operator
    // \param dim0: dimension size for the first operator
    // \param conj0: whether element-wise conjugate the first operator
    // \param v0: data for the first operator
    // \param o1: dimension labels for the second operator
    // \param dim1: dimension size for the second operator
    // \param conj1: whether element-wise conjugate the second operator
    // \param v1: data for the second operator
    // \param o_r: dimension labels for the output operator
    // \param dimr: dimension size for the output operator
    // \param vr: data for the second operator
    //
    // The order of the labels should be as following:
    //
    // - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
    // - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
    // - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
    // - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

    template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo, typename Vector>
    void local_contraction(const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0,
                           const Vector &v0, const Order<Nd1> &o1, const Coor<Nd1> &dim1,
                           bool conj1, const Vector &v1, const Order<Ndo> &o_r,
                           const Coor<Ndo> &dimr, Vector &vr) {

        // Check volumes
        assert(volume<Nd0>(dim0) == v0.size());
        assert(volume<Nd1>(dim1) == v1.size());
        assert(volume<Ndo>(dimr) == vr.size() || (dimr.size() == 0 && vr.size() == 1));

        // Check orders
        assert(check_order(o0));
        assert(check_order(o1));
        assert(check_order(o_r));

        // Find T, the common labels between o0, o1, and o_r
        unsigned int nT = 0; // size of the piece T
        char sT = 0;         // starting letter of the piece T
        char eT = 0;         // ending letter of the piece T
        largest_common_substring_order(o0, o1, o_r, sT, nT);
        if (nT > 0) eT = *(std::find(o0.begin(), o0.end(), sT) + nT - 1);

        // Find A, the common labels between o0 and o1
        unsigned int nA = 0; // size of the piece A
        char sA = 0;         // starting letter of the piece A
        largest_common_substring_order(o0, o1, sA, nA);

        // Find B, the common labels between o0 and o_r
        unsigned int nB = 0; // size of the piece B
        char sB = 0;         // starting letter of the piece B
        largest_common_substring_order(o0, o_r, sB, nB);

        // Find C, the common labels between o1 and o_r
        unsigned int nC = 0; // size of the piece C
        char sC = 0;         // starting letter of the piece C
        largest_common_substring_order(o1, o_r, sC, nC);

        // Check that o0 is made of the pieces T, A and B
        assert(o0.size() == nT + nA + nB);
        // Check that o1 is made of the pieces T, C and A
        assert(o1.size() == nT + nA + nC);
        // Check that o_r is made of the pieces T, C and B
        assert(o_r.size() == nT + nB + nC);

        // Check that no order ends with T
        assert(nT == 0 || o0.size() == 0 || o0.back() != eT);
        assert(nT == 0 || o1.size() == 0 || o1.back() != eT);
        assert(nT == 0 || o_r.size() == 0 || o_r.back() != eT);

        // Check whether each order starts with T
        bool o0_starts_with_T = (nT == 0 || o0.size() == 0 || o0[0] == sT);
        bool o1_starts_with_T = (nT == 0 || o1.size() == 0 || o1[0] == sT);
        bool or_starts_with_T = (nT == 0 || o_r.size() == 0 || o_r[0] == sT);

        // Check if o0 and o1 need transpose
        bool o0_trans = (o0.size() > 0 && o0[o0_starts_with_T ? nT : 0] == sB);
        bool o1_trans = (o1.size() > 0 && o1[o1_starts_with_T ? nT : 0] == sA);
        bool or_trans = (o_r.size() > 0 && o_r[o0_starts_with_T ? nT : 0] == sB);
        assert(!or_trans);          // Not supported this case for now
        assert(o0_trans || !conj0); // Not supported this case for now
        assert(o1_trans || !conj1); // Not supported this case for now

        // Compute the volume for each piece
        int volT = nT == 0 ? 1 : volume<Nd0>(o0, dim0, sT, nT);
        int volA = volume<Nd0>(o0, dim0, sA, nA);
        int volB = volume<Nd0>(o0, dim0, sB, nB);
        int volC = volume<Nd1>(o1, dim1, sC, nC);

        // Avoid issues with uninitialized memory by zeroing out
        thrust::fill(vr.begin(), vr.end(), 0.0);

        // Let's do (A, B) x (C, A) -> (C, B)
        char transab = o0_trans ? (conj0 ? 'C' : 'T') : 'N';
        char transca = o1_trans ? (conj1 ? 'C' : 'T') : 'N';
        int ldab = (o0_starts_with_T ? 1 : volT) * (!o0_trans ? volB : volA);
        int strideab = (o0_starts_with_T ? volume<Nd0>(dim0) / volT : (!o0_trans ? volB : volA));
        int ldca = (o1_starts_with_T ? 1 : volT) * (!o1_trans ? volA : volC);
        int strideca = (o1_starts_with_T ? volume<Nd1>(dim1) / volT : (!o1_trans ? volA : volC));
        int ldcb = (or_starts_with_T ? 1 : volT) * (!o0_trans ? volB : volC);
        int stridecb = (or_starts_with_T ? volume<Ndo>(dimr) / volT : (!o0_trans ? volB : volC));
        typename Vector::value_type one = 1.0, zero = 0.0;
        xgemm_batch_strided<typename Vector::value_type, xPU<Vector>>(
            transab, transca, (int)nB, (int)nC, (int)nA, one, get_const_pointer(v0), ldab, strideab,
            get_const_pointer(v1), ldca, strideca, zero, get_pointer(vr), ldcb, stridecb, volT);
    }

    // Copy the content of tensor o0 into o1
    // \param o0: dimension labels for the origin tensor
    // \param dim0: dimension size for the origin tensor
    // \param from0: first coordinate to copy from the origin tensor
    // \param to0: first coordinate not to copy from the origin tensor
    // \param v0: data for the origin tensor
    // \param o1: dimension labels for the destination tensor
    // \param dim1: dimension size for the destination tensor
    // \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    // \param v1: data for the destination tensor

    template <unsigned int Nd0, unsigned int Nd1, typename T>
    void local_copy(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Coor<Nd0> &from0,
                    const Coor<Nd0> &to0, const thrust::host_vector<T> &v0, const Order<Nd1> &o1,
                    const Coor<Nd1> &dim1, const Coor<Nd1> &from1, thrust::host_vector<T> &v1) {

        // Get the permutation vectors
        thrust::host_vector<IndexType> indices0(0), indices1(0);
        get_permutation<Nd0, Nd1>(o0, dim0, from0, to0, o1, dim1, from1, indices0, indices1);

        // Do the copy
        auto it0 = thrust::make_permutation_iterator(v0.begin(), indices0.begin());
        auto it1 = thrust::make_permutation_iterator(v1.begin(), indices1.begin());
        thrust::copy_n(it0, indices0.size(), it1);
     }

    template <unsigned int Nd0, unsigned int Nd1, typename T>
    void local_copy(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Coor<Nd0> &from0,
                    const Coor<Nd0> &to0, const thrust::device_vector<T> &v0, const Order<Nd1> &o1,
                    const Coor<Nd1> &dim1, const Coor<Nd1> &from1, thrust::device_vector<T> &v1) {

        // Get the permutation vectors
        thrust::device_vector<IndexType> indices0(0), indices1(0);
        get_permutation<Nd0, Nd1>(o0, dim0, from0, to0, o1, dim1, from1, indices0, indices1);

        // Do the copy
        auto it0 = thrust::make_permutation_iterator(v0.begin(), indices0.begin());
        auto it1 = thrust::make_permutation_iterator(v1.begin(), indices1.begin());
        thrust::copy_n(it0, indices0.size(), it1);
     }

    // Copy the content of tensor o0 into o1
    // \param o0: dimension labels for the origin tensor
    // \param dim0: dimension size for the origin tensor
    // \param from0: first coordinate to copy from the origin tensor
    // \param to0: first coordinate not to copy from the origin tensor
    // \param v0: data for the origin tensor
    // \param o1: dimension labels for the destination tensor
    // \param dim1: dimension size for the destination tensor
    // \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    // \param v1: data for the destination tensor

    template <unsigned int Nd0, unsigned int Nd1, typename T>
    void local_shift(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Coor<Nd0> &from0,
                    const Coor<Nd0> &to0, const thrust::host_vector<T> &v0, const Order<Nd1> &o1,
                    const Coor<Nd1> &dim1, const Coor<Nd1> &from1, thrust::host_vector<T> &v1) {

        // Get the permutation vectors
        thrust::host_vector<IndexType> indices0(0), indices1(0);
        get_permutation<Nd0, Nd1>(o0, dim0, from0, to0, o1, dim1, from1, indices0, indices1);

        // Do the copy
        auto it0 = thrust::make_permutation_iterator(v0.begin(), indices0.begin());
        auto it1 = thrust::make_permutation_iterator(v1.begin(), indices1.begin());
        thrust::copy_n(it0, indices0.size(), it1);
     }

     template <unsigned int Nd> struct AbstractTensor {
        AbstractTensor(const Order<Nd> &order, const Coor<Nd> &dim) : _order(order), _dim(dim) {}
        AbstractTensor(const Order<Nd> &order) : _order(order), _dim(fill_coor<Nd>(0)) {}

        Order<Nd> _order;
        Coor<Nd> _dim;
    };

    template <unsigned int Nd, typename T, template <typename> class Base>
    struct ThurstTensor : public AbstractTensor<Nd>, public Base<T> {
        ThurstTensor(const Order<Nd> &order, const Coor<Nd> &dim)
            : AbstractTensor<Nd>(order, dim), Base<T>(volume(dim)) {}
    };
}

using Complex = std::complex<double>;
using SpinColor = std::array<Complex, 12>;

int main(void) {
    //using SpinColorVectorCPU = Test::ThurstTensor<4,std::array<std::complex<double>,12>, thrust::host_vector>;
    using SpinColorVectorCPU = thrust::host_vector<SpinColor>;
    using SpinColorVectorGPU = thrust::device_vector<SpinColor>;

    SpinColor zero_sc = {1.0, 1.0};
    SpinColorVectorGPU v1(10, zero_sc);
    SpinColorVectorCPU v0 = v1;

    constexpr unsigned int Nd = 2;
    using LatticeCoor = Test::Coor<Nd>;
    using LatticeOrder = Test::Order<Nd>;
    //using TensorCPU = thrust::host_vector<int>;
    using TensorCPU = thrust::host_vector<std::complex<double>>;
    const LatticeCoor dim = {5};
    TensorCPU t0(Test::volume<Nd>(dim));
    TensorCPU t1(Test::volume<Nd>(dim));
    thrust::counting_iterator<unsigned int> it(0);
    thrust::copy(it, it + Test::volume<Nd>(dim), t0.begin());
    const LatticeCoor zero_coor = Test::fill_coor<Nd>(0);
    // const LatticeOrder o0 = {'x', 'y', 'z', 't', 's'};
    // const LatticeOrder o1 = {'t', 'x', 'y', 'z', 's'};
    const LatticeCoor dim1 = {5};
    {
        double t = omp_get_wtime();
        for (unsigned int rep = 0; rep < 10; ++rep) {
            Test::local_copy<Nd, Nd>({'x'}, dim, zero_coor, dim, t0, {'x'}, dim1, zero_coor, t1);
        }
        t = omp_get_wtime() - t;
        std::cout << "Time in permuting " << t/10 << std::endl;
    }

    //const LatticeCoor dimr = {2, 5};
    TensorCPU tc(1);
    {
        double t = omp_get_wtime();
        for (unsigned int rep = 0; rep < 10; ++rep) {
            Test::local_contraction<Nd, Nd, 0>({'x'}, dim, false, t0, {'x'}, dim1, false, t1, {},
                                               {}, tc);
        }
        t = omp_get_wtime() - t;
        std::cout << "Time in permuting " << t/10 << std::endl;
    }

    //print(t0, "t0");
    //print(t1, "t1");
    return 0;
}
