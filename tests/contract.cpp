#include "superbblas.h"
#include <iostream>
#include <vector>

using namespace superbblas;

template <std::size_t N, typename T> using Operator = std::tuple<Coor<N>, Order<N>, std::vector<T>>;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <typename T> T conj(T t) { return std::conj(t); }
template <> float conj<float>(float t) { return t; }
template <> double conj<double>(double t) { return t; }

template <typename T, typename T::value_type = 0>
T make_complex(typename T::value_type a, typename T::value_type b) {
    return T{a, b};
}
template <typename T> T make_complex(T a, T) { return a; }

template <std::size_t N> Order<N + 1> toStr(Order<N> o) {
    Order<N + 1> r{};
    std::copy(o.begin(), o.end(), r.begin());
    return r;
}

static std::size_t progress = 0;
static char progress_mark = 0;

template <std::size_t NA, std::size_t NB, std::size_t NC, typename T>
Operator<NA + NB + NC, T> generate_tensor(char a, char b, char c, const std::map<char, int> &dims) {
    // Build the operator with A,B,C
    constexpr std::size_t N = NA + NB + NC;
    Coor<N> dim{};
    for (std::size_t i = 0; i < NA; ++i) dim[i] = dims.at(a);
    for (std::size_t i = 0; i < NB; ++i) dim[i + NA] = dims.at(b);
    for (std::size_t i = 0; i < NC; ++i) dim[i + NA + NB] = dims.at(c);
    std::size_t vol = detail::volume(dim);
    std::vector<T> v(vol);
    for (std::size_t i = 0; i < vol; ++i) v[i] = make_complex<T>(i, i);
    Order<N> o{};
    for (std::size_t i = 0; i < NA; ++i) o[i] = a + i;
    for (std::size_t i = 0; i < NB; ++i) o[i + NA] = b + i;
    for (std::size_t i = 0; i < NC; ++i) o[i + NA + NB] = c + i;
    return {dim, o, v};
}

const char sT = 'A', sA = sT + 8, sB = sA + 8, sC = sB + 8;
enum distribution { OnMaster, OnEveryone, OnEveryoneReplicated };

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T>
void test_contraction(Operator<N0, T> op0, Operator<N1, T> op1, Operator<N2, T> op2, bool conj0,
                      bool conj1, char dist_dir) {
    std::array<distribution, 3> d{OnMaster, OnEveryone, OnEveryoneReplicated};
    for (unsigned int i = 0; i < d.size(); ++i)
        test_contraction(op0, d[i], op1, d[i], op2, d[i], conj0, conj1, dist_dir);
    for (unsigned int i = 0; i < d.size(); ++i)
        for (unsigned int j = i + 1; j < d.size(); ++j)
            test_contraction(op0, d[i], op1, d[j], op2, d[i], conj0, conj1, dist_dir);
}

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T>
void test_contraction(Operator<N0, T> op0, distribution d0, Operator<N1, T> op1, distribution d1,
                      Operator<N2, T> op2, distribution d2, bool conj0, bool conj1, char dist_dir) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nprocs = 1;
    rank = 0;
#endif

    const Coor<N0> dim0 = std::get<0>(op0);
    const Coor<N1> dim1 = std::get<0>(op1);
    const Coor<N2> dim2 = std::get<0>(op2);
    const auto o0 = toStr(std::get<1>(op0));
    const auto o1 = toStr(std::get<1>(op1));
    const auto o2 = toStr(std::get<1>(op2));
    const std::vector<T> v0_ = std::get<2>(op0);
    const std::vector<T> v1_ = std::get<2>(op1);
    const std::vector<T> v2_ = std::get<2>(op2);

    Context ctx = createCpuContext();

    // Distribute op0, op1, and a zeroed op2 along the `dist_dir` direction

    Coor<N0> procs0;
    for (std::size_t i = 0; i < N0; ++i)
        procs0[i] = (d0 == OnEveryone && o0[i] == dist_dir ? nprocs : 1);
    PartitionStored<N0> p0 = basic_partitioning(dim0, procs0, nprocs, d0 == OnEveryoneReplicated);
    std::vector<T> v0(detail::volume(p0[rank][1]));
    PartitionStored<N0> p0_(nprocs, {{{{}}, dim0}}); // tensor replicated partitioning
    T const *ptrv0_ = v0_.data();
    T *ptrv0 = v0.data();
    copy(1.0, p0_.data(), 1, &o0[0], {}, dim0, dim0, (const T **)&ptrv0_, nullptr, &ctx, p0.data(),
         1, &o0[0], {}, dim0, &ptrv0, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    Coor<N1> procs1;
    for (std::size_t i = 0; i < N1; ++i)
        procs1[i] = (d1 == OnEveryone && o1[i] == dist_dir ? nprocs : 1);
    PartitionStored<N1> p1 = basic_partitioning(dim1, procs1, nprocs, d1 == OnEveryoneReplicated);
    std::vector<T> v1(detail::volume(p1[rank][1]));
    PartitionStored<N1> p1_(nprocs, {{{{}}, dim1}}); // tensor replicated partitioning
    T const *ptrv1_ = v1_.data();
    T *ptrv1 = v1.data();
    copy(1.0, p1_.data(), 1, &o1[0], {}, dim1, dim1, (const T **)&ptrv1_, nullptr, &ctx, p1.data(),
         1, &o1[0], {}, dim1, &ptrv1, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    Coor<N2> procs2;
    for (std::size_t i = 0; i < N2; ++i)
        procs2[i] = (d2 == OnEveryone && o2[i] == dist_dir ? nprocs : 1);
    PartitionStored<N2> p2 = basic_partitioning(dim2, procs2, nprocs, d2 == OnEveryoneReplicated);
    std::vector<T> v2(detail::volume(p2[rank][1]));
    T *ptrv2 = v2.data();

    // Contract the distributed matrices

    contraction(T{1}, p0.data(), {{}}, dim0, dim0, 1, &o0[0], conj0, (const T **)&ptrv0, &ctx,
                p1.data(), {{}}, dim1, dim1, 1, &o1[0], conj1, (const T **)&ptrv1, &ctx, T{0},
                p2.data(), {{}}, dim2, dim2, 1, &o2[0], &ptrv2, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                MPI_COMM_WORLD,
#endif
                SlowToFast);

    // Move the result to proc 0
    PartitionStored<N2> pr(nprocs, {{{{}}, {{}}}});
    pr[0][1] = dim2; // tensor only supported on proc 0
    std::vector<T> vr(detail::volume(pr[rank][1]));
    T *ptrvr = vr.data();
    copy(1, p2.data(), 1, &o2[0], {}, dim2, dim2, (const T **)&ptrv2, nullptr, &ctx, pr.data(), 1,
         &o2[0], {}, dim2, &ptrvr, nullptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    // Test the resulting tensor

    int is_correct = 1;
    if (rank == 0) {
        double diff_fn = 0, fn = 0; // Frob-norm of the difference and the correct tensor
        for (std::size_t i = 0; i < v2_.size(); ++i)
            diff_fn += std::norm(v2_[i] - vr[i]), fn += std::norm(v2_[i]);
        diff_fn = std::sqrt(diff_fn);
        fn = std::sqrt(fn);
        if (diff_fn > fn * 1e-4) is_correct = 0;
        progress++;
        if (progress > 770000) {
            std::cout << std::string(1, '0' + progress_mark);
            std::cout.flush();
            progress = 0;
            progress_mark++;
        }
    }
#ifdef SUPERBBLAS_USE_MPI
    MPI_Bcast(&is_correct, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    if (!is_correct) {
        // NOTE: Put a breakpoint here to debug the cases producing wrong answers!
        contraction(T{1}, p0.data(), {{}}, dim0, dim0, 1, &o0[0], conj0, (const T **)&ptrv0, &ctx,
                    p1.data(), {{}}, dim1, dim1, 1, &o1[0], conj1, (const T **)&ptrv1, &ctx, T{0},
                    p2.data(), {{}}, dim2, dim2, 1, &o2[0], &ptrv2, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast);

        throw std::runtime_error("Result of contraction does not match with the correct answer");
    }
}

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T>
void test_contraction(Operator<N0, T> p0, Operator<N1, T> p1, Operator<N2, T> p2) {
    // Compute correct result of the contraction of p0 and p1
    const Coor<N0> dim0 = std::get<0>(p0);
    const Coor<N1> dim1 = std::get<0>(p1);
    const Coor<N2> dim2 = std::get<0>(p2);
    const Order<N0> o0 = std::get<1>(p0);
    const Order<N1> o1 = std::get<1>(p1);
    const Order<N2> o2 = std::get<1>(p2);
    const std::vector<T> v0 = std::get<2>(p0);
    const std::vector<T> v1 = std::get<2>(p1);
    std::vector<T> r0(detail::volume(dim2)); // p0 not conj, and p1 not conj
    std::vector<T> r1(detail::volume(dim2)); // p0 conj, and p1 not conj
    std::vector<T> r2(detail::volume(dim2)); // p0 not conj, and p1 conj
    std::vector<T> r3(detail::volume(dim2)); // p0 conj, and p1 conj
    Coor<N0> strides0 = detail::get_strides<IndexType>(dim0, SlowToFast);
    Coor<N1> strides1 = detail::get_strides<IndexType>(dim1, SlowToFast);
    Coor<N2> strides2 = detail::get_strides<IndexType>(dim2, SlowToFast);
    for (std::size_t i = 0, m = detail::volume(dim0); i < m; ++i) {
        std::vector<int> dim(128, -1);
        Coor<N0> c0 = detail::index2coor((IndexType)i, dim0, strides0);
        for (std::size_t d = 0; d < N0; ++d) dim[o0[d]] = c0[d];
        for (std::size_t j = 0, n = detail::volume(dim1); j < n; ++j) {
            std::vector<int> dim_ = dim;
            Coor<N1> c1 = detail::index2coor((IndexType)j, dim1, strides1);
            bool get_out = false;
            for (std::size_t d = 0; d < N1; ++d) {
                if (dim_[o1[d]] == -1)
                    dim_[o1[d]] = c1[d];
                else if (dim_[o1[d]] != c1[d]) {
                    get_out = true;
                    break;
                }
            }
            if (get_out) continue;
            Coor<N2> c2{};
            for (std::size_t d = 0; d < N2; ++d) c2[d] = dim_[o2[d]];
            IndexType k = detail::coor2index(c2, dim2, strides2);
            r0[k] += v0[i] * v1[j];
            r1[k] += conj(v0[i]) * v1[j];
            r2[k] += v0[i] * conj(v1[j]);
            r3[k] += conj(v0[i]) * conj(v1[j]);
        }
    }

    std::vector<char> labels({sT, sA, sB, sC});
    for (char c : labels) {
        // Test first operator no conj and second operator no conj
        std::get<2>(p2) = r0;
        test_contraction(p0, p1, p2, false, false, c);
        // Test first operator conj and second operator no conj
        std::get<2>(p2) = r1;
        test_contraction(p0, p1, p2, true, false, c);
        // Test first operator no conj and second operator conj
        std::get<2>(p2) = r2;
        test_contraction(p0, p1, p2, false, true, c);
        // Test first operator conj and second operator conj
        std::get<2>(p2) = r3;
        test_contraction(p0, p1, p2, true, true, c);
    }
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T>
void test_third_operator(Operator<NT + NA + NB, T> p0, Operator<NT + NA + NC, T> p1,
                         const std::map<char, int> &dims) {
    test_contraction(p0, p1, generate_tensor<NT, NB, NC, T>(sT, sB, sC, dims));
    test_contraction(p0, p1, generate_tensor<NT, NC, NB, T>(sT, sC, sB, dims));
    test_contraction(p0, p1, generate_tensor<NB, NC, NT, T>(sB, sC, sT, dims));
    test_contraction(p0, p1, generate_tensor<NB, NT, NC, T>(sB, sT, sC, dims));
    test_contraction(p0, p1, generate_tensor<NC, NB, NT, T>(sC, sB, sT, dims));
    test_contraction(p0, p1, generate_tensor<NC, NT, NB, T>(sC, sT, sB, dims));
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T>
void test_second_operator(Operator<NT + NA + NB, T> p0, const std::map<char, int> &dims) {
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NT, NA, NC, T>(sT, sA, sC, dims),
                                           dims);
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NT, NC, NA, T>(sT, sC, sA, dims),
                                           dims);
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NA, NC, NT, T>(sA, sC, sT, dims),
                                           dims);
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NA, NT, NC, T>(sA, sT, sC, dims),
                                           dims);
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NC, NA, NT, T>(sC, sA, sT, dims),
                                           dims);
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NC, NT, NA, T>(sC, sT, sA, dims),
                                           dims);
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T>
void test_first_operator(const std::map<char, int> &dims) {
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NT, NA, NB, T>(sT, sA, sB, dims), dims);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NT, NB, NA, T>(sT, sB, sA, dims), dims);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NA, NB, NT, T>(sA, sB, sT, dims), dims);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NA, NT, NB, T>(sA, sT, sB, dims), dims);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NB, NA, NT, T>(sB, sA, sT, dims), dims);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NB, NT, NA, T>(sB, sT, sA, dims), dims);
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T,
          typename std::enable_if<!(NT + NA + NB == 0 || NT + NA + NC == 0 || NT + NC + NB == 0),
                                  bool>::type = true>
void test_sizes() {
    if (NT + NA + NB == 0 || NT + NA + NC == 0 || NT + NC + NB == 0) return;
    for (int dimT = 1; dimT < 3; ++dimT)
        for (int dimA = 1; dimA < 3; ++dimA)
            for (int dimB = 1; dimB < 3; ++dimB)
                for (int dimC = 1; dimC < 3; ++dimC)
                    test_first_operator<NT, NA, NB, NC, T>(
                        {{sT, dimT}, {sA, dimA}, {sB, dimB}, {sC, dimC}});
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T,
          typename std::enable_if<(NT + NA + NB == 0 || NT + NA + NC == 0 || NT + NC + NB == 0),
                                  bool>::type = true>
void test_sizes() {}

template <std::size_t NT, std::size_t NA, std::size_t NB, typename T> void test_for_C() {
    test_sizes<NT, NA, NB, 0, T>();
    test_sizes<NT, NA, NB, 1, T>();
}

template <std::size_t NT, std::size_t NA, typename T> void test_for_B() {
    test_for_C<NT, NA, 0, T>();
    test_for_C<NT, NA, 1, T>();
}

template <std::size_t NT, typename T> void test_for_A() {
    test_for_B<NT, 0, T>();
    test_for_B<NT, 1, T>();
}

template <typename T> void test() {
    test_for_A<0, T>();
    test_for_A<1, T>();
}

int main(int argc, char **argv) {
    int rank = 0;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc;
    (void)argv;
#endif

    test<double>();
    test<std::complex<double>>();

    if (rank == 0) std::cout << " Everything went ok!" << std::endl;

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
