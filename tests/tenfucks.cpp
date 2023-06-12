#include "superbblas/tenfucks.h"
#include "superbblas.h"
#include <ccomplex>
#include <complex>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>

using namespace superbblas;
using namespace superbblas::detail;

template <typename T> void dummyFill(T* v, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) v[i] = i;
}

template <typename F> double show_time(double flops, std::size_t nreps, const F &f) {
    double t = w_time();
    for (std::size_t rep = 0; rep < nreps; ++rep) f();
    t = w_time() - t;
    return flops * nreps / t / 1e9;
}

std::vector<std::pair<std::string, double>> kernels;

template <typename F>
void show_time(const std::string &s, double flops, std::size_t nreps, const F &f) {
    double gflops = show_time(flops, nreps, f);
    std::cout << "   " << s << ": " << gflops << " gflops" << std::endl;
    kernels.push_back({s, gflops});
}

template <unsigned int M, unsigned int N, unsigned int K, typename T>
void test_blk(std::size_t m, std::size_t n, std::size_t k, std::size_t t, const T *a, const T *b,
              T *c, bool transa, bool transb, double flops, std::size_t nreps = 10) {

    T beta{1};
    double gflops_ijk = show_time(flops, nreps, [=] {
        for (std::size_t i = 0; i < t; ++i)
            gemm_blk_ijk_nobuffer<M, N, K, unsigned int>(
                m, n, k, T{1}, a + m * k * i, !transa ? 1 : k, !transa ? m : 1, b + k * n * i,
                !transb ? 1 : n, !transb ? k : 1, beta, c + m * n * i, 1, m);
    });
    double gflops_kij = show_time(flops, nreps, [=] {
        for (std::size_t i = 0; i < t; ++i)
            gemm_blk_kij_nobuffer<M, N, K, unsigned int>(
                m, n, k, T{1}, a + m * k * i, !transa ? 1 : k, !transa ? m : 1, b + k * n * i,
                !transb ? 1 : n, !transb ? k : 1, beta, c + m * n * i, 1, m);
    });
    double gflops_ikj = show_time(flops, nreps, [=] {
        for (std::size_t i = 0; i < t; ++i)
            gemm_blk_ikj_nobuffer<M, N, K, unsigned int>(
                m, n, k, T{1}, a + m * k * i, !transa ? 1 : k, !transa ? m : 1, b + k * n * i,
                !transb ? 1 : n, !transb ? k : 1, beta, c + m * n * i, 1, m);
    });
    std::stringstream ss_ijk;
    ss_ijk << "gemm_blk_ijk_" << M << "_" << N << "_" << K ;
    std::stringstream ss_kij;
    ss_kij << "gemm_blk_kij_" << M << "_" << N << "_" << K;
    std::stringstream ss_ikj;
    ss_ikj << "gemm_blk_ikj_" << M << "_" << N << "_" << K;
    std::cout << "   " << ss_ijk.str() << ": " << gflops_ijk << " gflops" //
              << "   " << ss_kij.str() << ": " << gflops_kij << " gflops" //
              << "   " << ss_ikj.str() << ": " << gflops_ikj << " gflops" << std::endl;
    kernels.push_back({ss_ijk.str(), gflops_ijk});
    kernels.push_back({ss_kij.str(), gflops_kij});
    kernels.push_back({ss_ikj.str(), gflops_ikj});
}

template <unsigned int M, unsigned int N, typename T>
void test_blk_2(std::size_t m, std::size_t n, std::size_t k, std::size_t t, const T *a, const T *b, T *c,
                bool transa, bool transb, double flops, std::size_t nreps) {
    if (1 <= k) test_blk<M, N, 1>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (2 <= k) test_blk<M, N, 2>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (3 <= k) test_blk<M, N, 3>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (4 <= k) test_blk<M, N, 4>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (6 <= k) test_blk<M, N, 6>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (8 <= k) test_blk<M, N, 8>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (16 <= k) test_blk<M, N, 16>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
}

template <unsigned int M, typename T>
void test_blk_1(std::size_t m, std::size_t n, std::size_t k, std::size_t t, const T *a, const T *b, T *c,
                bool transa, bool transb, double flops, std::size_t nreps) {
    if (1 <= n) test_blk_2<M, 1>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (2 <= n) test_blk_2<M, 2>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (3 <= n) test_blk_2<M, 3>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (4 <= n) test_blk_2<M, 4>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (6 <= n) test_blk_2<M, 6>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (8 <= n) test_blk_2<M, 8>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (16 <= n) test_blk_2<M, 16>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
}

template <typename T>
void test_blk_all(std::size_t m, std::size_t n, std::size_t k, std::size_t t, const T *a, const T *b, T *c,
                  bool transa, bool transb, double flops, std::size_t nreps) {
    if (1 <= m) test_blk_1<1>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (2 <= m) test_blk_1<2>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    if (3 <= m) test_blk_1<3>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (4 <= m) test_blk_1<4>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (6 <= m) test_blk_1<6>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (8 <= m) test_blk_1<8>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
    //if (16 <= m) test_blk_1<16>(m, n, k, t, a, b, c, transa, transb, flops, nreps);
}

template <typename T, typename Tc>
void test(std::size_t m, std::size_t n, std::size_t k, std::size_t t, const T *a, const T *b, T *c,
          bool transa, bool transb, std::size_t nreps = 10) {

    std::cout << " > " << (!transa ? 'n' : 't') << (!transb ? 'n' : 't') << std::endl;
    double flops = m * n * k * t;

    kernels.clear();
    T beta{1};
    show_time("blas", flops, nreps, [=] {
        for (std::size_t i = 0; i < t; ++i)
            xgemm(!transa ? 'n' : 't', !transb ? 'n' : 't', m, n, k, Tc{1},
                  (const Tc *)a + m * k * i, !transa ? m : k, (const Tc *)b + k * n * i,
                  !transb ? k : n, Tc{beta}, (Tc *)c + m * n * i, m, Cpu{});
    });

    show_time("gemm_basic", flops, nreps, [=] {
        for (std::size_t i = 0; i < t; ++i)
            gemm_basic<unsigned int>(m, n, k, T{1}, a + m * k * i, !transa ? 1 : k, !transa ? m : 1,
                                     b + k * n * i, !transb ? 1 : n, !transb ? k : 1, beta,
                                     c + m * n * i, 1, m);
    });

    test_blk_all(m, n, k, t, a, b, c, transa, transb, flops, nreps);

    std::sort(kernels.begin(), kernels.end(),
              [](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b) {
                  return a.second > b.second;
              });
    std::cout << "Best kernels:" << std::endl;
    for (unsigned int i = 0; i < 5 && i < kernels.size(); ++i)
        std::cout << kernels.at(i).first << ": " << kernels.at(i).second << " gflops" << std::endl;
}

template <typename T, typename Tc = T>
void test(std::size_t m, std::size_t n, std::size_t k, std::size_t t, std::size_t nreps = 10) {
    std::vector<T> a(m * k * t);
    std::vector<T> b(n * k * t);
    std::vector<T> c(m * n * t);
    dummyFill(a.data(), a.size());
    dummyFill(b.data(), b.size());

    test<T, Tc>(m, n, k, t, a.data(), b.data(), c.data(), false, false, nreps);
    test<T, Tc>(m, n, k, t, a.data(), b.data(), c.data(), true, false, nreps);
    test<T, Tc>(m, n, k, t, a.data(), b.data(), c.data(), false, true, nreps);
    test<T, Tc>(m, n, k, t, a.data(), b.data(), c.data(), true, true, nreps);
}

int main(int argc, char **argv) {
    std::size_t m = 4, n = 4, k = 1000000, t = 1, nreps = 10;
    if (argc == 6) {
        sscanf(argv[1], "%ld", &m);
        sscanf(argv[2], "%ld", &n);
        sscanf(argv[3], "%ld", &k);
        sscanf(argv[4], "%ld", &t);
        sscanf(argv[5], "%ld", &nreps);
    } else if (argc != 1) {
        std::cerr << "invalid number of args" << std::endl;
        return -1;
    }
    std::cout << "TESTING m= " << m << " n= " << n << " k=" << k << " t=" << t << std::endl;

    //std::cout << "*) half" << std::endl;
    //test<_Float16>(m, n, k, t);
    //std::cout << "*) complex half" << std::endl;
    //test<_Complex _Float16>(m, n, k, t);

    std::cout << "*) float" << std::endl;
    test<float>(m, n, k, t);
    std::cout << "*) complex float" << std::endl;
    test<_Complex float, std::complex<float>>(m, n, k, t);

    std::cout << "*) double" << std::endl;
    test<double>(m, n, k, t);
    std::cout << "*) complex double" << std::endl;
    test<_Complex double, std::complex<double>>(m, n, k, t);
    return 0;
}
