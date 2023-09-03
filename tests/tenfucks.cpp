#include "superbblas/tenfucks.h"
#include "superbblas.h"
#include <algorithm>
#include <ccomplex>
#include <chrono>
#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace superbblas;
using namespace superbblas::detail;

using SCALAR = std::complex<double>;

inline void xgemm_alt(char transa, char transb, int m, int n, int k, SCALAR alpha, const SCALAR *a,
                      int lda, const SCALAR *b, int ldb, SCALAR beta, SCALAR *c, int ldc, Cpu) {
    if (m == 0 || n == 0) return;
    (void)k;
    assert(k == 3 && (m == 3 || n == 3));

    bool ta = (transa != 'n' && transa != 'N');
    bool tb = (transb != 'n' && transb != 'N');
    if (m == 3) {
        superbblas::detail_xp::gemm_basic_3x3c_intr4(n, alpha, a, !ta ? 1 : lda, !ta ? lda : 1, b,
                                                     !tb ? 1 : ldb, !tb ? ldb : 1, beta, c, 1, ldc,
                                                     c, 1, ldc);
    } else if (n == 3) {
        superbblas::detail_xp::gemm_basic_3x3c_intr4(m, alpha, b, tb ? 1 : ldb, tb ? ldb : 1, a,
                                                     ta ? 1 : lda, ta ? lda : 1, beta, c, ldc, 1, c,
                                                     ldc, 1);
    } else
        throw std::runtime_error("wtf");
}

int main(int, char **) {
    std::vector<SCALAR> a(9);
    for (size_t i = 0; i < a.size(); ++i) a[i] = {1. * i, .5 * i};
    int n = 1;
    std::vector<SCALAR> b(3 * n);
    for (size_t i = 0; i < b.size(); ++i) b[i] = {1. * i, 1. * i};

    {
        std::vector<SCALAR> c(3 * n);

        xgemm_alt('n', 'n', 3, n, 3, 1, a.data(), 3, b.data(), 3, 0, c.data(), 3, Cpu{});

        std::vector<SCALAR> c0(3 * n);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < 3; ++k) c0[i + 3 * j] += a[i + 3 * k] * b[k + 3 * j];

        double r = 0;
        for (int i = 0; i < 3 * n; ++i) r += std::norm(c[i] - c0[i]);
        std::cout << "Error: " << std::sqrt(r) << std::endl;
    }
    {
        std::vector<SCALAR> c(3 * n);

        xgemm_alt('n', 'n', n, 3, 3, 1, b.data(), n, a.data(), 3, 0, c.data(), n, Cpu{});

        std::vector<SCALAR> c0(3 * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; ++k) c0[i + n * j] += b[i + n * k] * a[k + 3 * j];

        double r = 0;
        for (int i = 0; i < 3 * n; ++i) r += std::norm(c[i] - c0[i]);
        std::cout << "Error: " << std::sqrt(r) << std::endl;
    }

    return 0;
}
