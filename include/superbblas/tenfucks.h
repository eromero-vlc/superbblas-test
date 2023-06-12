/// TENsor FUture Contraction K-dimensional Subroutine (TenFuCKS)

#ifndef __SUPERBBLAS_TENFUCKS__
#define __SUPERBBLAS_TENFUCKS__

#include "platform.h"

template <typename Idx, typename T>
void gemm_basic(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar, Idx ldac,
                const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta, T *SB_RESTRICT c,
                Idx ldcr, Idx ldcc) {
    // d[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
    bool beta_is_one = (beta == T{1});
    bool beta_is_zero = (beta == T{0});
    for (Idx i = 0; i < M; ++i) {
        for (Idx j = 0; j < N; ++j) {
            T r{0};
            for (Idx k = 0; k < K; ++k) r += a[ldar * i + ldac * k] * b[ldbr * k + ldbc * j];
            if (beta_is_one)
                c[ldcr * i + ldcc * j] += alpha * r;
            else
                c[ldcr * i + ldcc * j] =
                    (!beta_is_zero ? beta * c[ldcr * i + ldcc * j] : T{0}) + alpha * r;
        }
    }
}

template <unsigned int MM, unsigned int NN, unsigned int KK, typename Idx, typename T>
void gemm_blk_ijk_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar, Idx ldac,
              const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta, T *SB_RESTRICT c, Idx ldcr,
              Idx ldcc) {
    // c[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
    if (beta != T{1})
        for (Idx i = 0; i < i; ++i)
            for (Idx j = 0; j < j; ++j)
                c[ldcr * i + ldcc * j] = (beta != T{0} ? beta * c[ldcr * i + ldcc * j] : T{0});
    for (Idx i = 0, ii = std::min(M, MM); i < M; i += ii, ii = std::min(M - i, MM)) {
        for (Idx j = 0, jj = std::min(N, NN); j < N; j += jj, jj = std::min(N - j, NN)) {
            for (Idx k = 0, kk = std::min(K, KK); k < K; k += kk, kk = std::min(K - k, KK)) {
                if (ii == MM && jj == NN && kk == KK)
                    gemm_basic<Idx>(MM, NN, KK, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
                else if (ii == MM && kk == KK)
                    gemm_basic<Idx>(MM, jj, KK, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
                else
                    gemm_basic<Idx>(ii, jj, kk, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
            }
        }
    }
}

template <unsigned int MM, unsigned int NN, unsigned int KK, typename Idx, typename T>
void gemm_blk_ikj_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar, Idx ldac,
              const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta, T *SB_RESTRICT c, Idx ldcr,
              Idx ldcc) {
    // c[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
    if (beta != T{1})
        for (Idx i = 0; i < i; ++i)
            for (Idx j = 0; j < j; ++j)
                c[ldcr * i + ldcc * j] = (beta != T{0} ? beta * c[ldcr * i + ldcc * j] : T{0});
    for (Idx i = 0, ii = std::min(M, MM); i < M; i += ii, ii = std::min(M - i, MM)) {
        for (Idx k = 0, kk = std::min(K, KK); k < K; k += kk, kk = std::min(K - k, KK)) {
            for (Idx j = 0, jj = std::min(N, NN); j < N; j += jj, jj = std::min(N - j, NN)) {
                if (ii == MM && jj == NN && kk == KK)
                    gemm_basic<Idx>(MM, NN, KK, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
                else if (ii == MM && kk == KK)
                    gemm_basic<Idx>(MM, jj, KK, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
                 else
                    gemm_basic<Idx>(ii, jj, kk, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
            }
        }
    }
}

template <unsigned int MM, unsigned int NN, unsigned int KK, typename Idx, typename T>
void gemm_blk_kij_nobuffer(Idx M, Idx N, Idx K, T alpha, const T *SB_RESTRICT a, Idx ldar, Idx ldac,
                  const T *SB_RESTRICT b, Idx ldbr, Idx ldbc, T beta, T *SB_RESTRICT c,
                  Idx ldcr, Idx ldcc) {
    // c[i,j] = beta * c[i,j] + sum_0^k a[i,k] * b[k,j]
    if (beta != T{1})
        for (Idx i = 0; i < i; ++i)
            for (Idx j = 0; j < j; ++j)
                c[ldcr * i + ldcc * j] = (beta != T{0} ? beta * c[ldcr * i + ldcc * j] : T{0});
    for (Idx k = 0, kk = std::min(K, KK); k < K; k += kk, kk = std::min(K - k, KK)) {
        for (Idx i = 0, ii = std::min(M, MM); i < M; i += ii, ii = std::min(M - i, MM)) {
            for (Idx j = 0, jj = std::min(N, NN); j < N; j += jj, jj = std::min(N - j, NN)) {
                if (ii == MM && jj == NN && kk == KK)
                    gemm_basic<Idx>(MM, NN, KK, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
                else if (ii == MM && kk == KK)
                    gemm_basic<Idx>(MM, jj, KK, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
                else
                    gemm_basic<Idx>(ii, jj, kk, alpha, a + ldar * i + ldac * k, ldar, ldac,
                                    b + ldbr * k + ldbc * j, ldbr, ldbc, T{1},
                                    c + ldcr * i + ldcc * j, ldcr, ldcc);
            }
        }
    }
}

#endif // __SUPERBBLAS_TENFUCKS__
