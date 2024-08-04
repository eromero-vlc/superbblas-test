/// TENsor Fix User-defined Contraction K-dimensional Subroutine (TenFuCKS)

/// Proposed strategy for matrix-matrix multiplication:
/// - Use tensor cores

#ifndef __SUPERBBLAS_TENFUCKS_GPU__
#define __SUPERBBLAS_TENFUCKS_GPU__

#include "platform.h"

#if defined(SUPERBBLAS_USE_GPU) && !defined(SUPERBBLAS_CREATING_FLAGS) &&                          \
    !defined(SUPERBBLAS_CREATING_LIB) && !defined(SUPERBBLAS_LIB)
#    define SUPERBBLAS_GENERATE_KERNELS
#endif

#if defined(SUPERBBLAS_USE_HIP) && defined(SUPERBBLAS_GENERATE_KERNELS)

#    include <hip/hip_ext.h>

/// Detect architectures with tensor cores
/// NOTE: all from GFX9

#    if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) ||                       \
        defined(__gfx941__) || defined(__gfx942__) || defined(__gfx1100__) ||                      \
        defined(__gfx1101__) || defined(__gfx1102__)
#        define SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES
#    endif

/// Detect architectures with tensor cores for double precision
/// NOTE: all from GFX9 excepting GFX908

#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES) && !defined(__gfx908__)
#        define SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES
#    endif

#endif

#if defined(SUPERBBLAS_CREATING_LIB)
/// Generate template instantiations for bsr_kron_nxn_4x4perm functions with template parameter T

#    define DECL_BSR_KRON_nxn_4x4PERM_T(...)                                                       \
        EMIT REPLACE1(bsr_kron_nxn_4x4perm, superbblas::detail::bsr_kron_nxn_4x4perm<T>)           \
            REPLACE_T template __VA_ARGS__;

/// Generate template instantiations for available_bsr_kron_nxn_4x4perm functions with template parameter T

#    define DECL_AVAILABLE_BSR_KRON_nxn_4x4PERM_T(...)                                             \
        EMIT REPLACE1(available_bsr_kron_nxn_4x4perm,                                              \
                      superbblas::detail::available_bsr_kron_nxn_4x4perm<T>)                       \
            REPLACE_T template __VA_ARGS__;

#else
#    define DECL_BSR_KRON_nxn_4x4PERM_T(...) __VA_ARGS__
#    define DECL_AVAILABLE_BSR_KRON_nxn_4x4PERM_T(...) __VA_ARGS__
#endif

namespace superbblas {
    namespace detail {

#ifdef SUPERBBLAS_USE_HIP

        template <typename T> struct bsr_kron_3x3_4x4perm_kernel;

        __host__ __device__ inline int get_a_idx(int a_ldr, int a_ldc, int num_dirs, int color_row,
                                                 int color_col, int block_row, int dir) {
            return a_ldr * color_row + a_ldc * color_col + 3 * 3 * dir +
                   3 * 3 * num_dirs * block_row;
        }

        __host__ __device__ inline int get_xy_idx(int ldr, int ncols, int color, int spin,
                                                  int block_row, int col) {
            //return color + 3*spin + 3*4*col + ldr*block_row;
            return spin + 4 * col + 4 * ncols * color + ldr * block_row;
        }

        __host__ __device__ inline int get_a_idx_complex(int a_ldr, int a_ldc, int a_n,
                                                         int num_dirs, int color_row, int color_col,
                                                         int block_row, int dir) {
            return (a_ldr * color_row + a_ldc * color_col + a_n * a_n * dir +
                    a_n * a_n * num_dirs * block_row) *
                   2;
        }

        __host__ __device__ inline int get_xy_idx_complex(int ldr, int ncols, int color, int spin,
                                                          int block_row, int col) {
            //return (color + 3*spin + 3*4*col + ldr*block_row)*2;
            return (spin + 4 * col + 4 * ncols * color + ldr * block_row) * 2;
        }

        __host__ __device__ inline int get_jj_idx(int num_dirs, int block_row, int dir) {
            return dir + num_dirs * block_row;
        }

        template <typename T> struct bsr_kron_3x3_4x4perm_kernel {
            static constexpr bool type_available() { return false; }

            using ptr = T *;

            static dim3 block_size() { return dim3(0, 0, 0); }

            static dim3 grid_size(int, int) { return dim3(0, 0, 0); }

            __global__ static void available(int *flag) { *flag = 0; }

            __global__ static void fun(const T *a, int a_ldr, int a_ldc, int *jj, int block_rows,
                                       int num_dirs, const T *perm_scalars, const int *perm,
                                       const T *x, int ldx, T *y, int ldy, int ncols) {
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
            }
        };

        template <> struct bsr_kron_3x3_4x4perm_kernel<std::complex<double>> {
            static constexpr bool type_available() { return true; }

            using ptr = double *;

            static dim3 block_size() { return dim3(4, 4, 4); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3((num_cols + 3) / 4, block_rows, 1);
            }

            __global__ static void available(int *flag) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
                *flag = 1;
#    else
                *flag = 0;
#    endif
            }

            __global__ static void fun(const double *a, int a_ldr, int a_ldc, int *jj,
                                       int block_rows, int num_dirs, const double *perm_scalars,
                                       const int *perm, const double *x, int ldx, double *y,
                                       int ldy, int ncols) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
                (void)block_rows;
                auto col = blockIdx.x * 4 + threadIdx.y;
                auto blk_row = blockIdx.y;
                auto a_row = threadIdx.x;
                auto a_col = threadIdx.z;
                auto x_color = threadIdx.z;
                auto x_spin = threadIdx.x;
                auto y_color = threadIdx.z;
                auto y_spin = threadIdx.x;
                double y_val_r = 0.0, y_val_i = 0.0;
                for (int dir = 0; dir < num_dirs; ++dir) {
                    // read a
                    bool a_is_zero = (a_row == 3 || a_col == 3);
                    int a_idx =
                        get_a_idx_complex(a_ldr, a_ldc, 3, num_dirs, a_row, a_col, blk_row, dir);
                    double a_val_r = 0.0, a_val_i = 0.0;
                    if (!a_is_zero) a_val_r = a[a_idx], a_val_i = a[a_idx + 1];

                    // read x
                    bool x_is_zero = (x_color == 3 || col >= ncols);
                    int x_idx = get_xy_idx_complex(ldx, ncols, x_color, perm[4 * dir + x_spin],
                                                   jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                    double x_val_r = 0.0, x_val_i = 0.0;
                    int s_dir = (4 * dir + x_spin) * 2;
                    const double s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                    if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                    x_val_r = x_val_r * s_r - x_val_i * s_i;
                    x_val_i = x_val_r * s_i + x_val_i * s_r;

                    // a[real] times x[real] -> y[real]
                    y_val_r =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_r, x_val_r, y_val_r, 0, 0, 0);

                    // a[real] times x[imag] -> y[imag]
                    y_val_i =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_r, x_val_i, y_val_i, 0, 0, 0);

                    // a[imag] times x[real] -> y[imag]
                    y_val_i =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_i, x_val_r, y_val_i, 0, 0, 0);

                    // a[imag] times x[imag] -> y[real]
                    y_val_r =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(-a_val_i, x_val_i, y_val_r, 0, 0, 0);
                }
                bool y_is_zero = (y_color == 3 || col >= ncols);
                int y_idx = get_xy_idx_complex(ldy, ncols, y_color, y_spin, blk_row, col);
                if (!y_is_zero) y[y_idx] = y_val_r, y[y_idx + 1] = y_val_i;
#    else
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
#    endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
            }
        };

        template <> struct bsr_kron_3x3_4x4perm_kernel<std::complex<float>> {
            static constexpr bool type_available() { return true; }

            using ptr = float *;

            static dim3 block_size() { return dim3(4, 16, 1); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3((num_cols + 15) / 16, block_rows, 1);
            }

            __global__ static void available(int *flag) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
                *flag = 1;
#    else
                *flag = 0;
#    endif
            }

            __global__ static void fun(const float *a, int a_ldr, int a_ldc, int *jj,
                                       int block_rows, int num_dirs, const float *perm_scalars,
                                       const int *perm, const float *x, int ldx, float *y, int ldy,
                                       int ncols) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
                (void)block_rows;
                using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
                auto col = blockIdx.x * 16 + threadIdx.y;
                auto blk_row = blockIdx.y;
                auto a_row = threadIdx.x;
                auto x_spin = threadIdx.x;
                auto y_spin = threadIdx.x;
                float4 y_val_r = {0}, y_val_i = {0};
                for (int dir = 0; dir < num_dirs; ++dir) {
                    for (int k = 0; k < 3; ++k) {
                        // read a
                        bool a_is_zero = (a_row == 3);
                        int a_idx =
                            get_a_idx_complex(a_ldr, a_ldc, 3, num_dirs, a_row, k, blk_row, dir);
                        float a_val_r = 0.0, a_val_i = 0.0;
                        if (!a_is_zero) a_val_r = a[a_idx], a_val_i = a[a_idx + 1];

                        // read x
                        bool x_is_zero = (col >= ncols);
                        int x_idx = get_xy_idx_complex(ldx, ncols, k, perm[4 * dir + x_spin],
                                                       jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                        float x_val_r = 0.0, x_val_i = 0.0;
                        int s_dir = (4 * dir + x_spin) * 2;
                        const float s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                        if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                        x_val_r = x_val_r * s_r - x_val_i * s_i;
                        x_val_i = x_val_r * s_i + x_val_i * s_r;

                        // a[real] times x[real] -> y[real]
                        y_val_r =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_r, x_val_r, y_val_r, 0, 0, 0);

                        // a[real] times x[imag] -> y[imag]
                        y_val_i =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_r, x_val_i, y_val_i, 0, 0, 0);

                        // a[imag] times x[real] -> y[imag]
                        y_val_i =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_i, x_val_r, y_val_i, 0, 0, 0);

                        // a[imag] times x[imag] -> y[real]
                        y_val_r =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(-a_val_i, x_val_i, y_val_r, 0, 0, 0);
                    }
                }
                bool y_is_zero = (col >= ncols);
                if (!y_is_zero) {
                    for (int k = 0; k < 3; ++k) {
                        int y_idx = get_xy_idx_complex(ldy, ncols, k, y_spin, blk_row, col);
                        y[y_idx] = y_val_r[k];
                        y[y_idx + 1] = y_val_i[k];
                    }
                }
#    else
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
#    endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
            }
        };

        template <typename T>
        void bsr_kron_3x3_4x4perm(const T *a, int a_ldr, int a_ldc, int *jj, int block_rows,
                                  int num_dirs, const T *perm_scalars, const int *perm, const T *x,
                                  int ldx, T *y, int ldy, int ncols, Gpu xpu) {
            if (!bsr_kron_3x3_4x4perm_kernel<T>::type_available()) throw std::runtime_error("wtf!");
            using ptr = typename bsr_kron_3x3_4x4perm_kernel<T>::ptr;
            hipExtLaunchKernelGGL(bsr_kron_3x3_4x4perm_kernel<T>::fun,
                                  bsr_kron_3x3_4x4perm_kernel<T>::grid_size(block_rows, ncols),
                                  bsr_kron_3x3_4x4perm_kernel<T>::block_size(),
                                  0,              // sharedMemBytes
                                  getStream(xpu), // stream
                                  0,              // Event start
                                  0,              // event stop
                                  0,              // flags
                                  (ptr)a, a_ldr, a_ldc, jj, 1, num_dirs, (ptr)perm_scalars, perm,
                                  (const ptr)x, ldx, (ptr)y, ldy, ncols);
            gpuCheck(hipGetLastError());
        }

        template <typename T> struct bsr_kron_nxn_4x4perm_kernel {
            static constexpr bool type_available() { return false; }

            using ptr = T *;

            static dim3 block_size() { return dim3(0, 0, 0); }

            static dim3 grid_size(int, int) { return dim3(0, 0, 0); }

            __global__ static void available(int *flag) { *flag = 0; }

            __global__ static void fun(const T *a, int a_ldr, int a_ldc, int a_n, int *jj,
                                       int block_rows, int num_dirs, const T *perm_scalars,
                                       const int *perm, const T *x, int ldx, T *y, int ldy,
                                       int ncols) {
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)a_n;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
            }
        };

        template <> struct bsr_kron_nxn_4x4perm_kernel<std::complex<double>> {
            static constexpr bool type_available() { return true; }

            using ptr = double *;

            static dim3 block_size() { return dim3(4, 4, 4); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3((num_cols + 3) / 4, block_rows, 1);
            }

            __global__ static void available(int *flag) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
                *flag = 1;
#    else
                *flag = 0;
#    endif
            }

            __global__ static void fun(const double *a, int a_ldr, int a_ldc, int a_n, int *jj,
                                       int block_rows, int num_dirs, const double *perm_scalars,
                                       const int *perm, const double *x, int ldx, double *y,
                                       int ldy, int ncols) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
                (void)block_rows;
                auto col = blockIdx.x * 4 + threadIdx.y;
                auto blk_row = blockIdx.y;
                auto a_row0 = threadIdx.x;
                auto a_col0 = threadIdx.z;
                auto x_spin = threadIdx.x;
                auto y_color0 = threadIdx.z;
                auto y_spin = threadIdx.x;
                for (int a_row = a_row0, y_color = y_color0; a_row < a_n + 4 - 1;
                     a_row += 4, y_color += 4) {
                    double y_val_r = 0.0, y_val_i = 0.0;
                    for (int dir = 0; dir < num_dirs; ++dir) {
                        for (int a_col = a_col0; a_col < a_n + 4 - 1; a_col += 4) {
                            // read a
                            bool a_is_zero = (a_row >= a_n || a_col >= a_n);
                            int a_idx = get_a_idx_complex(a_ldr, a_ldc, a_n, num_dirs, a_row, a_col,
                                                          blk_row, dir);
                            double a_val_r = 0.0, a_val_i = 0.0;
                            if (!a_is_zero) a_val_r = a[a_idx], a_val_i = a[a_idx + 1];

                            // read x
                            auto x_color = a_col;
                            bool x_is_zero = (x_color >= a_n || col >= ncols);
                            int x_idx =
                                get_xy_idx_complex(ldx, ncols, x_color, a_n, perm[4 * dir + x_spin],
                                                   jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                            double x_val_r = 0.0, x_val_i = 0.0;
                            int s_dir = (4 * dir + x_spin) * 2;
                            const double s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                            if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                            x_val_r = x_val_r * s_r - x_val_i * s_i;
                            x_val_i = x_val_r * s_i + x_val_i * s_r;

                            // a[real] times x[real] -> y[real]
                            y_val_r = __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_r, x_val_r, y_val_r,
                                                                         0, 0, 0);

                            // a[real] times x[imag] -> y[imag]
                            y_val_i = __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_r, x_val_i, y_val_i,
                                                                         0, 0, 0);

                            // a[imag] times x[real] -> y[imag]
                            y_val_i = __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_i, x_val_r, y_val_i,
                                                                         0, 0, 0);

                            // a[imag] times x[imag] -> y[real]
                            y_val_r = __builtin_amdgcn_mfma_f64_4x4x4f64(-a_val_i, x_val_i, y_val_r,
                                                                         0, 0, 0);
                        }
                    }
                    bool y_is_zero = (y_color >= a_n || col >= ncols);
                    int y_idx = get_xy_idx_complex(ldy, ncols, y_color, a_n, y_spin, blk_row, col);
                    if (!y_is_zero) y[y_idx] = y_val_r, y[y_idx + 1] = y_val_i;
                }
#    else
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)a_n;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
#    endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
            }
        };

        template <> struct bsr_kron_nxn_4x4perm_kernel<std::complex<float>> {
            static constexpr bool type_available() { return true; }

            using ptr = float *;

            static dim3 block_size() { return dim3(4, 16, 1); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3((num_cols + 15) / 16, block_rows, 1);
            }

            __global__ static void available(int *flag) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
                *flag = 1;
#    else
                *flag = 0;
#    endif
            }

            __global__ static void fun(const float *a, int a_ldr, int a_ldc, int a_n, int *jj,
                                       int block_rows, int num_dirs, const float *perm_scalars,
                                       const int *perm, const float *x, int ldx, float *y, int ldy,
                                       int ncols) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
                (void)block_rows;
                using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
                auto col = blockIdx.x * 16 + threadIdx.y;
                auto blk_row = blockIdx.y;
                auto a_row = threadIdx.x;
                auto x_spin = threadIdx.x;
                auto y_spin = threadIdx.x;
                for (int a_row = a_row0, y_color0 = 0; a_row < a_n + 4 - 1;
                     a_row += 4, y_color0 += 4) {
                    float4 y_val_r = {0}, y_val_i = {0};
                    for (int dir = 0; dir < num_dirs; ++dir) {
                        for (int k = 0; k < a_n; ++k) {
                            // read a
                            bool a_is_zero = (a_row >= a_n);
                            int a_idx = get_a_idx_complex(a_ldr, a_ldc, a_n, num_dirs, a_row, k,
                                                          blk_row, dir);
                            float a_val_r = 0.0, a_val_i = 0.0;
                            if (!a_is_zero) a_val_r = a[a_idx], a_val_i = a[a_idx + 1];

                            // read x
                            bool x_is_zero = (col >= ncols);
                            int x_idx =
                                get_xy_idx_complex(ldx, ncols, k, perm[4 * dir + x_spin],
                                                   jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                            float x_val_r = 0.0, x_val_i = 0.0;
                            int s_dir = (4 * dir + x_spin) * 2;
                            const float s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                            if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                            x_val_r = x_val_r * s_r - x_val_i * s_i;
                            x_val_i = x_val_r * s_i + x_val_i * s_r;

                            // a[real] times x[real] -> y[real]
                            y_val_r = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_r, x_val_r, y_val_r,
                                                                         0, 0, 0);

                            // a[real] times x[imag] -> y[imag]
                            y_val_i = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_r, x_val_i, y_val_i,
                                                                         0, 0, 0);

                            // a[imag] times x[real] -> y[imag]
                            y_val_i = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_i, x_val_r, y_val_i,
                                                                         0, 0, 0);

                            // a[imag] times x[imag] -> y[real]
                            y_val_r = __builtin_amdgcn_mfma_f32_4x4x1f32(-a_val_i, x_val_i, y_val_r,
                                                                         0, 0, 0);
                        }
                    }
                    bool y_is_zero = (col >= ncols);
                    if (!y_is_zero) {
                        for (int y_color1 = 0; y_color1 < 4 && y_color0 + y_color1 < a_n;
                             ++y_color1) {
                            int y_idx = get_xy_idx_complex(ldy, ncols, y_color0 + y_color1, y_spin,
                                                           blk_row, col);
                            y[y_idx] = y_val_r[y_color1];
                            y[y_idx + 1] = y_val_i[y_color1];
                        }
                    }
                }
#    else
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)a_n;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
#    endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
            }
        };

        template <typename T>
        DECL_BSR_KRON_nxn_4x4PERM_T(void bsr_kron_nxn_4x4perm(const T *a, int a_ldr, int a_ldc,
                                                              int a_n, int *jj, int block_rows,
                                                              int num_dirs, const T *perm_scalars,
                                                              const int *perm, const T *x, int ldx,
                                                              T *y, int ldy, int ncols, Gpu xpu))
            IMPL({
                if (a_n == 3) {
                    bsr_kron_3x3_4x4perm(a, a_ldr, a_ldc, jj, block_rows, num_dirs, perm_scalars,
                                         perm, x, ldx, y, ldy, ncols, xpu);
                    return;
                }
                if (!bsr_kron_nxn_4x4perm_kernel<T>::type_available())
                    throw std::runtime_error("wtf!");
                using ptr = typename bsr_kron_nxn_4x4perm_kernel<T>::ptr;
                hipExtLaunchKernelGGL(bsr_kron_nxn_4x4perm_kernel<T>::fun,
                                      bsr_kron_nxn_4x4perm_kernel<T>::grid_size(block_rows, ncols),
                                      bsr_kron_nxn_4x4perm_kernel<T>::block_size(),
                                      0,              // sharedMemBytes
                                      getStream(xpu), // stream
                                      0,              // Event start
                                      0,              // event stop
                                      0,              // flags
                                      (ptr)a, a_ldr, a_ldc, a_n, jj, 1, num_dirs, (ptr)perm_scalars,
                                      perm, (const ptr)x, ldx, (ptr)y, ldy, ncols);
                gpuCheck(hipGetLastError());
            })

        template <typename T>
        DECL_AVAILABLE_BSR_KRON_nxn_4x4PERM_T(bool available_bsr_kron_nxn_4x4perm(const Gpu &xpu))
            IMPL({
                if (!bsr_kron_nxn_4x4perm_kernel<T>::type_available()) return false;
                setDevice(xpu);
                int *flag;
                gpuCheck(hipMalloc(&flag, sizeof(int)));
                bsr_kron_nxn_4x4perm_kernel<T>::available<<<1, 1>>>(flag);
                gpuCheck(hipGetLastError());
                int flag_host = 0;
                gpuCheck(hipMemcpy(&flag_host, flag, sizeof(int), hipMemcpyDeviceToHost));
                gpuCheck(hipFree(flag));
                return flag_host != 0;
            })
#endif // SUPERBBLAS_USE_HIP
    }
}
#endif // __SUPERBBLAS_TENFUCKS_GPU__
