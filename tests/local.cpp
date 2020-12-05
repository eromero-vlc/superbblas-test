#include "superbblas.h"
#include <vector>
#include <iostream>
#include <omp.h>

#ifdef SUPERBBLAS_USE_CUDA
#    include <thrust/device_vector.h>
#endif

using namespace superbblas;

int main(void) {
    constexpr unsigned int Nd = 6; // xyztsc
    //const Coor<Nd> dim = {2, 2, 2, 2, 2, 2};
    constexpr unsigned int SC = 12; // length of dimension SC
    const Coor<Nd> dim = {16, 16, 16, 32, SC, 64};
    //using T = double;
    using T = std::complex<float>;
    {
        using Tensor = std::vector<T>;
        const Coor<Nd - 1> dim0 = {dim[0], dim[1], dim[2], dim[3], dim[4]}; // xyzts
        const Coor<Nd> dim1 = {dim[3], dim[5], dim[0], dim[1], dim[2], dim[4]}; // tcxyzs
        unsigned int vol0 = detail::volume<Nd-1>(dim0);
        unsigned int vol1 = detail::volume<Nd>(dim1);
        Tensor t0(vol0), t1(vol1);
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;
        Context ctx = createCpuContext();

        std::cout << ">>> CPU tests with " << omp_get_max_threads() << " threads" << std::endl;

        std::cout << "Maximum number of elements in a tested tensor: " << vol1 << " ( "
                  << vol1 * 1.0 * sizeof(T) / 1024 / 1024 << " MiB)" << std::endl;

        // Copy tensor t0 into tensor 1 (for reference)
        double tref = 0.0;
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                for (int c = 0; c < dim[5]; ++c) {
#ifdef _OPENMP
#    pragma omp parallel for
#endif
                    for (unsigned int i = 0; i < (unsigned int)vol0; ++i)
                        t1[i + c * (unsigned int)vol0] = t0[i];
                }
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in dummy copying from xyzts to tcxyzs " << t / 10 << std::endl;
            tref = t / 10; // time in copying a whole tensor with size dim1
        }


        // Copy tensor t0 into each of the c components of tensor 1
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                for (int c = 0; c < dim[5]; ++c) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, c, 0};
                    local_copy<Nd - 1, Nd>("xyzts", from0, dim0, dim0, t0.data(), ctx, "tcxyzs",
                                           from1, dim1, t1.data(), ctx);
                }
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in copying/permuting from xyzts to tcxyzs " << t / 10
                      << " (overhead " << t / 10 / tref << " )" << std::endl;
        }

        // Copy tensor t0 into each of the c components of tensor 1 (fast)
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                for (int c = 0; c < dim[5]; ++c) {
                    const Coor<Nd - 2> from0 = {0};
                    const Coor<Nd - 1> from1 = {0, c, 0};
                    Coor<Nd - 2> dim0a;
                    std::copy_n(dim0.begin(), Nd - 2, dim0a.begin());
                    Coor<Nd - 1> dim1a;
                    std::copy_n(dim1.begin(), Nd - 1, dim1a.begin());
                    local_copy<Nd - 2, Nd - 1>("xyzt", from0, dim0a, dim0a,
                                               (const std::array<T, SC> *)t0.data(), ctx, "tcxyz",
                                               from1, dim1a, (std::array<T, SC> *)t1.data(), ctx);
                }
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in copying/permuting from xyzt to tcxyz (fast) " << t / 10
                      << " (overhead " << t / 10 / tref << " )" << std::endl;
        }

        // Shift tensor 1 on the z-direction and store it on tensor 2
        Tensor t2(vol1);
         {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                local_copy<Nd, Nd>("tcxyzs", from0, dim1, dim1, t1.data(), ctx, "tcxyzs", from1,
                                       dim1, t1.data(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in shifting " << t / 10 << std::endl;
        }

        const Coor<3> dimc = {dim[3], dim[5], dim[5]}; // tcc
        unsigned int volc = detail::volume<3>(dimc); 
        Tensor tc(volc);
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                local_contraction<Nd, Nd, 3>("tcxyzs", dim1, false, t1.data(), "tCxyzs", dim1,
                                             false, t2.data(), "tCc", dimc, tc.data(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in contracting " << t / 10 << std::endl;
        }
    }
#ifdef SUPERBBLAS_USE_CUDA
    {
        using Tensor = thrust::device_vector<T>;
        const Coor<Nd - 1> dim0 = {dim[0], dim[1], dim[2], dim[3], dim[4]}; // xyzts
        const Coor<Nd> dim1 = {dim[3], dim[5], dim[0], dim[1], dim[2], dim[4]}; // tcxyzs
        unsigned int vol0 = detail::volume<Nd-1>(dim0);
        unsigned int vol1 = detail::volume<Nd>(dim1);
        Tensor t0(vol0), t1(vol1);
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;
        Context ctx = createCudaContext();

        std::cout << ">>> GPU tests" << std::endl;

        std::cout << "Maximum number of elements in a tested tensor: " << vol1 << " ( "
                  << vol1 * 1.0 * sizeof(T) / 1024 / 1024 << " MiB)" << std::endl;

        // Copy tensor t0 into tensor 1 (for reference)
        double tref = 0.0;
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                for (int c = 0; c < dim[5]; ++c) {
                    thrust::copy_n(t0.begin(), vol0, t1.begin() + c * vol0);
                }
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in dummy copying from xyzts to tcxyzs " << t / 10 << std::endl;
            tref = t / 10; // time in copying a whole tensor with size dim1
        }


        // Copy tensor t0 into each of the c components of tensor 1
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                for (int c = 0; c < dim[5]; ++c) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, c, 0};
                    local_copy<Nd - 1, Nd>("xyzts", from0, dim0, dim0, t0.data().get(), ctx,
                                           "tcxyzs", from1, dim1, t1.data().get(), ctx);
                }
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in copying/permuting from xyzts to tcxyzs " << t / 10
                      << " (overhead " << t / 10 / tref << " )" << std::endl;
        }

        // Copy tensor t0 into each of the c components of tensor 1 (fast?)
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                for (int c = 0; c < dim[5]; ++c) {
                    const Coor<Nd - 2> from0 = {0};
                    const Coor<Nd - 1> from1 = {0, c, 0};
                    Coor<Nd - 2> dim0a;
                    std::copy_n(dim0.begin(), Nd - 2, dim0a.begin());
                    Coor<Nd - 1> dim1a;
                    std::copy_n(dim1.begin(), Nd - 1, dim1a.begin());
                    local_copy<Nd - 2, Nd - 1>(
                        "xyzt", from0, dim0a, dim0a, (const std::array<T, SC> *)t0.data().get(),
                        ctx, "tcxyz", from1, dim1a, (std::array<T, SC> *)t1.data().get(), ctx);
                }
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in copying/permuting from xyzt to tcxyz (fast?) " << t / 10
                      << " (overhead " << t / 10 / tref << " )" << std::endl;
        }

        // Shift tensor 1 on the z-direction and store it on tensor 2
        Tensor t2(vol1);
         {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                local_copy<Nd, Nd>("tcxyzs", from0, dim1, dim1, t1.data().get(), ctx, "tcxyzs",
                                   from1, dim1, t1.data().get(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in shifting " << t / 10 << std::endl;
        }

        const Coor<3> dimc = {dim[3], dim[5], dim[5]}; // tcc
        unsigned int volc = detail::volume<3>(dimc); 
        Tensor tc(volc);
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                local_contraction<Nd, Nd, 3>("tcxyzs", dim1, false, t1.data().get(), "tCxyzs", dim1,
                                             false, t2.data().get(), "tCc", dimc, tc.data().get(),
                                             ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in contracting " << t / 10 << std::endl;
        }
    }
#endif

    return 0;
}
