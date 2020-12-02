#include "superbblas.h"
#include <vector>
#include <iostream>
#include <omp.h>

#ifdef SUPERBBLAS_USE_CUDA
#    include <thrust/device_vector.h>
#endif

using namespace superbblas;

int main(void) {
    {
        constexpr unsigned int Nd = 1;
        using LatticeCoor = Coor<Nd>;
        //using Tensor = std::vector<std::complex<double>>;
        using Tensor = std::vector<double>;
        const LatticeCoor dim = {5};
        unsigned int vol0 = detail::volume<Nd>(dim), vol1 = vol0;
        Tensor t0(vol0), t1(vol1);
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;
        const LatticeCoor zero_coor = {0};
        const LatticeCoor dim1 = {5};
        Context ctx = createCpuContext();
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                local_copy<Nd, Nd>("x", zero_coor, dim, dim, t0.data(), ctx, "x", zero_coor, dim1,
                                   t1.data(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in permuting " << t / 10 << std::endl;
        }

        Tensor tc(1);
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                local_contraction<Nd, Nd, 0>("x", dim, false, t0.data(), "x", dim1, false,
                                             t1.data(), "", {}, tc.data(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in permuting " << t / 10 << std::endl;
        }
    }
#ifdef SUPERBBLAS_USE_CUDA
    {
        constexpr unsigned int Nd = 1;
        using LatticeCoor = Coor<Nd>;
        //using Tensor = thrust::device_vector<std::complex<double>>;
        using Tensor = thrust::device_vector<double>;
        const LatticeCoor dim = {5};
        unsigned int vol0 = detail::volume<Nd>(dim), vol1 = vol0;
        Tensor t0(vol0), t1(vol1);
        thrust::counting_iterator<unsigned int> it(0);
        thrust::copy(it, it + vol0, t0.begin());
        const LatticeCoor zero_coor = {0};
        const LatticeCoor dim1 = {5};
        Context ctx = createCudaContext();
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                local_copy<Nd, Nd>("x", zero_coor, dim, dim, t0.data().get(), ctx, "x", zero_coor,
                                   dim1, t1.data().get(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in permuting " << t / 10 << std::endl;
        }

        Tensor tc(1);
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                local_contraction<Nd, Nd, 0>("x", dim, false, t0.data().get(), "x", dim1, false,
                                             t1.data().get(), "", {}, tc.data().get(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in permuting " << t / 10 << std::endl;
        }
    }
#endif

    return 0;
}
