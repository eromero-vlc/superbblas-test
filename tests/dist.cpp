#include "superbblas.h"
#include "superbblas/tensor.h"
#include <vector>
#include <iostream>
#include <omp.h>

#ifdef SUPERBBLAS_USE_CUDA
#    include <thrust/device_vector.h>
#endif

using namespace superbblas;

template<unsigned int Nd> From_size<Nd> disp_tensor(Coor<Nd> dim, Coor<Nd> procs) {
    int vol_procs = (int)detail::volume<Nd>(procs);
    From_size<Nd> fs(vol_procs);
    for (int rank = 0; rank < vol_procs; ++rank) {
        for (unsigned int i = 0; i < Nd; ++i) {
            fs[rank][0][i] = dim[i] / procs[i] * rank + std::min(rank, dim[i] % procs[i]);
            fs[rank][1][i] = dim[i] / procs[i] + (dim[i] % procs[i] > rank ? 1 : 0);
        }
    }
    return fs;
}
 
int main(void) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
#else
    nprocs = 1;
    rank = 0;
#endif
 
    {
        constexpr unsigned int Nd = 2;
        using LatticeCoor = Coor<Nd>;
        using Tensor = std::vector<double>;
        const LatticeCoor dim0 = {nprocs * 5, nprocs * 2};
        From_size<Nd> p0 = disp_tensor<Nd>(dim0, {nprocs, nprocs});
        const LatticeCoor local_dim0 = p0[rank][1];
        const LatticeCoor dim1 = {nprocs * 2, nprocs * 2};
        From_size<Nd> p1 = disp_tensor<Nd>(dim1, {nprocs, 1});
        const LatticeCoor local_dim1 = p1[rank][1];
        
        unsigned int vol0 = detail::volume<Nd>(local_dim0);;
        unsigned int vol1 = detail::volume<Nd>(local_dim1);;
        Tensor t0(vol0), t1(vol1);
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;
        const LatticeCoor zero_coor = {0};
        Context ctx = createCpuContext();
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
				double *ptr0 = t0.data(), *ptr1 = t1.data();
                copy<Nd, Nd>(p0, 1, {'t', 'x'}, zero_coor, {nprocs * 2, nprocs * 2},
                             (const double **)&ptr0, &ctx, p1, 1, {'t', 'x'}, zero_coor, &ptr1,
                             &ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in permuting " << t / 10 << std::endl;
        }

        const LatticeCoor local_dimr = {2, 1};
        unsigned int volr = detail::volume<Nd>(local_dimr);
        Tensor tc(volr);
        {
            double t = omp_get_wtime();
            for (unsigned int rep = 0; rep < 10; ++rep) {
                local_contraction<Nd, Nd, Nd>("tx", local_dim1, false, t1.data(), "tx", local_dim1,
                                              false, t1.data(), "t", local_dimr, tc.data(), ctx);
            }
            t = omp_get_wtime() - t;
            std::cout << "Time in permuting " << t / 10 << std::endl;
        }
    }
#ifdef SUPERBBLAS_USE_CUDA
#endif

    return 0;
}
