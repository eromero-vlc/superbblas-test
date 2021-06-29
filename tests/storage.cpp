#include "superbblas.h"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cstdio>
#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef SUPERBBLAS_USE_CUDA
#    include <thrust/device_vector.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <std::size_t Nd> PartitionStored<Nd> dist_tensor_on_root(Coor<Nd> dim, int nprocs) {
    PartitionStored<Nd> fs(nprocs);
    if (1 <= nprocs) fs[0][1] = dim;
    return fs;
}

int main(int argc, char **argv) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc;
    (void)argv;
    nprocs = 1;
    rank = 0;
#endif

    constexpr std::size_t Nd = 8; // mdtgsSnN
    constexpr unsigned int nS = 4, nG = 16; // length of dimension spin and number of gammas
    constexpr unsigned int M = 0, D = 1, T = 2, G = 3, S0 = 4, S1 = 5, N0 = 6, N1 = 7;
    Coor<Nd> dim = {1, 16, 16, nG, nS, nS, 4, 4}; // mdtgsSnN
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1, 1};
    const unsigned int nrep = 10;

    // Get options
    bool procs_was_set = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d %d %d", &dim[M], &dim[D], &dim[T], &dim[G],
                       &dim[N0]) != 5) {
                std::cerr << "--dim= should follow 5 numbers, for instance -dim='2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
            dim[N1] = dim[N0];
        } else if (std::strncmp("--procs=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, "%d", &procs[T]) != 1) {
                std::cerr << "--procs= should follow one number, for instance --procs=2"
                          << std::endl;
                return -1;
            }
            if (detail::volume(procs) != (std::size_t)nprocs) {
                std::cerr << "The total number of processes set by the option `--procs=` should "
                             "match the number of processes"
                          << std::endl;
                return -1;
            }
            procs_was_set = true;
         } else if(std::strncmp("--help", argv[i], 6) == 0) {
             std::cout << "Commandline option:\n  " << argv[0]
                       << " [--dim='m d t g n'] [--procs=t] [--help]" << std::endl;
             return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // If --procs isn't set, put all processes on the first dimension
    if (!procs_was_set) procs[T] = nprocs;

    // Show lattice dimensions and processes arrangement
    if (rank == 0) {
        std::cout << "Testing lattice dimensions mdtgsn= " << dim[M] << " " << dim[D] << " "
                  << dim[T] << " " << dim[G] << dim[S0] << " " << dim[N0] << std::endl;
        std::cout << "Processes arrangement t= " << procs[T] << std::endl;
    }

    // Samples of different S to request
    std::vector<int> nn(1, dim[N0]);
    while (nn.back() > 16) nn.push_back(nn.back() / 2);
    std::reverse(nn.begin(), nn.end());

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    // using Scalar = float;
    using Scalar = std::complex<float>;
    //using ScalarD = std::complex<double>;
    {
        using Tensor = std::vector<Scalar>;
        //using TensorD = std::vector<ScalarD>;

        // Create tensor t0 of Nd dims: a genprop
        const Coor<Nd - 1> dim0{dim[D], dim[T], dim[G], dim[S0], dim[S1], dim[N0], dim[N1]}; // dtgsSnN
        const Coor<Nd - 1> procs0 = {procs[D],  procs[T],  procs[G], procs[S0],
                                     procs[S1], procs[N0], procs[N1]}; // dtgsSnN
        PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
        const Coor<Nd - 1> local_size0 = p0[rank][1];
        std::size_t vol0 = detail::volume(local_size0);
        Tensor t0(vol0);

        // Create tensor t1 for reading the colorvec matrix
        const Coor<2> dim1 = {dim[N0], dim[N1]};
        const Coor<2> procs1 = {1, 1};
        PartitionStored<2> p1 = basic_partitioning(dim1, procs1);
        Tensor t1(detail::volume(p1[rank][1]));

        // Generate random requests
        std::vector<std::size_t> reqs(1000);
        {
            std::size_t hash = 5831;
            for (std::size_t c = 0; c < reqs.size(); ++c) {
                hash = hash * 33 + c;
                reqs[c] = hash % (vol0 / dim[N0] / dim[N1]);
            }
        }

        // Dummy initialization of t0
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;

        // Create a context in which the vectors live
        Context ctx = createCpuContext();

        if (rank == 0)
            std::cout << ">>> CPU tests with " << num_threads << " threads" << std::endl;

        std::size_t vol = detail::volume(dim);
        if (rank == 0)
            std::cout << "Maximum number of elements in a tested tensor per process: " << vol0
                      << " ( " << vol0 * 1.0 * sizeof(Scalar) / 1024 / 1024
                      << " MiB)   Expected file size: " << vol * 1.0 * sizeof(Scalar) / 1024 / 1024
                      << " MiB" << std::endl;

	const char *filename = "tensor.sed";

        // Create a file copying the content from a buffer; this should be the fastest way
	// to populate the file
        double trefw = 0.0;
        std::vector<double> trefr(nn.size(), 0.0);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                std::FILE *f = std::fopen(filename, "wb");
                for (int m = 0; m < dim[M]; ++m) {
                    if (fwrite(t0.data(), sizeof(Scalar), vol0, f) != vol0)
                        throw std::runtime_error("Error writing in a file");
                }
                std::fclose(f);
            }
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in dummy writing the tensor " << t / nrep
                          << std::endl;
            trefw = t / nrep; // time in copying a whole tensor with size dim1

            for (std::size_t nni = 0; nni < nn.size(); ++nni) {
                std::FILE *f = std::fopen(filename, "r");
                double t = w_time();
                for (unsigned int rep = 0; rep < nrep; ++rep) {
                    for (std::size_t r : reqs) {
                        if (fseek(f, sizeof(Scalar) * dim[N0] * dim[N1] * r, SEEK_SET) != 0)
                            throw std::runtime_error("Error setting file position");
                        if (fread(t1.data(), sizeof(Scalar), nn[nni] * nn[nni], f) !=
                            (std::size_t)nn[nni] * nn[nni])
                            throw std::runtime_error("Error reading in a file");
                    }
                }
                t = w_time() - t;
                if (rank == 0)
                    std::cout << "Time in dummy reading the tensor " << t / nrep << std::endl;
                trefr[nni] = t / nrep; // time in copying a whole tensor with size dim1
                std::fclose(f);
            }
        }

        // Save tensor t0 
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                Storage_handle stoh;
                create_storage<Nd, Scalar>(dim, SlowToFast, filename, "", 0,
#ifdef SUPERBBLAS_USE_MPI
                                           MPI_COMM_WORLD,
#endif
                                           &stoh);
                std::array<Coor<Nd>, 2> fs{dim, {}};
                append_block<Nd, Scalar>(&fs, 1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                         MPI_COMM_WORLD,
#endif
                                         SlowToFast);
                for (int m = 0; m < dim[M]; ++m) {
                    const Coor<Nd - 1> from0{};
                    const Coor<Nd> from1{m};
                    Scalar *ptr0 = t0.data();
                    save<Nd - 1, Nd, Scalar, Scalar>(1.0, p0.data(), 1, "dtgsSnN", from0, dim0,
                                                     (const Scalar **)&ptr0, &ctx, "mdtgsSnN",
                                                     from1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                     MPI_COMM_WORLD,
#endif
                                                     SlowToFast);
                }
		close_storage(stoh);
            }
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in writing " << t / nrep << " (overhead " << t / nrep / trefw
                          << " )" << std::endl;
        }

        // Load into tensor t1
        {
            const Coor<Nd - 2> dimr{dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // dtgsS
            Coor<Nd - 2> stride = detail::get_strides(dimr, SlowToFast);
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                Storage_handle stoh;
                open_storage<Nd, Scalar>(filename,
#ifdef SUPERBBLAS_USE_MPI
                                         MPI_COMM_WORLD,
#endif
                                         &stoh);
                for (auto n : nn) {
                    for (auto req : reqs) {
                        Coor<Nd> from0{};
                        std::copy_n(detail::index2coor(req, dimr, stride).begin(), Nd - 2,
                                    from0.begin());
                        Coor<Nd> size0{};
                        for (auto &c : size0) c = 1;
                        size0[Nd - 2] = size0[Nd - 1] = n;
                        const Coor<2> from1{};
                        Scalar *ptr1 = t1.data();
                        load<Nd, 2, Scalar, Scalar>(1.0, stoh, "mdtgsSnN", from0, size0, p1.data(),
                                                    1, "nN", from1, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                                    MPI_COMM_WORLD,
#endif
                                                    SlowToFast, Copy);
                    }
                }
                close_storage(stoh);
            }
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in writing " << t / nrep << " (overhead " << t / nrep / trefw
                          << " )" << std::endl;
        }

        if (rank == 0) reportTimings(std::cout);
        if (rank == 0) reportCacheUsage(std::cout);
    }

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
