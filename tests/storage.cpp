#include "superbblas.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <std::size_t Nd> PartitionStored<Nd> dist_tensor_on_root(Coor<Nd> dim, int nprocs) {
    PartitionStored<Nd> fs(nprocs);
    if (1 <= nprocs) fs[0][1] = dim;
    return fs;
}

void test_checksum() {
    const char *data = "Quixote was a great guy";
    const int n = std::strlen(data);
    checksum_t checksum_val0 = do_checksum(data, n);
    checksum_t checksum_val1 = 0;
    for (int i = 0; i < n; ++i) checksum_val1 = do_checksum(&data[i], 1, 0, checksum_val1);
    checksum_t checksum_val2 = 0;
    for (int i = 0; i < n; i += 2)
        checksum_val2 = do_checksum(&data[i], std::min(n - i, 2), 0, checksum_val2);
    if (checksum_val0 != checksum_val1 || checksum_val0 != checksum_val2)
        throw std::runtime_error("Checksum isn't associative");
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

    test_checksum();

    constexpr std::size_t Nd = 8;           // mdtgsSnN
    constexpr unsigned int nS = 4, nG = 16; // length of dimension spin and number of gammas
    constexpr unsigned int M = 0, D = 1, T = 2, G = 3, S0 = 4, S1 = 5, N0 = 6, N1 = 7;
    Coor<Nd> dim = {2, 3, 5, nG, nS, nS, 4, 4}; // mdtgsSnN
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1, 1};
    const unsigned int nrep = getDebugLevel() == 0 ? 10 : 1;
    const unsigned int num_reqs = 1000;
    std::string metadata = "S3T format!";
    const char *filename = "tensor.s3t";
    const char *filename_sp = "tensor_sp.s3t";

    // Get options
    bool procs_was_set = false;
    checksum_type checksum = NoChecksum;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6,
                       std::is_same<superbblas::IndexType, int>::value ? "%d %d %d %d %d"
                                                                       : "%ld %ld %ld %ld %ld",
                       &dim[M], &dim[D], &dim[T], &dim[G], &dim[N0]) != 5) {
                std::cerr << "--dim= should follow 5 numbers, for instance -dim='2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
            dim[N1] = dim[N0];
        } else if (std::strncmp("--procs=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, std::is_same<superbblas::IndexType, int>::value ? "%d" : "%ld",
                       &procs[T]) != 1) {
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
        } else if (std::strncmp("--checksum=", argv[i], 11) == 0) {
            int checksum_d = 0;
            if (sscanf(argv[i] + 11, "%d", &checksum_d) != 1 || checksum_d < 0 || checksum_d > 2) {
                std::cerr << "--checksum= should follow 0, 1, or 2, for instance --checksum=1"
                          << std::endl;
                return -1;
            }
            checksum =
                (checksum_d == 0 ? NoChecksum : (checksum_d == 1 ? GlobalChecksum : BlockChecksum));
            dim[N1] = dim[N0];
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
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
                  << dim[T] << " " << dim[G] << " " << dim[S0] << " " << dim[N0] << std::endl;
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
    using Scalar = std::complex<double>;
    //using ScalarD = std::complex<double>;
    {
        using Tensor = std::vector<Scalar>;
        //using TensorD = std::vector<ScalarD>;

        // Create tensor t0 of Nd dims: a genprop
        const Coor<Nd - 1> dim0{dim[D],  dim[T],  dim[G], dim[S0],
                                dim[S1], dim[N0], dim[N1]}; // dtgsSnN
        const Coor<Nd - 1> procs0 = {procs[D],  procs[T],  procs[G], procs[S0],
                                     procs[S1], procs[N0], procs[N1]}; // dtgsSnN
        PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
        const Coor<Nd - 1> local_size0 = p0[rank][1];

        // Generate random requests
        std::size_t vol = detail::volume(dim);
        std::vector<std::size_t> reqs(num_reqs);
        {
            std::size_t hash = 5831;
            for (std::size_t c = 0; c < reqs.size(); ++c) {
                hash = hash * 33 + c;
                reqs[c] = hash % (vol / dim[N0] / dim[N1]);
            }
        }

        // Create a context in which the vectors live
        Context ctx = createCpuContext();

        if (rank == 0) std::cout << ">>> CPU tests with " << num_threads << " threads" << std::endl;

        if (rank == 0)
            std::cout << "Maximum number of elements in a tested tensor per process: "
                      << detail::volume(local_size0) << " ( "
                      << detail::volume(local_size0) * 1.0 * sizeof(Scalar) / 1024 / 1024
                      << " MiB)   Expected file size: " << vol * 1.0 * sizeof(Scalar) / 1024 / 1024
                      << " MiB" << std::endl;

        // Create a file copying the content from a buffer; this should be the fastest way
        // to populate the file
        double trefw = 0.0;
        const bool dowrite = true;
        std::vector<double> trefr(nn.size(), 0.0);
        if (rank == 0) {
            std::FILE *f = std::fopen(filename, "w+");
            if (f == nullptr) superbblas::detail::gen_error("Error opening file for writing");

            // Dummy initialization of t0
            std::size_t vol0 = detail::volume(dim) / dim[M];
            Tensor t0(vol0);
            for (unsigned int i = 0; i < vol0; i++) t0[i] = i;

            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int m = 0; m < dim[M]; ++m) {
                    if (std::fwrite(t0.data(), sizeof(Scalar), vol0, f) != vol0)
                        superbblas::detail::gen_error("Error writing in a file");
                }
            }
            std::fflush(f);
            t = w_time() - t;
            std::cout << "Time in dummy writing the tensor " << t / nrep << " s  "
                      << dim[M] * vol0 * sizeof(Scalar) * nrep / t / 1024 / 1024 << " MiB/s"
                      << std::endl;
            trefw = t / nrep; // time in copying a whole tensor with size dim1

            std::fclose(f);
            f = std::fopen(filename, "rb");
            if (f == nullptr) superbblas::detail::gen_error("Error opening file for reading");

            for (std::size_t nni = 0; nni < nn.size(); ++nni) {
                Tensor t1(nn[nni] * nn[nni]);
                double t = w_time();
                for (unsigned int rep = 0; rep < nrep; ++rep) {
                    for (std::size_t r : reqs) {
                        if (std::fseek(f, sizeof(Scalar) * dim[N0] * dim[N1] * r, SEEK_SET) != 0)
                            throw std::runtime_error("Error setting file position");
                        std::size_t fread_out =
                            std::fread(t1.data(), sizeof(Scalar), nn[nni] * nn[nni], f);
                        if (fread_out != (std::size_t)nn[nni] * nn[nni])
                            superbblas::detail::gen_error("Error reading in a file");
                    }
                }
                t = w_time() - t;
                std::cout << "Time in dummy reading the tensor with " << nn[nni] << "^2 elements "
                          << t / nrep << " s  "
                          << nn[nni] * nn[nni] * sizeof(Scalar) * nrep * reqs.size() / t / 1024 /
                                 1024
                          << " MiB/s" << std::endl;
                trefr[nni] = t / nrep; // time in copying a whole tensor with size dim1
            }

            std::fclose(f);
        }

#ifdef SUPERBBLAS_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Save tensor t0
        if (dowrite) {
            std::size_t vol0 = detail::volume(local_size0);
            Tensor t0(vol0);
            for (unsigned int i = 0; i < vol0; i++) t0[i] = i;

            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                Storage_handle stoh;
                create_storage<Nd, Scalar>(dim, SlowToFast, filename, metadata.c_str(),
                                           metadata.size(), checksum,
#ifdef SUPERBBLAS_USE_MPI
                                           MPI_COMM_WORLD,
#endif
                                           &stoh);
                std::array<Coor<Nd>, 2> fs{Coor<Nd>{{}}, dim};
                append_blocks<Nd, Scalar>(&fs, 1, dim, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                          MPI_COMM_WORLD,
#endif
                                          SlowToFast);
                for (int m = 0; m < dim[M]; ++m) {
                    const Coor<Nd - 1> from0{};
                    const Coor<Nd> from1{m};
                    Scalar *ptr0 = t0.data();
                    save<Nd - 1, Nd, Scalar, Scalar>(1.0, p0.data(), 1, "dtgsSnN", from0, dim0,
                                                     dim0, (const Scalar **)&ptr0, &ctx, "mdtgsSnN",
                                                     from1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                     MPI_COMM_WORLD,
#endif
                                                     SlowToFast);
                }
                close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                          ,
                                          MPI_COMM_WORLD
#endif
                );
            }
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in writing " << t / nrep << " s (overhead " << t / nrep / trefw
                          << " )" << std::endl;
        }

        Storage_handle stoh;
        open_storage<Nd, Scalar>(filename,
#ifdef SUPERBBLAS_USE_MPI
                                 MPI_COMM_WORLD,
#endif
                                 &stoh);

        // Check storage
        check_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );

        // Load into tensor t1
        if (dowrite) {
            const Coor<Nd - 2> dimr{dim[M], dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // mdtgsS
            Coor<Nd - 2, std::size_t> stride = detail::get_strides<std::size_t>(dimr, SlowToFast);

            for (std::size_t nni = 0; nni < nn.size(); ++nni) {
                int n = nn[nni];
                // Create tensor t1 for reading the genprop on root process
                Coor<2> dim1{{n, n}};
                PartitionStored<2> p1(nprocs);
                p1[0][1] = dim1;
                std::size_t vol1 = detail::volume(p1[rank][1]);
                Tensor t1(vol1);

                double t = w_time();
                for (unsigned int rep = 0; rep < nrep; ++rep) {
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
                                                    1, "nN", from1, dim1, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                                    MPI_COMM_WORLD,
#endif
                                                    SlowToFast, Copy);
                    }
                }
                t = w_time() - t;
                if (rank == 0)
                    std::cout << "Time in reading the tensor with " << n << "^2 elements "
                              << t / nrep << " s  "
                              << " (overhead " << t / nrep / trefr[nni] << " )" << std::endl;
            }
        }

        close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );

        for (CoorOrder co : std::array<CoorOrder, 2>{SlowToFast, FastToSlow}) {
            Storage_handle stoh;
            create_storage<Nd, Scalar>(dim, co, filename, metadata.c_str(), metadata.size(),
                                       checksum,
#ifdef SUPERBBLAS_USE_MPI
                                       MPI_COMM_WORLD,
#endif
                                       &stoh);
            std::array<Coor<Nd>, 2> fs{Coor<Nd>{}, dim};
            append_blocks<Nd, Scalar>(&fs, 1, dim, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                      MPI_COMM_WORLD,
#endif
                                      co);

            // Store proper values to test the storage
            {
                PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
                const Coor<Nd - 1> local_size0 = p0[rank][1];
                std::size_t vol0 = detail::volume(local_size0);
                Tensor t0(vol0);

                Coor<Nd - 1, std::size_t> local_strides0 =
                    detail::get_strides<std::size_t>(local_size0, co);
                Coor<Nd, std::size_t> strides1 = detail::get_strides<std::size_t>(dim, co);
                for (int m = 0; m < dim[M]; ++m) {
                    const Coor<Nd - 1> from0{};
                    const Coor<Nd> from1{m};
                    Scalar *ptr0 = t0.data();
                    for (std::size_t i = 0; i < vol0; ++i) {
                        Coor<Nd - 1> c0 = index2coor(i, local_size0, local_strides0) + p0[rank][0];
                        Coor<Nd> c{m};
                        std::copy_n(c0.begin(), Nd - 1, c.begin() + 1);
                        t0[i] = coor2index(c, dim, strides1);
                    }
                    save<Nd - 1, Nd, Scalar, Scalar>(1.0, p0.data(), 1, "dtgsSnN", from0, dim0,
                                                     dim0, (const Scalar **)&ptr0, &ctx, "mdtgsSnN",
                                                     from1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                     MPI_COMM_WORLD,
#endif
                                                     co);
                }

                flush_storage(stoh);

                if (rank == 0) {
                    // The data of the only block should contain the numbers from zero to vol
                    std::size_t padding_size = (8 - metadata.size() % 8) % 8;
                    std::size_t header_size = sizeof(int) * 6 + metadata.size() + padding_size +
                                              sizeof(double) * (Nd + 1);
                    std::size_t disp = header_size + sizeof(double) * (2 + Nd * 2);
                    std::ifstream f(filename, std::ios::binary);
                    f.seekg(disp);
                    Scalar s;
                    for (std::size_t i = 0; i < vol; ++i) {
                        f.read((char *)&s, sizeof(s));
                        if (i != s.real()) throw std::runtime_error("Failing reading from storage");
                    }
                    f.close();
                }
            }

            // Check metadata
            {
                values_datatype dtype;
                std::vector<char> metadata0;
                std::vector<IndexType> dim0;
                read_storage_header(filename, co, dtype, metadata0, dim0);

                if (std::string(metadata0.begin(), metadata0.end()) != metadata)
                    throw std::runtime_error("Error recovering metadata");

                if (std::vector<IndexType>(dim.begin(), dim.end()) != dim0)
                    throw std::runtime_error("Error recovering tensor dimensions");

                if (dtype != CDOUBLE)
                    throw std::runtime_error("Error recovering the tensor datatype");
            }

            // Test the readings
            {
                const Coor<Nd - 2> dimr{dim[M], dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // mdtgsS
                Coor<Nd - 2, std::size_t> stridesr = detail::get_strides<std::size_t>(dimr, co);
                Coor<Nd, std::size_t> strides = detail::get_strides<std::size_t>(dim, co);

                for (auto n : nn) {
                    Coor<2> dimnn{n, n};
                    Coor<2, std::size_t> stridesnn = detail::get_strides<std::size_t>(dimnn, co);

                    // Create tensor t1 for reading the genprop on root process
                    PartitionStored<2> p1(nprocs);
                    p1[0][1] = dimnn;
                    std::size_t vol1 = detail::volume(p1[rank][1]);
                    Tensor t1(vol1);

                    for (auto req : reqs) {
                        Coor<Nd> from0{};
                        std::copy_n(detail::index2coor(req, dimr, stridesr).begin(), Nd - 2,
                                    from0.begin());
                        Coor<Nd> size0{};
                        for (auto &c : size0) c = 1;
                        size0[Nd - 2] = size0[Nd - 1] = n;
                        const Coor<2> from1{};
                        Scalar *ptr1 = t1.data();
                        for (std::size_t i = 0; i < vol1; ++i) t1[i] = -1;
                        load<Nd, 2, Scalar, Scalar>(1.0, stoh, "mdtgsSnN", from0, size0, p1.data(),
                                                    1, "nN", from1, dimnn, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                                    MPI_COMM_WORLD,
#endif
                                                    co, Copy);
                        for (std::size_t i = 0; i < vol1; ++i) {
                            Coor<2> cnn = index2coor(i, dimnn, stridesnn) + p1[rank][0];
                            Coor<Nd> c{};
                            c[Nd - 2] = cnn[0];
                            c[Nd - 1] = cnn[1];
                            if (t1[i].real() != coor2index(from0 + c, dim, strides))
                                throw std::runtime_error("Storage failed!");
                        }
                    }
                }
            }

            close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                      ,
                                      MPI_COMM_WORLD
#endif
            );

            create_storage<Nd, Scalar>(dim, co, filename_sp, metadata.c_str(), metadata.size(),
                                       checksum,
#ifdef SUPERBBLAS_USE_MPI
                                       MPI_COMM_WORLD,
#endif
                                       &stoh);

            // Store proper values to test the sparse storage
            {
                PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
                const Coor<Nd - 1> local_size0 = p0[rank][1];
                std::size_t vol0 = detail::volume(local_size0);
                Tensor t0(vol0);

                Coor<Nd - 1,std::size_t> local_strides0 = detail::get_strides<std::size_t>(local_size0, co);
                Coor<Nd, std::size_t> strides1 = detail::get_strides<std::size_t>(dim, co);
                for (int m = 0; m < dim[M]; ++m) {
                    const Coor<Nd - 1> from0{};
                    const Coor<Nd> from1{m};

                    append_blocks<Nd - 1, Nd, Scalar>(p0.data(), nprocs, "dtgsSnN", {}, dim0, dim0,
                                                      "mdtgsSnN", from1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                      MPI_COMM_WORLD,
#endif
                                                      co);

                    Scalar *ptr0 = t0.data();
                    for (std::size_t i = 0; i < vol0; ++i) {
                        Coor<Nd - 1> c0 = index2coor(i, local_size0, local_strides0) + p0[rank][0];
                        Coor<Nd> c{m};
                        std::copy_n(c0.begin(), Nd - 1, c.begin() + 1);
                        t0[i] = coor2index(c, dim, strides1);
                    }
                    save<Nd - 1, Nd, Scalar, Scalar>(1.0, p0.data(), 1, "dtgsSnN", from0, dim0,
                                                     dim0, (const Scalar **)&ptr0, &ctx, "mdtgsSnN",
                                                     from1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                     MPI_COMM_WORLD,
#endif
                                                     co);
                }
            }

            // Test the readings
            {
                const Coor<Nd - 2> dimr{dim[M], dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // mdtgsS
                Coor<Nd - 2, std::size_t> stridesr = detail::get_strides<std::size_t>(dimr, co);
                Coor<Nd, std::size_t> strides = detail::get_strides<std::size_t>(dim, co);

                for (auto n : nn) {
                    Coor<2> dimnn{n, n};
                    Coor<2, std::size_t> stridesnn = detail::get_strides<std::size_t>(dimnn, co);

                    // Create tensor t1 for reading the genprop on root process
                    PartitionStored<2> p1(nprocs);
                    p1[0][1] = dimnn;
                    std::size_t vol1 = detail::volume(p1[rank][1]);
                    Tensor t1(vol1);

                    for (auto req : reqs) {
                        Coor<Nd> from0{};
                        std::copy_n(detail::index2coor(req, dimr, stridesr).begin(), Nd - 2,
                                    from0.begin());
                        Coor<Nd> size0{};
                        for (auto &c : size0) c = 1;
                        size0[Nd - 2] = size0[Nd - 1] = n;
                        const Coor<2> from1{};
                        Scalar *ptr1 = t1.data();
                        for (std::size_t i = 0; i < vol1; ++i) t1[i] = -1;
                        load<Nd, 2, Scalar, Scalar>(1.0, stoh, "mdtgsSnN", from0, size0, p1.data(),
                                                    1, "nN", from1, dimnn, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                                    MPI_COMM_WORLD,
#endif
                                                    co, Copy);
                        for (std::size_t i = 0; i < vol1; ++i) {
                            Coor<2> cnn = index2coor(i, dimnn, stridesnn) + p1[rank][0];
                            Coor<Nd> c{};
                            c[Nd - 2] = cnn[0];
                            c[Nd - 1] = cnn[1];
                            if (t1[i].real() != coor2index(from0 + c, dim, strides))
                                throw std::runtime_error("Storage failed!");
                        }
                    }
                }
            }

            close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                      ,
                                      MPI_COMM_WORLD
#endif
            );
        }

        if (rank == 0) reportTimings(std::cout);
        if (rank == 0) reportCacheUsage(std::cout);
    }
#ifdef SUPERBBLAS_USE_CUDA
    {
        resetTimings();

        using Tensor = thrust::device_vector<Scalar>;

        // Create tensor t0 of Nd dims: a genprop
        const Coor<Nd - 1> dim0{dim[D],  dim[T],  dim[G], dim[S0],
                                dim[S1], dim[N0], dim[N1]}; // dtgsSnN
        const Coor<Nd - 1> procs0 = {procs[D],  procs[T],  procs[G], procs[S0],
                                     procs[S1], procs[N0], procs[N1]}; // dtgsSnN
        PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);

        // Generate random requests
        std::size_t vol = detail::volume(dim);
        std::vector<std::size_t> reqs(num_reqs);
        {
            std::size_t hash = 5831;
            for (std::size_t c = 0; c < reqs.size(); ++c) {
                hash = hash * 33 + c;
                reqs[c] = hash % (vol / dim[N0] / dim[N1]);
            }
        }

        // Create a context in which the vectors live
        Context ctx = createCudaContext();

        Storage_handle stoh;
        create_storage<Nd, Scalar>(dim, SlowToFast, filename, metadata.c_str(), metadata.size(),
                                   checksum,
#    ifdef SUPERBBLAS_USE_MPI
                                   MPI_COMM_WORLD,
#    endif
                                   &stoh);
        std::array<Coor<Nd>, 2> fs{{{}}, dim};
        append_blocks<Nd, Scalar>(&fs, 1, dim, stoh,
#    ifdef SUPERBBLAS_USE_MPI
                                  MPI_COMM_WORLD,
#    endif
                                  SlowToFast);

        // Store proper values to test the storage
        {
            PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
            const Coor<Nd - 1> local_size0 = p0[rank][1];
            std::size_t vol0 = detail::volume(local_size0);
            Tensor t0(vol0);
            std::vector<Scalar> t0_host(vol0);

            Coor<Nd - 1,std::size_t> local_strides0 = detail::get_strides<std::size_t>(local_size0, SlowToFast);
            Coor<Nd - 1, std::size_t> strides0 = detail::get_strides<std::size_t>(dim0, SlowToFast);
            for (int m = 0; m < dim[M]; ++m) {
                const Coor<Nd - 1> from0{};
                const Coor<Nd> from1{m};
                for (std::size_t i = 0; i < vol0; ++i)
                    t0_host[i] =
                        coor2index(index2coor(i, local_size0, local_strides0) + p0[rank][0], dim0,
                                   strides0) +
                        m * vol / dim[M];
                t0 = t0_host;
                Scalar *ptr0 = t0.data().get();
                save<Nd - 1, Nd, Scalar, Scalar>(1.0, p0.data(), 1, "dtgsSnN", from0, dim0, dim0,
                                                 (const Scalar **)&ptr0, &ctx, "mdtgsSnN", from1,
                                                 stoh,
#    ifdef SUPERBBLAS_USE_MPI
                                                 MPI_COMM_WORLD,
#    endif
                                                 SlowToFast);
            }

            flush_storage(stoh);

            if (rank == 0) {
                // The data of the only block should contain the numbers from zero to vol
                std::size_t padding_size = (8 - metadata.size() % 8) % 8;
                std::size_t header_size =
                    sizeof(int) * 6 + metadata.size() + padding_size + sizeof(double) * (Nd + 1);
                std::size_t disp = header_size + sizeof(double) * (2 + Nd * 2);
                std::ifstream f(filename, std::ios::binary);
                f.seekg(disp);
                Scalar s;
                for (std::size_t i = 0; i < vol; ++i) {
                    f.read((char *)&s, sizeof(s));
                    if (i != s.real()) throw std::runtime_error("Failing reading from storage");
                }
                f.close();
            }
        }

        // Test the readings
        {
            const Coor<Nd - 2> dimr{dim[M], dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // mdtgsS
            Coor<Nd - 2, std::size_t> stridesr = detail::get_strides<std::size_t>(dimr, SlowToFast);
            Coor<2> dimNN{dim[N0], dim[N1]};
            Coor<2, std::size_t> stridesNN = detail::get_strides<std::size_t>(dimNN, SlowToFast);
            Coor<Nd, std::size_t> strides = detail::get_strides<std::size_t>(dim, SlowToFast);

            for (auto n : nn) {
                Coor<2> dimnn{n, n};
                Coor<2, std::size_t> stridesnn =
                    detail::get_strides<std::size_t>(dimnn, SlowToFast);

                // Create tensor t1 for reading the genprop on root process
                PartitionStored<2> p1(nprocs);
                p1[0][1] = dimnn;
                std::size_t vol1 = detail::volume(p1[rank][1]);
                Tensor t1(vol1);
                std::vector<Scalar> t1_host(vol1);

                for (auto req : reqs) {
                    Coor<Nd> from0{};
                    std::copy_n(detail::index2coor(req, dimr, stridesr).begin(), Nd - 2,
                                from0.begin());
                    Coor<Nd> size0{};
                    for (auto &c : size0) c = 1;
                    size0[Nd - 2] = size0[Nd - 1] = n;
                    const Coor<2> from1{};
                    Scalar *ptr1 = t1.data().get();
                    for (std::size_t i = 0; i < vol1; ++i) t1[i] = -1;
                    load<Nd, 2, Scalar, Scalar>(1.0, stoh, "mdtgsSnN", from0, size0, p1.data(), 1,
                                                "nN", from1, dimnn, &ptr1, &ctx,
#    ifdef SUPERBBLAS_USE_MPI
                                                MPI_COMM_WORLD,
#    endif
                                                SlowToFast, Copy);
                    thrust::copy(t1.begin(), t1.end(), t1_host.begin());
                    for (std::size_t i = 0; i < vol1; ++i)
                        if (t1_host[i].real() !=
                            coor2index(index2coor(i, dimnn, stridesnn) + p1[rank][0], dimNN,
                                       stridesNN) +
                                coor2index(from0, dim, strides))
                            throw std::runtime_error("Storage failed!");
                }
            }
        }

        close_storage<Nd, Scalar>(stoh
#    ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#    endif
        );

        if (rank == 0) reportTimings(std::cout);
        if (rank == 0) reportCacheUsage(std::cout);
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
