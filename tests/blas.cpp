#include "superbblas.h"
#include "superbblas/blas.h"
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <omp.h>

#ifdef SUPERBBLAS_USE_CUDA
#    include <thrust/complex.h>
#    include <thrust/device_vector.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <typename T, typename XPU> struct gen_dummy_vector;

template <typename T> struct gen_dummy_vector<T, Cpu> {
    static vector<T, Cpu> get(std::size_t size, Cpu) {
        vector<T, Cpu> v(size);
        for (unsigned int i = 0; i < size; i++) v[i] = i;
        return v;
    }
};

template <typename T, std::size_t N>
struct gen_dummy_vector <std::array<T, N>, Cpu>{
    using Q = std::array<T, N>;
    static vector<Q, Cpu> get(std::size_t size, Cpu) {
        vector<Q, Cpu> v(size);
        for (unsigned int i = 0; i < size; i++)
            for (unsigned int j = 0; j < N; ++j) v[i][j] = i * N + j;
        return v;
    }
};

#ifdef SUPERBBLAS_USE_CUDA
template <typename T> struct gen_dummy_vector<T, Cuda> {
    static vector<T, Cuda> get(std::size_t size, Cuda cuda) {
        vector<T, Cpu> v = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
        vector<T, Cuda> r = new_vector<T>(size, cuda);
        copy_n<IndexType, T>(v.data(), Cpu{}, size, r.data(), cuda, EWOp::Copy{});
        return r;
    }
};
#endif

Indices<Cpu> gen_dummy_perm(std::size_t size, std::size_t max_size, Cpu) {
    Indices<Cpu> v(size);
    for (unsigned int i = 0; i < size; i++) v[i] = (i * 3) % max_size;
    return v;
}

#ifdef SUPERBBLAS_USE_CUDA
Indices<Cuda> gen_dummy_perm(std::size_t size, std::size_t max_size, Cuda cuda) {
    Indices<Cpu> v = gen_dummy_perm(size, max_size, Cpu{});
    Indices<Cuda> r = new_vector<IndexType>(size, cuda);
    copy_n<IndexType, IndexType>(v.data(), Cpu{}, size, r.data(), cuda, EWOp::Copy{});
    return r;
}
#endif

template <typename T> double myabs(T const &t) { return std::fabs(t); }

template <typename T> double myabs(std::complex<T> const &t) {
    return std::fabs(std::complex<T>(t.real(), t.imag()));
}

template <typename T, std::size_t N> double myabs(std::array<T, N> const &t) {
    double r = 0;
    for (const T &i : t) r += std::fabs(i);
    return r;
}

template <typename T> struct Epsilon {
    static double get(void) { return std::fabs(std::numeric_limits<T>::epsilon()); }
};

template <typename T, std::size_t N> struct Epsilon<std::array<T, N>> {
    static double get(void) { return std::fabs(std::numeric_limits<T>::epsilon()) * N; }
};

template <typename T> void check_are_equal(vector<T, Cpu> u, vector<T, Cpu> v) {
    if (u.size() != v.size()) throw std::runtime_error("Input vectors have different size!");
    double diff = 0, add = 0;
    for (unsigned int i = 0; i < u.size(); i++)
        diff += myabs(u[i] - v[i]), add += myabs(u[i]) + myabs(v[i]);
    const double bound = add * Epsilon<T>::get() * 10 * u.size() / 2;
    if (diff > bound) {
        std::stringstream ss;
        ss << "1-norm of the difference between the input vectors of size " << v.size() << " is "
           << diff << ", which is larger than the bound at " << bound;
    }
}

template<typename T> struct toStr;

template <> struct toStr<Cpu> { static constexpr const char *get = "cpu"; };
#ifdef SUPERBBLAS_USE_CUDA
template <> struct toStr<Cuda> { static constexpr const char *get = "cuda"; };
#endif
template <> struct toStr<EWOp::Add> { static constexpr const char *get = "add"; };
template <> struct toStr<EWOp::Copy> { static constexpr const char *get = "copy"; };
template <> struct toStr<float> { static constexpr const char *get = "float"; };
template <> struct toStr<double> { static constexpr const char *get = "double"; };
template <> struct toStr<std::complex<float>> { static constexpr const char *get = "cfloat"; };
template <> struct toStr<std::complex<double>> { static constexpr const char *get = "cdouble"; };
template <std::size_t N> struct toStr<std::array<std::complex<float>, N>> {
    static constexpr const char *get = "arraycfloat";
};
template <std::size_t N> struct toStr<std::array<std::complex<double>, N>> {
    static constexpr const char *get = "arraycdouble";
};

template <typename T, typename XPU, typename EWOP>
void test_copy(std::size_t size, XPU xpu, EWOP, unsigned int nrep = 10) {

    // Do once the operation for testing correctness
    vector<T, Cpu> t0 = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
    vector<T, XPU> t0_xpu = gen_dummy_vector<T, XPU>::get(size, xpu);
    vector<T, XPU> t1_xpu = gen_dummy_vector<T, XPU>::get(size, xpu);
    copy_n<IndexType, T>(t0.data(), Cpu{}, size, t0_xpu.data(), xpu, EWOP{});
    copy_n<IndexType, T>(t0_xpu.data(), xpu, size, t1_xpu.data(), xpu, EWOP{});
    vector<T, Cpu> t1 = new_vector<T>(size, Cpu{});
    copy_n<IndexType, T>(t0_xpu.data(), xpu, size, t1.data(), Cpu{}, EWOp::Copy{});

    vector<T, Cpu> r = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
    copy_n<IndexType, T>(t0.data(), Cpu{}, size, r.data(), Cpu{}, EWOP{});
    copy_n<IndexType, T>(t0.data(), Cpu{}, size, r.data(), Cpu{}, EWOP{});
    check_are_equal<T>(t1, r);

    // Test with indices
    Indices<Cpu> i0 = gen_dummy_perm(size/2, size, Cpu{});
    Indices<XPU> i0_xpu = gen_dummy_perm(size/2, size, xpu);
    zero_n<T>(t0_xpu.data(), size, xpu);
    zero_n<T>(t1_xpu.data(), size, xpu);
    // t0_xpu[i] = t0[i0[i]]
    copy_n<IndexType, T>(t0.data(), i0.begin(), Cpu{}, size/2, t0_xpu.data(), xpu, EWOp::Copy{});
    // t0_xpu[i0_xpu[i]] = t0[i]
    copy_n<IndexType, T>(t0.data(), Cpu{}, size / 2, t0_xpu.data(), i0_xpu.begin(), xpu,
                         EWOp::Copy{});
    // t1_xpu[i0_xpu[i]] = t0_xpu[i0_xpu[i]]
    copy_n<IndexType, T>(t0_xpu.data(), i0_xpu.begin(), xpu, size / 4, t1_xpu.data(),
                         i0_xpu.begin() + size / 4, xpu, EWOP{});
    // t0[i] = t0_xpu[i0[i]]
    copy_n<IndexType, T>(t1_xpu.data(), i0_xpu.begin(), xpu, size / 2, t1.data(), Cpu{},
                         EWOp::Copy{});

    vector<T, Cpu> r0(size), r1(size);
    zero_n<T>(r.data(), size, Cpu{});
    copy_n<IndexType, T>(t0.data(), i0.begin(), Cpu{}, size/2, r0.data(), Cpu{}, EWOp::Copy{});
    copy_n<IndexType, T>(t0.data(), Cpu{}, size / 2, r0.data(), i0.begin(), Cpu{}, EWOp::Copy{});
    copy_n<IndexType, T>(r0.data(), i0.begin(), Cpu{}, size / 4, r1.data(), i0.begin() + size / 4,
                         Cpu{}, EWOP{});
    copy_n<IndexType, T>(r1.data(), i0.begin(), Cpu{}, size / 2, r.data(), Cpu{}, EWOp::Copy{});
    check_are_equal<T>(t1, r);

    // Test performance
    double t;
    t  = omp_get_wtime();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(t0.data(), Cpu{}, size, t0_xpu.data(), xpu, EWOP{});
    }
    double t_cpu_xpu = (omp_get_wtime() - t) / nrep;

    t = omp_get_wtime();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(t0_xpu.data(), xpu, size, t1.data(), Cpu{}, EWOP{});
    }
    double t_xpu_cpu = (omp_get_wtime() - t) / nrep;

    t = omp_get_wtime();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(t0_xpu.data(), xpu, size, t1_xpu.data(), xpu, EWOP{});
    }
    sync(xpu);
    double t_xpu_xpu = (omp_get_wtime() - t) / nrep;

    Indices<XPU> p = gen_dummy_perm(size, size, xpu);
    t = omp_get_wtime();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(t0_xpu.data(), p.begin(), xpu, size, t1_xpu.data(), p.begin(), xpu,
                             EWOP{});
    }
    sync(xpu);
    double tp_xpu_xpu = (omp_get_wtime() - t) / nrep;

    std::cout << toStr<T>::get << " in " << toStr<EWOP>::get << " ("
              << sizeof(T) * size / 1024. / 1024 << " MiB)\t\t"

              << toStr<Cpu>::get << " -> " << toStr<XPU>::get << " : "
              << sizeof(T) * size / t_cpu_xpu / 1024 / 1024 / 1024 << " GiB/s"
              << "\t\t"

              << toStr<XPU>::get << " -> " << toStr<Cpu>::get << " : "
              << sizeof(T) * size / t_xpu_cpu / 1024 / 1024 / 1024 << " GiB/s"
              << "\t\t"

              << toStr<XPU>::get << " -> " << toStr<XPU>::get << " : "
              << sizeof(T) * size / t_xpu_xpu / 1024 / 1024 / 1024 << " GiB/s" 
              << "\t\t"

              << toStr<XPU>::get << "[i] -> " << toStr<XPU>::get << "[i] : "
              << sizeof(T) * size / tp_xpu_xpu / 1024 / 1024 / 1024 << " GiB/s" << std::endl;
}

int main(int argc, char **argv) {
    int size = 1000;
    int nrep = 10;

    // Get options
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--size=", argv[i], 7) == 0) {
            if (sscanf(argv[i] + 7, "%d", &size) != 1) {
                std::cerr << "--size= should follow 1 numbers, for instance --size=10" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--rep=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d", &nrep) != 1) {
                std::cerr << "--rep= should follow 1 numbers, for instance --rep=10" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--size=number] [--rep=number] [--help]" << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }
    std::cout << "Maximum number of elements in a tested array: " << size << std::endl;
    std::cout << "Doing " << nrep << " repetitions" << std::endl;

    {
        Context ctx = createCpuContext();
        test_copy<float, Cpu>(size, ctx.toCpu(), EWOp::Copy{}, nrep);
        test_copy<float, Cpu>(size, ctx.toCpu(), EWOp::Add{}, nrep);
        test_copy<double, Cpu>(size, ctx.toCpu(), EWOp::Copy{}, nrep);
        test_copy<double, Cpu>(size, ctx.toCpu(), EWOp::Add{}, nrep);
        test_copy<std::complex<float>, Cpu>(size, ctx.toCpu(), EWOp::Copy{}, nrep);
        test_copy<std::complex<float>, Cpu>(size, ctx.toCpu(), EWOp::Add{}, nrep);
        test_copy<std::complex<double>, Cpu>(size, ctx.toCpu(), EWOp::Copy{}, nrep);
        test_copy<std::complex<double>, Cpu>(size, ctx.toCpu(), EWOp::Add{}, nrep);
        test_copy<std::array<std::complex<float>, 12>, Cpu>(size, ctx.toCpu(), EWOp::Copy{}, nrep);
     }

#ifdef SUPERBBLAS_USE_CUDA
    {
        Context ctx = createCudaContext();
        test_copy<float, Cuda>(size, ctx.toCuda(), EWOp::Copy{}, nrep);
        test_copy<float, Cuda>(size, ctx.toCuda(), EWOp::Add{}, nrep);
        test_copy<double, Cuda>(size, ctx.toCuda(), EWOp::Copy{}, nrep);
        test_copy<double, Cuda>(size, ctx.toCuda(), EWOp::Add{}, nrep);
        test_copy<std::complex<float>, Cuda>(size, ctx.toCuda(), EWOp::Copy{}, nrep);
        test_copy<std::complex<float>, Cuda>(size, ctx.toCuda(), EWOp::Add{}, nrep);
        test_copy<std::complex<double>, Cuda>(size, ctx.toCuda(), EWOp::Copy{}, nrep);
        test_copy<std::complex<double>, Cuda>(size, ctx.toCuda(), EWOp::Add{}, nrep);
        test_copy<std::array<std::complex<float>, 12>, Cuda>(size, ctx.toCuda(), EWOp::Copy{}, nrep);
     }
#endif
    return 0;
}
