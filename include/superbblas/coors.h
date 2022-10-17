#ifndef __SUPERBBLAS_COOR__
#define __SUPERBBLAS_COOR__

#include "blas.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <initializer_list>

namespace superbblas {

    namespace detail {
        template <typename ClassN> unsigned int &array_size() {
            static unsigned int size;
            return size;
        }

        template <typename T> struct array_element_size_ {
            static std::size_t size() { return sizeof(T); }
        };

        template <typename T> struct is_array;
        template <typename T, typename ClassN, bool T_is_array = is_array<T>::value> struct array;

        template <typename T, typename ClassN> struct array_element_size_<array<T, ClassN>> {
            static std::size_t size() {
                return array_element_size_<T>::size() * array_size<ClassN>();
            }
        };
        template <typename T> std::size_t array_element_size() {
            return array_element_size_<T>::size();
        }

        template <typename T> struct is_array { static constexpr bool value = false; };
        template <typename T, typename ClassN> struct is_array<array<T, ClassN>> {
            static constexpr bool value = true;
        };

        /// Array container with run-time fixed size
        /// \tparam T: element type
        /// \tparam ClassN: tag struct to differentiate all instantiations with the same size

        template <typename T, typename ClassN> struct array<T, ClassN, false> {
        private:
            /// Pointer to the values array
            T *const p;

            /// Whether to deallocate the pointer `p` on the destructor
            const bool deallocate_pointer;

        public:
            /// Empty constructor, it does zero initialization

            array() : p((T *)std::calloc(size(), element_size())), deallocate_pointer(true) {}

            /// Reference to T constructor

            explicit array(T *p) : p(p), deallocate_pointer(false) {}

            /// Destructor

            ~array() {
                if (deallocate_pointer) std::free(p);
            }

            /// Copy constructor

            array(const array &c)
                : p((T *)std::malloc(size() * element_size())), deallocate_pointer(true) {
                assert(c.size() == size());
                std::copy_n(c.begin(), size(), begin());
            }

            template <typename Q, typename QClassN>
            array(const array<Q, QClassN, false> &c)
                : p((T *)std::malloc(size() * element_size())), deallocate_pointer(true) {
                assert(c.size() == size());
                std::copy_n(c.begin(), size(), begin());
            }

            /// Assignment

            array &operator=(const array &c) {
                std::copy_n(c.begin(), size(), begin());
                return *this;
            }

            template <typename Q, typename QClassN> array &operator=(const array<Q, QClassN, false> &c) {
                assert(c.size() == size());
                std::copy_n(c.begin(), size(), begin());
                return *this;
            }

            /// Comparison

            template <typename Q, typename QClassN> bool operator==(const array<Q, QClassN, false> &c) const {
                if (size() != c.size()) return false;
                for (unsigned int i = 0, n = size(); i < n; ++i)
                    if (p[i] != c[i]) return false;
                return true;
            }

            template <typename Q, typename QClassN> bool operator!=(const array<Q, QClassN, false> &c) const {
                return !(*this == c);
            }

            /// Return the element size
            static unsigned int element_size() { return array_element_size<T>(); }

            /// Return the array size

            static unsigned int size() { return array_size<T>(); }

            /// Iterator type
            using iterator = T *;

            /// Iterator type
            using const_iterator = const T *;

            /// Return the pointer to the first element

            T *begin() { return p; }

            /// Return the pointer to the first element

            const T *begin() const { return p; }

            /// Return the pointer to the first element not in the array

            T *end() { return p + size(); }

            /// Return the pointer to the first element not in the array

            const T *end() const { return p + size(); }

            /// Return the i-th element

            T &operator[](unsigned int i) {
                assert(i < size());
                return p[i];
            }

            /// Return the i-th element

            const T &operator[](unsigned int i) const {
                assert(i < size());
                return p[i];
            }

            /// Return the first element

            T &front() { return p[0]; }

            /// Return the first element

            const T &front() const { return p[0]; }

            /// Return the last element

            T &back() { return p[size() - 1]; }

            /// Return the last element

            const T &back() const { return p[size() - 1]; }
        };

        template <typename T, typename ClassN> struct array<T, ClassN, true> {
        private:
            /// Pointer to the values array
            void *const p;

            /// Whether to deallocate the pointer `p` on the destructor
            const bool deallocate_pointer;

        public:
            /// Empty constructor, it does zero initialization

            array() : p(std::calloc(size(), element_size())), deallocate_pointer(true) {}

            /// Reference to T constructor

            explicit array(T *p) : p(p), deallocate_pointer(false) {}

            /// Destructor

            ~array() {
                if (deallocate_pointer) std::free(p);
            }

            /// Copy constructor

            template <typename Q, typename QClassN>
            array(const array<Q, QClassN> &c)
                : p(std::calloc(size(), element_size())), deallocate_pointer(true) {
                assert(c.size() == size());
                std::copy_n(c.begin(), size(), begin());
            }

            /// Assignment

            template <typename Q, typename QClassN> array &operator=(const array<Q, QClassN> &c) {
                assert(c.size() == size());
                std::copy_n(c.begin(), size(), begin());
                return *this;
            }

            /// Comparison

            template <typename Q, typename QClassN> bool operator==(const array<Q, QClassN> &c) const {
                if (size() != c.size()) return false;
                for (unsigned int i = 0, n = size(); i < n; ++i)
                    if ((*this)[i] != c[i]) return false;
                return true;
            }

            template <typename Q, typename QClassN> bool operator!=(const array<Q, QClassN> &c) const {
                return !(*this == c);
            }


            /// Return the element size

            static unsigned int element_size() { return array_element_size<T>(); }

            /// Return the array size

            static unsigned int size() { return array_size<T>(); }

            template <typename Q>
            class Iterator
                : public std::iterator<std::random_access_iterator_tag, Q, std::ptrdiff_t, Q *, Q> {
                void *p;

            public:
                explicit Iterator(void *p) : p(p) {}
                Iterator &operator++() {
                    p += array_element_size<T>();
                    return *this;
                }
                Iterator operator++(int) {
                    iterator r = *this;
                    p += array_element_size<T>();
                    return r;
                }
                Iterator operator+(int n) { return Iterator(p + n * array_element_size<T>()); }
                Iterator operator-(int n) { return Iterator(p - n * array_element_size<T>()); }
                Iterator &operator+=(int n) {
                    p += n * array_element_size<T>();
                    return *this;
                }
                Iterator &operator-=(int n) {
                    p -= n * array_element_size<T>();
                    return *this;
                }

                std::ptrdiff_t operator-(Iterator it) const {
                    return ((char *)p - (char *)it.p) / element_size();
                }
                bool operator==(Iterator it) const { return p == it.p; }
                bool operator!=(Iterator it) const { return p != it.p; }
                bool operator<(Iterator it) const { return p < it.p; }
                bool operator>(Iterator it) const { return !(p <= it.p); }
                bool operator<=(Iterator it) const { return p <= it.p; }
                bool operator>=(Iterator it) const { return (!(p < it.p) | p == it.p); }
                Q operator*() const { return Q{p}; }
            };
            using iterator = Iterator<T>;
            using const_iterator = Iterator<const T>;

            /// Return the pointer to the first element

            iterator begin() { return iterator{p}; }

            /// Return the pointer to the first element

            const_iterator begin() const { return const_iterator{p}; }

            /// Return the pointer to the first element not in the array

            iterator end() { return iterator((void *)((char *)p + element_size() * size())); }

            /// Return the pointer to the first element not in the array

            const_iterator end() const {
                return const_iterator((void *)((char *)p + element_size() * size()));
            }

            /// Return the i-th element

            T operator[](unsigned int i) {
                assert(i < size());
                return T((void *)((char *)p + i * element_size()));
            }

            /// Return the i-th element

            const T operator[](unsigned int i) const {
                assert(i < size());
                return T((void *)((char *)p + i * element_size()));
            }

            /// Return the first element

            T front() { return T(p); }

            /// Return the first element

            const T front() const { return T(p); }

            /// Return the last element

            T back() { return T((void *)((char *)p + element_size() * (size() - 1))); }

            /// Return the last element

            const T back() const { return T((void *)((char *)p + element_size() * (size() - 1))); }
        };

        /// Coordinate Index type
        using IndexType = int;

        /// Coordinate type
        template <typename Nd, typename Idx = IndexType> using Coor = array<Idx, Nd>;

#ifdef SUPERBBLAS_USE_THRUST

        /// Thrust does not support std::array container; here we implement a quick-and-dirty array container based on tuples

        template <typename T, std::size_t N> struct tarray;
        template <typename T, std::size_t N> struct tarray {
            static const std::size_t size_left = (N + 1) / 2;
            static const std::size_t size_right = N - size_left;
            tarray<T, size_left> left;
            tarray<T, size_right> right;
        };
        template <typename T> struct tarray<T, 0ul> {};
        template <typename T> struct tarray<T, 1ul> { T leaf; };

        /// Return the I-th element on a tarray
        /// \tparam I: index of the element to return
        /// \param t: input array

        template <std::size_t I, typename T, std::size_t N,
                  typename std::enable_if<(I > 0 && I < N), bool>::type = true>
        inline __HOST__ __DEVICE__ T &tget(tarray<T, N> &t) {
            return (I < t.size_left ? tget<I>(t.left) : tget<I - t.size_left>(t.right));
        }

        template <std::size_t I, typename T, std::size_t N,
                  typename std::enable_if<(I == 0 && N == 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T &tget(tarray<T, N> &t) {
            return t.leaf;
        }

        /// Return the i-th element on a tarray
        /// \param i: index of the element to return
        /// \param t: input array

        template <typename T, typename Indx, std::size_t N,
                  typename std::enable_if<(N > 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T tget(Indx i, const tarray<T, N> &t) {
            return (i < Indx(t.size_left) ? tget(i, t.left) : tget(i - (Indx)t.size_left, t.right));
        }

        template <typename T, typename Indx, std::size_t N,
                  typename std::enable_if<(N == 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T tget(Indx i, const tarray<T, N> &t) {
            return (i == 0 ? t.leaf : T{0});
        }

        /// Coordinate based on tarray
        /// \tparam Nd: number of dimensions

        template <std::size_t Nd, typename Idx = IndexType> using TCoor = tarray<Idx, Nd>;
#endif

        //
        // Auxiliary functions
        //

        template <typename T, typename N>
        array<T, N> operator+(const array<T, N> &a, const array<T, N> &b) {
            array<T, N> r;
            for (std::size_t i = 0, n = a.size(); i < n; i++) r[i] = a[i] + b[i];
            return r;
        }

        template <typename T, typename N>
        array<T, N> operator-(const array<T, N> &a, const array<T, N> &b) {
            array<T, N> r;
            for (std::size_t i = 0, n = a.size(); i < n; i++) r[i] = a[i] - b[i];
            return r;
        }

        template <typename T, typename N>
        array<T, N> operator/(const array<T, N> &a, const array<T, N> &b) {
            array<T, N> r;
            for (std::size_t i = 0, n =a.size(); i < n; i++) r[i] = a[i] / b[i];
            return r;
        }

        template <typename T, typename N>
        bool all_less_or_equal(const array<T, N> &a, const array<T, N> &b) {
            for (std::size_t i = 0, n =a.size(); i < n; i++)
                if (a[i] > b[i]) return false;
            return true;
        }

        template <typename T, typename N>
        array<T, N> min_each(const array<T, N> &a, const array<T, N> &b) {
            array<T, N> r;
            for (std::size_t i = 0, n =a.size(); i < n; i++) r[i] = std::min(a[i], b[i]);
            return r;
        }

        template <typename T, typename N>
        array<T, N> max_each(const array<T, N> &a, const array<T, N> &b) {
            array<T, N> r;
            for (std::size_t i = 0, n = a.size(); i < n; i++) r[i] = std::max(a[i], b[i]);
            return r;
        }

        template <typename T, typename N> array<T, N> reverse(const array<T, N> v) {
            array<T, N> r = v;
            std::reverse(r.begin(), r.end());
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        struct ns_plus_aux {
            template <std::size_t Nd, typename std::enable_if<(Nd > 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd> plus_aux(const TCoor<Nd> &a,
                                                                 const TCoor<Nd> &b) {
                return {plus_aux(a.left, b.left), plus_aux(a.right, b.right)};
            }

            template <std::size_t Nd, typename std::enable_if<(Nd == 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd> plus_aux(const TCoor<Nd> &a,
                                                                 const TCoor<Nd> &b) {
                return {a.leaf + b.leaf};
            }
        };

        /// Add two arrays
        /// \param a: first array to add
        /// \param b: second array to add

        template <std::size_t Nd>
        __HOST__ __DEVICE__ inline TCoor<Nd> tplus(TCoor<Nd> a, TCoor<Nd> b) {
            return ns_plus_aux::plus_aux(a, b);
        }

        struct ns_toTCoor_aux {
            template <std::size_t I, std::size_t Nr, typename ClassN, typename IndexType,
                      typename std::enable_if<(1 < Nr), bool>::type = true>
            static inline TCoor<Nr, IndexType> toTCoor_aux(const Coor<ClassN, IndexType> &a) {
                const auto sl = TCoor<Nr, IndexType>::size_left;
                const auto sr = TCoor<Nr, IndexType>::size_right;
                return {toTCoor_aux<I, sl>(a), toTCoor_aux<I + sl, sr>(a)};
            }

            template <std::size_t I, std::size_t Nr, typename ClassN, typename IndexType,
                      typename std::enable_if<(1 == Nr), bool>::type = true>
            static inline TCoor<Nr, IndexType> toTCoor_aux(const Coor<ClassN, IndexType> &a) {
                return {a[I]};
            }
        };

        /// Convert from Coor to TCoor
        /// \param a: input coordinate

        template <std::size_t Nd, typename IndexType, typename ClassN>
        inline TCoor<Nd, IndexType> toTCoor(const Coor<ClassN, IndexType> &a) {
            assert(array_size<ClassN>() == Nd);
            return ns_toTCoor_aux::toTCoor_aux<0, Nd>(a);
        }
#endif


    }

}
#endif //  __SUPERBBLAS_COOR__
