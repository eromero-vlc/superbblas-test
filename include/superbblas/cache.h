#ifndef __SUPERBBLAS_CACHE__
#define __SUPERBBLAS_CACHE__

#include "performance.h"
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace superbblas {

    namespace detail {
        /// Cache with Least Recently Used eviction policy for heterogeneous objects

        class cache {
            /// Maximum storage allowed (in bytes)
            std::size_t maxCacheSize;

            /// Current storage (in bytes)
            std::size_t currentSize;

            /// Type for storing events
            using Timestamp = std::size_t;

            /// Event clock
            Timestamp timestamp;

        public:
            /// Values associated to a key
            template <typename V> struct Value {
                V value;
                Timestamp ts;
                std::size_t size;
            };

        private:
            /// Operations over each cache
            struct AbstractCache {
            protected:
                AbstractCache() {}

            public:
                virtual ~AbstractCache(){};
                virtual std::size_t deleteTs(Timestamp) {
                    throw std::runtime_error("Not implemented");
                };
            };

            /// Data associated to each cache for a particular K and V
            template <typename K, typename V, typename H> struct Cache : public AbstractCache {
                std::unordered_map<K, Value<V>, H> cache;
                std::unordered_map<Timestamp, K> keys;
                Cache() : cache(16), keys(16) {}
                ~Cache() {}
                std::size_t deleteTs(Timestamp ts) override {
                    auto k = keys.find(ts);
                    if (k == keys.end()) throw std::runtime_error("Timestamp not found");
                    auto v = cache.find(k->second);
                    if (v == cache.end()) throw std::runtime_error("This shouldn't happen");
                    std::size_t size = v->second.size;
                    cache.erase(v);
                    keys.erase(k);
                    return size;
                }
            };

            /// Record the Cache<K,V> associated to each typeid(tuple{K,V})
            std::map<std::type_index, AbstractCache *> caches;

            /// Record the cache associated to each timestamp
            std::map<Timestamp, AbstractCache *> timestamps;

        public:
            /// Create a cache
            /// \param maxCacheSize: maximum storage for all objects allocated on the this cache
            cache(std::size_t maxCacheSize = 0)
                : maxCacheSize(maxCacheSize), currentSize(0), timestamp(0) {}

            ~cache() { clear(); }

            /// Get the maximum storage for all objects allocated on the this cache

            std::size_t getMaxCacheSize() { return maxCacheSize; }

            /// Set the maximum storage for all objects allocated on the this cache
            /// \param maxCacheSize: value in bytes

            void setMaxCacheSize(std::size_t size) { maxCacheSize = size; }

        private:
            /// Return the unordered_map<K,V> associated to a type

            template <typename K, typename V, typename H, typename T = std::tuple<K, V>>
            Cache<K, V, H> &get() {
                std::type_index ti = std::type_index(typeid(T));
                auto it = caches.find(ti);
                if (it != caches.end()) return *reinterpret_cast<Cache<K, V, H> *>(it->second);
                return *reinterpret_cast<Cache<K, V, H> *>(caches[ti] = new Cache<K, V, H>());
            }

        public:
            /// Remove all entries in the cache and start over
            void clear() {
                for (const auto &it : caches) delete it.second;
                timestamps.clear();
                timestamp = 0;
            }

            /// Insert an entry into the cache, it may invalidate other iterators
            /// \param k: key
            /// \param v: value
            /// \param: size: memory footprint of the entry (in bytes)

            template <typename K, typename V, typename H, typename T = std::tuple<K, V>>
            void insert(const K &k, const V &v, std::size_t size) {
                Cache<K, V, H> &cache = get<K, V, H, T>();

                // Remove the entries associated to the key
                {
                    auto it = cache.cache.find(k);
                    if (it != cache.cache.end()) {
                        Timestamp ts = it->second.ts;
                        currentSize -= cache.deleteTs(ts);
                        timestamps.erase(ts);
                    }
                }

                // Don't store in cache entries bigger than the maximum cache size
                if (size > maxCacheSize) return;

                // Remove entries in the cache until the new entry fits in
                while (size + currentSize > maxCacheSize && !timestamps.empty()) {
                    auto it = timestamps.begin();
                    currentSize -= it->second->deleteTs(it->first);
                    timestamps.erase(it);
                }

                /// In the extremely unlikely situation that it gets out of timestamp, clean everything and start over
                if (timestamp == std::numeric_limits<Timestamp>::max()) {
                    clear();
                    insert<K, V, H, T>(k, v, size);
                    return;
                }

                // Associate a timestamp to the key and insert the key
                ++timestamp;
                timestamps[timestamp] = &cache;
                cache.cache[k] = Value<V>{v, timestamp, size};
                cache.keys[timestamp] = k;
                currentSize += size;
            }

            /// Return the entry given the key, it may invalidate other iterators
            /// \param k: key

            template <typename K, typename V, typename H, typename T = std::tuple<K, V>>
            typename std::unordered_map<K, Value<V>, H>::iterator find(const K &k) {
                Cache<K, V, H> &cache = get<K, V, H, T>();
                auto it = cache.cache.find(k);

                // Update the timestamp of the key
                if (it != cache.cache.end()) {
                    /// In the extremely unlikely situation that it gets out of timestamp, clean everything and start over
                    if (timestamp == std::numeric_limits<Timestamp>::max()) {
                        clear();
                        return cache.cache.end();
                    }

                    auto &value = it->second;
                    timestamps.erase(value.ts);
                    cache.keys.erase(value.ts);
                    ++timestamp;
                    cache.keys[timestamp] = k;
                    value.ts = timestamp;
                    timestamps[timestamp] = &cache;
                }

                return it;
            }

            /// Return the end iterator
            /// \param k: key

            template <typename K, typename V, typename H, typename T = std::tuple<K, V>>
            typename std::unordered_map<K, Value<V>, H>::iterator end() {
                return get<K, V, H, T>().cache.end();
            }
        };

        /// Easy way to call public methods of cache
        template <typename K, typename V, typename H, typename T = std::tuple<K, V>>
        struct cacheHelper {
            /// Reference to the cache
            cache &c;

            /// Insert an entry into the cache, it may invalidate other iterators
            /// \param k: key
            /// \param v: value
            /// \param: size: memory footprint of the entry (in bytes)
            void insert(const K &k, const V &v, std::size_t size) {
                tracker<Cpu> _t("cache insert", Cpu{});
                c.insert<K, V, H, T>(k, v, size);
            }

            /// Return the entry given the key, it may invalidate other iterators
            /// \param k: key
            typename std::unordered_map<K, cache::Value<V>, H>::iterator find(const K &k) {
                tracker<Cpu> _t("cache find", Cpu{});
                return c.find<K, V, H, T>(k);
            }

            /// Return the end iterator
            /// \param k: key
            typename std::unordered_map<K, cache::Value<V>, H>::iterator end() {
                return c.end<K, V, H, T>();
            }
        };

        /// Return all caches
        inline std::array<std::vector<cache>, 256> &getCaches() {
            // 2D array of caches, caches[session][deviceId]
            static std::array<std::vector<cache>, 256> caches{};
            return caches;
        }

        /// Return the caches associated to the devices
        inline std::vector<cache> &getCaches(Session session) {
            auto &caches = getCaches();
            static std::mutex m;

            // Initialize caches
            if (caches[255].size() == 0) {
                std::lock_guard<std::mutex> g(m);

                if (caches[255].size() == 0) {
                    for (Session s = 0; s < 256; ++s) {
                        // Get maximum memory use for CPU cache
                        std::size_t cacheMaxSizeCpu = 0;
                        if (getMaxCacheGiBCpu() >= 0) {
                            cacheMaxSizeCpu = std::size_t(getMaxCacheGiBCpu() * 1024 * 1024 * 1024);
                        } else {
                            cacheMaxSizeCpu =
                                std::size_t(sysconf(_SC_PAGESIZE)) * sysconf(_SC_PHYS_PAGES) / 10;
                        }

                        // Create the cache for the cpu objects and set the maximum size
                        std::vector<cache> cache_s(1);
                        cache_s[0].setMaxCacheSize(cacheMaxSizeCpu);

#ifdef SUPERBBLAS_USE_GPU
                        // Get maximum memory use for GPU cache
                        std::size_t cacheMaxSizeGpu = 0;
                        if (getMaxCacheGiBGpu() >= 0) {
                            cacheMaxSizeGpu = std::size_t(getMaxCacheGiBGpu() * 1024 * 1024 * 1024);
                        } else {
                            std::size_t free = 0, total = 0;
#    ifdef SUPERBBLAS_USE_CUDA
                            cudaCheck(cudaMemGetInfo(&free, &total));
#    else
                            hipCheck(hipMemGetInfo(&free, &total));
#    endif
                            cacheMaxSizeGpu = total / 10;
                        }

                        // Create the caches for the gpu objects and set the maximum size
                        int numDevices = getGpuDevicesCount();
                        cache_s.resize(numDevices + 1);
                        for (int d = 0; d < numDevices; ++d)
                            cache_s[d + 1].setMaxCacheSize(cacheMaxSizeGpu);
#endif
                        std::swap(caches[s], cache_s);
                    }
                }
            }
            return caches[session];
        }

        /// Return the cache to store objects on the device
        template <typename K, typename V, typename H, typename T = std::tuple<K, V>, typename XPU>
        inline cacheHelper<K, V, H, T> getCache(XPU xpu) {
            auto &caches = getCaches(xpu.session);
            int device = deviceId(xpu);
            if (device < -1 || device + 1 >= (int)caches.size())
                throw std::runtime_error("Invalid device");
            return cacheHelper<K, V, H, T>{caches[device + 1]};
        }
    }

    /// Clear all internal caches
    inline void clearCaches() {
        auto &caches = detail::getCaches();
        for (auto &i : caches) i.resize(0);
    }
}

#endif // __SUPERBBLAS_CACHE__
