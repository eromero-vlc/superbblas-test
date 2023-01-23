#ifndef __SUPERBBLAS_PERFORMANCE__
#define __SUPERBBLAS_PERFORMANCE__

#include "platform.h"
#include "runtime_features.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/// If SUPERBBLAS_USE_NVTX macro is defined, then the tracker reports to the NVIDIA profiler
/// the tracker name and duration

#ifdef SUPERBBLAS_USE_NVTX
#    include <nvToolsExt.h>
#endif

namespace superbblas {

    namespace detail {
        /// Return the relative cost of the multiplication with respect to real floating point
        /// \tparam T: type to consider
        template <typename T> struct multiplication_cost { static const int value = 0; };
        template <> struct multiplication_cost<float> { static const int value = 1; };
        template <> struct multiplication_cost<double> { static const int value = 2; };
        template <> struct multiplication_cost<std::complex<float>> { static const int value = 4; };
        template <> struct multiplication_cost<std::complex<double>> {
            static const int value = 8;
        };
    }

    /// Total time and number of invocations or cost factor
    struct Timing {
        double time;
        double cost;
        Timing() : time(0), cost(0) {}
    };

    /// Type for storing the timings
    using Timings = std::unordered_map<std::string, Timing>;

    /// Return the performance timings
    inline Timings &getTimings(Session session) {
        static std::vector<Timings> timings(256, Timings{16});
        return timings[session];
    }

    /// Type for storing the memory usage
    using CacheUsage = std::unordered_map<std::string, double>;

    /// Return the performance timings
    inline CacheUsage &getCacheUsage(Session session) {
        static std::vector<CacheUsage> cacheUsage(256, CacheUsage{16});
        return cacheUsage[session];
    }

    /// Get total memory allocated on the host/cpu if tracking memory consumption (see `getTrackingMemory`)

    inline double &getCpuMemUsed(Session session) {
        static std::array<double, 256> mem{{}};
        return mem[session];
    }

    /// Get total memory allocated on devices if tracking memory consumption (see `getTrackingMemory`)

    inline double &getGpuMemUsed(Session session) {
        static std::array<double, 256> mem{{}};
        return mem[session];
    }

    namespace detail {

        /// Stack of function calls being tracked
        using CallStack = std::vector<std::string>;

        /// Return the current function call stack begin tracked
        inline CallStack &getCallStackWithPath(Session session) {
            static std::vector<CallStack> callStack(256, CallStack{});
            return callStack[session];
        }

        /// Push function call to be tracked
        inline void pushCall(std::string funcName, Session session) {
            if (getCallStackWithPath(session).empty()) {
                // If the stack is empty, just append the function name
                getCallStackWithPath(session).push_back(funcName);
            } else {
                // Otherwise, push the previous one appending "/`funcName`"
                getCallStackWithPath(session).push_back(getCallStackWithPath(session).back() + "/" +
                                                        funcName);
            }
        }

        /// Pop function call from the stack
        inline std::string popCall(Session session) {
            assert(getCallStackWithPath(session).size() > 0);
            std::string back = getCallStackWithPath(session).back();
            getCallStackWithPath(session).pop_back();
            return back;
        }

        /// Return the number of seconds from some start
        inline double w_time() {
            return std::chrono::duration<double>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        }

        /// Track time between creation and destruction of the object
        template <typename XPU> struct tracker {
            /// Whether the tacker has been stopped
            bool stopped;
#ifdef SUPERBBLAS_USE_NVTX
            /// Whether the tracker has reported the end of the task
            bool reported;
#endif
            /// Name of the function being tracked
            const std::string funcName;
            /// Memory usage at that point
            const double mem_cpu, mem_gpu;
            /// Instant of the construction
            const std::chrono::time_point<std::chrono::system_clock> start;
            /// Context
            const XPU xpu;
            /// Elapsed time
            double elapsedTime;
            /// Equivalent units of cost
            double cost;

            /// Start a tracker
            tracker(std::string funcName, XPU xpu, bool timeAnyway = false)
                : stopped(!(timeAnyway || getTrackingTime())),
#ifdef SUPERBBLAS_USE_NVTX
                  reported(false),
#endif
                  funcName(!stopped ? funcName : std::string()),
                  mem_cpu(getTrackingMemory() ? getCpuMemUsed(xpu.session) : 0),
                  mem_gpu(getTrackingMemory() ? getGpuMemUsed(xpu.session) : 0),
                  start(!stopped ? std::chrono::system_clock::now()
                                 : std::chrono::time_point<std::chrono::system_clock>{}),
                  xpu(xpu),
                  elapsedTime(0),
                  cost(1) {
                if (!stopped) pushCall(funcName, xpu.session); // NOTE: well this is timed...
#ifdef SUPERBBLAS_USE_NVTX
                // Register this scope of time starting
                nvtxRangePushA(funcName.c_str());
#endif
            }

            ~tracker() { stop(); }

            /// Stop the tracker and store the timing
            void stop() {
#ifdef SUPERBBLAS_USE_NVTX
                if (!reported) {
                    // Register this scope of time finishing
                    nvtxRangePop();
                    reported = true;
                }
#endif

                if (stopped) return;
                stopped = true;

                // Enforce a synchronization
                if (getTrackingTimeSync()) sync(xpu);

                // Count elapsed time since the creation of the object
                elapsedTime =
                    std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();

                // Pop out this call and get a string representing the current call stack
                std::string funcNameWithStack = popCall(xpu.session);

                // Record the time
                auto &timing = getTimings(xpu.session)[funcNameWithStack];
                timing.time += elapsedTime;
                timing.cost += cost;

                // Record memory not released
                if (getTrackingMemory())
                    getCacheUsage(xpu.session)[funcNameWithStack] +=
                        getCpuMemUsed(xpu.session) - mem_cpu + getGpuMemUsed(xpu.session) - mem_gpu;
            }

            /// Stop the tracker and return timing
            double stopAndGetElapsedTime() {
                stop();
                return elapsedTime;
            }

            // Forbid copy constructor and assignment operator
            tracker(const tracker &) = delete;
            tracker &operator=(tracker const &) = delete;
        };
    }

    /// Reset all tracked timings
    inline void resetTimings() {
        for (Session session = 0; session < 256; ++session) getTimings(session).clear();
    }

    /// Report all tracked timings
    /// \param s: stream to write the report

    template <typename OStream> void reportTimings(OStream &s) {
        if (!getTrackingTime()) return;

        // Print the timings alphabetically
        s << "Timing of superbblas kernels:" << std::endl;
        s << "-----------------------------" << std::endl;
        std::vector<std::string> names;
        for (Session session = 0; session < 256; ++session)
            for (const auto &it : getTimings(session)) names.push_back(it.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names) {
            double total = 0, cost = 0;
            for (Session session = 0; session < 256; ++session) {
                auto it = getTimings(session).find(name);
                if (it != getTimings(session).end()) {
                    total += it->second.time;
                    cost += it->second.cost;
                }
            }
            s << name << " : " << total << " s (calls/cost: " << cost
              << " calls/cost_per_sec: " << (std::fabs(total) == 0 ? 0 : cost / total) << " )"
              << std::endl;
        }
    }

    /// Report all tracked cache memory usage
    /// \param s: stream to write the report

    template <typename OStream> void reportCacheUsage(OStream &s) {
        if (!getTrackingMemory()) return;

        // Print the timings alphabetically
        s << "Cache usage of superbblas kernels:" << std::endl;
        s << "-----------------------------" << std::endl;
        std::vector<std::string> names;
        for (Session session = 0; session < 256; ++session)
            for (const auto &it : getCacheUsage(session)) names.push_back(it.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names) {
            double total = 0;
            for (Session session = 0; session < 256; ++session) {
                auto it = getCacheUsage(session).find(name);
                if (it != getCacheUsage(session).end()) total += it->second;
            }
            s << name << " : " << total / 1024 / 1024 / 1024 << " GiB" << std::endl;
        }
    }

    namespace detail {
        /// Structure to store the memory allocations
        /// NOTE: the only instance is expected to be in `getAllocations`.

        struct Allocations : public std::unordered_map<void *, std::size_t> {
            Allocations(std::size_t num_backets)
                : std::unordered_map<void *, std::size_t>{num_backets} {}
        };

        /// Return all current allocations

        inline Allocations &getAllocations(Session session) {
            static std::vector<Allocations> allocs(256, Allocations{16});
            return allocs[session];
        }
    }
}

#endif // __SUPERBBLAS_PERFORMANCE__
