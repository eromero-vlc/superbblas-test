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

namespace superbblas {

    /// Type for storing the timings
    using Timings = std::unordered_map<std::string, double>;

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
        static std::array<double, 256> mem{};
        return mem[session];
    }

    /// Get total memory allocated on devices if tracking memory consumption (see `getTrackingMemory`)

    inline double &getGpuMemUsed(Session session) {
        static std::array<double, 256> mem{};
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

        /// Return the current function call stack begin tracked
        inline CallStack &getCallStack(Session session) {
            static std::vector<CallStack> callStack(256, CallStack{});
            return callStack[session];
        }

        /// Push function call to be tracked
        inline void pushCall(std::string funcName, Session session) {
            getCallStack(session).push_back(funcName);

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
            assert(getCallStack(session).size() > 0);
            std::string back = getCallStackWithPath(session).back();
            getCallStack(session).pop_back();
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
            /// Name of the function being tracked
            const std::string funcName;
            /// Memory usage at that point
            const double mem_cpu, mem_gpu;
            /// Instant of the construction
            const std::chrono::time_point<std::chrono::system_clock> start;
            /// Session
            const Session session;

            /// Start a tracker
            tracker(std::string funcName, XPU xpu)
                : stopped(!getTrackingTime()),
                  funcName(!stopped ? funcName : std::string()),
                  mem_cpu(getTrackingMemory() ? getCpuMemUsed(xpu.session) : 0),
                  mem_gpu(getTrackingMemory() ? getGpuMemUsed(xpu.session) : 0),
                  start(!stopped ? std::chrono::system_clock::now()
                                 : std::chrono::time_point<std::chrono::system_clock>{}),
                  session(xpu.session) {
                if (!stopped) pushCall(funcName, session); // NOTE: well this is timed...
            }

            ~tracker() { stop(); }

            /// Stop the tracker and store the timing
            void stop() {
                if (stopped) return;
                stopped = true;

                // Count elapsed time since the creation of the object
                double elapsedTime =
                    std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();

                // Pop out this call
                std::string category = popCall(session);

                // If this function is not recursive, store the timings in the category with its name only
                const auto &stack = getCallStack(session);
                if (std::find(stack.begin(), stack.end(), funcName) == stack.end())
                    getTimings(session)[funcName] += elapsedTime;

                // If this is not the first function being tracked, store the timings in the
                // category with its path name
                if (category != funcName) getTimings(session)[category] += elapsedTime;

                // Record memory not released
                if (getTrackingMemory())
                    getCacheUsage(session)[funcName] +=
                        getCpuMemUsed(session) - mem_cpu + getGpuMemUsed(session) - mem_gpu;
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
            double total = 0;
            for (Session session = 0; session < 256; ++session) {
                auto it = getTimings(session).find(name);
                if (it != getTimings(session).end()) total += it->second;
            }
            s << name << " : " << total << std::endl;
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
            // Make sure of no usage of the instance after its destruction
            ~Allocations() { getTrackingMemory() = false; }
        };

        /// Return all current allocations

        inline Allocations &getAllocations(Session session) {
            static std::vector<Allocations> allocs(256, Allocations{16});
            return allocs[session];
        }
    }
}

#endif // __SUPERBBLAS_PERFORMANCE__
