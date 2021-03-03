#ifndef __SUPERBBLAS_PERFORMANCE__
#define __SUPERBBLAS_PERFORMANCE__

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
    Timings &getTimings() {
        static Timings timings(16);
        return timings;
    }

    namespace detail {

        /// Stack of function calls being tracked
        using CallStack = std::vector<std::string>;

        /// Return the current function call stack begin tracked
        inline CallStack &getCallStackWithPath() {
            static CallStack callStack{};
            return callStack;
        }

        /// Return the current function call stack begin tracked
        inline CallStack &getCallStack() {
            static CallStack callStack{};
            return callStack;
        }

        /// Push function call to be tracked
        inline void pushCall(std::string funcName) {
            getCallStack().push_back(funcName);

            if (getCallStackWithPath().empty()) {
                // If the stack is empty, just append the function name
                getCallStackWithPath().push_back(funcName);
            } else {
                // Otherwise, push the previous one appending "/`funcName`"
                getCallStackWithPath().push_back(getCallStackWithPath().back() + "/" + funcName);
            }
        }

        /// Pop function call from the stack
        inline std::string popCall() {
            assert(getCallStack().size() > 0);
            std::string back = getCallStackWithPath().back();
            getCallStack().pop_back();
            getCallStackWithPath().pop_back();
            return back;
        }

        /// Return the number of seconds from some start
        inline double w_time() {
            return std::chrono::duration<double>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        }

        /// Track time between creation and destruction of the object
        struct tracker {
            /// Whether the tacker has been stopped
            bool stopped;
            /// Name of the function being tracked
            const std::string funcName;
            /// Instant of the construction
            const std::chrono::time_point<std::chrono::system_clock> start;

            /// Start a tracker
            tracker(std::string funcName)
                : stopped(!getTrackingTime()),
                  funcName(!stopped ? funcName : std::string()),
                  start(!stopped ? std::chrono::system_clock::now()
                                 : std::chrono::time_point<std::chrono::system_clock>{}) {
                if (!stopped) pushCall(funcName); // NOTE: well this is timed...
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
                std::string category = popCall();

                // If this function is not recursive, store the timings in the category with its name only
                const auto &stack = getCallStack();
                if (std::find(stack.begin(), stack.end(), funcName) == stack.end())
                    getTimings()[funcName] += elapsedTime;

                // If this is not the first function being tracked, store the timings in the
                // category with its path name
                if (category != funcName) getTimings()[category] += elapsedTime;
            }
        };
    }

    /// Reset all tracked timings
    inline void resetTimings() { getTimings().clear(); }

    /// Report all tracked timings
    /// \param s: stream to write the report

    template <typename OStream> void reportTimings(OStream &s) {
        if (!getTrackingTime()) return;

        // Print the timings alphabetically
        s << "Timing of superbblas kernels:" << std::endl;
        s << "-----------------------------" << std::endl;
        std::vector<std::string> names;
        for (const auto &it : getTimings()) names.push_back(it.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names) s << name << " : " << getTimings()[name] << std::endl;
    }

    /// Get total memory allocated on the host/cpu if tracking memory consumption (see `getTrackingMemory`)

    inline double &getCpuMemUsed() {
        static double mem = 0;
        return mem;
    }

    /// Get total memory allocated on devices if tracking memory consumption (see `getTrackingMemory`)

    inline double &getGpuMemUsed() {
        static double mem = 0;
        return mem;
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

        inline Allocations &getAllocations() {
            static Allocations allocs(16);
            return allocs;
        }
    }
}

#endif // __SUPERBBLAS_PERFORMANCE__
