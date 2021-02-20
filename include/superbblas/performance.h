#ifndef __SUPERBBLAS_PERFORMANCE__
#define __SUPERBBLAS_PERFORMANCE__

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
        CallStack &getCallStack() {
            static CallStack callStack{};
            return callStack;
        }

        /// Push function call to be tracked
        void pushCall(std::string funcName) {
            if (getCallStack().empty()) {
		// If the stack is empty, just append the function name
                getCallStack().push_back(funcName);
            } else {
		// Otherwise, push the previous one appending "/`funcName`"
                getCallStack().push_back(getCallStack().back() + "/" + funcName);
            }
        }

        /// Pop function call from the stack
        std::string popCall() {
            assert(getCallStack().size() > 0);
            std::string back = getCallStack().back();
            getCallStack().pop_back();
            return back;
        }

        enum TrackSubfunctions { DoTrackSubfunction, NotTrackSubfunctions };

        /// Track time between creation and destruction of the object
        struct tracker {
            /// Name of the function being tracked
            const std::string funcName;
            /// Track subfunctions
            const TrackSubfunctions trackSubfunctions;
            /// Instant of the construction
            const std::chrono::time_point<std::chrono::system_clock> start;

            /// Start a tracker
            tracker(std::string funcName, TrackSubfunctions trackSubfunctions = DoTrackSubfunction)
                : funcName(funcName),
                  trackSubfunctions(trackSubfunctions),
                  start(std::chrono::system_clock::now()) {
                if (trackSubfunctions == DoTrackSubfunction) pushCall(funcName);
            }

            /// Stop the tracker and store the timing
            ~tracker() {
                double elapsedTime =
                    std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
                getTimings()[funcName] += elapsedTime;
                if (trackSubfunctions == DoTrackSubfunction) {
                    std::string category = popCall();
                    if (category != funcName) getTimings()[category] += elapsedTime;
                }
            }
        };
    }

    void resetTiming() { getTimings().clear(); }

    template <typename OStream> void reportTimings(OStream &s) {
        // Print the timings alphabetically
	s << "Timing of superbblas kernels:" << std::endl;
	s << "-----------------------------" << std::endl;
        std::vector<std::string> names;
        for (const auto &it : getTimings()) names.push_back(it.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names)
            s << name << " : " << getTimings()[name] << std::endl;
    }
}

#endif // __SUPERBBLAS_PERFORMANCE__
