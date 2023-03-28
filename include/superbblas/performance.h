#ifndef __SUPERBBLAS_PERFORMANCE__
#define __SUPERBBLAS_PERFORMANCE__

#include "platform.h"
#include "runtime_features.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <complex>
#include <iomanip>
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

#ifdef SUPERBBLAS_USE_GPU
    using GpuEvent = SUPERBBLAS_GPU_SYMBOL(Event_t);
    using TimingGpuEvent = std::array<GpuEvent, 2>;

    /// A list of pairs of starting and ending timing events
    using TimingGpuEvents = std::vector<TimingGpuEvent>;
#endif

    /// Performance metrics, time, memory usage, etc
    struct Metric {
        double cpu_time;   ///< wall-clock time for the cpu
        double gpu_time;   ///< wall-clock time for the gpu
        double cost;       ///< flops or entities processed (eg, rhs for matvecs)
        double max_mem;    ///< memory usage in bytes
        std::size_t calls; ///< number of times the function was called
#ifdef SUPERBBLAS_USE_GPU
        /// List of start-end gpu events for calls of this function in course
        TimingGpuEvents timing_events;
#endif
        Metric() : cpu_time(0), gpu_time(0), cost(0), max_mem(0), calls(0) {}
    };

    /// Type for storing the timings
    using Timings = std::unordered_map<std::string, Metric>;

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

    namespace detail {

        /// Template namespace for managing the gpu timings
        template <typename XPU> struct TimingEvents;

#ifdef SUPERBBLAS_USE_GPU
        template <> struct TimingEvents<Gpu> {
            /// Gpu timing event
            using TimingEvent = TimingGpuEvent;

            /// Extract the timings from the recorded events just finished, remove them from the vector,
            /// and return the accumulated time
            /// \param events: (in/out) vector of events to inspect

            static double processEvents(TimingGpuEvents &events) {
                double new_time = 0;
                events.erase(std::remove_if(events.begin(), events.end(),
                                            [&](const TimingGpuEvent &ev) {
                                                // Try to get the elapsed time between the two events
                                                float ms = 0;
                                                auto err = SUPERBBLAS_GPU_SYMBOL(EventElapsedTime)(
                                                    &ms, ev[0], ev[1]);
                                                if (err == SUPERBBLAS_GPU_SYMBOL(Success)) {
                                                    // If successful, register the time and erase the entry in the vector
                                                    new_time += ms / 1000.0;
                                                    return true;
                                                } else {
                                                    // Otherwise, do nothing
                                                    return false;
                                                }
                                            }),
                             events.end());
                return new_time;
            }

            /// Return a gpu timing event and start counting
            /// \param xpu: context

            static TimingEvent startRecordingEvent(const Gpu &xpu) {
                TimingEvent tev;
                setDevice(xpu);
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventCreate)(&tev[0]));
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventCreate)(&tev[1]));
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventRecord)(tev[0], getStream(xpu)));
                return tev;
            }

            /// Mark the end of a recording
            /// \param tev: gpu timing event
            /// \param xpu: context

            static void endRecordingEvent(const TimingEvent &tev, const Gpu &xpu) {
                setDevice(xpu);
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventRecord)(tev[1], getStream(xpu)));
            }

            static void updateGpuTimingEvents(Metric &metric) {
                metric.gpu_time += processEvents(metric.timing_events);
            }

            static void updateGpuTimingEvents(Metric &metric, const TimingEvent &tev) {
                updateGpuTimingEvents(metric);
                metric.timing_events.push_back(tev);
            }
        };
#endif

        /// Dummy implementation of `TimingEvents` for cpu

        template <> struct TimingEvents<Cpu> {
            using TimingEvent = char;
            static TimingEvent startRecordingEvent(const Cpu &) { return 0; }
            static void endRecordingEvent(const TimingEvent &, const Cpu &) {}
            static void updateGpuTimingEvents(Metric &) {}
            static void updateGpuTimingEvents(Metric &, const TimingEvent &) {}
        };

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
            /// Cpu elapsed time
            double elapsedTime;
            /// Gpu starting and ending events
            typename TimingEvents<XPU>::TimingEvent timingEvent;
            /// Equivalent units of cost
            double cost;

            /// Start a tracker
            tracker(const std::string &funcName, XPU xpu, bool timeAnyway = false)
                : stopped(!(timeAnyway || getTrackingTime())),
#ifdef SUPERBBLAS_USE_NVTX
                  reported(false),
#endif
                  funcName(funcName),
                  mem_cpu(getTrackingMemory() ? getCpuMemUsed(xpu.session) : 0),
                  mem_gpu(getTrackingMemory() ? getGpuMemUsed(xpu.session) : 0),
                  start(!stopped ? std::chrono::system_clock::now()
                                 : std::chrono::time_point<std::chrono::system_clock>{}),
                  xpu(xpu),
                  elapsedTime(0),
                  cost(0) {

                if (!stopped) {
                    pushCall(funcName, xpu.session);
                    timingEvent = TimingEvents<XPU>::startRecordingEvent(xpu);
                }
#ifdef SUPERBBLAS_USE_NVTX
                // Register this scope of time starting
                nvtxRangePushA(this->funcName.c_str());
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

                // Record gpu ending event
                TimingEvents<XPU>::endRecordingEvent(timingEvent, xpu);

                // Enforce a synchronization
                if (getTrackingTimeSync()) sync(xpu);

                // Count elapsed time since the creation of the object
                elapsedTime =
                    std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();

                // Pop out this call and get a string representing the current call stack
                std::string funcNameWithStack = popCall(xpu.session);

                // Record the time
                auto &timing = getTimings(xpu.session)[funcNameWithStack];
                timing.cpu_time += elapsedTime;
                timing.cost += cost;
                timing.calls++;
                TimingEvents<XPU>::updateGpuTimingEvents(timing, timingEvent);

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
            double cpu_time = 0, gpu_time = 0, cost = 0, calls = 0;
            for (Session session = 0; session < 256; ++session) {
                auto it = getTimings(session).find(name);
                if (it != getTimings(session).end()) {
#ifdef SUPERBBLAS_USE_GPU
                    detail::TimingEvents<detail::Gpu>::updateGpuTimingEvents(it->second);
                    gpu_time += it->second.gpu_time;
#endif
                    cpu_time += it->second.cpu_time;
                    cost += it->second.cost;
                    calls += it->second.calls;
                }
            }
            double time = (gpu_time > 0 ? gpu_time : cpu_time);
            double gcost_per_sec = (time > 0 ? cost / time : 0) / 1024u / 1024u / 1024u;
            s << name << " : " << std::fixed << std::setprecision(3) << cpu_time << " s ("
#ifdef SUPERBBLAS_USE_GPU
              << "gpu_time: " << gpu_time << " "
#endif
              << "calls: " << std::setprecision(0) << calls << " cost: " << cost << std::scientific
              << std::setprecision(3) << " gcost_per_sec: " << gcost_per_sec << " )" << std::endl;
        }
        s << std::defaultfloat;
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
