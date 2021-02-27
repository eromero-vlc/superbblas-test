#ifndef __SUPERBBLAS_RUNTIME_FEATURES__
#define __SUPERBBLAS_RUNTIME_FEATURES__

#include <algorithm>
#include <cstdlib>

namespace superbblas {

    /// Return the log level that may have been set by the environment variable SB_LOG
    /// \return int: log level
    /// The current log levels are:
    ///   * 0: no log (default)
    ///   * 1: some log
    
    int getLogLevel() {
        static int log_level = []() {
            const char *l = std::getenv("SB_LOG");
            if (l) return std::max(0, std::atoi(l));
            return 0;
        }();
        return log_level;
    }

    /// Return the debug level that may have been set by the environment variable SB_DEBUG
    /// \return int: debug level
    /// The current log levels are:
    ///   * 0: no extra checking (default)
    ///   * >= 1: GPU sync and MPI barriers before and after `copy` and `contraction`
    ///   * >= 2: verify all `copy` calls (expensive)
    
    int getDebugLevel() {
        static int debug_level = []() {
            const char *l = std::getenv("SB_DEBUG");
            if (l) return std::max(0, std::atoi(l));
            return 0;
        }();
        return debug_level;
    }

    /// Return whether to track memory consumption, which may have been set by the environment variable SB_TRACK_MEM
    /// \return bool: whether to track memory consumption
    /// The accepted value in the environment variable SB_TRACK_MEM are:
    ///   * 0: no tracking memory consumption (default)
    ///   * != 0: tracking memory consumption

    bool& getTrackingMemory() {
        static bool track_mem = []() {
            const char *l = std::getenv("SB_TRACK_MEM");
            if (l) return (0 != std::atoi(l));
            return false;
        }();
        return track_mem;
    }

    /// Return whether to track timings, which may have been set by the environment variable SB_TRACK_TIME
    /// \return bool: whether to track the time that critical functions take
    /// The accepted value in the environment variable SB_TRACK_TIME are:
    ///   * 0: no tracking time (default)
    ///   * != 0: tracking time

    bool& getTrackingTime() {
        static bool track_time = []() {
            const char *l = std::getenv("SB_TRACK_TIME");
            if (l) return (0 != std::atoi(l));
            return false;
        }();
        return track_time;
    }

    /// Return whether to use asynchronous MPI_Alltoall in `copy`, which may have been set by the environment variable SB_ASYNC_ALLTOALL
    /// \return bool: whether to use the asynchronous version of MPI_Alltoall
    /// The accepted value in the environment variable SB_ASYNC_ALLTOALL are:
    ///   * 0: use the synchronous version MPI_Alltoallv
    ///   * != 0: use the asynchronous version MPI_Ialltoallv (default)

    bool getUseAsyncAlltoall() {
        static bool async_alltoall = []() {
            const char *l = std::getenv("SB_ASYNC_ALLTOALL");
            if (l) return (0 != std::atoi(l));
            return true;
        }();
        return async_alltoall;
    }
}

#endif // __SUPERBBLAS_RUNTIME_FEATURES__
