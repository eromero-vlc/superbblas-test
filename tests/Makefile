###
# For each of the source files in ${SOURCES}, this makefile defines the
# actions %_cpu and %_cuda to compile with CXX and NVCC respectively.
# If ${CXX} contains mpi, then the macro SUPERBBLAS_USE_MPI is defined to
# activate the MPI portion of the tests and the library.
#
# Examples of actions:
#
#  make local_cpu             # builds local.cpp with ${CXX} compiler
#  make local_cuda dist_cuda  # builds both sources with ${NVCC} compiler
#  make dist_cuda CXX=mpicxx  # builds dist.cpp with ${NVCC} adding the flags
#                             # and linking information for MPI

SOURCES := local.cpp dist.cpp

CPU_TARGETS := $(patsubst %.cpp,%_cpu,$(SOURCES))
CUDA_TARGETS := $(patsubst %.cpp,%_cuda,$(SOURCES))

SB_INCLUDE := -I../include
MPISBFLAG ?= $(if $(findstring mpi,${CXX}),-DSUPERBBLAS_USE_MPI,)

CXXFLAGS ?= -g -O0 -Wall -Wextra -fopenmp
MPIINCFLAGS ?= $(if $(findstring mpi,${CXX}),$(shell ${CXX} -showme:compile),)
comma := ,
MPILDFLAGS ?= $(if $(findstring mpi,${CXX}),\
        $(filter-out -pthread -fexceptions, \
            $(subst -Wl$(comma),-Xlinker ,$(shell ${CXX} -showme:link))),)
LDFLAGS ?= -lopenblas
NVCCLDFLAGS ?= -lcublas -lopenblas
NVCC ?= nvcc
NVCCFLAGS ?= -x cu -g -G -arch=sm_75
NVCCSTDLANG ?= -std c++14

$(CPU_TARGETS): %_cpu: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} ${STDLANG} $< -o $@ ${LDFLAGS}

$(CUDA_TARGETS): %_cuda: %.cpp
	${NVCC} ${SB_INCLUDE} ${MPISBFLAG} ${NVCCFLAGS} ${NVCCSTDLANG} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' \
	        ${NVCCLDFLAGS} ${MPILDFLAGS} $< -o $@

clean:
	rm -f ${CPU_TARGETS} ${CUDA_TARGETS}

.PHONY: ${CPU_TARGETS} ${CUDA_TARGETS}
