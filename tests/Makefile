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

BUILDDIR ?= ..
SB_LDFLAGS ?=

include ../make.inc

SOURCES := blas.cpp local.cpp dist.cpp storage.cpp

CPU_TARGETS := $(patsubst %.cpp,%_cpu,$(SOURCES))
CUDA_TARGETS := $(patsubst %.cpp,%_cuda,$(SOURCES))
CUDA_LIB_TARGETS := $(patsubst %.cpp,%_cuda_lib,$(SOURCES))
HIP_TARGETS := $(patsubst %.cpp,%_hip,$(SOURCES))

SB_INCLUDE ?= -I$(BUILDDIR)/include
MPISBFLAG ?= $(if $(findstring mpi,${CXX}),-DSUPERBBLAS_USE_MPI,)

MPIINCFLAGS ?= $(if $(findstring mpi,${CXX}),$(shell ${CXX} -showme:compile),)
comma := ,
MPILDFLAGS ?= $(if $(findstring mpi,${CXX}),\
        $(filter-out -pthread -fexceptions, \
            $(subst -Wl$(comma),-Xlinker ,$(shell ${CXX} -showme:link))),)
SB_LDFLAGS ?= -L$(BUILDDIR)/lib -lsuperbblas
CUDAINCLUDE ?= -I$(CUDADIR)/include
CUDALDFLAGS ?= -L$(CUDADIR)/lib64
NVCCSTDLANG ?= -std c++14

all_cpu: $(CPU_TARGETS)
all_cuda: $(CUDA_TARGETS)
all_cuda_lib: $(CUDA_LIB_TARGETS)
all_hip: $(HIP_TARGETS)

$(CPU_TARGETS): %_cpu: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${SB_LDFLAGS} ${LDFLAGS}

$(CUDA_TARGETS): %_cuda: %.cpp
	${NVCC} ${SB_INCLUDE} ${MPISBFLAG} ${NVCCFLAGS} ${NVCCSTDLANG} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' \
	        ${SB_LDFLAGS} ${NVCCLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} $< -o $@

$(CUDA_LIB_TARGETS): %_cuda_lib: %.cpp
	${CXX} ${SB_INCLUDE} ${CUDAINCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${SB_LDFLAGS} ${CUDALDFLAGS} ${NVCCLDFLAGS} ${LDFLAGS}

$(HIP_TARGETS): %_hip: %.cpp
	${HIP} ${SB_INCLUDE} ${MPISBFLAG} ${HIPFLAGS} ${HIPSTDLANG} ${CXXFLAGS} ${MPIINCFLAGS} \
	        ${SB_LDFLAGS} ${HIPLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} $< -o $@

clean:
	rm -f ${CPU_TARGETS} ${CUDA_TARGETS} ${CUDA_LIB_TARGETS} ${HIP_TARGETS}

known_issues:
	# MPI_Ialltoallv hungs
	mpirun -np 4 ./dist_cpu --dim='2 1 1 2 1' --procs='2 1 1 2'

.PHONY: clean ${CPU_TARGETS} ${CUDA_TARGETS} ${CUDA_LIB_TARGETS} ${HIP_TARGETS}
