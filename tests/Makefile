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

SHELL = /bin/bash
BUILDDIR ?= ..
SB_LDFLAGS ?=

include ../make.inc

SOURCES := blas.cpp dist.cpp contract.cpp storage.cpp storage_details.cpp bsr.cpp dense.cpp

CPU_TARGETS := $(patsubst %.cpp,%_cpu,$(SOURCES))
CUDA_TARGETS := $(patsubst %.cpp,%_cuda,$(SOURCES))
HIP_TARGETS := $(patsubst %.cpp,%_hip,$(SOURCES))
CPU_LIB_TARGETS := $(patsubst %.cpp,%_cpu_lib,$(SOURCES))
CUDA_LIB_TARGETS := $(patsubst %.cpp,%_cuda_lib,$(SOURCES))
HIP_LIB_TARGETS := $(patsubst %.cpp,%_hip_lib,$(SOURCES))

SB_INCLUDE ?= -I$(BUILDDIR)/include
MPISBFLAG ?= $(if $(findstring mpi,${CXX}),-DSUPERBBLAS_USE_MPI,)

comma := ,
MPIINCFLAGS ?= $(if $(findstring mpi,${CXX}),$(patsubst -Wl$(comma)%,,$(shell ${CXX} -showme:compile 2>/dev/null || ${CXX} -show -CC=)),)
MPILDFLAGS ?= $(if $(findstring mpi,${CXX}),\
        $(filter-out -pthread -fexceptions, \
            $(subst -Wl$(comma),-Xlinker ,$(shell ${CXX} -showme:link 2>/dev/null || ${CXX} -show -CC=))),)
SB_LDFLAGS := -L$(BUILDDIR)/lib -lsuperbblas
CUDAINCLUDE ?= -I$(CUDADIR)/include
CUDALDFLAGS ?= -L$(CUDADIR)/lib64
NVCCSTDLANG ?= -std c++14

all_cpu: $(CPU_TARGETS)
all_cuda: $(CUDA_TARGETS)
all_hip: $(HIP_TARGETS)
all_cpu_lib: $(CPU_LIB_TARGETS)
all_cuda_lib: $(CUDA_LIB_TARGETS)
all_hip_lib: $(HIP_LIB_TARGETS)

$(CPU_TARGETS): %_cpu: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${LDFLAGS}

$(CPU_LIB_TARGETS): %_cpu_lib: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${SB_LDFLAGS} ${LDFLAGS}

storage_details: storage_details.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${LDFLAGS}

$(CUDA_TARGETS): %_cuda: %.cpp
	${NVCC} ${SB_INCLUDE} ${MPISBFLAG} ${NVCCFLAGS} ${NVCCSTDLANG} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' \
	        ${NVCCLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} $< -o $@

$(CUDA_LIB_TARGETS): %_cuda_lib: %.cpp
	${CXX} ${SB_INCLUDE} ${CUDAINCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${SB_LDFLAGS} ${CUDALDFLAGS} ${NVCCLDFLAGS} ${LDFLAGS}

$(HIP_TARGETS): %_hip: %.cpp
	${HIP} ${SB_INCLUDE} ${MPISBFLAG} ${HIPFLAGS} ${HIPSTDLANG} ${CXXFLAGS} ${MPIINCFLAGS} \
	       ${HIPLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} $< -o $@

$(HIP_LIB_TARGETS): %_hip_lib: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${HIPFLAGS} ${HIPSTDLANG} ${CXXFLAGS} ${MPIINCFLAGS} \
	       ${SB_LDFLAGS} ${HIPLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} $< -o $@

all_cpu all_cpu_lib all_cuda all_cuda_lib all_hip all_hip_lib: all_%:
	./blas_$*
	./storage_$*
	SB_DEBUG=5 ./bsr_$* --dim='2 2 2 2 2 2'
ifeq ($(SUPERBBLAS_WITH_MPI), yes)
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np 6 --oversubscribe ./dist_$*  --procs='1 1 2 3' --dim='4 4 4 4 64'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np 3 --oversubscribe ./dist_$*  --procs='1 1 1 3' --dim='4 4 4 2 64'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np 6 --oversubscribe ./contract_$*  --procs='1 1 2 3' --dim='2 2 4 4 2'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np 4 --oversubscribe ./dense_$*  --procs='1 1 2 2' --dim='4 4 4 4 4 4'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np 4 --oversubscribe ./bsr_$*  --procs='1 1 2 2' --dim='4 4 4 4 4 4'
endif

clean:
	rm -f ${CPU_TARGETS} ${CPU_LIB_TARGETS} ${CUDA_TARGETS} ${CUDA_LIB_TARGETS} ${HIP_TARGETS} ${HIP_LIB_TARGETS} storage_details

known_issues:
	# MPI_Ialltoallv hungs
	SB_DEBUG=5 mpirun -np 4 ./dist_cpu --dim='2 1 1 2 1' --procs='2 1 1 2'
	SB_DEBUG=5 mpirun -np 6 ./dist_cpu  --procs='1 1 2 3' --dim='4 4 4 4 64'

.PHONY: clean ${CPU_TARGETS} ${CPU_LIB_TARGETS} ${CUDA_TARGETS} ${CUDA_LIB_TARGETS} ${HIP_TARGETS} ${HIP_LIB_TARGETS} storage_details
