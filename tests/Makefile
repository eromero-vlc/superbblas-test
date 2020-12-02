
SOURCES := local.cpp dist.cpp

CPU_TARGETS := $(patsubst %.cpp,%_cpu,$(SOURCES))
CUDA_TARGETS := $(patsubst %.cpp,%_cuda,$(SOURCES))

SB_INCLUDE := -I../include
MPISBFLAG ?= $(if $(findstring mpi,${CXX}),-DSUPERBBLAS_USE_MPI,)

CXXFLAGS ?= -g -O0 -Wall -Wextra -fopenmp
MPIINCFLAGS ?= $(if $(findstring mpi,${CXX}),$(shell $CXX -show:compile),)
MPILDFLAGS ?= $(if $(findstring mpi,${CXX}),$(shell $CXX -show:link),)
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
