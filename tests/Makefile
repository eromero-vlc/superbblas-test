
SOURCES := local.cpp

CPU_TARGETS := $(patsubst %.cpp,%_cpu,$(SOURCES))
CUDA_TARGETS := $(patsubst %.cpp,%_cuda,$(SOURCES))

SB_INCLUDE := -I../include

CXXFLAGS ?= -g -O0 -Wall -Wextra -fopenmp
LDFLAGS ?= -lopenblas
NVCCLDFLAGS ?= -lcublas -lopenblas
NVCC ?= nvcc
NVCCFLAGS ?= -x cu -g -G -arch=sm_61 
NVCCSTDLANG ?= -std c++11

$(CPU_TARGETS): %_cpu: %.cpp
	${CXX} ${SB_INCLUDE} ${CXXFLAGS} ${STDLANG} $< -o $@ ${LDFLAGS}

$(CUDA_TARGETS): %_cuda: %.cpp
	${NVCC} ${SB_INCLUDE} ${NVCCFLAGS} ${NVCCSTDLANG} -Xcompiler '${CXXFLAGS}' ${NVCCLDFLAGS} $< -o $@

clean:
	rm -f ${CPU_TARGETS} ${CUDA_TARGETS}
