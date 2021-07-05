
#
# If SUPERBBLAS_WITH_MPI is yes, then include support for MPI
#
SUPERBBLAS_WITH_MPI ?= $(if $(findstring mpi,${CXX}),yes,no)
ifeq ($(SUPERBBLAS_WITH_MPI), yes)
    MPISBFLAG ?= -DSUPERBBLAS_USE_MPI
endif

#
# Compiler for cpu target, CXX, compiler flags when compiling for cpu target and for NVCC -Xcompiler
# when compiling for cuda, and general linking flags for all targets
#
CXX ?= c++
CXXFLAGS ?= -g -O0 -Wall -Wextra -fopenmp $(MPISBFLAG)
#CXXFLAGS ?= -Ofast -march=native -DNDEBUG -Wall -Wextra -fopenmp $(MPISBFLAG)
LDFLAGS ?= -lopenblas

#
# Compiler for cuda target, compiler flags, and linking flags
#
NVCC ?= $(shell which nvcc)
CUDADIR ?= $(shell dirname $(NVCC))/..
#NVCCFLAGS ?= -x cu -g -G -arch=sm_75
NVCCFLAGS ?= -x cu -O3 -arch=sm_75
NVCCLDFLAGS ?= -lcublas -lcudart

#
# Building directory
#
BUILDDIR ?= $(PWD)/build

#
# Installation
#
PREFIX ?= $(BUILDDIR)
includedir ?= $(PREFIX)/include
libdir ?= $(PREFIX)/lib
LIBRARY = libsuperbblas.a

#
# Operating system
#
UNAME ?= $(shell uname)

#
# Other tools
#
RANLIB ?= ranlib
PYTHON ?= $(if $(findstring 1984, $(shell python -c 'print(1984)')), \
  python, \
  $(if $(findstring 1984, $(shell python2 -c 'print(1984)')),python2,python3))

