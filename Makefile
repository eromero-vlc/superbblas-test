
include make.inc

install_cpu:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean TARGET=cpu
	@$(MAKE) -C src install TARGET=cpu

install_cuda:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean TARGET=cuda CXX=
	@$(MAKE) -C src install TARGET=cuda CXX=

install_hip:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean TARGET=hip CXX=
	@$(MAKE) -C src install TARGET=hip CXX=

test_cpu:
	@$(MAKE) -C tests clean all_cpu

test_cuda: install_cuda
	@$(MAKE) -C tests clean all_cuda_lib

test_header_only_cuda:
	@$(MAKE) -C tests clean all_cuda

test_header_only_hip:
	@$(MAKE) -C tests clean all_hip

test_header_only_cpu test_header_only_cuda test_header_only_hip: export SB_LDFLAGS :=
test_header_only_cpu test_header_only_cuda test_header_only_hip: export SB_INCLUDE := -I../include
test_header_only_cpu: test_cpu

format:
	${MAKE} -C src format

.PHONY: install_cpu install_cuda install_hip test_cpu test_cuda test_hip test_header_only_cuda test_header_only_hip format
.NOTPARALLEL:
