
include make.inc

install_cpu:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean install TARGET=cpu

install_cuda:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean install TARGET=cuda

test_cpu:
	@$(MAKE) -C tests clean all_cpu

test_cuda:
	@$(MAKE) -C tests clean all_cuda_lib

test_header_only_cuda:
	@$(MAKE) -C tests clean all_cuda

test_header_only_cpu test_header_only_cuda: export SB_LDFLAGS :=
test_header_only_cpu test_header_only_cuda: export SB_INCLUDE := -I../include
test_header_only_cpu: test_cpu

format:
	${MAKE} -C src format

.PHONY: lib_cpu lib_cuda clean tests
