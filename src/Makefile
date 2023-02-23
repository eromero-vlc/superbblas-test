include ../make.inc

MAIN_HEADER := ../include/superbblas.h
OTHER_HEADERS := \
  ../include/superbblas/superbblas_lib.h \
  ../include/superbblas/alloc.h \
  ../include/superbblas/blas_cpu_tmpl.hpp \
  ../include/superbblas/blas.h \
  ../include/superbblas/bsr.h \
  ../include/superbblas/cblas.h \
  ../include/superbblas/cache.h \
  ../include/superbblas/copy_n.h \
  ../include/superbblas/crc32.h \
  ../include/superbblas/dense.h \
  ../include/superbblas/dist.h \
  ../include/superbblas/performance.h \
  ../include/superbblas/platform.h \
  ../include/superbblas/runtime_features.h \
  ../include/superbblas/storage.h \
  ../include/superbblas/template.h \
  ../include/superbblas/template_types.h \
  ../include/superbblas/template_undef.h \
  ../include/superbblas/tensor.h
TESTS := \
  ../tests/blas.cpp \
  ../tests/bsr.cpp \
  ../tests/bsr_hist.cpp \
  ../tests/contract.cpp \
  ../tests/dense.cpp \
  ../tests/dist.cpp \
  ../tests/storage.cpp \
  ../tests/storage_details.cpp

SOURCE := $(MAIN_HEADER) $(OTHER_HEADERS)

INCLUDE := -I../include

#
# Go throw the source code and filter out generated code for superbblas_flags.h
#

$(BUILDDIR)/superbblas_flags.h: $(SOURCE)
	echo "// This file is generated automatically. Please don't modify" > $@
	echo "#ifndef __SUPERBBLAS_SUPERBBLAS_FLAGS__" >> $@
	echo "#define __SUPERBBLAS_SUPERBBLAS_FLAGS__" >> $@
ifeq ($(TARGET), cpu)
	$(CXX) -E $(INCLUDE) ${CXXFLAGS} -DSUPERBBLAS_CREATING_FLAGS ../include/superbblas.h | $(PYTHON) tools/ctemplate >> $@
endif
ifeq ($(TARGET), cuda)
	${NVCC} -E ${MPISBFLAG} ${NVCCFLAGS} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' -DSUPERBBLAS_CREATING_FLAGS ../include/superbblas.h | $(PYTHON) tools/ctemplate >> $@
endif
ifeq ($(TARGET), hip)
	${HIP} -x c++ -E ${MPISBFLAG} ${HIPFLAGS} ${CXXFLAGS} ${MPIINCFLAGS} -DSUPERBBLAS_CREATING_FLAGS ../include/superbblas.h | $(PYTHON) tools/ctemplate >> $@
endif
	echo "#endif // __SUPERBBLAS_SUPERBBLAS_FLAGS__" >> $@

#
# Go throw the source code and filter out generated code for superbblas_lib.cpp
# The file is split in ten pieces to cut down the compilation time
#

LIB_SOURCE_STEM := $(BUILDDIR)/superbblas_lib
MAX_FRAGMENT := 50
LIB_SOURCE_NUMS := $(shell seq 0 $(MAX_FRAGMENT) )
SOURCE_LIB := $(patsubst %,$(LIB_SOURCE_STEM)_%.cpp,$(LIB_SOURCE_NUMS))
OBJECT_LIB := $(patsubst %.cpp,%.o,$(SOURCE_LIB))

$(LIB_SOURCE_STEM): $(SOURCE)
ifeq ($(TARGET), cpu)
	$(CXX) -E $(INCLUDE) ${CXXFLAGS} -DSUPERBBLAS_CREATING_LIB ../include/superbblas.h | $(PYTHON) tools/ctemplate > $@
endif
ifeq ($(TARGET), cuda)
	${NVCC} -E ${MPISBFLAG} ${NVCCFLAGS} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' -DSUPERBBLAS_CREATING_LIB ../include/superbblas.h | $(PYTHON) tools/ctemplate > $@
endif
ifeq ($(TARGET), hip)
	${HIP} -x c++ -E ${MPISBFLAG} ${HIPFLAGS} ${CXXFLAGS} ${MPIINCFLAGS} -DSUPERBBLAS_CREATING_LIB ../include/superbblas.h | $(PYTHON) tools/ctemplate > $@
endif

$(SOURCE_LIB): $(LIB_SOURCE_STEM).split

$(LIB_SOURCE_STEM).split: $(LIB_SOURCE_STEM)
	lines_frag="$$(( ($$( wc -l < $(LIB_SOURCE_STEM)) + $(MAX_FRAGMENT)) / ($(MAX_FRAGMENT)+1) ))"; \
	for i in $(LIB_SOURCE_NUMS) ; do \
		f="$(LIB_SOURCE_STEM)_$${i}.cpp"; \
		echo "// This file is generated automatically. Please, do not modify" > $$f; \
		echo "#include \"superbblas.h\"" >> $$f; \
		echo "using namespace superbblas; using namespace detail;" >> $$f;  \
		l=0; \
		cat $(LIB_SOURCE_STEM) | while read line; do \
			[ $$(( l/lines_frag )) -ne $$i ] && echo extern $$line; \
			l="$$(( l+1 ))"; \
		done >> $$f; \
		l=0; \
		cat $(LIB_SOURCE_STEM) | while read line; do \
			[ $$(( l/lines_frag )) -eq $$i ] && echo $$line; \
			l="$$(( l+1 ))"; \
		done >> $$f; \
	done; \
	touch $(LIB_SOURCE_STEM).split

%.o: %.cpp
ifeq ($(TARGET), cpu)
	${CXX} -c $(INCLUDE) ${CXXFLAGS} $< -o $@
endif
ifeq ($(TARGET), cuda)
	${NVCC} -c ${INCLUDE} ${MPISBFLAG} ${NVCCFLAGS} ${NVCCSTDLANG} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' $< -o $@
endif
ifeq ($(TARGET), hip)
	${HIP} -c ${INCLUDE} ${MPISBFLAG} ${HIPFLAGS} ${HIPSTDLANG} ${CXXFLAGS} ${MPIINCFLAGS} $< -o $@
endif


$(OBJECT_LIB): $(BUILDDIR)/superbblas_flags.h

pkgconfig_file := $(libdir)/pkgconfig/superbblas.pc

install: $(OBJECT_LIB) install_headers $(pkgconfig_file)
	@rm -f $(LIBRARY)
	install -d $(libdir)
ifeq ($(UNAME), Darwin)
	libtool -static -o $(LIBRARY) $(OBJECT_LIB) -no_warning_for_no_symbols
else
	$(AR) r $(LIBRARY) $(OBJECT_LIB)
	$(RANLIB) $(LIBRARY)
endif
	install -m 644 $(LIBRARY) $(libdir)

install_headers: $(BUILDDIR)/superbblas_flags.h
	install -d $(includedir)
	install -d $(includedir)/superbblas
	install -m 644 $(MAIN_HEADER) $(includedir)
	install -m 644 $(BUILDDIR)/superbblas_flags.h $(includedir)/superbblas
ifeq ($(INSTALL_LINK_SOURCE), yes)
	for i in $(OTHER_HEADERS) ; do ln -fs $$PWD/$$i $(includedir)/superbblas/ ; done
else
	install -m 644 $(OTHER_HEADERS) $(includedir)/superbblas
endif

$(pkgconfig_file):
	install -d $(libdir)/pkgconfig
	echo prefix=$(PREFIX) > $@
	echo libdir=$(libdir) >> $@
	echo includedir=$(includedir) >> $@
	echo superbblas_config=$(CXXFLAGS) $(MPISBFLAG) >> $@
	echo "Name: superbblas" >> $@
	echo "Description: tensor library based on BLAS" >> $@
	echo "Version: 0.1" >> $@
	echo "URL: https://github.com/eromero-vlc/superbblas" >> $@
	echo "Libs: -L\$${libdir} -lsuperbblas" >> $@
ifeq ($(TARGET), cuda)
	echo "Libs.private: cublas" >> $@
endif
ifeq ($(TARGET), hip)
	echo "Libs.private: hipblas" >> $@
endif

format:
	clang-format -i ${MAIN_HEADER} ${OTHER_HEADERS} $(TESTS)

clean:
	rm -f $(BUILDDIR)/superbblas_flags.h $(BUILDDIR)/superbblas_lib.cpp $(BUILDDIR)/superbblas_lib.o $(LIBRARY)

.PHONY: install install_headers format clean $(pkgconfig_file)
