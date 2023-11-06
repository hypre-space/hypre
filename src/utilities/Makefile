# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = ${TIMERDEFS}

C_COMPILE_FLAGS =\
   -I..\
   -I$(srcdir)/..\
   -I$(srcdir)/../struct_mv\
   -I$(srcdir)\
   ${CINCLUDES}\
   ${CDEFS}

CXX_COMPILE_FLAGS = ${C_COMPILE_FLAGS}

HEADERS =\
 amg_linklist.h\
 exchange_data.h\
 fortran.h\
 fortran_matrix.h\
 general.h\
 int_array.h\
 memory.h\
 memory_tracker.h\
 HYPRE_utilities.h\
 mpistubs.h\
 threading.h\
 timing.h\
 magma.h\
 _hypre_onedpl.hpp\
 _hypre_utilities.h\
 _hypre_utilities.hpp

FILES =\
 F90_HYPRE_error.c\
 F90_HYPRE_general.c\
 HYPRE_handle.c\
 HYPRE_version.c\
 amg_linklist.c\
 binsearch.c\
 exchange_data.c\
 fortran_matrix.c\
 ap.c\
 log.c\
 complex.c\
 error.c\
 int_array.c\
 hopscotch_hash.c\
 matrix_stats.c\
 magma.c\
 memory_tracker.c\
 merge_sort.c\
 mmio.c\
 mpi_comm_f2c.c\
 prefix_sum.c\
 printf.c\
 qsort.c\
 utilities.c\
 mpistubs.c\
 qsplit.c\
 random.c\
 state.c\
 threading.c\
 timer.c\
 timing.c

CUFILES=\
 device_utils.c\
 general.c\
 handle.c\
 int_array_device.c\
 memory.c\
 omp_device.c\
 nvtx.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

SONAME = libHYPRE_utilities-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_onedpl.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_utilities.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_utilities.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/fortran*.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_utilities.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/fortran*.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_utilities.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_utilities.so libHYPRE_utilities.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
