# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
 hypre_general.h\
 hypre_memory.h\
 HYPRE_utilities.h\
 mpistubs.h\
 threading.h\
 timing.h\
 _hypre_utilities.h

FILES =\
 F90_HYPRE_error.c\
 F90_HYPRE_general.c\
 HYPRE_version.c\
 amg_linklist.c\
 binsearch.c\
 exchange_data.c\
 fortran_matrix.c\
 hypre_ap.c\
 hypre_complex.c\
 hypre_error.c\
 hypre_hopscotch_hash.c\
 hypre_merge_sort.c\
 hypre_mpi_comm_f2c.c\
 hypre_nvtx.c\
 hypre_prefix_sum.c\
 hypre_printf.c\
 hypre_qsort.c\
 mpistubs.c\
 qsplit.c\
 random.c\
 threading.c\
 timer.c\
 timing.c

CUFILES=\
 hypre_cuda_utils.c\
 hypre_general.c\
 hypre_memory.c\
 hypre_omp_device.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

SONAME = libHYPRE_utilities-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_utilities.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_utilities.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_hopscotch_hash.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/fortran*.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_utilities.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/fortran*.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o* libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_utilities.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ *.o
	${RANLIB} $@

libHYPRE_utilities.so libHYPRE_utilities.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
