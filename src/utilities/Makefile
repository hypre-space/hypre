#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.29 $
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = ${TIMERDEFS} -DHYPRE_TIMING

C_COMPILE_FLAGS =\
   -I..\
   -I$(srcdir)/..\
   -I$(srcdir)\
   ${CINCLUDES}\
   ${CDEFS}

HEADERS =\
 amg_linklist.h\
 exchange_data.h\
 fortran.h\
 fortran_matrix.h\
 general.h\
 hypre_memory.h\
 hypre_smp_forloop.h\
 HYPRE_utilities.h\
 mpistubs.h\
 threading.h\
 thread_mpistubs.h\
 timing.h\
 umalloc_local.h\
 _hypre_utilities.h

FILES =\
 amg_linklist.c\
 binsearch.c\
 exchange_data.c\
 fortran_matrix.c\
 hypre_ap.c\
 hypre_printf.c\
 hypre_memory.c\
 hypre_qsort.c\
 memory_dmalloc.c\
 mpistubs.c\
 qsplit.c\
 random.c\
 threading.c\
 thread_mpistubs.c\
 timer.c\
 timing.c\
 umalloc_local.c\
 hypre_mpi_comm_f2c.c\
 hypre_error.c\
 F90_HYPRE_error.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_utilities-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fpPR $(srcdir)/_hypre_utilities.h $(HYPRE_BUILD_DIR)/include
	cp -fpPR $(srcdir)/fortran*.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fpPR $(srcdir)/_hypre_utilities.h $(HYPRE_INC_INSTALL)
	cp -fpPR $(srcdir)/fortran*.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE* f2c.h *blas.h *lapack.h 
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################
 
libHYPRE_utilities.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ *.o
	${RANLIB} $@
 
libHYPRE_utilities.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
