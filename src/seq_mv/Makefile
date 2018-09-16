#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************


include ../config/Makefile.config

#FILES_NVCC = gpukernels.cu
#include $(HYPRE_NVCC_MAKEFILE)

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS =\
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

CU_COMPILE_FLAGS = ${C_COMPILE_FLAGS}

HEADERS =\
 csr_matrix.h\
 HYPRE_seq_mv.h\
 mapped_matrix.h\
 multiblock_matrix.h\
 seq_mv.h\
 vector.h

FILES =\
 csr_matop.c\
 csr_matrix.c\
 csr_matvec.c\
 genpart.c\
 HYPRE_csr_matrix.c\
 HYPRE_mapped_matrix.c\
 HYPRE_multiblock_matrix.c\
 HYPRE_vector.c\
 mapped_matrix.c\
 multiblock_matrix.c\
 vector.c\
 csr_matvec_oomp.c\
 gpukernels.c

OBJS = ${FILES:.c=.o}
#CUOBJS = ${FILES_NVCC:.cu=.o}

SONAME = libHYPRE_seq_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_seq_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/seq_mv.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_seq_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/seq_mv.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_seq_mv.a: ${OBJS} ${CUOBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_seq_mv.so libHYPRE_seq_mv.dylib: ${OBJS} ${CUOBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
