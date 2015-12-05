#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.15 $
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../seq_mv\
 ${CINCLUDES}

EXTRA_HEADERS = internal_protos.h

HEADERS =\
 distributed_matrix.h\
 HYPRE_distributed_matrix_mv.h\
 HYPRE_distributed_matrix_protos.h\
 HYPRE_distributed_matrix_types.h

FILES =\
 distributed_matrix.c\
 HYPRE_distributed_matrix.c\
 distributed_matrix_ISIS.c\
 distributed_matrix_PETSc.c\
 distributed_matrix_parcsr.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_DistributedMatrix-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_DistributedMatrix${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/distributed_matrix.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_DistributedMatrix${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/distributed_matrix.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_DistributedMatrix.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_DistributedMatrix.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
