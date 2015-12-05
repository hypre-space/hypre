#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.18 $
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../IJ_mv\
 ${CINCLUDES}

HEADERS =\
 HYPRE_sstruct_mv.h\
 _hypre_sstruct_mv.h\
 headers.h\
 sstruct_graph.h\
 sstruct_grid.h\
 sstruct_matrix.h\
 sstruct_stencil.h\
 sstruct_vector.h

FILES =\
 F90_HYPRE_sstruct_graph.c\
 F90_HYPRE_sstruct_grid.c\
 F90_HYPRE_sstruct_matrix.c\
 F90_HYPRE_sstruct_stencil.c\
 F90_HYPRE_sstruct_vector.c\
 HYPRE_sstruct_graph.c\
 HYPRE_sstruct_grid.c\
 HYPRE_sstruct_matrix.c\
 HYPRE_sstruct_stencil.c\
 HYPRE_sstruct_vector.c\
 sstruct_axpy.c\
 sstruct_copy.c\
 sstruct_graph.c\
 sstruct_grid.c\
 sstruct_innerprod.c\
 sstruct_matrix.c\
 sstruct_matvec.c\
 sstruct_overlap_innerprod.c\
 sstruct_scale.c\
 sstruct_stencil.c\
 sstruct_vector.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_sstruct_mv-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_sstruct_mv${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fpPR $(srcdir)/_hypre_sstruct_mv.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_sstruct_mv${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fpPR $(srcdir)/_hypre_sstruct_mv.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_sstruct_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_sstruct_mv.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
