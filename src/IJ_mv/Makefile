#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.16 $
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS =\
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../parcsr_ls\
 ${CINCLUDES}

HEADERS =\
 aux_parcsr_matrix.h\
 aux_par_vector.h\
 headers.h\
 HYPRE_IJ_mv.h\
 _hypre_IJ_mv.h\
 IJ_matrix.h\
 IJ_vector.h

EXTRA_FILES =\
 IJMatrix_isis.c\
 IJMatrix_petsc.c

FILES =\
 aux_parcsr_matrix.c\
 aux_par_vector.c\
 F90_HYPRE_IJMatrix.c\
 F90_HYPRE_IJVector.c\
 F90_IJMatrix.c\
 HYPRE_IJMatrix.c\
 HYPRE_IJVector.c\
 IJMatrix.c\
 IJMatrix_parcsr.c\
 IJVector.c\
 IJVector_parcsr.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_IJ_mv-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_IJ_mv${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fpPR $(srcdir)/_hypre_IJ_mv.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_IJ_mv${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fpPR $(srcdir)/_hypre_IJ_mv.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_IJ_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_IJ_mv.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
