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


include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
 ${CINCLUDES}

HEADERS =\
 Common.h\
 ConjGrad.h\
 DiagScale.h\
 Hash.h\
 hypre_ParaSails.h\
 LoadBal.h\
 Matrix.h\
 Mem.h\
 Numbering.h\
 OrderStat.h\
 ParaSails.h\
 PrunedRows.h\
 RowPatt.h\
 StoredRows.h

FILES =\
 ConjGrad.c\
 DiagScale.c\
 FGmres.c\
 Hash.c\
 hypre_ParaSails.c\
 LoadBal.c\
 Matrix.c\
 Mem.c\
 Numbering.c\
 OrderStat.c\
 ParaSails.c\
 PrunedRows.c\
 RowPatt.c\
 StoredRows.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_ParaSails-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_ParaSails${HYPRE_LIB_SUFFIX}
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_ParaSails${HYPRE_LIB_SUFFIX}
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_ParaSails.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_ParaSails.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
