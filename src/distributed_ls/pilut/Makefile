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


include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
${CINCLUDES}

HEADERS = \
 HYPRE_DistributedMatrixPilutSolver_protos.h\
 HYPRE_DistributedMatrixPilutSolver_types.h\
 DistributedMatrixPilutSolver.h\
 const.h\
 ilu.h\
 internal_protos.h\
 macros.h\
 struct.h

FILES =\
 comm.c\
 debug.c\
 distributed_qsort.c\
 distributed_qsort_si.c\
 HYPRE_DistributedMatrixPilutSolver.c\
 ilut.c\
 parilut.c\
 parutil.c\
 pblas1.c\
 serilut.c\
 trifactor.c\
 util.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_DistributedMatrixPilutSolver-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Main rules
##################################################################

libHYPRE_DistributedMatrixPilutSolver.a: ${OBJS}
	@echo "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_DistributedMatrixPilutSolver.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
