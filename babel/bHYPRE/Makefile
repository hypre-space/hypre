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


include ../../config/Makefile.config
include babel.make

IORSRCS_NONNULL = $(filter-out bHYPRE_SStructVariable_IOR.c, ${IORSRCS})
# ... ignore zero-length file because it will confuse the gcc/ar interaction

CINCLUDES=${INCLUDES} ${MPIINCLUDE}
CDEFS = 
 
BABEL_RUNTIME_INCLUDES=\
  -I$(HYPRE_SRC_TOP_DIR)/babel-runtime/sidl
BABEL_RUNTIME_SIDL_LIBS=\
  -L$(HYPRE_SRC_TOP_DIR)/babel-runtime/sidl/.libs\
  -lsidl-1.0.4 -lsidl
# >>> the reference to a specific Babel version (1.0.4) needs to be cleaned up <<<
# ... note re the use of the .libs directory above.  Its contents will
# get copied (by babel/Makefile) to hypre/lib, but the copy hasn't
# happened yet, and we need to point to the sidl libraries now.

C_COMPILE_FLAGS=\
 -I..\
 -I$(HYPRE_SRC_TOP_DIR)\
 -I$(HYPRE_SRC_TOP_DIR)/hypre/include\
 ${BABEL_RUNTIME_INCLUDES}\
 ${CINCLUDES}\
 ${CDEFS}
 
LIBFLAGS = ${LIBDIRS} ${LIBS}
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS}
 
LFLAGS =\
 -L.\
 -L../seq_mv\
 -L../utilities\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${LIBFLAGS} ${LDLIBFLAGS}

HEADERS = ${STUBHDRS} ${IORHDRS} ${IMPLHDRS}

FILES = ${IMPLSRCS} ${IORSRCS_NONNULL} ${SKELSRCS} ${STUBSRCS} \
 bHYPRE_Extensions.c

OBJS = ${FILES:.c=.o}
# SHARED_OBJS = ${FILES:.c=.lo}

##################################################################
# Targets
##################################################################

all: libbHYPRE${HYPRE_LIB_SUFFIX}
	cp -fa $(srcdir)/*.h ${HYPRE_BUILD_DIR}/include
	cp -fa libbHYPRE* ${HYPRE_BUILD_DIR}/lib
 
install: all
	cp -fa $(srcdir)/*.h ${HYPRE_INC_INSTALL}
	cp -fa libbHYPRE* ${HYPRE_LIB_INSTALL}
 
clean:
	rm -f *.o *.lo libbHYPRE*

distclean: clean

##################################################################
# Rules
##################################################################

libbHYPRE.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libbHYPRE.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o $@ ${OBJS} -L${HYPRE_LIB_INSTALL} \
	-lHYPRE_DistributedMatrixPilutSolver\
	-lHYPRE_DistributedMatrixPilutSolver\
	-lHYPRE_DistributedMatrix\
	-lHYPRE_Euclid\
	-lHYPRE_IJ_mv\
	-lHYPRE_krylov\
	-lHYPRE_MatrixMatrix\
	-lHYPRE_multivector\
	-lHYPRE_ParaSails\
	-lHYPRE_parcsr_block_mv\
	-lHYPRE_parcsr_ls\
	-lHYPRE_parcsr_mv\
	-lHYPRE_seq_mv\
	-lHYPRE_sstruct_ls\
	-lHYPRE_sstruct_mv\
	-lHYPRE_struct_ls\
	-lHYPRE_struct_mv\
	-lHYPRE_superlu\
	-lHYPRE_utilities\
	${BABEL_RUNTIME_SIDL_LIBS}

${OBJS}: ${HEADERS}


