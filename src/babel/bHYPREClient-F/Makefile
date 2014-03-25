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

C_COMPILE_FLAGS=\
 -I..\
 -I$(HYPRE_SRC_TOP_DIR)\
 -I$(HYPRE_SRC_TOP_DIR)/utilities\
 -I$(HYPRE_SRC_TOP_DIR)/seq_mv\
 -I$(HYPRE_SRC_TOP_DIR)/krylov\
 -I$(HYPRE_SRC_TOP_DIR)/IJ_mv\
 -I$(HYPRE_SRC_TOP_DIR)/parcsr_mv\
 -I$(HYPRE_SRC_TOP_DIR)/parcsr_ls\
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

FILES = ${IMPLSRCS} ${IORSRCS_NONNULL} ${SKELSRCS} ${STUBSRCS} 

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libbHYPREClient-F.a
	cp -fR $(srcdir)/*.h ${HYPRE_BUILD_DIR}/include/.
	cp -fR $(srcdir)/*.inc ${HYPRE_BUILD_DIR}/include/.
	cp -fR libbHYPRE* ${HYPRE_BUILD_DIR}/lib/.
 
install: all
	cp -fR $(srcdir)/*.h ${HYPRE_INC_INSTALL}/.
	cp -fR libbHYPRE* ${HYPRE_LIB_INSTALL}/.
 
clean:
	rm -f *.o libbHYPRE*

distclean: clean

##################################################################
# Rules
##################################################################

libbHYPREClient-F.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
