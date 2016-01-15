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

CINCLUDES=${INCLUDES} ${MPIINCLUDE}
CDEFS = 
 
BABEL_RUNTIME_INCLUDES=\
  -I$(HYPRE_SRC_TOP_DIR)/babel-runtime/sidl
BABEL_RUNTIME_SIDL_LIBS=\
  -L$(HYPRE_SRC_TOP_DIR)/babel-runtime/sidl/.libs\
  -lsidl-1.0.4 -lsidl
BABEL_RUNTIME_LIBS=\
  -L{HYPRE_SRC_TOP_DIR}/hypre/lib\
  -lbHYPRE

OUTSIDE_INCLUDES=\
  -I$(HYPRE_SRC_TOP_DIR)/../babel-1.0.4/lib/sidlstubs/Java
# ...<<<< this is needed for sidl_BaseClass_jniStub.h and probably more <<<
# >>> This and the other hard-coded references to the main Babel tree, or to the
# >>> Babel version number, need to be cleaned up.

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
 ${OUTSIDE_INCLUDES}\
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

FILES = ${IMPLSRCS} ${IORSRCS} ${SKELSRCS} ${STUBSRCS} 

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libbHYPRE.scl libbHYPREClient-J${HYPRE_LIB_SUFFIX} classes
	cp -fR $(srcdir)/*.h ${HYPRE_BUILD_DIR}/include/.
	cp -fR libbHYPRE* ${HYPRE_BUILD_DIR}/lib/.
 
install: all
	cp -fR $(srcdir)/*.h ${HYPRE_INC_INSTALL}/.
	cp -fR libbHYPRE* ${HYPRE_LIB_INSTALL}/.
 
clean:
	rm -f *.o *.a *.so libbHYPRE.scl
	cd bHYPRE; rm -f *.class

distclean: clean

##################################################################
# Rules
##################################################################

libbHYPRE.scl: libbHYPRE.scl.in
	sed -e 's,@@DIR@@,$(HYPRE_LIB_INSTALL),g' < $(srcdir)/libbHYPRE.scl.in > libbHYPRE.scl

libbHYPREClient-J.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libbHYPREClient-J.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o $@ ${OBJS} -L${HYPRE_LIB_INSTALL} \
	${BABEL_RUNTIME_SIDL_LIBS}\
	${BABEL_RUNTIME_LIBS}

classes:
	cd bHYPRE && javac *.java

${OBJS}: ${HEADERS}
