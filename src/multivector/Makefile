#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.15 $
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I.\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 multivector.h\
 temp_multivector.h

FILES =\
 multivector.c\
 temp_multivector.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_multivector-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_multivector${HYPRE_LIB_SUFFIX}
	cp -fpPR *.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_multivector${HYPRE_LIB_SUFFIX}
	cp -fpPR *.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_multivector.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_multivector.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
