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


all: lib

install:all

#Headers
HEADERS =\
 PETScSolverParILUT.h

HYPREFILES = HYPRE_PETScSolverParILUT.c HYPRE_PETScSolverParILUTSetup.c

LOCALFILES =

FILES = $(HYPREFILES) $(LOCALFILES)

OBJS = ${FILES:.c=.o}


${OBJS}: ${HEADERS}

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = ${HYPRE_PETSCDEFS}

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../PETScMat_ls/pilut \
 ${PETSCINCLUDE} \
 ${CINCLUDES} ${CDEFS}

CXX_COMPILE_FLAGS = \
 ${PETSCINCLUDE} \
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 ${CINCLUDES}\
 ${COPT}\
 ${CDEBUG} ${CDEFS}

##################################################################
# Main rules
##################################################################

driver: driver.o libHYPRE${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

link:
	rm driver
	$(MAKE) driver

lib: ${PROTOS} ${OBJS}
	@echo  "Building $@ ... "
	@${AR} lib_HYPRE_PETScSolverParILUT${HYPRE_LIB_SUFFIX} ${OBJS}
	@${RANLIB} lib_HYPRE_PETScSolverParILUT${HYPRE_LIB_SUFFIX}


##################################################################
# Targets
##################################################################
clean:
	@rm -f *.o lib_HYPRE*

distclean: clean
