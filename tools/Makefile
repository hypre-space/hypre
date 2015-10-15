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

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = 

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}\
 ${CDEFS}

MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
LIBFLAGS = ${LIBDIRS} ${LIBS}

LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE\
 ${MPILIBFLAGS}\
 ${LIBFLAGS}

##################################################################
# Targets
##################################################################

all: stencilmult

install: all
	cp -f V[1-9]* $$HYPRE_INSTALL_DIR/bin/.

clean:
	rm -f *.o

distclean: clean
	rm -f stencilmult
	rm -f one_to_many
	rm -f one_to_many_vector
	rm -f create_2d_laplacian 
	rm -f create_3d_laplacian

##################################################################
# Rules
##################################################################

stencilmult: stencilmult.o
	@echo  "Linking" $@ "... "
	${CC} -o stencilmult stencilmult.o ${LFLAGS}

one_to_many: one_to_many.o
	@echo  "Linking" $@ "... "
	${CC} -o one_to_many one_to_many.o ${LFLAGS}

one_to_many_vector: one_to_many_vector.o
	@echo  "Linking" $@ "... "
	${CC} -o one_to_many_vector one_to_many_vector.o ${LFLAGS}

create_2d_laplacian: create_2d_laplacian.o
	@echo  "Linking" $@ "... "
	${CC} -o create_2d_laplacian create_2d_laplacian.o ${LFLAGS}

create_3d_laplacian: create_3d_laplacian.o
	@echo  "Linking" $@ "... "
	${CC} -o create_3d_laplacian create_3d_laplacian.o ${LFLAGS}
