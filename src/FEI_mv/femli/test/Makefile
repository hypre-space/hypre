#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.6 $
#EHEADER**********************************************************************








include ../../../config/Makefile.config


CC= mpicc

C_COMPILE_FLAGS=-O2
LD_LINK_FLAGS=
CINCLUDES= 
CDEFS = -DHYPRE_TIMING
 
CFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../seq_mv\
 ${CINCLUDES}\
 ${CDEFS}
 
MPILIBFLAGS =     
LIBFLAGS =   -lm -lg2c
LDLIBFLAGS =   @LIBM@
LAPACKLIBFLAGS =  -L/usr/lib -L/usr/local/lib -llapack
BLASLIBFLAGS =  ${BLASLIBDIRS} ${BLASLIBS}


##################################################################
# Targets
##################################################################

all: driver

install:

clean:
	@rm -f *.o

distclean: clean

##################################################################
# Rules
##################################################################

driver: driver.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}
	@echo  " "
