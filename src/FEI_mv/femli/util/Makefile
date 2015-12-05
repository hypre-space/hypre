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

CC =  mpicc
F77 = mpif77

C_COMPILE_FLAGS=-O2 
CINCLUDES= 

F_COMPILE_FLAGS=-O
FINCLUDES= 
 
CFLAGS =\
 ${C_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../fei-hypre\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../eigensolvers/krylov\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../distributed_ls\
 -I../../../distributed_matrix\
 -I../../../seq_mv\
 ${CINCLUDES}
 
FFLAGS =\
 ${F_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../fei-hypre\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../eigensolvers/krylov\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../distributed_ls\
 -I../../../distributed_matrix\
 -I../../../seq_mv\
 ${FINCLUDES}

HEADERS =\
 mli_utils.h

FILES =\
 mli_olapdd.c\
 mli_utils.c\
 mli_utils_fortran.f

TEMP = ${FILES:.f=.o}
OBJS = ${TEMP:.c=.o}

##################################################################
# Targets
##################################################################

all: ${OBJS} 

install: all

clean:
	@rm -f *.o

distclean: clean


##################################################################
# Rules
##################################################################

${OBJS}: ${HEADERS}
