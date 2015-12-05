#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.7 $
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
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../seq_mv\
 ${CINCLUDES}
 
FFLAGS =\
 ${F_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../seq_mv\
 ${FINCLUDES}

HEADERS = f2c.h

CFILES =\
 dmyblas.c\
 s_cmp.c\
 s_copy.c

FFILES =\
 dasum.f\
 daxpy.f\
 dcabs1.f\
 dcopy.f\
 ddot.f\
 dgbmv.f\
 dgemm.f\
 dgemv.f\
 dger.f\
 dnrm2.f\
 drot.f\
 drotg.f\
 drotm.f\
 drotmg.f\
 dsbmv.f\
 dscal.f\
 dsdot.f\
 dspmv.f\
 dspr2.f\
 dspr.f\
 dswap.f\
 dsymm.f\
 dsymv.f\
 dsyr2.f\
 dsyr2k.f\
 dsyr.f\
 dsyrk.f\
 dtbmv.f\
 dtbsv.f\
 dtpmv.f\
 dtpsv.f\
 dtrmm.f\
 dtrmv.f\
 dtrsm.f\
 dtrsv.f\
 dzasum.f\
 dznrm2.f\
 icamax.f\
 idamax.f\
 isamax.f\
 izamax.f\
 lsame.f\
 xerbla.f

FILES = ${CFILES} ${FFILES}

COBJS = ${FILES:.c=.o}
OBJS = ${COBJS:.f=.o}

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
