#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.20 $
#EHEADER**********************************************************************


###  Internal BLAS routines may be compiled in this sub-directory
###  Generally, the Makefile in the utilities subdirectory controls
###  the inclusion of these routines based on the configure options.
###

include ../config/Makefile.config

C_COMPILE_FLAGS = ${INCLUDES} -I.. -I../utilities

BLAS_HEADERS =  f2c.h  hypre_blas.h
BLAS_FILES = \
blas_utils.c\
dasum.c\
daxpy.c\
dcopy.c\
ddot.c\
dgemm.c\
dgemv.c\
dger.c\
dnrm2.c\
drot.c\
dscal.c\
dswap.c\
dsymm.c\
dsymv.c\
dsyr2.c\
dsyr2k.c\
dsyrk.c\
dtrmm.c\
dtrmv.c\
dtrsm.c\
dtrsv.c\
idamax.c

OBJS = ${BLAS_FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: ${OBJS}
	cp -fp *.o ${HYPRE_SRC_TOP_DIR}/utilities

install: all
	cp -fp *.h ${HYPRE_INC_INSTALL}/include

clean:
	rm -rf *.o

distclean: clean

##################################################################
# Rules
##################################################################

${OBJS}: ${BLAS_HEADERS}
