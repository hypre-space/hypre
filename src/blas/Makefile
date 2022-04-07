# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

###  Internal BLAS routines may be compiled in this sub-directory
###  Generally, the Makefile in the utilities subdirectory controls
###  the inclusion of these routines based on the configure options.
###

include ../config/Makefile.config

C_COMPILE_FLAGS = ${INCLUDES} -I.. -I../utilities

BLAS_HEADERS =  f2c.h  hypre_blas.h
BLAS_FILES = \
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
 f2c.c\
 idamax.c\
 lsame.c\
 xerbla.c

OBJS = ${BLAS_FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: ${OBJS}

install: all
	cp -fR $(srcdir)/_hypre_blas.h $(HYPRE_INC_INSTALL)

clean:
	rm -rf *.o

distclean: clean

##################################################################
# Rules
##################################################################

${OBJS}: ${BLAS_HEADERS}
