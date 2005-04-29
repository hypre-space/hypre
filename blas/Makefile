#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

###  Internal BLAS routines may be compiled in this sub-directory
###  Generally, the Makefile in the utilities subdirectory controls
###  the inclusion of these routines based on the configure options.
###

include ../config/Makefile.config


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

install: all

clean:
	rm -f *.o

distclean: clean

##################################################################
# Rules
##################################################################

${OBJS}: ${BLAS_HEADERS}
