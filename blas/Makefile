#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

###  Internal BLAS routines are compiled in this sub-directory then copied
###  to the utilities and lib directories to be included in the appropriate
###  libHYPRE* libraries.
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
	@\
	for file in ${OBJS}; \
	do \
	  (cp -fp $$file ../utilities); \
	  (cp -fp $$file ../lib); \
	done
	cp -fp ${BLAS_HEADERS} ../utilities

install: all

clean:
	rm -f *.o

distclean: clean

##################################################################
# Rules
##################################################################

${OBJS}: ${BLAS_HEADERS}
