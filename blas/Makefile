#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

###  If internal BLAS routines are needed, compile in this sub-directory 
###  then copy the files to the utilities directory to be included in
###  libHYPRE_utilities.a 
###

include ../config/Makefile.config


${BLASNEED}BLAS_HEADERS =  f2c.h  hypre_blas.h
${BLASNEED}BLAS_FILES = \
${BLASNEED}blas_utils.c\
${BLASNEED}dasum.c\
${BLASNEED}daxpy.c\
${BLASNEED}dcopy.c\
${BLASNEED}ddot.c\
${BLASNEED}dgemm.c\
${BLASNEED}dgemv.c\
${BLASNEED}dger.c\
${BLASNEED}dnrm2.c\
${BLASNEED}drot.c\
${BLASNEED}dscal.c\
${BLASNEED}dswap.c\
${BLASNEED}dsymm.c\
${BLASNEED}dsymv.c\
${BLASNEED}dsyr2.c\
${BLASNEED}dsyr2k.c\
${BLASNEED}dsyrk.c\
${BLASNEED}dtrmm.c\
${BLASNEED}dtrmv.c\
${BLASNEED}dtrsm.c\
${BLASNEED}dtrsv.c\
${BLASNEED}idamax.c

OBJS = ${BLAS_FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: ${OBJS}
	@\
	${BLASNEED}for file in ${OBJS}; \
	${BLASNEED}do \
	${BLASNEED}  (cp -fp $$file ../utilities); \
	${BLASNEED}done
	${BLASNEED}cp -fp ${BLAS_HEADERS} ../utilities

install: all

clean:
	rm -f *.o

distclean: clean

##################################################################
# Rules
##################################################################

${OBJS}: ${BLAS_HEADERS}
