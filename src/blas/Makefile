#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.2 $
#EHEADER***********************************************************************

###  This is the template file for the blas sub-directory.  When configure
###  determines whether or not these files are needed, a macro will be
###  automatically run to replace the # variable with a comment
###  symbol or a space and create the appropriate file named Makefile.
###
###
###  If internal BLAS routines are needed, compile in this sub-directory 
###  then copy the files to the utilities directory to be included in
###  libHYPRE_utilities.a 
###

include ../config/Makefile.config


#BLAS_HEADERS =  f2c.h  hypre_blas.h
#BLAS_FILES = \
#blas_utils.c\
#dasum.c\
#daxpy.c\
#dcopy.c\
#ddot.c\
#dgemm.c\
#dgemv.c\
#dger.c\
#dnrm2.c\
#drot.c\
#dscal.c\
#dswap.c\
#dsymm.c\
#dsymv.c\
#dsyr2.c\
#dsyr2k.c\
#dsyrk.c\
#dtrmm.c\
#dtrmv.c\
#dtrsm.c\
#dtrsv.c\
#idamax.c

OBJS = ${BLAS_FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: ${OBJS}
	@\
	#for file in ${OBJS}; \
	#do \
	#  (cp -fp $$file ../utilities); \
	#done
	#cp -fp ${BLAS_HEADERS} ../utilities

install: all

clean:
	rm -f *.o

distclean: clean
	rm -f Makefile

##################################################################
# Rules
##################################################################

${OBJS}: ${BLAS_HEADERS}
