# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.2 $
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .o

srcdir = .


CC =  mpicc

C_COMPILE_FLAGS= -O 
CXX_COMPILE_FLAGS= -O 
F77_COMPILE_FLAGS= -O -silent 
CINCLUDES=

CFLAGS =\
 -I$(srcdir)\
 ${C_COMPILE_FLAGS}\
 ${CINCLUDES}

RANLIB= ranlib

LIBFLAGS =   -lm
LDLIBFLAGS =  

HEADERS =\
 f2c.h

FILES =\
 dcopy.c \
 ddot.c \
 dnrm2.c \
 dscal.c \
 daxpy.c \
 dpotrf.c \
 dpotrs.c \
 dtrsm.c \
 dgemm.c \
 dgemv.c \
 dsyrk.c \
 s_cmp.c \
 s_copy.c \
 dgels_all.c \
 dasum.c \
 dmyblas2.c \
 drot.c \
 dsymv.c \
 dsyr2.c \
 dtrsv.c \
 idamax.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_blas.a 

install: all
	@cp -f libHYPRE_*.a $$HYPRE_INSTALL_DIR/lib/.

clean:
	@rm -f *.o

veryclean: clean
	@rm -f libHYPRE_*.a

##################################################################
# Rules
##################################################################

libHYPRE_blas.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

.f.o:
	${F77} -o $@ -c ${FFLAGS} $<

