# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .o

srcdir = .


CC = purify gcc

C_COMPILE_FLAGS= -g  
CXX_COMPILE_FLAGS= -g  
F77_COMPILE_FLAGS=  
CINCLUDES=
CDEFS = -DHYPRE_NO_PTHREAD_MANGLING

CFLAGS =\
 -I.\
 ${C_COMPILE_FLAGS}\
 ${CINCLUDES}\
 ${CDEFS}

RANLIB= ranlib

LIBFLAGS =   -lm
LDLIBFLAGS =  

HEADERS =\
 f2c.h

FILES =\
 dcopy.c \
 ddot.c \
 dnrm2.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_blas.a 

install: all
	@cp -f f2c.h $$HYPRE_INSTALL_DIR/include/.
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

