# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1999   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .o

srcdir = .


CC =  mpicc

C_COMPILE_FLAGS=  -g
CXX_COMPILE_FLAGS=  -g
F77_COMPILE_FLAGS=  -g -silent
CINCLUDES= 
CDEFS =  -DHYPRE_NO_PTHREAD_MANGLING

CFLAGS =\
 ${C_COMPILE_FLAGS}\
 -I../utilities\
 -I../struct_matrix_vector\
 -I../seq_matrix_vector\
 -I../parcsr_matrix_vector\
 -I../parcsr_linear_solvers\
 ${CINCLUDES}\
 ${CDEFS}

RANLIB= ranlib

MPILIBFLAGS =   
LIBFLAGS =   -lm
LDLIBFLAGS =  

LFLAGS =\
 -L.\
 -L../utilities\
 -L../struct_matrix_vector\
 -L../seq_matrix_vector\
 -L../parcsr_matrix_vector\
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_mv\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

HEADERS =\
 aux_parcsr_matrix.h\
 headers.h\
 fortran.h\
 IJ_matrix.h\
 IJ_matrix_vector.h\
 HYPRE_IJMatrix.h

FILES =\
 aux_parcsr_matrix.c\
 hypre_IJMatrix_parcsr.c\
 hypre_IJMatrix_petsc.c\
 hypre_IJMatrix_isis.c\
 IJMatrix_parcsr.c\
 IJMatrix_petsc.c\
 IJMatrix_isis.c\
 HYPRE_IJMatrix.c\
 F90_HYPRE_IJMatrix.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_IJ_mv.a

install: all

clean:
	@rm -f *.o

veryclean: clean
	@rm -f libHYPRE_*.a
	@rm -f driver

##################################################################
# Rules
##################################################################

driver: driver.o libHYPRE_IJ_mv.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

libHYPRE_IJ_mv.a: ${OBJS}
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
