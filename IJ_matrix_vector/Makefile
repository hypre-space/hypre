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
 
CC =  mpicc

C_COMPILE_FLAGS= -O 
CXX_COMPILE_FLAGS= -O 
F77_COMPILE_FLAGS= -O -silent  
CINCLUDES= 
CDEFS = 
 
CFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I../utilities\
 -I../seq_matrix_vector\
 -I../parcsr_matrix_vector\
 ${CINCLUDES}\
 ${CDEFS}
 
RANLIB = ranlib
 
MPILIBFLAGS =    
LIBFLAGS =   -lm
LDLIBFLAGS =  
 
LFLAGS =\
 -L.\
 -L../parcsr_matrix_vector\
 -L../seq_matrix_vector\
 -L../utilities\
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
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
 F90_HYPRE_IJMatrix.c\
 aux_parcsr_matrix.c\
 hypre_IJMatrix_parcsr.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_IJ_mv.a
 
install: all
 
clean:
	@rm -f *.o

veryclean: clean
	@rm -f libHYPRE_IJ_mv.a
	@rm -f driver 

##################################################################
# Rules
##################################################################

libHYPRE_IJ_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	ranlib $@

driver: driver.o libHYPRE_IJ_mv.a 
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS} 

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

