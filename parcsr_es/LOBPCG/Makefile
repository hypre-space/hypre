# Generated automatically from Makefile.in by configure.
# Makefile for lobpcg; AK 080802, MEA 081802

.SUFFIXES:
.SUFFIXES: .c .C .f .o .CXX.o

srcdir = .

# Compiler and compiler options

CC =  mpicc
C_COMPILE_FLAGS=  -O 
CINCLUDES= 

FORTRAN_LIB  = 
#FORTRAN_LIB = -lg2c -L/opt/scali/lib -lmpi 
#FORTRAN_LIB = -lg2c -L/opt/mpich/lib -lmpich 
#FORTRAN_LIB = -L/opt/pgi/linux86/lib -lpgftnrtl -llapack -L/opt/scali/lib -lmpi
#FORTRAN_LIB = -lxlf -lxlf90 -lessl -lm 
#FORTRAN_LIB = -lcxmlp -lm
#FORTRAN_LIB = -lg2c

#LAPACK_LIB  = -llapack 
#BLAS_LIB    = -lblas
LAPACK_LIB   = -llapack
BLAS_LIB     = -lHYPRE_blas

HYPRE_LIB = -L../../hypre/lib

CDEFS = -DHYPRE_TIMING

CFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I../../hypre/include\
 ${CINCLUDES}\
 ${CDEFS}

RANLIB      = ranlib
MPILIBFLAGS =   
LIBFLAGS    =   -lm
LDLIBFLAGS  =  

##################################################################
# Targets
##################################################################

HYPRE_STANDARD_DRIVERS = lobpcg.c lobpcg_matrix.c lobpcg_utilities.c

CTARGET = libHYPRE_lobpcg.a
OBJS    = ${HYPRE_STANDARD_DRIVERS:.c=.o}
INCS    = HYPRE_lobpcg.h lobpcg.h mmio.h

HYPRE_DRIVERS = IJ_eigen_solver.c ${HYPRE_STANDARD_DRIVERS}
HYPRE_DRIVER_OBJS=${HYPRE_DRIVERS:.c=.o}

all: ${CTARGET}

install: all
	@cp -f $(srcdir)/HYPRE_*.h $$HYPRE_INSTALL_DIR/include/.
	@cp -f $(srcdir)/lobpcg.h $$HYPRE_INSTALL_DIR/include/.
	@cp -f libHYPRE_*.a $$HYPRE_INSTALL_DIR/lib/.

clean:
	@rm -f *.o

veryclean: clean
	@rm -f libHYPRE_lobpcg.a

##################################################################
# Rules
##################################################################

${HYPRE_DRIVER_OBJS}: ${INCS} Makefile.dec

libHYPRE_lobpcg.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} -rcu $@ ${OBJS}
	ranlib $@

IJ_eigen_solver: ${HYPRE_DRIVER_OBJS}
	${CC} ${CFLAGS} -o IJ_eigen_solver ${HYPRE_DRIVER_OBJS}\
	  ${HYPRE_LIB} -lHYPRE_LSI ${LAPACK_LIB} ${BLAS_LIB}\
          ${FORTRAN_LIB} ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}


##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

