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

HEADERS =\
 box.h\
 struct_grid.h\
 grid_to_coord.h\
 headers.h\
 struct_stencil.h\
 struct_matrix.h\
 struct_vector.h\
 struct_solver.h
 
FILES =\
 box.c\
 struct_grid.c\
 grid_to_coord.c\
 struct_stencil.c\
 struct_matrix.c\
 struct_matrix_PETSc.c\
 struct_vector.c\
 struct_vector_PETSc.c\
 struct_solver.c\
 struct_solver_PETSc.c\
 ZZZ_struct_grid.c\
 ZZZ_struct_stencil.c\
 ZZZ_struct_matrix.c\
 ZZZ_struct_vector.c\
 ZZZ_struct_solver.c

OBJS = ${FILES:.c=.o}

CC = gcc

F77 = /home/casc/g77/bin/g77

PETSC_CFLAGS =\
 -I${PETSC_DIR}/src/mat/impls/aij/mpi\
 -I${PETSC_DIR}/src/mat/impls/aij/seq\
 -I${PETSC_DIR}/src/mat\
 -I${PETSC_DIR}\
 -I${PETSC_DIR}/include\
 -I/usr/local/mpi/mpich/include
CFLAGS =\
 ${PETSC_CFLAGS}\
 ${PAR_SOLVER_FLAGS}\
 ${SEQ_SOLVER_FLAGS}\
 -I/usr/local/include\
 -I/home/casc/include\
 -I../includes\
 -g

FFLAGS = -g

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base.site
PETSC_OPT = g
PETSC_LFLAGS =\
 -L${PETSC_DIR}/lib/lib${PETSC_OPT}/${PETSC_ARCH}\
 -lpetscsles\
 -lpetscmat\
 -lpetscvec\
 -lpetscsys\
 ${SUPERLU_LIB}\
 ${BS_LIB}\
 ${LAPACK_LIB}\
 ${BLAS_LIB}\
 ${X11_LIB}\
 ${MPE_LIB}\
 ${MPI_LIB}\
 ${SYS_LIB}
LFLAGS =\
 -Wl,"-zmuldefs"\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
 -L../PETSc_linear_solvers/Block_Preconditioned\
 -L../seq_linear_solvers/ilut\
 -L../seq_linear_solvers/ict\
 -L../seq_linear_solvers/sparskit\
 -L.\
 -lZZZ\
 -lZZZILU -lZZZIC\
 ${PETSC_LFLAGS}\
 -lilu -lic -ldsparskit\
 -lf2c\
 -lcegdb\
 -lmalloc\
 -lm


##################################################################
# Main rules
##################################################################

driver: driver.o libZZZ.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

libZZZ.a: ${OBJS}
	@echo  "Building $@ ... "
	@ar -ru $@ ${OBJS}
	@ranlib $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	@echo "Making (c) " $@
	${CC} -o $@ -c ${CFLAGS} $<

#	@${CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

veryclean: clean
	@rm -f driver libZZZ.a

clean:
	@rm -f *.o

