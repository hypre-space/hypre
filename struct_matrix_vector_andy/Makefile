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
 struct_vector.h
 
LOCALFILES =\
 box.c\
 struct_grid.c\
 grid_to_coord.c\
 struct_stencil.c\
 struct_matrix.c\
 struct_matrix_PETSc.c\
 struct_vector.c\
 struct_vector_PETSc.c

HYPREFILES =\
 HYPRE_struct_grid.c\
 HYPRE_struct_stencil.c\
 HYPRE_struct_matrix.c\
 HYPRE_struct_vector.c

FILES = $(HYPREFILES) $(LOCALFILES)

#Automatically generate prototypes when necessary
HYPRE_protos.h: $(HYPREFILES)
	mkproto HYPRE*.c > HYPRE_protos.h

hypre_protos.h: $(FILES)
	mkproto hypre*.c > hypre_protos.h

internal_protos.h: $(FILES)
	mkproto *.c > internal_protos.h

PROTOS =\
 HYPRE_protos.h\
 hypre_protos.h\
 internal_protos.h
 
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
 -I/usr/local/include\
 -I/home/casc/include\
 -I../includes\
 -DHYPRE_MEMORY_CHECK\
 -DHYPRE_DEBUG\
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
 -L../PETScMat_linear_solvers/pilut\
 -L../PETSc_linear_solvers/Block_Preconditioned\
 -L../PETSc_linear_solvers/ParILUT\
 -L../distributed_linear_solvers/pilut\
 -L../distributed_matrix\
 -L../matrix_matrix\
 -L../seq_linear_solvers/ilut\
 -L../seq_linear_solvers/ict\
 -L../seq_linear_solvers/sparskit\
 -L../seq_matrix_vector \
 -L../struct_linear_solvers_andy \
 -L../struct_matrix_vector_andy\
 -L../utilities\
 -lHYPRE_StructSolver \
 -lHYPREPETScSolverParILUT \
 -lHYPREPETScMatPilutSolver \
 -lHYPREDistributedMatrixPilutSolver \
 -lilu -lic -ldsparskit\
 -lHYPRE_StructMatrix \
 -lHYPREDistributedMatrix \
 -lHYPRE_MatrixMatrix\
 -lHYPREseq_mat_vec\
 -lHYPRE_memory\
 ${PETSC_LFLAGS}\
 -lf2c\
 -lcegdb\
 -lmalloc\
 -lm

# This line taken out for now... refers to Petsc block solvers -lHYPREILU -lHYPREIC\

##################################################################
# Main rules
##################################################################

driver: driver.o lib
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

link:
	rm driver
	make driver

lib: libHYPRE_StructMatrix.a

libHYPRE_StructMatrix.a: $(PROTOS) ${OBJS}
	@echo  "Building $@ ... "
	@ar -ru libHYPRE_StructMatrix.a ${OBJS}
	@ranlib libHYPRE_StructMatrix.a

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
	@rm -f driver libHYPRE.a

clean:
	@rm -f *.o

