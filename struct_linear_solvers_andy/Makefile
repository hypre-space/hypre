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
 struct_solver.h
 
LOCALFILES =\
 struct_solver.c\
 struct_solver_PETSc.c

HYPREFILES =\
 HYPRE_struct_solver.c

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
 ${PAR_SOLVER_FLAGS}\
 ${SEQ_SOLVER_FLAGS}\
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
 -L../PETSc_linear_solvers/Block_Preconditioned\
 -L../PETSc_linear_solvers/ParILUT\
 -L../PETSc_linear_solvers/ParILUT/pilut\
 -L../seq_linear_solvers/ilut\
 -L../seq_linear_solvers/ict\
 -L../seq_linear_solvers/sparskit\
 -L../seq_matrix_vector \
 -L.\
 -lHYPRE\
 -lHYPREILU -lHYPREIC\
 -lHYPREParILUT \
 -lHYPREpilut\
 -lilu -lic -ldsparskit\
 -lHYPREseq_mat_vec\
 ${PETSC_LFLAGS}\
 -lf2c\
 -lcegdb\
 -lmalloc\
 -lm


${OBJS}: ${HEADERS}

##################################################################
# Main rules
##################################################################

driver: driver.o libHYPRE.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

link:
	rm driver
	make driver

lib: $(PROTOS) ${OBJS}
	@echo  "Building $@ ... "
	@ar -ru libHYPRE_StructSolver.a ${OBJS}
	@ranlib libHYPRE_StructSolver.a

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

