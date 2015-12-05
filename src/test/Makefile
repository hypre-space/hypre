# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.7 $
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .C .f .o .CXX.o

srcdir = .

CC= mpicc
CXX= mpiCC
F77=mpif77

C_COMPILE_FLAGS= -O 
CXX_COMPILE_FLAGS= -O 
F77_COMPILE_FLAGS= -O -silent 
LD_LINK_FLAGS=
CINCLUDES= 
CDEFS = -DHYPRE_TIMING
BABEL_HYPRE_INCLUDES = 
 
CFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I$(srcdir)\
 -I$(srcdir)/..\
 ${BABEL_HYPRE_INCLUDES}\
 -I../hypre/include\
 ${CINCLUDES}\
 ${CDEFS}
 
FFLAGS = \
 ${F77_COMPILE_FLAGS}\
 ${BABEL_HYPRE_INCLUDES}
 
MPILIBFLAGS =    
LIBFLAGS =   -lm
LDLIBFLAGS =  
BLASLIBFLAGS =  -lHYPRE_blas
PETSCLIBFLAGS =  
BABELLIBFLAGS =  

LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L../hypre/lib\
 -lHYPRE_sstruct_ls\
 -lHYPRE_sstruct_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lkrylov\
 -lHYPRE_utilities\
 ${PETSCLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

###########################################
# Beta definitions
###########################################

BETA_CFLAGS = ${CFLAGS}

BETA_LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L../hypre/lib\
 -lHYPRE_sstruct_ls\
 -lHYPRE_sstruct_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lkrylov\
 -lHYPRE_utilities\
 ${PETSCLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

BETA77_LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L.\
 -L../hypre/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_utilities\
 -lkrylov\
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

STRUCT_LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L../hypre/lib\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lkrylov\
 -lHYPRE_utilities\
 ${PETSCLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

SSTRUCT_LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L../hypre/lib\
 -lHYPRE_sstruct_ls\
 -lHYPRE_sstruct_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lkrylov\
 -lHYPRE_utilities\
 ${PETSCLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

IJ_LS_LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L../hypre/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lkrylov\
 -lHYPRE_utilities\
 ${PETSCLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

IJ_MV_LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L../hypre/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lkrylov\
 -lHYPRE_utilities\
 ${PETSCLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

FEICXX_LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L../hypre/lib\
 -lHYPRE_LSI\
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

HYPRE_DIR = /home/treadway/hypre-1.6.0/src/linear_solvers

##################################################################
# Targets
##################################################################

HYPRE_STANDARD_DRIVERS =\
 IJ_linear_solvers.c\
 IJ_matrix_vector.c\
 struct_linear_solvers.c\
 sstruct_linear_solvers.c

HYPRE_BABEL_DRIVERS =\
 struct_linear_solvers_b.c\
 IJ_linear_solvers_b.c

HYPRE_DRIVERS_CXX =\
 ${HYPRE_DRIVERS:.c=.CXX.C}

HYPRE_FEI_DRIVERS_CXX =\
 fei_linear_solvers.C

HYPRE_STANDARD_DRIVERS_F77 =\
 f77_IJ_linear_solvers.f\
 f77_IJ_matrix_vector.f\
 f77_struct_linear_solvers.f

HYPRE_BABEL_DRIVERS_CXX =

HYPRE_BABEL_DRIVERS_F77 =\
 f77_struct_linear_solvers_b.f

HYPRE_BETA_DRIVERS =

HYPRE_BETA_DRIVERS_F77 =

#ifndef is gmake only, can't be used...
#ifndef BABELLIBFLAGS
 HYPRE_DRIVERS = ${HYPRE_STANDARD_DRIVERS}
 HYPRE_DRIVERS_F77 = ${HYPRE_STANDARD_DRIVERS_F77}
#else
# HYPRE_DRIVERS = ${HYPRE_STANDARD_DRIVERS} ${HYPRE_BABEL_DRIVERS}
# HYPRE_DRIVERS_F77 = ${HYPRE_STANDARD_DRIVERS_F77} ${HYPRE_BABEL_DRIVERS_F77}
#endif

HYPRE_DRIVER_OBJS=${HYPRE_DRIVERS:.c=.o}
HYPRE_DRIVER_CXX_OBJS=${HYPRE_DRIVERS_CXX:.C=.o}
HYPRE_DRIVER_F77_OBJS=${HYPRE_DRIVERS_F77:.f=.o}
HYPRE_BETA_DRIVER_OBJS=${HYPRE_BETA_DRIVERS:.c=.o}
HYPRE_BETA_DRIVER_F77_OBJS=${HYPRE_BETA_DRIVERS_F77:.f=.o}
HYPRE_FEI_DRIVER_CXX_OBJS=${HYPRE_FEI_DRIVERS_CXX:.C=.o}

HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.c=}
HYPRE_DRIVER_CXX_EXECS=${HYPRE_DRIVERS_CXX:.C=}
HYPRE_DRIVER_F77_EXECS=${HYPRE_DRIVERS_F77:.f=}
HYPRE_BETA_DRIVER_EXECS=${HYPRE_BETA_DRIVERS:.c=}
HYPRE_BETA_DRIVER_F77_EXECS=${HYPRE_BETA_DRIVERS_F77:.f=}
HYPRE_FEI_DRIVER_CXX_EXECS=${HYPRE_FEI_DRIVERS_CXX:.C=}

HYPRE_BABEL_DRIVER_OBJS=${HYPRE_BABEL_DRIVERS:.c=.o}
HYPRE_BABEL_DRIVER_CXX_OBJS=${HYPRE_BABEL_DRIVERS_CXX:.C=.o}
HYPRE_BABEL_DRIVER_F77_OBJS=${HYPRE_BABEL_DRIVERS_F77:.f=.o}

HYPRE_BABEL_DRIVER_EXECS=${HYPRE_BABEL_DRIVERS:.c=}
HYPRE_BABEL_DRIVER_CXX_EXECS=${HYPRE_BABEL_DRIVERS_CXX:.C=}
HYPRE_BABEL_DRIVER_F77_EXECS=${HYPRE_BABEL_DRIVERS_F77:.f=}

all: ${HYPRE_DRIVER_EXECS}

all++: ${HYPRE_DRIVER_CXX_EXECS}

all77: ${HYPRE_DRIVER_F77_EXECS}

beta: all ${HYPRE_BETA_DRIVER_EXECS}

beta++: all++

fei++: ${HYPRE_FEI_DRIVER_CXX_EXECS}

beta77: all77 ${HYPRE_BETA_DRIVER_F77_EXECS}

babel: ${HYPRE_BABEL_DRIVER_EXECS} ${HYPRE_BABEL_DRIVER_F77_EXECS}

install:

clean:
	@rm -f *.o

veryclean: clean
	@rm -f ${HYPRE_DRIVER_EXECS}
	@rm -f ${HYPRE_DRIVER_CXX_EXECS}
	@rm -f ${HYPRE_DRIVER_F77_EXECS}
	@rm -f ${HYPRE_BABEL_DRIVER_EXECS}
	@rm -f ${HYPRE_BABEL_DRIVER_CXX_EXECS}
	@rm -f ${HYPRE_BABEL_DRIVER_F77_EXECS}
	@rm -f ${HYPRE_BETA_DRIVER_EXECS}
	@rm -f ${HYPRE_BETA_DRIVER_F77_EXECS}
	@rm -f ${HYPRE_FEI_DRIVER_CXX_EXECS}

##################################################################
# Rules
##################################################################

PURIFY = purify
PURIFY_TO_FILE = purify \
 -log-file=struct_linear_solvers.purify \
 -append-logfile=yes

struct_linear_solvers: struct_linear_solvers.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${STRUCT_LFLAGS}

sstruct_linear_solvers: sstruct_linear_solvers.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${SSTRUCT_LFLAGS}

IJ_linear_solvers: IJ_linear_solvers.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_LS_LFLAGS}

IJ_matrix_vector: IJ_matrix_vector.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_MV_LFLAGS}

#${HYPRE_DRIVER_EXECS}: ${HYPRE_DRIVER_OBJS}
#	@echo  "Building" $@ "... "
#	${CC} -o $@ $@.o ${LFLAGS}

struct_linear_solvers.CXX: struct_linear_solvers.CXX.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${STRUCT_LFLAGS}

sstruct_linear_solvers.CXX: sstruct_linear_solvers.CXX.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${SSTRUCT_LFLAGS}

IJ_linear_solvers.CXX: IJ_linear_solvers.CXX.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_LS_LFLAGS}

IJ_matrix_vector.CXX: IJ_matrix_vector.CXX.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_MV_LFLAGS}

#${HYPRE_DRIVER_CXX_EXECS}: ${HYPRE_DRIVER_CXX_OBJS}
#	@echo  "Building" $@ "... "
#	${CXX} -o $@ $@.o ${LFLAGS}

f77_struct_linear_solvers: f77_struct_linear_solvers.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${STRUCT_LFLAGS}

f77_IJ_linear_solvers: f77_IJ_linear_solvers.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${IJ_LS_LFLAGS}

f77_IJ_matrix_vector: f77_IJ_matrix_vector.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${IJ_MV_LFLAGS}

#${HYPRE_DRIVER_F77_EXECS}: ${HYPRE_DRIVER_F77_OBJS}
#	@echo  "Building" $@ "... "
#	${F77} -o $@ $@.o ${FFLAGS} ${LFLAGS}

# special autotest target
${HYPRE_FEI_DRIVER_CXX_EXECS}: ${HYPRE_FEI_DRIVER_CXX_OBJS}
	@echo  "Building" $@ "... "
	${CXX} -o $@ $@.o ${FEICXX_LFLAGS}

##################################################################
# Babel version of above
##################################################################
${HYPRE_BABEL_DRIVER_EXECS}: ${HYPRE_BABEL_DRIVER_OBJS}
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${BABELLIBFLAGS} ${LFLAGS}

#${HYPRE_BABEL_DRIVER_CXX_EXECS}: ${HYPRE_BABEL_DRIVER_CXX_OBJS}
#	@echo  "Building" $@ "... "
#	${CXX} -o $@ $@.o ${LFLAGS}

${HYPRE_BABEL_DRIVER_F77_EXECS}: ${HYPRE_BABEL_DRIVER_F77_OBJS}
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${LFLAGS}

##################################################################
# Beta rules
##################################################################

#${HYPRE_BETA_DRIVER_EXECS}: ${HYPRE_BETA_DRIVER_OBJS}
#	@echo  "Building" $@ "... "
#	${CC} -o $@ $@.o ${BETA_CFLAGS} ${BETA_LFLAGS}

#${HYPRE_BETA_DRIVER_F77_EXECS}: ${HYPRE_BETA_DRIVER_F77_OBJS}
#	@echo  "Building" $@ "... "
#	${F77} -o $@ $@.o ${FFLAGS} ${BETA77_LFLAGS}

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

.C.o:
	${CXX} -o $@ -c -DNOFEI ${CFLAGS} $<

.f.o:
	${F77} -o $@ -c ${FFLAGS} $<

# rule to compile C files with C++
.c.CXX.o:
	cp -f $< ${@:.o=.C}
	${CXX} -o $@ -c ${CFLAGS} ${@:.o=.C}
	rm -f ${@:.o=.C}
