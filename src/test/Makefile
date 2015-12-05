#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.11 $
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = -DHYPRE_TIMING
CXXDEFS = -DNOFEI -DHYPRE_TIMING -DMPICH_SKIP_MPICXX

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}\
 ${CDEFS}

CXX_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../FEI_mv/fei-base\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}\
 ${CXXDEFS}

F77_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}
 
MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS} 
LIBFLAGS = ${LIBDIRS} ${LIBS}
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS} ${LIBM}
LAPACKLIBFLAGS = ${LAPACKLIBDIRS} ${LAPACKLIBS}
BLASLIBFLAGS = ${BLASLIBDIRS} ${BLASLIBS}

LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
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
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

###########################################
# Commpkg definitions
###########################################
COMMPKG_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

###########################################
# Struct definitions
###########################################
STRUCT_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

###########################################
# Struct_es definitions
###########################################
STRUCT_ES_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
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
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 -lHYPRE_lobpcg \
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

###########################################
# Sstruct definitions
###########################################
SSTRUCT_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
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
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

###########################################
# Sstruct_es definitions
###########################################
SSTRUCT_ES_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
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
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 -lHYPRE_lobpcg\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

###########################################
# IJ definitions
###########################################
IJ_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

###########################################
# IJ_ES definitions
###########################################
IJ_ES_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 -lHYPRE_lobpcg\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

###########################################
# IJ_MV definitions
###########################################
IJ_MV_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

###########################################
# FEI definitions
###########################################
FEICXX_LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE_LSI\
 -lHYPRE_superlu\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LIBFLAGS} ${FLIBS} ${LDLIBFLAGS}

##################################################################
# Targets
##################################################################
HYPRE_STANDARD_DRIVERS =\
 ij.c\
 ij_es.c\
 ij_mv.c\
 new_ij.c\
 sstruct.c\
 sstruct_es.c\
 sstruct_fac.c\
 struct.c\
 struct_es.c

HYPRE_BABEL_DRIVERS =\
 struct_b.c\
 ij_b.c

HYPRE_FEI_DRIVERS_CXX =\
 fei.cxx

HYPRE_STANDARD_DRIVERS_F77 =\
 f77_ij.f\
 f77_ij_mv.f\
 f77_struct.f

HYPRE_BABEL_DRIVERS_CXX =

HYPRE_BABEL_DRIVERS_F77 =\
 f77_ij_b.f\
 f77_struct_b.f

#ifndef is gmake only, can't be used...
#ifndef BABELLIBFLAGS
 HYPRE_DRIVERS = ${HYPRE_STANDARD_DRIVERS}
 HYPRE_DRIVERS_F77 = ${HYPRE_STANDARD_DRIVERS_F77}
#else
# HYPRE_DRIVERS = ${HYPRE_STANDARD_DRIVERS} ${HYPRE_BABEL_DRIVERS}
# HYPRE_DRIVERS_F77 = ${HYPRE_STANDARD_DRIVERS_F77} ${HYPRE_BABEL_DRIVERS_F77}
#endif

HYPRE_DRIVER_OBJS=${HYPRE_DRIVERS:.c=.o}
HYPRE_DRIVER_F77_OBJS=${HYPRE_DRIVERS_F77:.f=.o}
HYPRE_FEI_DRIVER_CXX_OBJS=${HYPRE_FEI_DRIVERS_CXX:.cxx=.o}

HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.c=}
HYPRE_DRIVER_F77_EXECS=${HYPRE_DRIVERS_F77:.f=}
HYPRE_FEI_DRIVER_CXX_EXECS=${HYPRE_FEI_DRIVERS_CXX:.cxx=}

HYPRE_BABEL_DRIVER_OBJS=${HYPRE_BABEL_DRIVERS:.c=.o}
HYPRE_BABEL_DRIVER_CXX_OBJS=${HYPRE_BABEL_DRIVERS_CXX:.cxx=.o}
HYPRE_BABEL_DRIVER_F77_OBJS=${HYPRE_BABEL_DRIVERS_F77:.f=.o}

HYPRE_BABEL_DRIVER_EXECS=${HYPRE_BABEL_DRIVERS:.c=}
HYPRE_BABEL_DRIVER_CXX_EXECS=${HYPRE_BABEL_DRIVERS_CXX:.cxx=}
HYPRE_BABEL_DRIVER_F77_EXECS=${HYPRE_BABEL_DRIVERS_F77:.f=}

all: ${HYPRE_DRIVER_EXECS}

all77: ${HYPRE_DRIVER_F77_EXECS}

all++: ${HYPRE_FEI_DRIVER_CXX_EXECS}

babel: ${HYPRE_BABEL_DRIVER_EXECS} ${HYPRE_BABEL_DRIVER_F77_EXECS}

install:

clean:
	rm -f *.o

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}
	rm -f ${HYPRE_DRIVER_F77_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_CXX_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_F77_EXECS}
	rm -f ${HYPRE_FEI_DRIVER_CXX_EXECS}

##################################################################
# Rules
##################################################################
PURIFY = purify
PURIFY_TO_FILE = purify \
 -log-file=struct_linear_solvers.purify \
 -append-logfile=yes

struct: struct.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${STRUCT_LFLAGS}

struct_es: struct_es.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${STRUCT_ES_LFLAGS}

sstruct: sstruct.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${SSTRUCT_LFLAGS}

sstruct_fac: sstruct_fac.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${SSTRUCT_LFLAGS}

sstruct_es: sstruct_es.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${SSTRUCT_ES_LFLAGS}

ij: ij.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_LFLAGS}

ij_es: ij_es.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_ES_LFLAGS}

ij_mv: ij_mv.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_MV_LFLAGS}

new_ij: new_ij.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${IJ_LFLAGS}

fei: fei.o
	@echo  "Building" $@ "... "
	${CXX} -o $@ $@.o ${FEICXX_LFLAGS}

f77_struct: f77_struct.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${STRUCT_LFLAGS}

f77_ij: f77_ij.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${IJ_LFLAGS}

f77_ij_mv: f77_ij_mv.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${IJ_MV_LFLAGS}

driver_commpkg: driver_commpkg.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${COMMPKG_LFLAGS}

##################################################################
# Babel version of above
##################################################################
${HYPRE_BABEL_DRIVER_EXECS}: ${HYPRE_BABEL_DRIVER_OBJS}
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${BABELLIBFLAGS} ${LFLAGS}

${HYPRE_BABEL_DRIVER_F77_EXECS}: ${HYPRE_BABEL_DRIVER_F77_OBJS}
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${LFLAGS}
