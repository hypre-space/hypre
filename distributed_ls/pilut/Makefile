.SUFFIXES:
.SUFFIXES: .c .f .o

all: lib

#Driver stuff; ignore if just concerned about library
WANG  = test/wang4.bin
G40   = test/g40x40x40.10.bin
TORSO = test/torso.bin
ALE0  = -ysmp test/test0_full.matx
ALE1  = -ysmp test/test1_full.matx
ARGS  = 5 1e-4 1 20 10 1e-8

# Gnu gcc
# OPTIMIZATION = -O3
OTHER_LIBS = /home/casc/lib/blas_solaris.a -R/opt/SUNWspro/lib \
             -L/opt/SUNWspro/lib -lF77 -lM77 -lsunmath

# MPICH on the CASC cluster @ LLNL
MPI_INCLUDES = -I/usr/local/mpi/mpich/include
MPI_LIBDIR   = -L/usr/local/mpi/mpich/lib/solaris/ch_p4/
MPI_LIBS     = -lmpi -lnsl -lsocket -lgen
MPIRUN       = mpirun -np 4

#PETSc; not used here, but used by objects called from here 
PETSC_INCLUDES = -I${PETSC_DIR}/include
# General
HYPRE_INCLUDES = -I../.. -I../../utilities -I/home/casc/include
INCLUDES = $(HYPRE_INCLUDES) $(MPI_INCLUDES) $(PETSC_INCLUDES)
LIBDIR   = $(MPI_LIBDIR)
LIBS     = -L. -lHYPREpilut -lm $(MPI_LIBS) $(OTHER_LIBS)

# Debugging via asserts
DEBUGFLAGS = -DHYPRE_DEBUG -DHYPRE_MEMORY_CHECK #-DNDEBUG #-DTIME
CFLAGS = $(OPTIMIZATION) $(INCLUDES) $(DEBUGFLAGS)

HYPREFILES =\
 HYPRE_DistributedMatrixPilutSolver.c

LOCALFILES = debug.c comm.c parutil.c serilut.c parilut.c util.c qsort.c\
       qsort_si.c trifactor.c ilut.c pblas1.c 

FILES = $(HYPREFILES) $(LOCALFILES)

HEADERS = \
       DistributedMatrixPilutSolver.h const.h ilu.h macros.h struct.h

OBJS = ${FILES:.c=.o}

${OBJS}: ${HEADERS}

#include site-specific information
include ../../bmakes/base

##################################################################
# Generic rules
##################################################################

.c.o:
	@echo "Making (c) " $@
	${CC} -o $@ -c ${CFLAGS} $<


##################################################################
# Main rules
##################################################################

lib:    $(PROTOS) $(OBJS)
	ar -r libHYPREDistributedMatrixPilutSolver.a $(OBJS)

