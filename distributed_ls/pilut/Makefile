#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
${CINCLUDES}

HEADERS = \
 HYPRE_DistributedMatrixPilutSolver_protos.h\
 HYPRE_DistributedMatrixPilutSolver_types.h\
 DistributedMatrixPilutSolver.h\
 const.h\
 ilu.h\
 internal_protos.h\
 macros.h\
 struct.h

FILES =\
 comm.c\
 debug.c\
 distributed_qsort.c\
 distributed_qsort_si.c\
 HYPRE_DistributedMatrixPilutSolver.c\
 ilut.c\
 parilut.c\
 parutil.c\
 pblas1.c\
 serilut.c\
 trifactor.c\
 util.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
	cp -f $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -f libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Main rules
##################################################################

libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
