# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../blas\
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

ifeq (${MP_BUILD}, 1) 

COBJS_single      = ${FILES:.c=.o_flt}
COBJS_double      = ${FILES:.c=.o_dbl}
COBJS_longdouble  = ${FILES:.c=.o_ldbl}

OBJS = ${COBJS_single} ${COBJS_double} ${COBJS_longdouble}

endif

SONAME = libHYPRE_DistributedMatrixPilutSolver-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -f *.o_flt *.o_dbl *.o_ldbl
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Main rules
##################################################################

libHYPRE_DistributedMatrixPilutSolver.a: ${OBJS}
	@echo "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_DistributedMatrixPilutSolver.so libHYPRE_DistributedMatrixPilutSolver.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
