# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../seq_mv\
 ${CINCLUDES}

EXTRA_HEADERS = internal_protos.h

HEADERS =\
 distributed_matrix.h\
 HYPRE_distributed_matrix_mv.h\
 HYPRE_distributed_matrix_protos.h\
 HYPRE_distributed_matrix_types.h

FILES =\
 distributed_matrix.c\
 HYPRE_distributed_matrix.c\
 distributed_matrix_ISIS.c\
 distributed_matrix_PETSc.c\
 distributed_matrix_parcsr.c

OBJS = ${FILES:.c=.o}

ifeq (${MP_BUILD}, 1) 

COBJS_single      = ${FILES:.c=.o_flt}
COBJS_double      = ${FILES:.c=.o_dbl}
COBJS_longdouble  = ${FILES:.c=.o_ldbl}

OBJS = ${COBJS_single} ${COBJS_double} ${COBJS_longdouble}

endif

SONAME = libHYPRE_DistributedMatrix-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_DistributedMatrix${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_DistributedMatrix${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -f *.o_flt *.o_dbl *.o_ldbl
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_DistributedMatrix.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_DistributedMatrix.so libHYPRE_DistributedMatrix.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
