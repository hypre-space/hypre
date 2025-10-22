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
 -I$(srcdir)/../../lapack\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
 ${CINCLUDES}

HEADERS =\
 _hypre_ParaSails.h\
 Common.h\
 ConjGrad.h\
 DiagScale.h\
 Hash.h\
 LoadBal.h\
 Matrix.h\
 Mem.h\
 Numbering.h\
 OrderStat.h\
 ParaSails.h\
 PrunedRows.h\
 RowPatt.h\
 StoredRows.h

FILES =\
 ConjGrad.c\
 DiagScale.c\
 FGmres.c\
 Hash.c\
 hypre_ParaSails.c\
 LoadBal.c\
 Matrix.c\
 Mem.c\
 Numbering.c\
 OrderStat.c\
 ParaSails.c\
 PrunedRows.c\
 RowPatt.c\
 StoredRows.c

OBJS = ${FILES:.c=.o}

ifeq (${MP_BUILD}, 1) 

COBJS_single      = ${FILES:.c=.o_flt}
COBJS_double      = ${FILES:.c=.o_dbl}
COBJS_longdouble  = ${FILES:.c=.o_ldbl}

OBJS = ${COBJS_single} ${COBJS_double} ${COBJS_longdouble}

endif

SONAME = libHYPRE_ParaSails-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_ParaSails${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_ParaSails${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -f *.o_flt *.o_dbl *.o_ldbl
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_ParaSails.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_ParaSails.so libHYPRE_ParaSails.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
