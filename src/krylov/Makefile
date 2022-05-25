# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../blas\
 -I$(srcdir)/../lapack\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 HYPRE_krylov.h\
 krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 cogmres.h\
 flexgmres.h\
 lgmres.h\
 pcg.h\
 HYPRE_lobpcg.h\
 HYPRE_MatvecFunctions.h\
 lobpcg.h

FILES =\
 bicgstab.c\
 cgnr.c\
 gmres.c\
 cogmres.c\
 flexgmres.c\
 lgmres.c\
 HYPRE_bicgstab.c\
 HYPRE_cgnr.c\
 HYPRE_gmres.c\
 HYPRE_cogmres.c\
 HYPRE_lgmres.c\
 HYPRE_flexgmres.c\
 HYPRE_pcg.c\
 pcg.c\
 HYPRE_lobpcg.c\
 lobpcg.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_krylov-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_krylov${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/krylov.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/lobpcg.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: all
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/krylov.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/lobpcg.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_krylov.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_krylov.so libHYPRE_krylov.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
