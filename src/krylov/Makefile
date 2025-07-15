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
 pcg.c\
 lobpcg.c\
 HYPRE_bicgstab.c\
 HYPRE_cgnr.c\
 HYPRE_gmres.c\
 HYPRE_cogmres.c\
 HYPRE_lgmres.c\
 HYPRE_flexgmres.c\
 HYPRE_pcg.c\
 HYPRE_lobpcg.c

# Mixed precision files
MP_FILES = \
 mup_fixed.c\
 mup_functions.c\
 mup_pre.c\
 mup_lobpcg_fixed.c\
 mup_lobpcg_functions.c\
 mup_lobpcg_pre.c

COBJS = ${FILES:.c=.o}
OBJS = ${COBJS}

ifeq (${MP_BUILD}, 1) 

COBJS_single     = ${FILES:.c=.o_flt}
COBJS_double     = ${FILES:.c=.o_dbl}
COBJS_longdouble = ${FILES:.c=.o_ldbl}
COBJS_default    = ${FILES:.c=.o_def}
MP_COBJS         = ${MP_FILES:.c=.o}

OBJS = ${COBJS_single} ${COBJS_double} ${COBJS_longdouble} ${COBJS_default} ${MP_COBJS}

endif

SONAME = libHYPRE_krylov-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_krylov${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/krylov.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/lobpcg.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: all
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/krylov.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/lobpcg.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -f *.o_flt *.o_dbl *.o_ldbl *.o_def
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Build Rules
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
