# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS =\
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../seq_block_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../parcsr_ls\
 ${CINCLUDES}

HEADERS =\
 aux_parcsr_matrix.h\
 aux_par_vector.h\
 HYPRE_IJ_mv.h\
 _hypre_IJ_mv.h\
 IJ_matrix.h\
 IJ_vector.h

EXTRA_FILES =\
 IJMatrix_isis.c\
 IJMatrix_petsc.c

FILES =\
 aux_parcsr_matrix.c\
 aux_par_vector.c\
 F90_HYPRE_IJMatrix.c\
 F90_HYPRE_IJVector.c\
 F90_IJMatrix.c\
 HYPRE_IJMatrix.c\
 HYPRE_IJVector.c\
 IJ_assumed_part.c\
 IJMatrix.c\
 IJMatrix_parcsr.c\
 IJVector.c\
 IJVector_parcsr.c

CUFILES =\
 IJMatrix_parcsr_device.c\
 IJVector_parcsr_device.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

SONAME = libHYPRE_IJ_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_IJ_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_IJ_mv.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_IJ_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_IJ_mv.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_IJ_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_IJ_mv.so libHYPRE_IJ_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
