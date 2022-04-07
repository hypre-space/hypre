# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 HYPRE_struct_mv.h\
 _hypre_struct_mv.h\
 _hypre_struct_mv.hpp\
 assumed_part.h\
 box.h\
 box_manager.h\
 computation.h\
 struct_communication.h\
 struct_grid.h\
 struct_matrix.h\
 struct_stencil.h\
 struct_vector.h

FILES =\
 assumed_part.c\
 box_algebra.c\
 box_boundary.c\
 box.c\
 box_manager.c\
 communication_info.c\
 computation.c\
 F90_HYPRE_struct_grid.c\
 F90_HYPRE_struct_matrix.c\
 F90_HYPRE_struct_stencil.c\
 F90_HYPRE_struct_vector.c\
 HYPRE_struct_grid.c\
 HYPRE_struct_matrix.c\
 HYPRE_struct_stencil.c\
 HYPRE_struct_vector.c\
 project.c\
 struct_grid.c\
 struct_io.c\
 struct_matrix_mask.c\
 struct_stencil.c

CUFILES =\
 struct_axpy.c\
 struct_communication.c\
 struct_copy.c\
 struct_innerprod.c\
 struct_matrix.c\
 struct_matvec.c\
 struct_scale.c\
 struct_vector.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

SONAME = libHYPRE_struct_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_struct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_struct_mv.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_struct_mv.hpp $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_struct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_struct_mv.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_struct_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_struct_mv.so libHYPRE_struct_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
