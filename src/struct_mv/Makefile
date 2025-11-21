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
 box_ds.h\
 box_manager.h\
 computation.h\
 struct_communication.h\
 struct_grid.h\
 struct_matrix.h\
 struct_stencil.h\
 struct_stmatrix.h\
 struct_vector.h\
 struct_matmult.h

FILES =\
 assumed_part.c\
 box_algebra.c\
 box_boundary.c\
 box.c\
 box_ds.c\
 box_manager.c\
 coarsen.c\
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
 struct_stencil.c\
 struct_stmatrix.c

CUFILES =\
 struct_matvec.c\
 struct_matvec_cc.c\
 struct_matvec_icc.c\
 struct_matvec_ivc.c\
 struct_matvec_ivcc.c\
 struct_matvec_vc.c\
 struct_matvec_vcc.c\
 struct_matmult.c\
 struct_matmult_core.c\
 struct_matmult_fuse.c\
 struct_matmult_fuse_fff.c\
 struct_matmult_fuse_ffc.c\
 struct_matmult_fuse_fcc.c\
 struct_matmult_fuse_ff.c\
 struct_matmult_fuse_fc.c\
 struct_matmult_fuse_cc.c\
 struct_matmult_fuse_c.c\
 struct_matmult_fuse_f.c\
 struct_axpy.c\
 struct_communication.c\
 struct_copy.c\
 struct_data.c\
 struct_innerprod.c\
 struct_matop.c\
 struct_matrix.c\
 struct_scale.c\
 struct_vector.c

# Mixed precision files
MP_FILES = \
 mup_fixed.c\
 mup_functions.c\
 mup_pre.c\
 HYPRE_struct_mv_mp.c\
 struct_mv_mp.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

ifeq (${MP_BUILD}, 1)

COBJS_single      = ${FILES:.c=.o_flt}
COBJS_double      = ${FILES:.c=.o_dbl}
COBJS_longdouble  = ${FILES:.c=.o_ldbl}
CUOBJS_single     = ${CUFILES:.c=.obj_flt}
CUOBJS_double     = ${CUFILES:.c=.obj_dbl}
CUOBJS_longdouble = ${CUFILES:.c=.obj_ldbl}
MP_COBJS          = ${MP_FILES:.c=.o}

OBJS = ${COBJS_single} ${COBJS_double} ${COBJS_longdouble} ${MP_COBJS}
OBJS += ${CUOBJS_single} ${CUOBJS_double} ${CUOBJS_longdouble}

endif

SONAME = libHYPRE_struct_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_struct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_struct_mv.hpp $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_struct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -f *.o_flt *.o_dbl *.o_ldbl *.o_def
	rm -f *.obj_flt *.obj_dbl *.obj_ldbl *.obj_def
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
