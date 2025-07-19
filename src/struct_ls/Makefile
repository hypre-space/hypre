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
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../krylov\
 -I$(srcdir)/../struct_mv\
 ${CINCLUDES}

HEADERS =\
 HYPRE_struct_ls.h\
 _hypre_struct_ls.h\
 pfmg.h\
 red_black_gs.h\
 smg.h

FILES =\
 coarsen.c\
 F90_HYPRE_struct_bicgstab.c\
 F90_HYPRE_struct_cycred.c\
 F90_HYPRE_struct_flexgmres.c\
 F90_HYPRE_struct_gmres.c\
 F90_HYPRE_struct_lgmres.c\
 F90_HYPRE_struct_hybrid.c\
 F90_HYPRE_struct_int.c\
 F90_HYPRE_struct_jacobi.c\
 F90_HYPRE_struct_pcg.c\
 F90_HYPRE_struct_pfmg.c\
 F90_HYPRE_struct_smg.c\
 hybrid.c\
 HYPRE_struct_bicgstab.c\
 HYPRE_struct_cycred.c\
 HYPRE_struct_flexgmres.c\
 HYPRE_struct_gmres.c\
 HYPRE_struct_hybrid.c\
 HYPRE_struct_jacobi.c\
 HYPRE_struct_lgmres.c\
 HYPRE_struct_pfmg.c\
 HYPRE_struct_smg.c\
 jacobi.c\
 pcg_struct.c\
 pfmg.c\
 pfmg_relax.c\
 pfmg_setup_rap.c\
 pfmg_solve.c\
 semi.c\
 smg_relax.c\
 smg_setup.c\
 smg_setup_rap.c\
 smg_setup_restrict.c\
 smg_solve.c

CUFILES =\
 cyclic_reduction.c\
 setrandom.c\
 diagscale.c\
 HYPRE_struct_int.c\
 HYPRE_struct_pcg.c\
 pfmg2_setup_rap.c\
 pfmg3_setup_rap.c\
 pfmg_setup.c\
 pfmg_setup_interp.c\
 pfmg_setup_rap5.c\
 pfmg_setup_rap7.c\
 point_relax.c\
 red_black_constantcoef_gs.c\
 red_black_gs.c\
 semi_interp.c\
 semi_restrict.c\
 semi_setup_rap.c\
 smg2_setup_rap.c\
 smg3_setup_rap.c\
 smg.c\
 smg_axpy.c\
 smg_residual.c\
 smg_setup_interp.c

# Mixed precision files
MP_FILES = \
 mup_fixed.c\
 mup_functions.c\
 mup_pre.c
# HYPRE_struct_ls_mp.c    # Add this later

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

ifeq (${MP_BUILD}, 1) 

COBJS_single      = ${FILES:.c=.o_flt}
COBJS_double      = ${FILES:.c=.o_dbl}
COBJS_longdouble  = ${FILES:.c=.o_ldbl}
COBJS_default     = ${FILES:.c=.o_def}
CUOBJS_single     = ${CUFILES:.c=.obj_flt}
CUOBJS_double     = ${CUFILES:.c=.obj_dbl}
CUOBJS_longdouble = ${CUFILES:.c=.obj_ldbl}
CUOBJS_default    = ${CUFILES:.c=.obj_def}
MP_COBJS          = ${MP_FILES:.c=.o}

OBJS = ${COBJS_single} ${COBJS_double} ${COBJS_longdouble} ${COBJS_default} ${MP_COBJS}
OBJS += ${CUOBJS_single} ${CUOBJS_double} ${CUOBJS_longdouble} ${CUOBJS_default}

endif

SONAME = libHYPRE_struct_ls-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_struct_ls${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_struct_ls${HYPRE_LIB_SUFFIX}
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

libHYPRE_struct_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_struct_ls.so libHYPRE_struct_ls.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
