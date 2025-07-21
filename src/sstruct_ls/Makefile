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
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../seq_block_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../parcsr_block_mv\
 -I$(srcdir)/../sstruct_mv\
 -I$(srcdir)/../struct_ls\
 -I$(srcdir)/../parcsr_ls\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}

HEADERS =\
 HYPRE_sstruct_ls.h\
 _hypre_sstruct_ls.h\
 sys_pfmg.h

FILES =\
 F90_HYPRE_sstruct_bicgstab.c\
 F90_HYPRE_sstruct_gmres.c\
 F90_HYPRE_sstruct_flexgmres.c\
 F90_HYPRE_sstruct_lgmres.c\
 F90_HYPRE_sstruct_int.c\
 F90_HYPRE_sstruct_pcg.c\
 F90_HYPRE_sstruct_split.c\
 F90_HYPRE_sstruct_sys_pfmg.c\
 HYPRE_sstruct_bicgstab.c\
 HYPRE_sstruct_gmres.c\
 HYPRE_sstruct_flexgmres.c\
 HYPRE_sstruct_lgmres.c\
 HYPRE_sstruct_int.c\
 HYPRE_sstruct_pcg.c\
 HYPRE_sstruct_split.c\
 HYPRE_sstruct_sys_pfmg.c\
 krylov.c\
 krylov_sstruct.c\
 sys_pfmg.c\
 sys_pfmg_relax.c\
 sys_pfmg_setup.c\
 sys_pfmg_setup_interp.c\
 sys_pfmg_setup_rap.c\
 sys_pfmg_solve.c\
 sys_semi_interp.c\
 sys_semi_restrict.c

CUFILES =\
 node_relax.c\

# Mixed precision files
MP_FILES = \
 mup_fixed.c\
 mup_functions.c\
 mup_pre.c

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

SONAME = libHYPRE_sstruct_ls-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_sstruct_ls${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_sstruct_ls${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -f *.o_flt *.o_dbl *.o_ldbl
	rm -f *.obj_flt *.obj_dbl *.obj_ldbl
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_sstruct_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_sstruct_ls.so libHYPRE_sstruct_ls.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
