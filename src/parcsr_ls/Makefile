# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

## Euclid specific macros taken from distributed_ls/Euclid

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS =\
 -I..\
 -I../distributed_ls/Euclid\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../blas\
 -I$(srcdir)/../lapack\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../krylov\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../distributed_matrix\
 -I$(srcdir)/../matrix_matrix\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../parcsr_block_mv\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}

HEADERS =\
 HYPRE_parcsr_ls.h\
 _hypre_parcsr_ls.h\
 aux_interp.h\
 block_tridiag.h\
 Common.h\
 par_amg.h\
 par_amgdd.h\
 par_amgdd_comp_grid.h\
 par_fsai.h\
 schwarz.h

FILES =\
 HYPRE_parcsr_ls_mp.c

CUFILES =\
 ads.c\
 ams.c\
 ame.c\
 dsuperlu.c\
 par_amg_setup.c\
 par_ge_device.c\
 par_ilu_setup_device.c\
 par_ilu_solve_device.c\
 par_ilu_solve.c \
 par_cheby_device.c\
 par_relax_more_device.c\
 par_coarsen_device.c\
 par_coarse_parms_device.c\
 par_indepset_device.c\
 par_interp_device.c\
 par_lr_restr_device.c\
 par_interp_trunc_device.c\
 par_lr_interp_device.c\
 par_strength_device.c\
 par_strength2nd_device.c\
 par_amgdd_fac_cycle_device.c\
 par_2s_interp_device.c\
 par_relax_device.c\
 par_mod_multi_interp_device.c\
 par_mgr_device.c\
 par_fsai_device.c

#######################
# Multiprecision files
#######################
MuP_FILES = \
 amg_hybrid.c\
 aux_interp.c\
 F90_hypre_laplace.c\
 F90_HYPRE_parcsr_amg.c\
 F90_HYPRE_parcsr_bicgstab.c\
 F90_HYPRE_parcsr_block.c\
 F90_HYPRE_parcsr_cgnr.c\
 F90_HYPRE_parcsr_Euclid.c\
 F90_HYPRE_parcsr_gmres.c\
 F90_HYPRE_parcsr_cogmres.c\
 F90_HYPRE_parcsr_flexgmres.c\
 F90_HYPRE_parcsr_lgmres.c\
 F90_HYPRE_parcsr_hybrid.c\
 F90_HYPRE_parcsr_int.c\
 F90_HYPRE_parcsr_ParaSails.c\
 F90_HYPRE_parcsr_pcg.c\
 F90_HYPRE_parcsr_pilut.c\
 F90_HYPRE_parcsr_schwarz.c\
 F90_HYPRE_parcsr_mgr.c\
 F90_HYPRE_parcsr_ilu.c\
 F90_HYPRE_ams.c\
 gen_redcs_mat.c\
 HYPRE_parcsr_amg.c\
 HYPRE_parcsr_amgdd.c\
 HYPRE_parcsr_bicgstab.c\
 HYPRE_parcsr_block.c\
 HYPRE_parcsr_cgnr.c\
 HYPRE_parcsr_Euclid.c\
 HYPRE_parcsr_gmres.c\
 HYPRE_parcsr_cogmres.c\
 HYPRE_parcsr_flexgmres.c\
 HYPRE_parcsr_lgmres.c\
 HYPRE_parcsr_hybrid.c\
 HYPRE_parcsr_int.c\
 HYPRE_parcsr_mgr.c\
 HYPRE_parcsr_ilu.c \
 HYPRE_parcsr_fsai.c \
 HYPRE_parcsr_ParaSails.c\
 HYPRE_parcsr_pcg.c\
 HYPRE_parcsr_pilut.c\
 HYPRE_parcsr_schwarz.c\
 HYPRE_ams.c\
 HYPRE_ads.c\
 HYPRE_ame.c\
 par_2s_interp.c\
 par_amg.c\
 par_amgdd.c\
 par_amgdd_comp_grid.c\
 par_amgdd_setup.c\
 par_amgdd_solve.c\
 par_amgdd_fac_cycle.c\
 par_amgdd_helpers.c\
 par_amg_solve.c\
 par_amg_solveT.c\
 par_fsai.c\
 par_fsai_setup.c\
 par_fsai_solve.c\
 par_cg_relax_wt.c\
 par_coarsen.c\
 par_cgc_coarsen.c\
 par_cheby.c\
 par_coarse_parms.c\
 par_coordinates.c\
 par_cr.c\
 par_cycle.c\
 par_add_cycle.c\
 par_difconv.c\
 par_gauss_elim.c\
 par_gsmg.c\
 par_indepset.c\
 par_interp.c\
 par_jacobi_interp.c\
 par_krylov_func.c\
 par_ilu.c\
 par_ilu_setup.c \
 par_mod_lr_interp.c\
 par_multi_interp.c\
 par_mod_multi_interp.c\
 par_laplace.c\
 par_laplace_27pt.c\
 par_laplace_9pt.c\
 par_lr_interp.c\
 par_mgr.c\
 par_mgr_coarsen.c\
 par_mgr_setup.c\
 par_mgr_solve.c\
 par_mgr_stats.c\
 par_nongalerkin.c\
 par_nodal_systems.c\
 par_rap.c\
 par_rap_communication.c\
 par_rotate_7pt.c\
 par_relax.c\
 par_relax_more.c\
 par_relax_interface.c\
 par_scaled_matnorm.c\
 par_schwarz.c\
 par_stats.c\
 par_strength.c\
 par_sv_interp.c\
 par_sv_interp_ln.c\
 par_vardifconv.c\
 par_vardifconv_rs.c\
 partial.c\
 schwarz.c\
 block_tridiag.c\
 par_restr.c\
 par_lr_restr.c

ifeq (${MP_BUILD}, 1) 
   COBJS = ${FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
   MuP_COBJS = ${MuP_FILES:.c=.o}
else
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
endif

SONAME = libHYPRE_parcsr_ls-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_ls${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_parcsr_ls.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/parcsr_ls_mup_func.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_parcsr_ls_mup.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/HYPRE_parcsr_ls_mp.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_parcsr_ls${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_parcsr_ls.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/parcsr_ls_mup_func.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/hypre_parcsr_ls_mup.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/HYPRE_parcsr_ls_mp.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean: clean_mup
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

ifeq (${MP_BUILD}, 1) 
clean_mup:
	rm -rf ${MuP_OBJDIR}	
else
clean_mup:
endif

##################################################################
# Build Rules
##################################################################

ifeq (${MP_BUILD}, 1) 

MuP_OBJDIR = MuP_obj.dir
MuP_OBJDIR_single = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_flt}
MuP_OBJDIR_double = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_dbl}
MuP_OBJDIR_longdouble = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_ldbl}
#
ADD_OBJS_single = ${addprefix ${MuP_OBJDIR_single}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_single}: CFLAGS += -DMP_BUILD_SINGLE=1
${MuP_OBJDIR_single}/%.o ${MuP_OBJDIR_single}/%.obj: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_double = ${addprefix ${MuP_OBJDIR_double}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_double}: CFLAGS += -DMP_BUILD_DOUBLE=1
${MuP_OBJDIR_double}/%.o ${MuP_OBJDIR_double}/%.obj: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_longdouble = ${addprefix ${MuP_OBJDIR_longdouble}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_longdouble}: CFLAGS += -DMP_BUILD_LONGDOUBLE=1
${MuP_OBJDIR_longdouble}/%.o ${MuP_OBJDIR_longdouble}/%.obj: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<

OBJS = ${COBJS} ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

else
OBJS = ${COBJS} ${CUOBJS}
endif
#
libHYPRE_parcsr_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_parcsr_ls.so libHYPRE_parcsr_ls.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
