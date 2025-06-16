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
 fac.h\
 maxwell_TV.h\
 nd1_amge_interpolation.h\
 sstruct_owninfo.h\
 sstruct_recvinfo.h\
 sstruct_sendinfo.h\
 sys_pfmg.h

FILES =\

CUFILES =\
 fac_amr_fcoarsen.c\
 fac_amr_rap.c\
 fac_restrict2.c\
 fac_setup2.c\
 fac_zero_stencilcoef.c\
 node_relax.c\

#######################
# Multiprecision files
#######################
MuP_FILES = \
 F90_HYPRE_sstruct_bicgstab.c\
 F90_HYPRE_sstruct_gmres.c\
 F90_HYPRE_sstruct_flexgmres.c\
 F90_HYPRE_sstruct_lgmres.c\
 F90_HYPRE_sstruct_InterFAC.c\
 F90_HYPRE_sstruct_int.c\
 F90_HYPRE_sstruct_maxwell.c\
 F90_HYPRE_sstruct_pcg.c\
 F90_HYPRE_sstruct_split.c\
 F90_HYPRE_sstruct_sys_pfmg.c\
 bsearch.c\
 fac.c\
 fac_amr_zero_data.c\
 fac_cf_coarsen.c\
 fac_cfstencil_box.c\
 fac_CFInterfaceExtents.c\
 fac_interp2.c\
 fac_relax.c\
 fac_solve3.c\
 fac_zero_cdata.c\
 krylov.c\
 krylov_sstruct.c\
 eliminate_rowscols.c\
 maxwell_grad.c\
 maxwell_physbdy.c\
 maxwell_PNedelec.c\
 maxwell_PNedelec_bdy.c\
 maxwell_semi_interp.c\
 maxwell_solve.c\
 maxwell_solve2.c\
 maxwell_TV.c\
 maxwell_TV_setup.c\
 maxwell_zeroBC.c\
 nd1_amge_interpolation.c\
 sstruct_amr_intercommunication.c\
 sstruct_owninfo.c\
 sstruct_recvinfo.c\
 sstruct_sendinfo.c\
 sstruct_sharedDOFComm.c\
 sys_pfmg.c\
 sys_pfmg_relax.c\
 sys_pfmg_setup.c\
 sys_pfmg_setup_interp.c\
 sys_pfmg_setup_rap.c\
 sys_pfmg_solve.c\
 sys_semi_interp.c\
 sys_semi_restrict.c\
 HYPRE_sstruct_bicgstab.c\
 HYPRE_sstruct_gmres.c\
 HYPRE_sstruct_flexgmres.c\
 HYPRE_sstruct_lgmres.c\
 HYPRE_sstruct_InterFAC.c\
 HYPRE_sstruct_int.c\
 HYPRE_sstruct_maxwell.c\
 HYPRE_sstruct_pcg.c\
 HYPRE_sstruct_split.c\
 HYPRE_sstruct_sys_pfmg.c

ifeq (${MP_BUILD}, 1) 
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
   MuP_COBJS = ${MuP_FILES:.c=.o}
else
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
endif

SONAME = libHYPRE_sstruct_ls-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_sstruct_ls${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_sstruct_ls.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/sstruct_ls_mup_func.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_sstruct_ls${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_sstruct_ls.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/sstruct_ls_mup_func.h $(HYPRE_INC_INSTALL)
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
#
${COBJS}: CFLAGS += -DMP_BUILD_DEFAULT=1
${CUOBJS}: CFLAGS += -DMP_BUILD_DEFAULT=1

OBJS = ${COBJS} ${CUOBJS} ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

else
OBJS = ${COBJS} ${CUOBJS}
endif
#
libHYPRE_sstruct_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_sstruct_ls.so libHYPRE_sstruct_ls.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
