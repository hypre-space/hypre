#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.3 $
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../krylov\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../sstruct_mv\
 -I$(srcdir)/../struct_ls\
 -I$(srcdir)/../parcsr_ls\
 ${CINCLUDES}

HEADERS =\
 headers.h\
 HYPRE_sstruct_ls.h\
 sstruct_ls.h\
 sys_pfmg.h\
 fac.h\
 sstruct_owninfo.h\
 sstruct_recvinfo.h\
 sstruct_sendinfo.h

FILES =\
 F90_HYPRE_sstruct_bicgstab.c\
 F90_HYPRE_sstruct_gmres.c\
 F90_HYPRE_sstruct_pcg.c\
 F90_HYPRE_sstruct_split.c\
 F90_HYPRE_sstruct_sys_pfmg.c\
 HYPRE_sstruct_bicgstab.c\
 HYPRE_sstruct_gmres.c\
 HYPRE_sstruct_InterFAC.c\
 HYPRE_sstruct_pcg.c\
 HYPRE_sstruct_split.c\
 HYPRE_sstruct_sys_pfmg.c\
 fac.c\
 fac_amr_rap.c\
 fac_amr_fcoarsen.c\
 fac_amr_zero_data.c\
 fac_cf_coarsen.c\
 fac_cfstencil_box.c\
 fac_CFInterfaceExtents.c\
 fac_interp2.c\
 fac_relax.c\
 fac_restrict2.c\
 fac_setup2.c\
 fac_solve3.c\
 fac_zero_cdata.c\
 fac_zero_stencilcoef.c\
 hypre_bsearch.c\
 krylov.c\
 krylov_sstruct.c\
 node_relax.c\
 sstruct_amr_intercommunication.c\
 sstruct_owninfo.c\
 sstruct_recvinfo.c\
 sstruct_sendinfo.c\
 sys_pfmg.c\
 sys_pfmg_relax.c\
 sys_pfmg_setup.c\
 sys_pfmg_setup_interp.c\
 sys_pfmg_setup_rap.c\
 sys_pfmg_solve.c\
 sys_semi_interp.c\
 sys_semi_restrict.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_sstruct_ls.a
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include/.
	cp -fp $(srcdir)/sstruct_ls.h $(HYPRE_BUILD_DIR)/include/.
	cp -fp libHYPRE_*.a $(HYPRE_BUILD_DIR)/lib/.

install: libHYPRE_sstruct_ls.a
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)/.
	cp -fp $(srcdir)/sstruct_ls.h $(HYPRE_INC_INSTALL)/.
	cp -fp libHYPRE_*.a $(HYPRE_LIB_INSTALL)/.

clean:
	rm -f *.o lib*.a

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_sstruct_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
