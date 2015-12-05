#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision: 2.16 $
#EHEADER**********************************************************************




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
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../sstruct_mv\
 -I$(srcdir)/../struct_ls\
 -I$(srcdir)/../parcsr_ls\
 ${CINCLUDES}

HEADERS =\
 headers.h\
 HYPRE_sstruct_int.h\
 HYPRE_sstruct_ls.h\
 sstruct_ls.h\
 sys_pfmg.h\
 fac.h\
 sstruct_owninfo.h\
 sstruct_recvinfo.h\
 sstruct_sendinfo.h\
 nd1_amge_interpolation.h\
 maxwell_TV.h

FILES =\
 F90_HYPRE_sstruct_bicgstab.c\
 F90_HYPRE_sstruct_gmres.c\
 F90_HYPRE_sstruct_InterFAC.c\
 F90_HYPRE_sstruct_int.c\
 F90_HYPRE_sstruct_maxwell.c\
 F90_HYPRE_sstruct_pcg.c\
 F90_HYPRE_sstruct_split.c\
 F90_HYPRE_sstruct_sys_pfmg.c\
 HYPRE_sstruct_bicgstab.c\
 HYPRE_sstruct_gmres.c\
 HYPRE_sstruct_InterFAC.c\
 HYPRE_sstruct_int.c\
 HYPRE_sstruct_maxwell.c\
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
 eliminate_rowscols.c\
 maxwell_grad.c\
 maxwell_physbdy.c\
 maxwell_PNedelec.c\
 maxwell_PNedelec_bdy.c\
 maxwell_semi_interp.c\
 maxwell_TV.c\
 maxwell_TV_setup.c\
 maxwell_zeroBC.c\
 hypre_MaxwellSolve.c\
 hypre_MaxwellSolve2.c\
 nd1_amge_interpolation.c\
 node_relax.c\
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
 sys_semi_restrict.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_sstruct_ls${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/sstruct_ls.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_sstruct_ls${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/sstruct_ls.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_sstruct_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_sstruct_ls.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o $@ ${OBJS}

${OBJS}: ${HEADERS}
