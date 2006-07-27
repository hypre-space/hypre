#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer and the GNU Lesser General Public License.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
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
 ${CINCLUDES}

HEADERS =\
 headers.h\
 HYPRE_struct_int.h\
 HYPRE_struct_ls.h\
 pfmg.h\
 red_black_gs.h\
 smg.h\
 sparse_msg.h\
 struct_ls.h

FILES =\
 coarsen.c\
 cyclic_reduction.c\
 F90_HYPRE_struct_bicgstab.c\
 F90_HYPRE_struct_gmres.c\
 F90_HYPRE_struct_hybrid.c\
 F90_HYPRE_struct_int.c\
 F90_HYPRE_struct_jacobi.c\
 F90_HYPRE_struct_pcg.c\
 F90_HYPRE_struct_pfmg.c\
 F90_HYPRE_struct_smg.c\
 F90_HYPRE_struct_sparse_msg.c\
 general.c\
 hybrid.c\
 HYPRE_struct_bicgstab.c\
 HYPRE_struct_hybrid.c\
 HYPRE_struct_int.c\
 HYPRE_struct_jacobi.c\
 HYPRE_struct_pfmg.c\
 HYPRE_struct_smg.c\
 HYPRE_struct_sparse_msg.c\
 HYPRE_struct_pcg.c\
 HYPRE_struct_gmres.c\
 jacobi.c\
 pcg_struct.c\
 pfmg2_setup_rap.c\
 pfmg3_setup_rap.c\
 pfmg.c\
 pfmg_relax.c\
 pfmg_setup.c\
 pfmg_setup_interp.c\
 pfmg_setup_rap5.c\
 pfmg_setup_rap7.c\
 pfmg_setup_rap.c\
 pfmg_solve.c\
 point_relax.c\
 red_black_constantcoef_gs.c\
 red_black_gs.c\
 semi_interp.c\
 semi_restrict.c\
 semi_setup_rap.c\
 smg2_setup_rap.c\
 smg3_setup_rap.c\
 smg_axpy.c\
 smg.c\
 smg_relax.c\
 smg_residual.c\
 smg_setup.c\
 smg_setup_interp.c\
 smg_setup_rap.c\
 smg_setup_restrict.c\
 smg_solve.c\
 sparse_msg2_setup_rap.c\
 sparse_msg3_setup_rap.c\
 sparse_msg.c\
 sparse_msg_filter.c\
 sparse_msg_interp.c\
 sparse_msg_restrict.c\
 sparse_msg_setup.c\
 sparse_msg_setup_rap.c\
 sparse_msg_solve.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_struct_ls${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/struct_ls.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_struct_ls${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/struct_ls.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_struct_ls${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
