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


include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/../../\
 -I$(srcdir)/../../utilities\
 ${CINCLUDES}

HEADERS =\
 Cnames.h\
 colamd.h\
 dsp_defs.h\
 superlu_util.h\
 supermatrix.h

FILES =\
 colamd.c \
 dcolumn_bmod.c \
 dcolumn_dfs.c \
 dcopy_to_ucol.c \
 dgscon.c \
 dgsequ.c \
 dgsrfs.c \
 dgssv.c \
 dgssvx.c \
 dgstrf.c \
 dgstrs.c \
 dlacon.c \
 dlangs.c \
 dlaqgs.c \
 dmemory.c \
 dpanel_bmod.c \
 dpanel_dfs.c \
 dpivotgrowth.c \
 dpivotL.c \
 dpruneL.c \
 dreadhb.c \
 dsnode_bmod.c \
 dsnode_dfs.c \
 dsp_blas2.c \
 dsp_blas3.c \
 dutil.c \
 get_perm_c.c \
 mmd.c \
 relax_snode.c \
 sp_coletree.c \
 sp_ienv.c \
 sp_preorder.c \
 superlu_memory.c \
 superlu_timer.c \
 superlu_util.c

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libHYPRE_superlu${HYPRE_LIB_SUFFIX}
	cp -fp *.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_superlu${HYPRE_LIB_SUFFIX}
	cp -f *.h $(HYPRE_INC_INSTALL)
	cp -f libHYPRE* $(HYPRE_LIB_INSTALL)
	@echo  " "

clean:
	rm -f *.o *.lo libHYPRE*
	rm -rf pchdir tca.map *inslog* .libs

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_superlu${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
