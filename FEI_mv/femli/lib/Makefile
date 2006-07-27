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


include ../../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

MLI_INCLUDES = \
 -I.\
 -I../../..\
 -I$(srcdir)/..\
 -I$(srcdir)/../../../utilities\
 -I$(srcdir)/../../../IJ_mv\
 -I$(srcdir)/../../../krylov\
 -I$(srcdir)/../../../multivector\
 -I$(srcdir)/../../../parcsr_mv\
 -I$(srcdir)/../../../parcsr_ls\
 -I$(srcdir)/../../../seq_mv\
 -I$(srcdir)/../../../distributed_matrix\
 -I$(srcdir)/../../../distributed_ls\
 -I$(srcdir)/../../../FEI_mv/fei-hypre\
 -I$(srcdir)/../../../FEI_mv/femli\
 -I$(srcdir)/../../../FEI_mv/SuperLU

C_COMPILE_FLAGS =\
 -DMLI_SUPERLU\
 ${MLI_INCLUDES}\
 ${CINCLUDES}
 
CXX_COMPILE_FLAGS =\
 -DMLI_SUPERLU -DMPICH_SKIP_MPICXX\
 ${MLI_INCLUDES}\
 ${CINCLUDES}


HEADERS =\
 ../amgs/mli_method.h\
 ../amgs/mli_method_amgcr.h\
 ../amgs/mli_method_amgrs.h\
 ../amgs/mli_method_amgsa.h\
 ../base/mli.h\
 ../base/mli_defs.h\
 ../base/mli_oneLevel.h\
 ../cintface/cmli.h\
 ../fedata/mli_febase.h\
 ../fedata/mli_fedata.h\
 ../fedata/mli_sfei.h\
 ../fedata/mli_fedata_utils.h\
 ../mapper/mli_mapper.h\
 ../matrix/mli_matrix.h\
 ../matrix/mli_matrix_misc.h\
 ../solver/mli_solver.h\
 ../solver/mli_solver_jacobi.h\
 ../solver/mli_solver_bjacobi.h\
 ../solver/mli_solver_gs.h\
 ../solver/mli_solver_sgs.h\
 ../solver/mli_solver_bsgs.h\
 ../solver/mli_solver_hsgs.h\
 ../solver/mli_solver_hschwarz.h\
 ../solver/mli_solver_parasails.h\
 ../solver/mli_solver_mls.h\
 ../solver/mli_solver_mli.h\
 ../solver/mli_solver_amg.h\
 ../solver/mli_solver_chebyshev.h\
 ../solver/mli_solver_cg.h\
 ../solver/mli_solver_gmres.h\
 ../solver/mli_solver_kaczmarz.h\
 ../solver/mli_solver_superlu.h\
 ../solver/mli_solver_seqsuperlu.h\
 ../solver/mli_solver_arpacksuperlu.h\
 ../util/mli_utils.h\
 ../vector/mli_vector.h

FILES =\
 mli_utils.c\
 mli_amgsa_calib.cxx\
 mli_amgsa_coarsen1.cxx\
 mli_amgsa_dd_fedata.cxx\
 mli_amgsa_dd_sfei.cxx\
 mli_method.cxx\
 mli_method_amgcr.cxx\
 mli_method_amgrs.cxx\
 mli_method_amgsa.cxx\
 mli.cxx\
 mli_oneLevel.cxx\
 cmli.cxx\
 mli_febase.cxx\
 mli_fedata.cxx\
 mli_fedata_utils.cxx\
 mli_sfei.cxx\
 mli_mapper.cxx\
 mli_matrix.cxx\
 mli_matrix_mult.cxx\
 mli_matrix_utils.cxx\
 mli_solver.cxx\
 mli_solver_jacobi.cxx\
 mli_solver_bjacobi.cxx\
 mli_solver_gs.cxx\
 mli_solver_sgs.cxx\
 mli_solver_bsgs.cxx\
 mli_solver_hsgs.cxx\
 mli_solver_hschwarz.cxx\
 mli_solver_parasails.cxx\
 mli_solver_mls.cxx\
 mli_solver_mli.cxx\
 mli_solver_amg.cxx\
 mli_solver_chebyshev.cxx\
 mli_solver_cg.cxx\
 mli_solver_gmres.cxx\
 mli_solver_kaczmarz.cxx\
 mli_solver_superlu.cxx\
 mli_solver_seqsuperlu.cxx\
 mli_solver_arpacksuperlu.cxx\
 mli_vector.cxx

OBJSC = ${FILES:.c=.o}
OBJSCXX = ${OBJSC:.cxx=.o}
OBJS = ${OBJSCXX:.f=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_mli${HYPRE_LIB_SUFFIX}
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_mli${HYPRE_LIB_SUFFIX}
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)
	@echo  " "

clean:
	rm -f *.o *.h *.c *.f *.cxx libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_mli${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
