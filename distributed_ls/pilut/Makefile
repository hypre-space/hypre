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
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
${CINCLUDES}

HEADERS = \
 HYPRE_DistributedMatrixPilutSolver_protos.h\
 HYPRE_DistributedMatrixPilutSolver_types.h\
 DistributedMatrixPilutSolver.h\
 const.h\
 ilu.h\
 internal_protos.h\
 macros.h\
 struct.h

FILES =\
 comm.c\
 debug.c\
 distributed_qsort.c\
 distributed_qsort_si.c\
 HYPRE_DistributedMatrixPilutSolver.c\
 ilut.c\
 parilut.c\
 parutil.c\
 pblas1.c\
 serilut.c\
 trifactor.c\
 util.c

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}
	cp -f $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -f libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.lo libHYPRE*
	rm -rf pchdir tca.map *inslog* .libs

distclean: clean

##################################################################
# Main rules
##################################################################

libHYPRE_DistributedMatrixPilutSolver${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
