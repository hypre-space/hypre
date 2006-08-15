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

C_COMPILE_FLAGS =\
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../parcsr_ls\
 ${CINCLUDES}

HEADERS =\
 aux_parcsr_matrix.h\
 aux_par_vector.h\
 headers.h\
 HYPRE_IJ_mv.h\
 IJ_matrix.h\
 IJ_mv.h\
 IJ_vector.h

EXTRA_FILES =\
 IJMatrix_isis.c\
 IJMatrix_petsc.c

FILES =\
 aux_parcsr_matrix.c\
 aux_par_vector.c\
 F90_HYPRE_IJMatrix.c\
 F90_HYPRE_IJVector.c\
 F90_IJMatrix.c\
 HYPRE_IJMatrix.c\
 HYPRE_IJVector.c\
 IJMatrix.c\
 IJMatrix_parcsr.c\
 IJVector.c\
 IJVector_parcsr.c

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libHYPRE_IJ_mv${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/IJ_mv.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_IJ_mv${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/IJ_mv.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.lo libHYPRE*
	rm -rf pchdir tca.map *inslog* .libs

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_IJ_mv${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
