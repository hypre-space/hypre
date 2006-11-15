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
# $Revision$
#EHEADER**********************************************************************




include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../IJ_mv\
 ${CINCLUDES}

HEADERS =\
 HYPRE_sstruct_mv.h\
 _hypre_sstruct_mv.h\
 box_map.h\
 headers.h\
 sstruct_graph.h\
 sstruct_grid.h\
 sstruct_matrix.h\
 sstruct_stencil.h\
 sstruct_vector.h

FILES =\
 box_map.c\
 F90_HYPRE_sstruct_graph.c\
 F90_HYPRE_sstruct_grid.c\
 F90_HYPRE_sstruct_matrix.c\
 F90_HYPRE_sstruct_stencil.c\
 F90_HYPRE_sstruct_vector.c\
 HYPRE_sstruct_graph.c\
 HYPRE_sstruct_grid.c\
 HYPRE_sstruct_matrix.c\
 HYPRE_sstruct_stencil.c\
 HYPRE_sstruct_vector.c\
 sstruct_axpy.c\
 sstruct_copy.c\
 sstruct_graph.c\
 sstruct_grid.c\
 sstruct_innerprod.c\
 sstruct_matrix.c\
 sstruct_matvec.c\
 sstruct_overlap_innerprod.c\
 sstruct_scale.c\
 sstruct_stencil.c\
 sstruct_vector.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_sstruct_mv${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/_hypre_sstruct_mv.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_sstruct_mv${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/_hypre_sstruct_mv.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_sstruct_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_sstruct_mv.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o $@ ${OBJS}

${OBJS}: ${HEADERS}
