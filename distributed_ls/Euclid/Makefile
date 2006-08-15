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

########Euclid specific macros:
EUCLID_CFLAGS =  -DMPI_MODE -DMPI_TIMING -DUSING_MPI -DHYPRE_GET_ROW \
                 -DHYPRE_MODE -DOPTIMIZED_DH

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 ${EUCLID_CFLAGS} \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../hypre/include\
 ${CINCLUDES}

HEADERS =\
 blas_dh.h\
 euclid_common.h\
 euclid_config.h\
 Euclid_dh.h\
 euclid.h\
 ExternalRows_dh.h\
 Factor_dh.h\
 getRow_dh.h\
 guards_dh.h\
 Hash_dh.h\
 Hash_i_dh.h\
 ilu_dh.h\
 io_dh.h\
 krylov_dh.h\
 macros_dh.h\
 Mat_dh.h\
 mat_dh_private.h\
 MatGenFD.h\
 Mem_dh.h\
 Numbering_dh.h\
 Parser_dh.h\
 shellSort_dh.h\
 sig_dh.h\
 SortedList_dh.h\
 SortedSet_dh.h\
 SubdomainGraph_dh.h\
 TimeLog_dh.h\
 Timer_dh.h\
 Vec_dh.h

FILES =\
 blas_dh.c\
 Euclid_apply.c\
 Euclid_dh.c\
 ExternalRows_dh.c\
 Factor_dh.c\
 getRow_dh.c\
 globalObjects.c\
 Hash_dh.c\
 Hash_i_dh.c\
 ilu_mpi_bj.c\
 ilu_mpi_pilu.c\
 ilu_seq.c\
 io_dh.c\
 krylov_dh.c\
 Mat_dh.c\
 mat_dh_private.c\
 MatGenFD.c\
 Mem_dh.c\
 Numbering_dh.c\
 Parser_dh.c\
 shellSort_dh.c\
 sig_dh.c\
 SortedList_dh.c\
 SortedSet_dh.c\
 SubdomainGraph_dh.c\
 TimeLog_dh.c\
 Timer_dh.c\
 Vec_dh.c

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libHYPRE_Euclid${HYPRE_LIB_SUFFIX}
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_Euclid${HYPRE_LIB_SUFFIX}
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.lo libHYPRE*
	rm -rf pchdir tca.map *inslog* .libs

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_Euclid${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
