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

 
SUPERLUFILES = ${HYPRE_FEI_SUPERLU_FILES}
FEIHYPREFILES = ${HYPRE_FEI_HYPRE_FILES}
FEMLIFILES = ${HYPRE_FEI_FEMLI_FILES}
IJMVFILES = ${HYPRE_SRC_TOP_DIR}/IJ_mv/*${HYPRE_OBJ_SUFFIX}
EUCLIDFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/Euclid/*${HYPRE_OBJ_SUFFIX}
PARASAILSFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/ParaSails/*${HYPRE_OBJ_SUFFIX}
PILUTFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/pilut/*${HYPRE_OBJ_SUFFIX}
DISTMATRIXFILES = ${HYPRE_SRC_TOP_DIR}/distributed_matrix/*${HYPRE_OBJ_SUFFIX}
KRYLOVFILES = ${HYPRE_SRC_TOP_DIR}/krylov/*${HYPRE_OBJ_SUFFIX}
MATMATFILES = ${HYPRE_SRC_TOP_DIR}/matrix_matrix/*${HYPRE_OBJ_SUFFIX}
MULTIVECFILES = ${HYPRE_SRC_TOP_DIR}/multivector/*${HYPRE_OBJ_SUFFIX}
PARCSRLSFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*${HYPRE_OBJ_SUFFIX}
PARCSRMVFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*${HYPRE_OBJ_SUFFIX}
PARCSRBLOCKMVFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*${HYPRE_OBJ_SUFFIX}
SEQMVFILES = ${HYPRE_SRC_TOP_DIR}/seq_mv/*${HYPRE_OBJ_SUFFIX}
SSTRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*${HYPRE_OBJ_SUFFIX}
SSTRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*${HYPRE_OBJ_SUFFIX}
STRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/struct_ls/*${HYPRE_OBJ_SUFFIX}
STRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/struct_mv/*${HYPRE_OBJ_SUFFIX}
UTILITIESFILES = ${HYPRE_SRC_TOP_DIR}/utilities/*${HYPRE_OBJ_SUFFIX}

UTILFILES = \
 ${HYPRE_SRC_TOP_DIR}/utilities/amg_linklist${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/binsearch${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/exchange_data${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/fortran_matrix${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_memory${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_qsort${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/memory_dmalloc${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/mpistubs${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/qsplit${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/random${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/thread_mpistubs${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/threading${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/timer${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/timing${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/umalloc_local${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_mpi_comm_f2c${HYPRE_OBJ_SUFFIX}\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_error${HYPRE_OBJ_SUFFIX}

FILES1 = \
$(SUPERLUFILES)\
$(FEIHYPREFILES)\
$(FEMLIFILES)\
$(IJMVFILES)\
$(EUCLIDFILES)\
$(PARASAILSFILES)\
$(PILUTFILES)\
$(DISTMATRIXFILES)\
$(KRYLOVFILES)\
$(MATMATFILES)

FILES2 = \
$(MULTIVECFILES)\
$(PARCSRLSFILES)\
$(PARCSRMVFILES)\
$(PARCSRBLOCKMVFILES)\
$(SEQMVFILES)\
$(SSTRUCTLSFILES)\
$(SSTRUCTMVFILES)\
$(STRUCTLSFILES)\
$(STRUCTMVFILES)\
$(UTILITIESFILES)

FILES_LSI = \
$(FEIHYPREFILES)\
$(FEMLIFILES)\
$(IJMVFILES)\
$(EUCLIDFILES)\
$(PARASAILSFILES)\
$(PILUTFILES)\
$(DISTMATRIXFILES)\
$(KRYLOVFILES)\
$(MATMATFILES)\
$(MULTIVECFILES)\
$(PARCSRLSFILES)\
$(PARCSRMVFILES)\
$(PARCSRBLOCKMVFILES)\
$(SEQMVFILES)\
$(SSTRUCTLSFILES)\
$(SSTRUCTMVFILES)\
$(STRUCTLSFILES)\
$(STRUCTMVFILES)\
$(UTILFILES)

##################################################################
# Targets
##################################################################

all:
	@echo  "Building libHYPRE ... "
	cp -fp ${FILES1} .
	cp -fp ${FILES2} .
	${AR} libHYPRE${HYPRE_LIB_SUFFIX} *${HYPRE_OBJ_SUFFIX}
	${RANLIB} libHYPRE${HYPRE_LIB_SUFFIX}
	@echo  "Building libHYPRE_LSI ... "
	${AR} libHYPRE_LSI${HYPRE_LIB_SUFFIX} ${FILES_LSI}
	${RANLIB} libHYPRE_LSI${HYPRE_LIB_SUFFIX}
	cp -fp libHYPRE* ${HYPRE_BUILD_DIR}/lib

install: all
	cp -fp libHYPRE* ${HYPRE_LIB_INSTALL}

clean:
	rm -f *.o *.lo libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean
