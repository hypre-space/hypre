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
# $Revision: 1.24 $
#EHEADER**********************************************************************




include ../config/Makefile.config

 
SUPERLUFILES = ${HYPRE_FEI_SUPERLU_FILES}
FEIHYPREFILES = ${HYPRE_FEI_HYPRE_FILES}
FEMLIFILES = ${HYPRE_FEI_FEMLI_FILES}
IJMVFILES = ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.o
EUCLIDFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/Euclid/*.o
PARASAILSFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/ParaSails/*.o
PILUTFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/pilut/*.o
DISTMATRIXFILES = ${HYPRE_SRC_TOP_DIR}/distributed_matrix/*.o
KRYLOVFILES = ${HYPRE_SRC_TOP_DIR}/krylov/*.o
MATMATFILES = ${HYPRE_SRC_TOP_DIR}/matrix_matrix/*.o
MULTIVECFILES = ${HYPRE_SRC_TOP_DIR}/multivector/*.o
PARCSRLSFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o
PARCSRMVFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o
PARCSRBLOCKMVFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*.o
SEQMVFILES = ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o
SSTRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o
SSTRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o
STRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o
STRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o
UTILITIESFILES = ${HYPRE_SRC_TOP_DIR}/utilities/*.o

UTILFILES = \
 ${HYPRE_SRC_TOP_DIR}/utilities/amg_linklist.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/exchange_data.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/fortran_matrix.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_memory.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_qsort.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/memory_dmalloc.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/mpistubs.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/qsplit.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/random.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/thread_mpistubs.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/threading.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/timer.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/timing.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/umalloc_local.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_mpi_comm_f2c.*o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_error.*o

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

all: libHYPRE${HYPRE_LIB_SUFFIX} libHYPRE_LSI${HYPRE_LIB_SUFFIX}
	cp -fp libHYPRE* ${HYPRE_BUILD_DIR}/lib

install: all
	cp -fp libHYPRE* ${HYPRE_LIB_INSTALL}

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE.a: ${FILES1} ${FILES2}
	@echo  "Building libHYPRE ... "
	cp -fp ${FILES1} .
	cp -fp ${FILES2} .
	${AR} $@ *.o
	${RANLIB} $@

libHYPRE.so: ${FILES1} ${FILES2}
	@echo  "Building libHYPRE ... "
	cp -fp ${FILES1} .
	cp -fp ${FILES2} .
	${BUILD_CC_SHARED} -o $@ *.o

libHYPRE_LSI.a: ${FILES_LSI}
	@echo  "Building libHYPRE_LSI ... "
	${AR} $@ ${FILES_LSI}
	${RANLIB} $@

libHYPRE_LSI.so: ${FILES_LSI}
	@echo  "Building libHYPRE_LSI ... "
	${BUILD_CC_SHARED} -o $@ ${FILES_LSI}
