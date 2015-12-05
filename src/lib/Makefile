#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.32 $
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

FILES_HYPRE = \
$(SUPERLUFILES)\
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
$(UTILITIESFILES)

SONAME = libHYPRE-${HYPRE_RELEASE_VERSION}.so
SOLIBS = ${MPILIBDIRS} ${MPILIBS} ${LAPACKLIBDIRS} ${LAPACKLIBS}\
 ${BLASLIBDIRS} ${BLASLIBS} ${LIBS} ${FLIBS}


##################################################################
# Targets
##################################################################

all: libHYPRE${HYPRE_LIB_SUFFIX}
	cp -fpPR libHYPRE* ${HYPRE_BUILD_DIR}/lib

install: all
	cp -fpPR libHYPRE* ${HYPRE_LIB_INSTALL}

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE.a: ${FILES_HYPRE}
	@echo  "Building libHYPRE ... "
	${AR} $@ $(SUPERLUFILES) $(FEIHYPREFILES) $(FEMLIFILES) $(IJMVFILES)
	${AR} $@ $(EUCLIDFILES)
	${AR} $@ $(PARASAILSFILES)
	${AR} $@ $(PILUTFILES)
	${AR} $@ $(DISTMATRIXFILES)
	${AR} $@ $(KRYLOVFILES)
	${AR} $@ $(MATMATFILES)
	${AR} $@ $(MULTIVECFILES)
	${AR} $@ $(PARCSRLSFILES)
	${AR} $@ $(PARCSRMVFILES)
	${AR} $@ $(PARCSRBLOCKMVFILES)
	${AR} $@ $(SEQMVFILES)
	${AR} $@ $(SSTRUCTLSFILES)
	${AR} $@ $(SSTRUCTMVFILES)
	${AR} $@ $(STRUCTLSFILES)
	${AR} $@ $(STRUCTMVFILES)
	${AR} $@ $(UTILITIESFILES)
	${RANLIB} $@

libHYPRE.so: ${FILES_HYPRE}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${FILES_HYPRE} ${SOLIBS} ${SHARED_SET_SONAME}${SONAME} ${SHARED_OPTIONS}
	ln -s ${SONAME} $@
