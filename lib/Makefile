#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

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
PARCSRBLOCKMVFILES =  ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*.o
SEQMVFILES = ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o
SSTRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o
SSTRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o
STRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o
STRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o
UTILITIESFILES = ${HYPRE_SRC_TOP_DIR}/utilities/*.o

UTILFILES = \
 ${HYPRE_SRC_TOP_DIR}/utilities/amg_linklist.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/binsearch.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/exchange_data.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/fortran_matrix.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_memory.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_qsort.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/memory_dmalloc.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/mpistubs.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/qsplit.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/random.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/thread_mpistubs.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/threading.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/timer.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/timing.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/umalloc_local.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_mpi_comm_f2c.o\
 ${HYPRE_SRC_TOP_DIR}/utilities/hypre_error.o

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
	@echo  "Building libHYPRE.a ... "
	cp -fp ${FILES1} .
	cp -fp ${FILES2} .
	${AR} libHYPRE.a *.o
	${RANLIB} libHYPRE.a
	@echo  "Building libHYPRE_LSI.a ... "
	${AR} libHYPRE_LSI.a ${FILES_LSI}
	${RANLIB} libHYPRE_LSI.a
	cp -fp libHYPRE* ${HYPRE_BUILD_DIR}/lib

install: all
	cp -fp libHYPRE* ${HYPRE_LIB_INSTALL}

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean
