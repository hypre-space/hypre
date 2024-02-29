# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

FEIHYPREFILES = ${HYPRE_FEI_HYPRE_FILES}
FEMLIFILES = ${HYPRE_FEI_FEMLI_FILES}

ifeq (${MP_BUILD}, 1) 
PILUTFILES =  ${HYPRE_SRC_TOP_DIR}/distributed_ls/pilut/MuP_obj.dir/pilut_*/*.o
EUCLIDFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/Euclid/MuP_obj.dir/Euclid_*/*.o
PARASAILSFILES = ${HYPRE_SRC_TOP_DIR}/distributed_ls/ParaSails/MuP_obj.dir/ParaSails_*/*.o
IJMVFILES = ${HYPRE_SRC_TOP_DIR}/IJ_mv/MuP_obj.dir/IJ_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/IJ_mv/MuP_obj.dir/IJ_mv_*/*.obj
DISTMATRIXFILES = ${HYPRE_SRC_TOP_DIR}/distributed_matrix/MuP_obj.dir/distributed_matrix_*/*.o
MATMATFILES = ${HYPRE_SRC_TOP_DIR}/matrix_matrix/MuP_obj.dir/matrix_matrix_*/*.o
KRYLOVFILES = ${HYPRE_SRC_TOP_DIR}/krylov/MuP_obj.dir/krylov_*/*.o
SEQMVFILES = ${HYPRE_SRC_TOP_DIR}/seq_mv/MuP_obj.dir/seq_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o ${HYPRE_SRC_TOP_DIR}/seq_mv/MuP_obj.dir/seq_mv_*/*.obj
PARCSRMVFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_mv/MuP_obj.dir/parcsr_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_mv/MuP_obj.dir/parcsr_mv_*/*.obj
PARCSRBLOCKMVFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/MuP_obj.dir/parcsr_block_mv_*/*.o
PARCSRLSFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_ls/MuP_obj.dir/parcsr_ls_*/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_ls/MuP_obj.dir/parcsr_ls_*/*.obj
MULTIVECFILES = ${HYPRE_SRC_TOP_DIR}/multivector/MuP_obj.dir/multivector_*/*.o
SSTRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_ls/MuP_obj.dir/sstruct_ls_*/*.o ${HYPRE_SRC_TOP_DIR}/sstruct_ls/MuP_obj.dir/sstruct_ls_*/*.obj
STRUCTLSFILES = ${HYPRE_SRC_TOP_DIR}/struct_ls/MuP_obj.dir/struct_ls_*/*.o ${HYPRE_SRC_TOP_DIR}/struct_ls/MuP_obj.dir/struct_ls_*/*.obj
SSTRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/sstruct_mv/MuP_obj.dir/sstruct_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/sstruct_mv/MuP_obj.dir/sstruct_mv_*/*.obj
STRUCTMVFILES = ${HYPRE_SRC_TOP_DIR}/struct_mv/MuP_obj.dir/struct_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/struct_mv/MuP_obj.dir/struct_mv_*/*.obj
UTILITIESFILES = ${HYPRE_SRC_TOP_DIR}/utilities/MuP_obj.dir/utilities_*/*.o ${HYPRE_SRC_TOP_DIR}/utilities/MuP_obj.dir/utilities_*/*.obj
BLASFILES = ${HYPRE_SRC_TOP_DIR}/blas/MuP_obj.dir/blas_*/*.o
LAPACKFILES = ${HYPRE_SRC_TOP_DIR}/lapack/MuP_obj.dir/lapack_*/*.o

else
IJMVFILES          = ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.o ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.obj
EUCLIDFILES        = ${HYPRE_EUCLID_FILES}
PARASAILSFILES     = ${HYPRE_PARASAILS_FILES}
PILUTFILES         = ${HYPRE_PILUT_FILES}
DISTMATRIXFILES    = ${HYPRE_SRC_TOP_DIR}/distributed_matrix/*.o
KRYLOVFILES        = ${HYPRE_SRC_TOP_DIR}/krylov/*.o
MATMATFILES        = ${HYPRE_SRC_TOP_DIR}/matrix_matrix/*.o
MULTIVECFILES      = ${HYPRE_SRC_TOP_DIR}/multivector/*.o
PARCSRLSFILES      = ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.obj
PARCSRMVFILES      = ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.obj
PARCSRBLOCKMVFILES = ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*.o
SEQMVFILES         = ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o ${HYPRE_SRC_TOP_DIR}/seq_mv/*.obj
SEQBLOCKMVFILES    = ${HYPRE_SRC_TOP_DIR}/seq_block_mv/*.o
SSTRUCTLSFILES     = ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj
SSTRUCTMVFILES     = ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj
STRUCTLSFILES      = ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o ${HYPRE_SRC_TOP_DIR}/struct_ls/*.obj
STRUCTMVFILES      = ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o ${HYPRE_SRC_TOP_DIR}/struct_mv/*.obj
UTILITIESFILES     = ${HYPRE_SRC_TOP_DIR}/utilities/*.o ${HYPRE_SRC_TOP_DIR}/utilities/*.obj
BLASFILES          = ${HYPRE_SRC_TOP_DIR}/blas/*.o
LAPACKFILES        = ${HYPRE_SRC_TOP_DIR}/lapack/*.o
endif

FILES_HYPRE = \
$(FEIHYPREFILES)\
$(FEMLIFILES)\
$(IJMVFILES)\
$(EUCLIDFILES)\
$(PARASAILSFILES)\
$(PILUTFILES)\
$(DISTMATRIXFILES)\
$(MATMATFILES)\
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
$(UTILITIESFILES)\
$(BLASFILES)\
$(LAPACKFILES)

SONAME = libHYPRE-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}
SOLIBS = ${DSUPERLU_LIBS} ${HYPRE_MAGMA_LIB_DIR} ${HYPRE_MAGMA_LIB} ${MPILIBDIRS} ${MPILIBS} ${LAPACKLIBDIRS} ${LAPACKLIBS}\
 ${BLASLIBDIRS} ${BLASLIBS} ${LIBS} ${FLIBS}


##################################################################
# Targets
##################################################################

all: libHYPRE${HYPRE_LIB_SUFFIX}
	cp -fR libHYPRE* ${HYPRE_BUILD_DIR}/lib

install: all
	cp -fR libHYPRE* ${HYPRE_LIB_INSTALL}

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

# NOTE: Some of the FILES symbols below can be empty, so they are listed on an
# AR line with at least one nonempty symbol
libHYPRE.a: ${FILES_HYPRE}
	@echo  "Building libHYPRE ... "
	rm -f $@
	${AR} $@ $(EUCLIDFILES) $(PARASAILSFILES) $(PILUTFILES) $(DISTMATRIXFILES)
	${AR} $@ $(IJMVFILES)
	${AR} $@ $(KRYLOVFILES)
	${AR} $@ $(MATMATFILES)
	${AR} $@ $(MULTIVECFILES)
	${AR} $@ $(PARCSRLSFILES)
	${AR} $@ $(PARCSRMVFILES)
	${AR} $@ $(PARCSRBLOCKMVFILES)
	${AR} $@ $(SEQMVFILES)
	${AR} $@ $(SEQBLOCKMVFILES)
	${AR} $@ $(SSTRUCTLSFILES)
	${AR} $@ $(SSTRUCTMVFILES)
	${AR} $@ $(STRUCTLSFILES)
	${AR} $@ $(STRUCTMVFILES)
	${AR} $@ $(UTILITIESFILES)
	${AR} $@ $(BLASFILES) $(LAPACKFILES)
	${RANLIB} $@

libHYPRE.so libHYPRE.dylib: ${FILES_HYPRE}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${FILES_HYPRE} ${SOLIBS} ${SHARED_SET_SONAME}${SONAME} ${SHARED_OPTIONS} ${LDFLAGS}
	ln -s -f ${SONAME} $@
