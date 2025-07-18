# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

# Redefine AR to allow for files with the same name (but in different directories)
AR = ar -cq

FEIHYPREFILES = ${HYPRE_FEI_HYPRE_FILES}
FEMLIFILES = ${HYPRE_FEI_FEMLI_FILES}

# multiprecision build
ifeq (${MP_BUILD}, 1) 

IJMVFILES           = ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.o ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.obj
IJMVFILES          += ${HYPRE_SRC_TOP_DIR}/IJ_mv/MuP_obj.dir/IJ_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/IJ_mv/MuP_obj.dir/IJ_mv_*/*.obj

EUCLIDFILES         = ${HYPRE_EUCLID_FILES}
EUCLIDFILES        += ${HYPRE_SRC_TOP_DIR}/distributed_ls/Euclid/MuP_obj.dir/Euclid_*/*.o

PARASAILSFILES      = ${HYPRE_PARASAILS_FILES}
PARASAILSFILES     += ${HYPRE_SRC_TOP_DIR}/distributed_ls/ParaSails/MuP_obj.dir/ParaSails_*/*.o

PILUTFILES          = ${HYPRE_PILUT_FILES}
PILUTFILES         += ${HYPRE_SRC_TOP_DIR}/distributed_ls/pilut/MuP_obj.dir/pilut_*/*.o

DISTMATRIXFILES     = ${HYPRE_SRC_TOP_DIR}/distributed_matrix/*.o
DISTMATRIXFILES    += ${HYPRE_SRC_TOP_DIR}/distributed_matrix/MuP_obj.dir/distributed_matrix_*/*.o

KRYLOVFILES         = ${HYPRE_SRC_TOP_DIR}/krylov/*.o
KRYLOVFILES        += ${HYPRE_SRC_TOP_DIR}/krylov/*.o_flt
KRYLOVFILES        += ${HYPRE_SRC_TOP_DIR}/krylov/*.o_dbl
KRYLOVFILES        += ${HYPRE_SRC_TOP_DIR}/krylov/*.o_ldbl
KRYLOVFILES        += ${HYPRE_SRC_TOP_DIR}/krylov/*.o_def

MATMATFILES         = ${HYPRE_SRC_TOP_DIR}/matrix_matrix/*.o
MATMATFILES        += ${HYPRE_SRC_TOP_DIR}/matrix_matrix/MuP_obj.dir/matrix_matrix_*/*.o

MULTIVECFILES       = ${HYPRE_SRC_TOP_DIR}/multivector/*.o
MULTIVECFILES      += ${HYPRE_SRC_TOP_DIR}/multivector/MuP_obj.dir/multivector_*/*.o

PARCSRLSFILES       = ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.obj
PARCSRLSFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_ls/MuP_obj.dir/parcsr_ls_*/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_ls/MuP_obj.dir/parcsr_ls_*/*.obj

PARCSRMVFILES       = ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.obj
PARCSRMVFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_mv/MuP_obj.dir/parcsr_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/parcsr_mv/MuP_obj.dir/parcsr_mv_*/*.obj

PARCSRBLOCKMVFILES  = ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*.o
PARCSRBLOCKMVFILES += ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/MuP_obj.dir/parcsr_block_mv_*/*.o

SEQMVFILES          = ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o ${HYPRE_SRC_TOP_DIR}/seq_mv/*.obj
SEQMVFILES         += ${HYPRE_SRC_TOP_DIR}/seq_mv/MuP_obj.dir/seq_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/seq_mv/MuP_obj.dir/seq_mv_*/*.obj

SEQBLOCKMVFILES     = ${HYPRE_SRC_TOP_DIR}/seq_block_mv/*.o
SEQBLOCKMVFILES    += ${HYPRE_SRC_TOP_DIR}/seq_block_mv/MuP_obj.dir/seq_block_mv_*/*.o

SSTRUCTLSFILES      = ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o_flt
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o_dbl
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o_ldbl
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o_def
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj_flt
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj_dbl
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj_ldbl
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj_def

SSTRUCTMVFILES      = ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o_flt
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o_dbl
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o_ldbl
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o_def
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj_flt
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj_dbl
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj_ldbl
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj_def

STRUCTLSFILES       = ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o ${HYPRE_SRC_TOP_DIR}/struct_ls/*.obj
STRUCTLSFILES      += ${HYPRE_SRC_TOP_DIR}/struct_ls/MuP_obj.dir/struct_ls_*/*.o ${HYPRE_SRC_TOP_DIR}/struct_ls/MuP_obj.dir/struct_ls_*/*.obj

STRUCTMVFILES       = ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o ${HYPRE_SRC_TOP_DIR}/struct_mv/*.obj
STRUCTMVFILES      += ${HYPRE_SRC_TOP_DIR}/struct_mv/MuP_obj.dir/struct_mv_*/*.o ${HYPRE_SRC_TOP_DIR}/struct_mv/MuP_obj.dir/struct_mv_*/*.obj

UTILITIESFILES      = ${HYPRE_SRC_TOP_DIR}/utilities/*.o ${HYPRE_SRC_TOP_DIR}/utilities/*.obj
UTILITIESFILES     += ${HYPRE_SRC_TOP_DIR}/utilities/MuP_obj.dir/utilities_*/*.o ${HYPRE_SRC_TOP_DIR}/utilities/MuP_obj.dir/utilities_*/*.obj

BLASFILES           = ${HYPRE_SRC_TOP_DIR}/blas/*.o
BLASFILES          += ${HYPRE_SRC_TOP_DIR}/blas/MuP_obj.dir/blas_*/*.o

LAPACKFILES         = ${HYPRE_SRC_TOP_DIR}/lapack/*.o
LAPACKFILES        += ${HYPRE_SRC_TOP_DIR}/lapack/MuP_obj.dir/lapack_*/*.o

# normal build
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
 $(KRYLOVFILES)\
 $(MATMATFILES)\
 $(MULTIVECFILES)\
 $(PARCSRLSFILES)\
 $(PARCSRMVFILES)\
 $(PARCSRBLOCKMVFILES)\
 $(SEQMVFILES)\
 $(SEQBLOCKMVFILES)\
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
	${AR} $@ $(FEIHYPREFILES) $(FEMLIFILES) $(IJMVFILES)
	${AR} $@ $(EUCLIDFILES) $(PARASAILSFILES) $(PILUTFILES) $(DISTMATRIXFILES)
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
