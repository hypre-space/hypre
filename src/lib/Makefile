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

IJMVFILES           = ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.o
IJMVFILES          += ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.o_flt
IJMVFILES          += ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.o_dbl
IJMVFILES          += ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.o_ldbl
IJMVFILES          += ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.obj_flt
IJMVFILES          += ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.obj_dbl
IJMVFILES          += ${HYPRE_SRC_TOP_DIR}/IJ_mv/*.obj_ldbl

EUCLIDFILES         = ${HYPRE_SRC_TOP_DIR}/distributed_ls/Euclid/*.o_flt
EUCLIDFILES        += ${HYPRE_SRC_TOP_DIR}/distributed_ls/Euclid/*.o_dbl
EUCLIDFILES        += ${HYPRE_SRC_TOP_DIR}/distributed_ls/Euclid/*.o_ldbl

PARASAILSFILES      = ${HYPRE_SRC_TOP_DIR}/distributed_ls/ParaSails/*.o_flt
PARASAILSFILES     += ${HYPRE_SRC_TOP_DIR}/distributed_ls/ParaSails/*.o_dbl
PARASAILSFILES     += ${HYPRE_SRC_TOP_DIR}/distributed_ls/ParaSails/*.o_ldbl

PILUTFILES          = ${HYPRE_SRC_TOP_DIR}/distributed_ls/pilut/*.o_flt
PILUTFILES         += ${HYPRE_SRC_TOP_DIR}/distributed_ls/pilut/*.o_dbl
PILUTFILES         += ${HYPRE_SRC_TOP_DIR}/distributed_ls/pilut/*.o_ldbl

DISTMATRIXFILES     = ${HYPRE_SRC_TOP_DIR}/distributed_matrix/*.o_flt
DISTMATRIXFILES    += ${HYPRE_SRC_TOP_DIR}/distributed_matrix/*.o_dbl
DISTMATRIXFILES    += ${HYPRE_SRC_TOP_DIR}/distributed_matrix/*.o_ldbl

KRYLOVFILES         = ${HYPRE_SRC_TOP_DIR}/krylov/*.o
KRYLOVFILES        += ${HYPRE_SRC_TOP_DIR}/krylov/*.o_flt
KRYLOVFILES        += ${HYPRE_SRC_TOP_DIR}/krylov/*.o_dbl
KRYLOVFILES        += ${HYPRE_SRC_TOP_DIR}/krylov/*.o_ldbl

MATMATFILES         = ${HYPRE_SRC_TOP_DIR}/matrix_matrix/*.o
MATMATFILES        += ${HYPRE_SRC_TOP_DIR}/matrix_matrix/MuP_obj.dir/matrix_matrix_*/*.o

MULTIVECFILES       = ${HYPRE_SRC_TOP_DIR}/multivector/*.o
MULTIVECFILES      += ${HYPRE_SRC_TOP_DIR}/multivector/MuP_obj.dir/multivector_*/*.o

PARCSRLSFILES       = ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o
PARCSRLSFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o_flt
PARCSRLSFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o_dbl
PARCSRLSFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.o_ldbl
PARCSRLSFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.obj_flt
PARCSRLSFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.obj_dbl
PARCSRLSFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_ls/*.obj_ldbl

PARCSRMVFILES       = ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o
PARCSRMVFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o_flt
PARCSRMVFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o_dbl
PARCSRMVFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.o_ldbl
PARCSRMVFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.obj_flt
PARCSRMVFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.obj_dbl
PARCSRMVFILES      += ${HYPRE_SRC_TOP_DIR}/parcsr_mv/*.obj_ldbl

PARCSRBLOCKMVFILES  = ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*.o_flt
PARCSRBLOCKMVFILES += ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*.o_dbl
PARCSRBLOCKMVFILES += ${HYPRE_SRC_TOP_DIR}/parcsr_block_mv/*.o_ldbl

SEQMVFILES          = ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o
SEQMVFILES         += ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o_flt
SEQMVFILES         += ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o_dbl
SEQMVFILES         += ${HYPRE_SRC_TOP_DIR}/seq_mv/*.o_ldbl
SEQMVFILES         += ${HYPRE_SRC_TOP_DIR}/seq_mv/*.obj_flt
SEQMVFILES         += ${HYPRE_SRC_TOP_DIR}/seq_mv/*.obj_dbl
SEQMVFILES         += ${HYPRE_SRC_TOP_DIR}/seq_mv/*.obj_ldbl

SEQBLOCKMVFILES     = ${HYPRE_SRC_TOP_DIR}/seq_block_mv/*.o
SEQBLOCKMVFILES    += ${HYPRE_SRC_TOP_DIR}/seq_block_mv/*.o_flt
SEQBLOCKMVFILES    += ${HYPRE_SRC_TOP_DIR}/seq_block_mv/*.o_dbl
SEQBLOCKMVFILES    += ${HYPRE_SRC_TOP_DIR}/seq_block_mv/*.o_ldbl

SSTRUCTLSFILES      = ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o_flt
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o_dbl
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.o_ldbl
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj_flt
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj_dbl
SSTRUCTLSFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_ls/*.obj_ldbl

SSTRUCTMVFILES      = ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o_flt
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o_dbl
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.o_ldbl
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj_flt
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj_dbl
SSTRUCTMVFILES     += ${HYPRE_SRC_TOP_DIR}/sstruct_mv/*.obj_ldbl

STRUCTLSFILES       = ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o
STRUCTLSFILES      += ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o_flt
STRUCTLSFILES      += ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o_dbl
STRUCTLSFILES      += ${HYPRE_SRC_TOP_DIR}/struct_ls/*.o_ldbl
STRUCTLSFILES      += ${HYPRE_SRC_TOP_DIR}/struct_ls/*.obj_flt
STRUCTLSFILES      += ${HYPRE_SRC_TOP_DIR}/struct_ls/*.obj_dbl
STRUCTLSFILES      += ${HYPRE_SRC_TOP_DIR}/struct_ls/*.obj_ldbl

STRUCTMVFILES       = ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o
STRUCTMVFILES      += ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o_flt
STRUCTMVFILES      += ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o_dbl
STRUCTMVFILES      += ${HYPRE_SRC_TOP_DIR}/struct_mv/*.o_ldbl
STRUCTMVFILES      += ${HYPRE_SRC_TOP_DIR}/struct_mv/*.obj_flt
STRUCTMVFILES      += ${HYPRE_SRC_TOP_DIR}/struct_mv/*.obj_dbl
STRUCTMVFILES      += ${HYPRE_SRC_TOP_DIR}/struct_mv/*.obj_ldbl

UTILITIESFILES      = ${HYPRE_SRC_TOP_DIR}/utilities/*.o
UTILITIESFILES     += ${HYPRE_SRC_TOP_DIR}/utilities/*.o_flt
UTILITIESFILES     += ${HYPRE_SRC_TOP_DIR}/utilities/*.o_dbl
UTILITIESFILES     += ${HYPRE_SRC_TOP_DIR}/utilities/*.o_ldbl
UTILITIESFILES     += ${HYPRE_SRC_TOP_DIR}/utilities/*.obj_flt
UTILITIESFILES     += ${HYPRE_SRC_TOP_DIR}/utilities/*.obj_dbl
UTILITIESFILES     += ${HYPRE_SRC_TOP_DIR}/utilities/*.obj_ldbl

BLASFILES           = ${HYPRE_SRC_TOP_DIR}/blas/*.o
BLASFILES          += ${HYPRE_SRC_TOP_DIR}/blas/*.o_flt
BLASFILES          += ${HYPRE_SRC_TOP_DIR}/blas/*.o_dbl
BLASFILES          += ${HYPRE_SRC_TOP_DIR}/blas/*.o_ldbl

LAPACKFILES         = ${HYPRE_SRC_TOP_DIR}/lapack/*.o
LAPACKFILES        += ${HYPRE_SRC_TOP_DIR}/lapack/*.o_flt
LAPACKFILES        += ${HYPRE_SRC_TOP_DIR}/lapack/*.o_dbl
LAPACKFILES        += ${HYPRE_SRC_TOP_DIR}/lapack/*.o_ldbl

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
