# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

MLI_INCLUDES = \
 -I.\
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../../blas\
 -I$(srcdir)/../../lapack\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../IJ_mv\
 -I$(srcdir)/../../krylov\
 -I$(srcdir)/../../multivector\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../parcsr_ls\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../seq_block_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../distributed_ls\
 -I$(srcdir)/../../FEI_mv/fei-hypre\
 -I$(srcdir)/../../FEI_mv/femli\
 ${SUPERLU_INCLUDE}
# ${DSUPERLU_INCLUDE}

C_COMPILE_FLAGS =\
 -DMLI_SUPERLU\
 ${MLI_INCLUDES}\
 ${CINCLUDES}
 
CXX_COMPILE_FLAGS =\
 -DMLI_SUPERLU -DMPICH_SKIP_MPICXX\
 ${MLI_INCLUDES}\
 ${CINCLUDES}

HEADERS =\
 mli_method.h\
 mli_method_amgcr.h\
 mli_method_amgrs.h\
 mli_method_amgsa.h\
 mli.h\
 mli_defs.h\
 mli_oneLevel.h\
 cmli.h\
 mli_febase.h\
 mli_fedata.h\
 mli_sfei.h\
 mli_fedata_utils.h\
 mli_mapper.h\
 mli_matrix.h\
 mli_matrix_misc.h\
 mli_solver.h\
 mli_solver_jacobi.h\
 mli_solver_bjacobi.h\
 mli_solver_gs.h\
 mli_solver_sgs.h\
 mli_solver_bsgs.h\
 mli_solver_hsgs.h\
 mli_solver_hschwarz.h\
 mli_solver_parasails.h\
 mli_solver_mls.h\
 mli_solver_mli.h\
 mli_solver_amg.h\
 mli_solver_chebyshev.h\
 mli_solver_cg.h\
 mli_solver_gmres.h\
 mli_solver_kaczmarz.h\
 mli_solver_superlu.h\
 mli_solver_seqsuperlu.h\
 mli_solver_arpacksuperlu.h\
 mli_utils.h\
 mli_vector.h

FILES =\
 mli_utils.c\
 mli_amgsa_calib.cxx\
 mli_amgsa_coarsen1.cxx\
 mli_amgsa_dd_fedata.cxx\
 mli_amgsa_dd_sfei.cxx\
 mli_method.cxx\
 mli_method_amgcr.cxx\
 mli_method_amgrs.cxx\
 mli_method_amgsa.cxx\
 mli.cxx\
 mli_oneLevel.cxx\
 cmli.cxx\
 mli_febase.cxx\
 mli_fedata.cxx\
 mli_fedata_utils.cxx\
 mli_sfei.cxx\
 mli_mapper.cxx\
 mli_matrix.cxx\
 mli_matrix_mult.cxx\
 mli_matrix_utils.cxx\
 mli_solver.cxx\
 mli_solver_jacobi.cxx\
 mli_solver_bjacobi.cxx\
 mli_solver_gs.cxx\
 mli_solver_sgs.cxx\
 mli_solver_bsgs.cxx\
 mli_solver_hsgs.cxx\
 mli_solver_hschwarz.cxx\
 mli_solver_parasails.cxx\
 mli_solver_mls.cxx\
 mli_solver_mli.cxx\
 mli_solver_amg.cxx\
 mli_solver_chebyshev.cxx\
 mli_solver_cg.cxx\
 mli_solver_gmres.cxx\
 mli_solver_kaczmarz.cxx\
 mli_solver_superlu.cxx\
 mli_solver_seqsuperlu.cxx\
 mli_solver_arpacksuperlu.cxx\
 mli_vector.cxx

OBJSC = ${FILES:.c=.o}
OBJSCXX = ${OBJSC:.cxx=.o}
OBJS = ${OBJSCXX:.f=.o}

SONAME = libHYPRE_mli-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_mli${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_mli${HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)
	@echo  " "

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_mli.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_mli.so libHYPRE_mli.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
