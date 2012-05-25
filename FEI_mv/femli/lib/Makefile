#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************


include ../../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

MLI_INCLUDES = \
 -I.\
 -I../../..\
 -I$(srcdir)/..\
 -I$(srcdir)/../../../utilities\
 -I$(srcdir)/../../../IJ_mv\
 -I$(srcdir)/../../../krylov\
 -I$(srcdir)/../../../multivector\
 -I$(srcdir)/../../../parcsr_mv\
 -I$(srcdir)/../../../parcsr_ls\
 -I$(srcdir)/../../../seq_mv\
 -I$(srcdir)/../../../distributed_matrix\
 -I$(srcdir)/../../../distributed_ls\
 -I$(srcdir)/../../../FEI_mv/fei-hypre\
 -I$(srcdir)/../../../FEI_mv/femli\
 -I$(srcdir)/../../../FEI_mv/SuperLU

C_COMPILE_FLAGS =\
 -DMLI_SUPERLU\
 ${MLI_INCLUDES}\
 ${CINCLUDES}
 
CXX_COMPILE_FLAGS =\
 -DMLI_SUPERLU -DMPICH_SKIP_MPICXX\
 ${MLI_INCLUDES}\
 ${CINCLUDES}

HEADERS =\
 ../amgs/mli_method.h\
 ../amgs/mli_method_amgcr.h\
 ../amgs/mli_method_amgrs.h\
 ../amgs/mli_method_amgsa.h\
 ../base/mli.h\
 ../base/mli_defs.h\
 ../base/mli_oneLevel.h\
 ../cintface/cmli.h\
 ../fedata/mli_febase.h\
 ../fedata/mli_fedata.h\
 ../fedata/mli_sfei.h\
 ../fedata/mli_fedata_utils.h\
 ../mapper/mli_mapper.h\
 ../matrix/mli_matrix.h\
 ../matrix/mli_matrix_misc.h\
 ../solver/mli_solver.h\
 ../solver/mli_solver_jacobi.h\
 ../solver/mli_solver_bjacobi.h\
 ../solver/mli_solver_gs.h\
 ../solver/mli_solver_sgs.h\
 ../solver/mli_solver_bsgs.h\
 ../solver/mli_solver_hsgs.h\
 ../solver/mli_solver_hschwarz.h\
 ../solver/mli_solver_parasails.h\
 ../solver/mli_solver_mls.h\
 ../solver/mli_solver_mli.h\
 ../solver/mli_solver_amg.h\
 ../solver/mli_solver_chebyshev.h\
 ../solver/mli_solver_cg.h\
 ../solver/mli_solver_gmres.h\
 ../solver/mli_solver_kaczmarz.h\
 ../solver/mli_solver_superlu.h\
 ../solver/mli_solver_seqsuperlu.h\
 ../solver/mli_solver_arpacksuperlu.h\
 ../util/mli_utils.h\
 ../vector/mli_vector.h

FILES =\
 ../util/mli_utils.c\
 ../amgs/mli_amgsa_calib.cxx\
 ../amgs/mli_amgsa_coarsen1.cxx\
 ../amgs/mli_amgsa_dd_fedata.cxx\
 ../amgs/mli_amgsa_dd_sfei.cxx\
 ../amgs/mli_method.cxx\
 ../amgs/mli_method_amgcr.cxx\
 ../amgs/mli_method_amgrs.cxx\
 ../amgs/mli_method_amgsa.cxx\
 ../base/mli.cxx\
 ../base/mli_oneLevel.cxx\
 ../cintface/cmli.cxx\
 ../fedata/mli_febase.cxx\
 ../fedata/mli_fedata.cxx\
 ../fedata/mli_fedata_utils.cxx\
 ../fedata/mli_sfei.cxx\
 ../mapper/mli_mapper.cxx\
 ../matrix/mli_matrix.cxx\
 ../matrix/mli_matrix_mult.cxx\
 ../matrix/mli_matrix_utils.cxx\
 ../solver/mli_solver.cxx\
 ../solver/mli_solver_jacobi.cxx\
 ../solver/mli_solver_bjacobi.cxx\
 ../solver/mli_solver_gs.cxx\
 ../solver/mli_solver_sgs.cxx\
 ../solver/mli_solver_bsgs.cxx\
 ../solver/mli_solver_hsgs.cxx\
 ../solver/mli_solver_hschwarz.cxx\
 ../solver/mli_solver_parasails.cxx\
 ../solver/mli_solver_mls.cxx\
 ../solver/mli_solver_mli.cxx\
 ../solver/mli_solver_amg.cxx\
 ../solver/mli_solver_chebyshev.cxx\
 ../solver/mli_solver_cg.cxx\
 ../solver/mli_solver_gmres.cxx\
 ../solver/mli_solver_kaczmarz.cxx\
 ../solver/mli_solver_superlu.cxx\
 ../solver/mli_solver_seqsuperlu.cxx\
 ../solver/mli_solver_arpacksuperlu.cxx\
 ../vector/mli_vector.cxx

OBJSC = ${FILES:.c=.o}
OBJSCXX = ${OBJSC:.cxx=.o}
OBJS = ${OBJSCXX:.f=.o}

SONAME = libHYPRE_mli-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_mli${HYPRE_LIB_SUFFIX}
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_mli${HYPRE_LIB_SUFFIX}
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)
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
	${AR} $@ *.o
	${RANLIB} $@

libHYPRE_mli.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} *.o ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
