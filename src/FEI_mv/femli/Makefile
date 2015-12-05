#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.9 $
#EHEADER**********************************************************************


###  Copy needed header and source files to lib sub-directory to be compiled
###  and create libHYPRE_mli*

include ../../config/Makefile.config

HEADERS = \
 ./cintface/cmli.h\
 ./solver/mli_solver.h\
 ./solver/mli_solver_jacobi.h\
 ./solver/mli_solver_bjacobi.h\
 ./solver/mli_solver_gs.h\
 ./solver/mli_solver_sgs.h\
 ./solver/mli_solver_bsgs.h\
 ./solver/mli_solver_hsgs.h\
 ./solver/mli_solver_hschwarz.h\
 ./solver/mli_solver_parasails.h\
 ./solver/mli_solver_mls.h\
 ./solver/mli_solver_mli.h\
 ./solver/mli_solver_amg.h\
 ./solver/mli_solver_chebyshev.h\
 ./solver/mli_solver_cg.h\
 ./solver/mli_solver_gmres.h\
 ./solver/mli_solver_kaczmarz.h\
 ./solver/mli_solver_superlu.h\
 ./solver/mli_solver_seqsuperlu.h\
 ./solver/mli_solver_arpacksuperlu.h

FILES =\
 ./util/mli_utils.c\
 ./util/mli_utils_fortran.f\
 ./amgs/mli_amgsa_calib.cxx\
 ./amgs/mli_amgsa_coarsen1.cxx\
 ./amgs/mli_amgsa_dd_fedata.cxx\
 ./amgs/mli_amgsa_dd_sfei.cxx\
 ./amgs/mli_method.cxx\
 ./amgs/mli_method_amgcr.cxx\
 ./amgs/mli_method_amgrs.cxx\
 ./amgs/mli_method_amgsa.cxx\
 ./base/mli.cxx\
 ./base/mli_oneLevel.cxx\
 ./cintface/cmli.cxx\
 ./fedata/mli_febase.cxx\
 ./fedata/mli_fedata.cxx\
 ./fedata/mli_fedata_utils.cxx\
 ./fedata/mli_sfei.cxx\
 ./mapper/mli_mapper.cxx\
 ./matrix/mli_matrix.cxx\
 ./matrix/mli_matrix_mult.cxx\
 ./matrix/mli_matrix_utils.cxx\
 ./solver/mli_solver.cxx\
 ./solver/mli_solver_jacobi.cxx\
 ./solver/mli_solver_bjacobi.cxx\
 ./solver/mli_solver_gs.cxx\
 ./solver/mli_solver_sgs.cxx\
 ./solver/mli_solver_bsgs.cxx\
 ./solver/mli_solver_hsgs.cxx\
 ./solver/mli_solver_hschwarz.cxx\
 ./solver/mli_solver_parasails.cxx\
 ./solver/mli_solver_mls.cxx\
 ./solver/mli_solver_mli.cxx\
 ./solver/mli_solver_amg.cxx\
 ./solver/mli_solver_chebyshev.cxx\
 ./solver/mli_solver_cg.cxx\
 ./solver/mli_solver_gmres.cxx\
 ./solver/mli_solver_kaczmarz.cxx\
 ./solver/mli_solver_superlu.cxx\
 ./solver/mli_solver_seqsuperlu.cxx\
 ./solver/mli_solver_arpacksuperlu.cxx\
 ./vector/mli_vector.cxx

all: 
	@ \
	for head in ${HEADERS}; \
	do \
	    (cp -fpPR $$head ./lib); \
	done;\
	for file in ${FILES}; \
	do \
	    (cp -fpPR $$file ./lib); \
	done;\
	(cd ./lib && $(MAKE) $@)

install:
	@ \
	for head in ${HEADERS}; \
	do \
	    (cp -fpPR $$head ./lib); \
	done;\
	for file in ${FILES}; \
	do \
	    (cp -fpPR $$file ./lib); \
	done;\
	(cd ./lib && $(MAKE) $@)

clean:
	@(cd ./lib && $(MAKE) $@)

distclean: clean
