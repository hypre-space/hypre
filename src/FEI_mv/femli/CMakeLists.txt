# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


set(HYPRE_mli_SRCS 
  mli_utils.c
  mli_amgsa_calib.cxx
  mli_amgsa_coarsen1.cxx
  mli_amgsa_dd_fedata.cxx
  mli_amgsa_dd_sfei.cxx
  mli_method.cxx
  mli_method_amgcr.cxx
  mli_method_amgrs.cxx
  mli_method_amgsa.cxx
  mli.cxx
  mli_oneLevel.cxx
  cmli.cxx
  mli_febase.cxx
  mli_fedata.cxx
  mli_fedata_utils.cxx
  mli_sfei.cxx
  mli_mapper.cxx
  mli_matrix.cxx
  mli_matrix_mult.cxx
  mli_matrix_utils.cxx
  mli_solver.cxx
  mli_solver_jacobi.cxx
  mli_solver_bjacobi.cxx
  mli_solver_gs.cxx
  mli_solver_sgs.cxx
  mli_solver_bsgs.cxx
  mli_solver_hsgs.cxx
  mli_solver_hschwarz.cxx
  mli_solver_parasails.cxx
  mli_solver_mls.cxx
  mli_solver_mli.cxx
  mli_solver_amg.cxx
  mli_solver_chebyshev.cxx
  mli_solver_cg.cxx
  mli_solver_gmres.cxx
  mli_solver_kaczmarz.cxx
  mli_solver_superlu.cxx
  mli_solver_seqsuperlu.cxx
  mli_solver_arpacksuperlu.cxx
  mli_vector.cxx
)

add_library(HYPRE_mli OBJECT ${HYPRE_mli_SRCS})
