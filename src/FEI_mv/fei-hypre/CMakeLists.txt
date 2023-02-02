# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


set(HYPRE_fei_HEADERS
  cfei-hypre.h
  cfei_hypre.h
  hypre_cfei.h
  HYPRE_LinSysCore.h
  HYPRE_FEI_includes.h
  LLNL_FEI_Fei.h
  LLNL_FEI_Impl.h
  LLNL_FEI_LSCore.h
  LLNL_FEI_Matrix.h
  LLNL_FEI_Solver.h
)

set(HYPRE_fei_SRCS 
  HYPRE_LSI_Dsuperlu.c
  HYPRE_LSI_ddict.c
  HYPRE_LSI_ddilut.c
  HYPRE_LSI_ml.c
  HYPRE_LSI_mlmaxwell.c
  HYPRE_LSI_poly.c
  HYPRE_LSI_schwarz.c
  HYPRE_parcsr_TFQmr.c
  HYPRE_parcsr_bicgs.c
  HYPRE_parcsr_bicgstabl.c
  HYPRE_parcsr_fgmres.c
  HYPRE_parcsr_lsicg.c
  HYPRE_parcsr_symqmr.c
  HYPRE_parcsr_maxwell.c
  SymQMR.c
  TFQmr.c
  bicgs.c
  bicgstabl.c
  fgmres.c
  hypre_lsi_amge.c
  hypre_lsi_ddamg.c
  hypre_lsi_misc.c
  lsicg.c
  FEI_HYPRE_Impl.cxx
  LLNL_FEI_Impl.cxx
  LLNL_FEI_Fei.cxx
  LLNL_FEI_LSCore.cxx
  LLNL_FEI_Solver.cxx
  LLNL_FEI_Matrix.cxx
  HYPRE_LSC_aux.cxx
  HYPRE_LSI_UZAWA.cxx
  HYPRE_LSI_blkprec.cxx
  HYPRE_LSI_mli.cxx
  HYPRE_LinSysCore.cxx
  HYPRE_SlideReduction.cxx
  cfei_hypre.cxx
  hypre_cfei.cxx
  hypre_schur_reduce.cxx
  hypre_slide_reduce.cxx
  HYPRE_fei_mesh.cxx
  HYPRE_fei_matrix.cxx
  HYPRE_fei_vector.cxx 
)

install (FILES ${HYPRE_fei_HEADERS} DESTINATION include)

target_sources(${PROJECT_NAME}
  PRIVATE ${HYPRE_fei_SRCS}
)
