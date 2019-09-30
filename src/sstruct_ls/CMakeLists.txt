# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_sstruct_ls.h
  _hypre_sstruct_ls.h
)

set(SRCS
  F90_HYPRE_sstruct_bicgstab.c
  F90_HYPRE_sstruct_gmres.c
  F90_HYPRE_sstruct_flexgmres.c
  F90_HYPRE_sstruct_lgmres.c
  F90_HYPRE_sstruct_InterFAC.c
  F90_HYPRE_sstruct_int.c
  F90_HYPRE_sstruct_maxwell.c
  F90_HYPRE_sstruct_pcg.c
  F90_HYPRE_sstruct_split.c
  F90_HYPRE_sstruct_sys_pfmg.c
  HYPRE_sstruct_bicgstab.c
  HYPRE_sstruct_gmres.c
  HYPRE_sstruct_flexgmres.c
  HYPRE_sstruct_lgmres.c
  HYPRE_sstruct_InterFAC.c
  HYPRE_sstruct_int.c
  HYPRE_sstruct_maxwell.c
  HYPRE_sstruct_pcg.c
  HYPRE_sstruct_split.c
  HYPRE_sstruct_sys_pfmg.c
  fac.c
  fac_amr_rap.c
  fac_amr_fcoarsen.c
  fac_amr_zero_data.c
  fac_cf_coarsen.c
  fac_cfstencil_box.c
  fac_CFInterfaceExtents.c
  fac_interp2.c
  fac_relax.c
  fac_restrict2.c
  fac_setup2.c
  fac_solve3.c
  fac_zero_cdata.c
  fac_zero_stencilcoef.c
  hypre_bsearch.c
  krylov.c
  krylov_sstruct.c
  eliminate_rowscols.c
  maxwell_grad.c
  maxwell_physbdy.c
  maxwell_PNedelec.c
  maxwell_PNedelec_bdy.c
  maxwell_semi_interp.c
  maxwell_TV.c
  maxwell_TV_setup.c
  maxwell_zeroBC.c
  hypre_MaxwellSolve.c
  hypre_MaxwellSolve2.c
  nd1_amge_interpolation.c
  node_relax.c
  sstruct_amr_intercommunication.c
  sstruct_owninfo.c
  sstruct_recvinfo.c
  sstruct_sendinfo.c
  sstruct_sharedDOFComm.c
  sys_pfmg.c
  sys_pfmg_relax.c
  sys_pfmg_setup.c
  sys_pfmg_setup_interp.c
  sys_pfmg_setup_rap.c
  sys_pfmg_solve.c
  sys_semi_interp.c
  sys_semi_restrict.c
)

convert_filenames_to_full_paths(HDRS)
convert_filenames_to_full_paths(SRCS)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


