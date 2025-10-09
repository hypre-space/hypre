# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_parcsr_ls.h
  _hypre_parcsr_ls.h
)

# Add MUP headers when mixed precision is enabled
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_parcsr_ls_mp.h
    HYPRE_parcsr_ls_mup.h
    _hypre_parcsr_ls_mup.h
    _hypre_parcsr_ls_mup_def.h
    _hypre_parcsr_ls_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  amg_hybrid.c
  aux_interp.c
  F90_HYPRE_parcsr_amg.c
  F90_HYPRE_parcsr_bicgstab.c
  F90_HYPRE_parcsr_block.c
  F90_HYPRE_parcsr_cgnr.c
  F90_HYPRE_parcsr_Euclid.c
  F90_HYPRE_parcsr_gmres.c
  F90_HYPRE_parcsr_cogmres.c
  F90_HYPRE_parcsr_flexgmres.c
  F90_HYPRE_parcsr_lgmres.c
  F90_HYPRE_parcsr_hybrid.c
  F90_HYPRE_parcsr_int.c
  F90_HYPRE_parcsr_ParaSails.c
  F90_HYPRE_parcsr_pcg.c
  F90_HYPRE_parcsr_pilut.c
  F90_HYPRE_parcsr_schwarz.c
  F90_HYPRE_parcsr_mgr.c
  F90_HYPRE_parcsr_ilu.c
  F90_HYPRE_ams.c
  gen_redcs_mat.c
  HYPRE_parcsr_amg.c
  HYPRE_parcsr_amgdd.c
  HYPRE_parcsr_bicgstab.c
  HYPRE_parcsr_block.c
  HYPRE_parcsr_cgnr.c
  HYPRE_parcsr_cheby.c
  HYPRE_parcsr_Euclid.c
  HYPRE_parcsr_gmres.c
  HYPRE_parcsr_cogmres.c
  HYPRE_parcsr_flexgmres.c
  HYPRE_parcsr_lgmres.c
  HYPRE_parcsr_hybrid.c
  HYPRE_parcsr_int.c
  HYPRE_parcsr_ilu.c
  HYPRE_parcsr_mgr.c
  HYPRE_parcsr_ParaSails.c
  HYPRE_parcsr_pcg.c
  HYPRE_parcsr_pilut.c
  HYPRE_parcsr_schwarz.c
  HYPRE_parcsr_fsai.c
  HYPRE_ams.c
  HYPRE_ads.c
  HYPRE_ame.c
  par_2s_interp.c
  par_amg.c
  par_amgdd.c
  par_amgdd_comp_grid.c
  par_amgdd_solve.c
  par_amgdd_helpers.c
  par_amgdd_fac_cycle.c
  par_amgdd_setup.c
  par_amg_setup.c
  par_amg_solve.c
  par_amg_solveT.c
  par_cg_relax_wt.c
  par_coarsen.c
  par_cgc_coarsen.c
  par_cheby.c
  par_cheby_setup.c
  par_cheby_solve.c
  par_cheby_device.c
  par_coarse_parms.c
  par_cr.c
  par_cycle.c
  par_add_cycle.c
  par_fsai.c
  par_fsai_setup.c
  par_fsai_solve.c
  par_gauss_elim.c
  par_gsmg.c
  par_indepset.c
  par_interp.c
  par_jacobi_interp.c
  par_krylov_func.c
  par_mod_lr_interp.c
  par_mod_multi_interp.c
  par_multi_interp.c
  par_lr_interp.c
  par_mgr.c
  par_mgr_coarsen.c
  par_mgr_interp.c
  par_mgr_rap.c
  par_mgr_setup.c
  par_mgr_solve.c
  par_mgr_stats.c
  par_nongalerkin.c
  par_nodal_systems.c
  par_rap.c
  par_rap_communication.c
  par_relax.c
  par_relax_more.c
  par_relax_interface.c
  par_scaled_matnorm.c
  par_schwarz.c
  par_stats.c
  par_strength.c
  par_sv_interp.c
  par_sv_interp_ln.c
  partial.c
  schwarz.c
  block_tridiag.c
  ams.c
  ads.c
  ame.c
  par_restr.c
  par_lr_restr.c
  par_ilu.c
  par_ilu_setup.c
  par_ilu_solve.c
  par_cheby_device.c
  par_coarse_parms_device.c
  par_fsai_device.c
  par_ge_device.c
  par_relax_more_device.c
  par_coarsen_device.c
  par_indepset_device.c
  par_interp_device.c
  par_interp_trunc_device.c
  par_lr_interp_device.c
  par_lr_restr_device.c
  par_strength_device.c
  par_strength2nd_device.c
  par_amgdd_fac_cycle_device.c
  par_2s_interp_device.c
  par_relax_device.c
  par_mod_multi_interp_device.c
  par_mgr_device.c
  par_ilu_setup_device.c
  par_ilu_solve_device.c
)

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
  HYPRE_parcsr_ls_mp.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("parcsr_ls" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    ams.c
    ads.c
    ame.c
    par_amg_setup.c
    par_ge_device.c
    par_ilu.c
    par_ilu_setup.c
    par_ilu_setup_device.c
    par_ilu_solve.c
    par_ilu_solve_device.c
    par_coarsen_device.c
    par_coarse_parms_device.c
    par_indepset_device.c
    par_interp_device.c
    par_interp_trunc_device.c
    par_lr_interp_device.c
    par_lr_restr_device.c
    par_mod_multi_interp_device.c
    par_strength_device.c
    par_strength2nd_device.c
    par_relax_more_device.c
    par_cheby_device.c
    par_amgdd_fac_cycle_device.c
    par_2s_interp_device.c
    par_relax_device.c
    par_mgr_device.c
    par_fsai_device.c
    dsuperlu.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
