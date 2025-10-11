# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_struct_ls.h
  _hypre_struct_ls.h
)

# Add MUP headers when mixed precision is enabled
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_struct_ls_mp.h
    HYPRE_struct_ls_mup.h
    _hypre_struct_ls_mup.h
    _hypre_struct_ls_mup_def.h
    _hypre_struct_ls_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  cyclic_reduction.c
  diagscale.c
  F90_HYPRE_struct_bicgstab.c
  F90_HYPRE_struct_cycred.c
  F90_HYPRE_struct_flexgmres.c
  F90_HYPRE_struct_lgmres.c
  F90_HYPRE_struct_gmres.c
  F90_HYPRE_struct_lgmres.c
  F90_HYPRE_struct_hybrid.c
  F90_HYPRE_struct_int.c
  F90_HYPRE_struct_jacobi.c
  F90_HYPRE_struct_pcg.c
  F90_HYPRE_struct_pfmg.c
  F90_HYPRE_struct_smg.c
  hybrid.c
  HYPRE_struct_bicgstab.c
  HYPRE_struct_cycred.c
  HYPRE_struct_flexgmres.c
  HYPRE_struct_gmres.c
  HYPRE_struct_hybrid.c
  HYPRE_struct_jacobi.c
  HYPRE_struct_lgmres.c
  HYPRE_struct_pfmg.c
  HYPRE_struct_smg.c
  jacobi.c
  pcg_struct.c
  pfmg.c
  pfmg_cxyz_cc.c
  pfmg_cxyz_vc.c
  pfmg_coarsen.c
  pfmg_relax.c
  pfmg_setup_rap.c
  pfmg_solve.c
  semi.c
  smg_relax.c
  smg_setup.c
  smg_setup_rap.c
  smg_setup_restrict.c
  smg_solve.c
  cyclic_reduction.c
  diagscale.c
  HYPRE_struct_int.c
  HYPRE_struct_pcg.c
  pfmg2_setup_rap.c
  pfmg3_setup_rap.c
  pfmg_setup.c
  pfmg_setup_interp.c
  pfmg_setup_rap5.c
  pfmg_setup_rap7.c
  pfmg_setup_rap.c
  pfmg_solve.c
  red_black_constantcoef_gs.c
  red_black_gs.c
  semi_interp.c
  semi_restrict.c
  semi_setup_rap.c
  smg2_setup_rap.c
  smg3_setup_rap.c
  smg.c
  smg_axpy.c
  smg_residual.c
  smg_setup_interp.c
  smg_setup_rap.c
  smg_setup_restrict.c
  smg_solve.c
)

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("struct_ls" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    cyclic_reduction.c
    diagscale.c
    jacobi.c
    HYPRE_struct_int.c
    HYPRE_struct_pcg.c
    pfmg2_setup_rap.c
    pfmg3_setup_rap.c
    pfmg_cxyz_cc.c
    pfmg_cxyz_vc.c
    pfmg_coarsen.c
    pfmg_setup.c
    pfmg_setup_interp.c
    pfmg_setup_rap5.c
    pfmg_setup_rap7.c
    point_relax.c
    red_black_constantcoef_gs.c
    red_black_gs.c
    semi_interp.c
    semi_restrict.c
    semi_setup_rap.c
    smg2_setup_rap.c
    smg3_setup_rap.c
    smg.c
    smg_axpy.c
    smg_residual.c
    smg_setup_interp.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
