# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_sstruct_ls.h
  _hypre_sstruct_ls.h
  sys_pfmg.h
)

# Add MUP headers when mixed precision is enabled
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_sstruct_ls_mup.h
    _hypre_sstruct_ls_mup.h
    _hypre_sstruct_ls_mup_def.h
    _hypre_sstruct_ls_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  F90_HYPRE_sstruct_bicgstab.c
  F90_HYPRE_sstruct_gmres.c
  F90_HYPRE_sstruct_flexgmres.c
  F90_HYPRE_sstruct_lgmres.c
  F90_HYPRE_sstruct_int.c
  F90_HYPRE_sstruct_pcg.c
  F90_HYPRE_sstruct_split.c
  F90_HYPRE_sstruct_sys_pfmg.c
  F90_HYPRE_sstruct_ssamg.c
  HYPRE_sstruct_bicgstab.c
  HYPRE_sstruct_gmres.c
  HYPRE_sstruct_flexgmres.c
  HYPRE_sstruct_lgmres.c
  HYPRE_sstruct_int.c
  HYPRE_sstruct_pcg.c
  HYPRE_sstruct_split.c
  HYPRE_sstruct_sys_pfmg.c
  HYPRE_sstruct_ssamg.c
  krylov.c
  krylov_sstruct.c
  sys_pfmg.c
  sys_pfmg_relax.c
  sys_pfmg_setup.c
  sys_pfmg_setup_interp.c
  sys_pfmg_solve.c
  node_relax.c
  ssamg.c
  ssamg_csolver.c
  ssamg_interp.c
  ssamg_relax.c
  ssamg_setup.c
  ssamg_setup_rap.c
  ssamg_solve.c
  ssamg_stats.c
  ssamg_uinterp.c
)

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("sstruct_ls" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    fac_amr_fcoarsen.c
    fac_amr_rap.c
    fac_restrict2.c
    fac_setup2.c
    fac_zero_stencilcoef.c
    node_relax.c
    ssamg_interp.c
    ssamg_relax.c
    ssamg_uinterp.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
