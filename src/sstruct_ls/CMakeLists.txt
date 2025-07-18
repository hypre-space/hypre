# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
  F90_HYPRE_sstruct_int.c
  F90_HYPRE_sstruct_pcg.c
  F90_HYPRE_sstruct_split.c
  F90_HYPRE_sstruct_sys_pfmg.c
  HYPRE_sstruct_bicgstab.c
  HYPRE_sstruct_gmres.c
  HYPRE_sstruct_flexgmres.c
  HYPRE_sstruct_lgmres.c
  HYPRE_sstruct_int.c
  HYPRE_sstruct_pcg.c
  HYPRE_sstruct_split.c
  HYPRE_sstruct_sys_pfmg.c
  krylov.c
  krylov_sstruct.c
  node_relax.c
  sys_pfmg.c
  sys_pfmg_relax.c
  sys_pfmg_setup.c
  sys_pfmg_setup_interp.c
  sys_pfmg_setup_rap.c
  sys_pfmg_solve.c
  sys_semi_interp.c
  sys_semi_restrict.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    node_relax.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
