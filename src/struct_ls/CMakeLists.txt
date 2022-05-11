# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_struct_ls.h
  _hypre_struct_ls.h
)

set(SRCS
  coarsen.c
  cyclic_reduction.c
  F90_HYPRE_struct_bicgstab.c
  F90_HYPRE_struct_cycred.c
  F90_HYPRE_struct_gmres.c
  F90_HYPRE_struct_hybrid.c
  F90_HYPRE_struct_int.c
  F90_HYPRE_struct_jacobi.c
  F90_HYPRE_struct_pcg.c
  F90_HYPRE_struct_pfmg.c
  F90_HYPRE_struct_smg.c
  F90_HYPRE_struct_sparse_msg.c
  hybrid.c
  HYPRE_struct_bicgstab.c
  HYPRE_struct_cycred.c
  HYPRE_struct_hybrid.c
  HYPRE_struct_int.c
  HYPRE_struct_jacobi.c
  HYPRE_struct_pfmg.c
  HYPRE_struct_smg.c
  HYPRE_struct_sparse_msg.c
  HYPRE_struct_pcg.c
  HYPRE_struct_gmres.c
  HYPRE_struct_flexgmres.c
  HYPRE_struct_lgmres.c
  jacobi.c
  pcg_struct.c
  pfmg2_setup_rap.c
  pfmg3_setup_rap.c
  pfmg.c
  pfmg_relax.c
  pfmg_setup.c
  pfmg_setup_interp.c
  pfmg_setup_rap5.c
  pfmg_setup_rap7.c
  pfmg_setup_rap.c
  pfmg_solve.c
  point_relax.c
  red_black_constantcoef_gs.c
  red_black_gs.c
  semi.c
  semi_interp.c
  semi_restrict.c
  semi_setup_rap.c
  smg2_setup_rap.c
  smg3_setup_rap.c
  smg_axpy.c
  smg.c
  smg_relax.c
  smg_residual.c
  smg_setup.c
  smg_setup_interp.c
  smg_setup_rap.c
  smg_setup_restrict.c
  smg_solve.c
  sparse_msg2_setup_rap.c
  sparse_msg3_setup_rap.c
  sparse_msg.c
  sparse_msg_filter.c
  sparse_msg_interp.c
  sparse_msg_restrict.c
  sparse_msg_setup.c
  sparse_msg_setup_rap.c
  sparse_msg_solve.c
)
target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (HYPRE_USING_CUDA OR HYPRE_USING_SYCL)
  set(GPU_SRCS
    cyclic_reduction.c
    HYPRE_struct_int.c
    HYPRE_struct_pcg.c
    pfmg2_setup_rap.c
    pfmg3_setup_rap.c
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
    sparse_msg2_setup_rap.c
    sparse_msg3_setup_rap.c
    sparse_msg_filter.c
    sparse_msg_interp.c
    sparse_msg_restrict.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
