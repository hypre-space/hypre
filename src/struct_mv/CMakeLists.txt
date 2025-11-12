# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_struct_mv.h
  _hypre_struct_mv.h
  _hypre_struct_mv.hpp
)

# Add MUP headers when mixed precision is enabled
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_struct_mv_mp.h
    HYPRE_struct_mv_mup.h
    _hypre_struct_mv_mup.h
    _hypre_struct_mv_mup_def.h
    _hypre_struct_mv_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  struct_matvec.c
  struct_matvec_icc.c
  struct_matvec_ivc.c
  struct_matvec_ivcc.c
  struct_matvec_cc.c
  struct_matvec_vc.c
  struct_matvec_vcc.c
  struct_matmult.c
  struct_matmult_core.c
  struct_matmult_fuse.c
  struct_matmult_fuse_fff.c
  struct_matmult_fuse_ffc.c
  struct_matmult_fuse_fcc.c
  struct_matmult_fuse_ff.c
  struct_matmult_fuse_fc.c
  struct_matmult_fuse_cc.c
  struct_matmult_fuse_f.c
  struct_matmult_fuse_c.c
  assumed_part.c
  box_algebra.c
  box_boundary.c
  box.c
  box_ds.c
  box_manager.c
  coarsen.c
  communication_info.c
  computation.c
  F90_HYPRE_struct_grid.c
  F90_HYPRE_struct_matrix.c
  F90_HYPRE_struct_stencil.c
  F90_HYPRE_struct_vector.c
  HYPRE_struct_grid.c
  HYPRE_struct_matrix.c
  HYPRE_struct_stencil.c
  HYPRE_struct_vector.c
  project.c
  struct_grid.c
  struct_io.c
  struct_matrix_mask.c
  struct_stencil.c
  struct_axpy.c
  struct_communication.c
  struct_copy.c
  struct_data.c
  struct_grid.c
  struct_innerprod.c
  struct_matrix.c
  struct_matop.c
  struct_stmatrix.c
  struct_scale.c
  struct_vector.c
)

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("struct_mv" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    struct_matvec.c
    struct_matvec_cc.c
    struct_matvec_icc.c
    struct_matvec_ivc.c
    struct_matvec_ivcc.c
    struct_matvec_vc.c
    struct_matvec_vcc.c
    struct_matmult.c
    struct_matmult_core.c
    struct_matmult_fuse.c
    struct_matmult_fuse_fff.c
    struct_matmult_fuse_ffc.c
    struct_matmult_fuse_fcc.c
    struct_matmult_fuse_ff.c
    struct_matmult_fuse_fc.c
    struct_matmult_fuse_cc.c
    struct_matmult_fuse_f.c
    struct_matmult_fuse_c.c
    struct_axpy.c
    struct_communication.c
    struct_copy.c
    struct_data.c
    struct_innerprod.c
    struct_matop.c
    struct_matrix.c
    struct_scale.c
    struct_vector.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
