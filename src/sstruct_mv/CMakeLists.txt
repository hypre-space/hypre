# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_sstruct_mv.h
  _hypre_sstruct_mv.h
  sstruct_graph.h
  sstruct_grid.h
  sstruct_matrix.h
  sstruct_stencil.h
  sstruct_vector.h
)

# Add MUP headers when mixed precision is enabled
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_sstruct_mv_mup.h
    _hypre_sstruct_mv_mup.h
    _hypre_sstruct_mv_mup_def.h
    _hypre_sstruct_mv_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  F90_HYPRE_sstruct_graph.c
  F90_HYPRE_sstruct_grid.c
  F90_HYPRE_sstruct_matrix.c
  F90_HYPRE_sstruct_stencil.c
  F90_HYPRE_sstruct_vector.c
  HYPRE_sstruct_graph.c
  HYPRE_sstruct_grid.c
  HYPRE_sstruct_matrix.c
  HYPRE_sstruct_stencil.c
  HYPRE_sstruct_vector.c
  sstruct_axpy.c
  sstruct_copy.c
  sstruct_graph.c
  sstruct_grid.c
  sstruct_innerprod.c
  sstruct_matvec.c
  sstruct_matmult.c
  sstruct_matop.c
  sstruct_scale.c
  sstruct_stencil.c
  sstruct_matrix.c
  sstruct_vector.c
)

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("sstruct_mv" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    sstruct_matrix.c
    sstruct_vector.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
