# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_struct_mv.h
  _hypre_struct_mv.h
)

set(SRCS
  assumed_part.c
  box_algebra.c
  box_boundary.c
  box.c
  box_manager.c
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
  struct_axpy.c
  struct_communication.c
  struct_copy.c
  struct_grid.c
  struct_innerprod.c
  struct_io.c
  struct_matrix.c
  struct_matrix_mask.c
  struct_matvec.c
  struct_scale.c
  struct_stencil.c
  struct_vector.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (HYPRE_USING_CUDA OR HYPRE_USING_SYCL)
  set(GPU_SRCS
    struct_axpy.c
    struct_communication.c
    struct_copy.c
    struct_innerprod.c
    struct_matrix.c
    struct_matvec.c
    struct_scale.c
    struct_vector.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
