# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  aux_parcsr_matrix.h
  aux_par_vector.h
  HYPRE_IJ_mv.h
  _hypre_IJ_mv.h
  IJ_matrix.h
  IJ_vector.h
)

set(SRCS
  aux_parcsr_matrix.c
  aux_par_vector.c
  F90_HYPRE_IJMatrix.c
  F90_HYPRE_IJVector.c
  F90_IJMatrix.c
  HYPRE_IJMatrix.c
  HYPRE_IJVector.c
  IJ_assumed_part.c
  IJMatrix.c
  IJMatrix_parcsr.c
  IJVector.c
  IJVector_parcsr.c
  IJMatrix_parcsr_device.c
  IJVector_parcsr_device.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (HYPRE_USING_CUDA OR HYPRE_USING_SYCL)
  set(GPU_SRCS
    IJMatrix_parcsr_device.c
    IJVector_parcsr_device.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
