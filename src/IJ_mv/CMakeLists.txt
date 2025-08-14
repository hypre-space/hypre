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

# Add MUP headers when mixed precision is enabled
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    HYPRE_IJ_mv_mup.h
    _hypre_IJ_mv_mup.h
    _hypre_IJ_mv_mup_def.h
    _hypre_IJ_mv_mup_undef.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
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

# Mixed precision sources
set(MUP_SRCS
  mup_fixed.c
  mup_functions.c
  mup_pre.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("IJ_mv" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

if (HYPRE_USING_GPU)
  set(GPU_SRCS
    IJMatrix_parcsr_device.c
    IJVector_parcsr_device.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
