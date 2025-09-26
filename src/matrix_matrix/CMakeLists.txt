# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS "")

# Add MUP headers
if(HYPRE_ENABLE_MIXED_PRECISION)
  list(APPEND HDRS
    _hypre_matrix_matrix_mup_def.h
  )
endif()

# Regular sources
set(REGULAR_SRCS
  HYPRE_ConvertParCSRMatrixToDistributedMatrix.c
  HYPRE_ConvertPETScMatrixToDistributedMatrix.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("matrix_matrix" SRCS "${REGULAR_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
endif()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
