# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  _hypre_seq_block_mv.h
)

set(SRCS
  dense_block_matrix.c
  dense_block_matmult.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

# if (HYPRE_USING_CUDA OR HYPRE_USING_HIP)
#   set(GPU_SRCS
#      # File name here
#   )
#   convert_filenames_to_full_paths(GPU_SRCS)
#   set(HYPRE_GPU_SOURCES ${HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
# endif ()

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
