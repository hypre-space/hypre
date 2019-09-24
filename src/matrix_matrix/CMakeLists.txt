# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_matrix_matrix_protos.h
)

set(SRCS
  HYPRE_ConvertParCSRMatrixToDistributedMatrix.c
  HYPRE_ConvertPETScMatrixToDistributedMatrix.c
)
  
convert_filenames_to_full_paths(HDRS)
convert_filenames_to_full_paths(SRCS)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


