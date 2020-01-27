# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  distributed_matrix.h
)

set(SRCS
  distributed_matrix.c
  distributed_matrix_ISIS.c
  distributed_matrix_parcsr.c
  distributed_matrix_PETSc.c
  HYPRE_distributed_matrix.c
)

target_sources(HYPRE
  PRIVATE ${SRCS}
          ${HDRS}
  )

convert_filenames_to_full_paths(HDRS)
set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
