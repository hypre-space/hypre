# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_seq_mv.h
  seq_mv.h
)

set(SRCS
  csr_matop.c
  csr_matrix.c
  csr_matvec.c
  genpart.c
  HYPRE_csr_matrix.c
  HYPRE_mapped_matrix.c
  HYPRE_multiblock_matrix.c
  HYPRE_vector.c
  mapped_matrix.c
  multiblock_matrix.c
  vector.c
  vector_batched.c
)
  
convert_filenames_to_full_paths(HDRS)
convert_filenames_to_full_paths(SRCS)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


