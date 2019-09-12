# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_IJ_mv.h
  _hypre_IJ_mv.h
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
)
  
convert_filenames_to_full_paths(HDRS)
convert_filenames_to_full_paths(SRCS)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


