# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_parcsr_mv.h
  _hypre_parcsr_mv.h
)

set(SRCS
  communicationT.c
  F90_HYPRE_parcsr_matrix.c
  F90_HYPRE_parcsr_vector.c
  F90_parcsr_matrix.c
  F90_par_vector.c
  HYPRE_parcsr_matrix.c
  HYPRE_parcsr_vector.c
  new_commpkg.c
  numbers.c
  par_csr_aat.c
  par_csr_assumed_part.c
  par_csr_bool_matop.c
  par_csr_bool_matrix.c
  par_csr_communication.c
  par_csr_matop.c
  par_csr_matrix.c
  par_csr_matop_marked.c
  par_csr_matvec.c
  par_vector.c
  par_make_system.c
  par_csr_triplemat.c
)
  
convert_filenames_to_full_paths(HDRS)
convert_filenames_to_full_paths(SRCS)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


