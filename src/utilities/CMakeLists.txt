# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  HYPRE_utilities.h
  _hypre_utilities.h
  hypre_hopscotch_hash.h
  HYPRE_error_f.h
  fortran.h
  fortran_matrix.h
)

set(SRCS
  HYPRE_version.c
  amg_linklist.c
  binsearch.c
  exchange_data.c
  F90_HYPRE_error.c
  fortran_matrix.c
  hypre_ap.c
  hypre_complex.c
  hypre_error.c
  hypre_general.c
  hypre_hopscotch_hash.c
  hypre_memory.c
  hypre_merge_sort.c
  hypre_mpi_comm_f2c.c
  hypre_prefix_sum.c
  hypre_printf.c
  hypre_qsort.c
  mpistubs.c
  qsplit.c
  random.c
  threading.c
  timer.c
  timing.c
)
  
convert_filenames_to_full_paths(HDRS)
convert_filenames_to_full_paths(SRCS)

set(HYPRE_HEADERS ${HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
set(HYPRE_SOURCES ${HYPRE_SOURCES} ${SRCS} PARENT_SCOPE)


