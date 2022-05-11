# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(SRCS
  blas_dh.c
  Euclid_apply.c
  Euclid_dh.c
  ExternalRows_dh.c
  Factor_dh.c
  getRow_dh.c
  globalObjects.c
  Hash_dh.c
  Hash_i_dh.c
  ilu_mpi_bj.c
  ilu_mpi_pilu.c
  ilu_seq.c
  io_dh.c
  krylov_dh.c
  Mat_dh.c
  mat_dh_private.c
  MatGenFD.c
  Mem_dh.c
  Numbering_dh.c
  Parser_dh.c
  shellSort_dh.c
  sig_dh.c
  SortedList_dh.c
  SortedSet_dh.c
  SubdomainGraph_dh.c
  TimeLog_dh.c
  Timer_dh.c
  Vec_dh.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
)
