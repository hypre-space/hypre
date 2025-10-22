# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  blas_dh.h
  euclid_common.h
  euclid_config.h
  Euclid_dh.h
  ExternalRows_dh.h
  Factor_dh.h
  getRow_dh.h
  guards_dh.h
  Hash_dh.h
  Hash_i_dh.h
  ilu_dh.h
  io_dh.h
  krylov_dh.h
  macros_dh.h
  Mat_dh.h
  mat_dh_private.h
  MatGenFD.h
  Mem_dh.h
  Numbering_dh.h
  Parser_dh.h
  shellSort_dh.h
  sig_dh.h
  SortedList_dh.h
  SortedSet_dh.h
  SubdomainGraph_dh.h
  TimeLog_dh.h
  Timer_dh.h
  Vec_dh.h
  _hypre_Euclid.h
)

# Sources that need multi-precision compilation (FILES from Makefile)
set(MUP_SRCS
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

# Sources that don't need multi-precision compilation (none for Euclid)
set(REGULAR_SRCS
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("Euclid" SRCS "${MUP_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS} ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${REGULAR_SRCS} ${HDRS})
endif()
