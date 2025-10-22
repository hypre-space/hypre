# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Sources that need multi-precision compilation (FILES from Makefile)
set(MUP_SRCS
  comm.c
  debug.c
  distributed_qsort.c
  distributed_qsort_si.c
  HYPRE_DistributedMatrixPilutSolver.c
  ilut.c
  parilut.c
  parutil.c
  pblas1.c
  serilut.c
  trifactor.c
  util.c
)

# Sources that don't need multi-precision compilation (none for pilut)
set(REGULAR_SRCS
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("pilut" SRCS "${MUP_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${REGULAR_SRCS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${REGULAR_SRCS})
endif()
