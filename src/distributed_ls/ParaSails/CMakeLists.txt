# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  _hypre_ParaSails.h
  Common.h
  ConjGrad.h
  DiagScale.h
  Hash.h
  LoadBal.h
  Matrix.h
  Mem.h
  Numbering.h
  OrderStat.h
  ParaSails.h
  PrunedRows.h
  RowPatt.h
  StoredRows.h
)

# Sources that need multi-precision compilation (FILES from Makefile)
set(MUP_SRCS
  ConjGrad.c
  DiagScale.c
  FGmres.c
  Hash.c
  hypre_ParaSails.c
  LoadBal.c
  Matrix.c
  Mem.c
  Numbering.c
  OrderStat.c
  ParaSails.c
  PrunedRows.c
  RowPatt.c
  StoredRows.c
)

if(HYPRE_ENABLE_MIXED_PRECISION)
  setup_mixed_precision_compilation("ParaSails" SRCS "${MUP_SRCS}")
  target_sources(${PROJECT_NAME} PRIVATE ${HDRS})
else()
  target_sources(${PROJECT_NAME} PRIVATE ${MUP_SRCS} ${HDRS})
endif()
