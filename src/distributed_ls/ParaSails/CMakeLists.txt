# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(SRCS
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

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
)
