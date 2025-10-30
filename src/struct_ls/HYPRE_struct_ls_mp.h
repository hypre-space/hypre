/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_STRUCT_LS_MP_HEADER
#define HYPRE_STRUCT_LS_MP_HEADER

#include "_hypre_struct_ls.h"

/* Mixed precision function protos */

#ifdef HYPRE_MIXED_PRECISION

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int
HYPRE_StructSMGSetup_mp( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

HYPRE_Int
HYPRE_StructSMGSolve_mp( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);
HYPRE_Int
HYPRE_StructPFMGSetup_mp( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x);

HYPRE_Int
HYPRE_StructPFMGSolve_mp( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x);

#ifdef __cplusplus
}
#endif

#endif
#endif

