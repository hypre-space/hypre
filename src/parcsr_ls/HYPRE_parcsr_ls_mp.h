/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Mixed precision function protos
 *
 *****************************************************************************/

#ifndef HYPRE_PARCSR_LS_MP_HEADER
#define HYPRE_PARCSR_LS_MP_HEADER

#ifdef HYPRE_MIXED_PRECISION

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int HYPRE_BoomerAMGSetup_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);

HYPRE_Int HYPRE_BoomerAMGSolve_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);

#ifdef __cplusplus
}
#endif

#endif

#endif
