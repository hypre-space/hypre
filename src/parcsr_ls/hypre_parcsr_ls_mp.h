/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/* Mixed precision function protos */
/* hypre_parcsr_ls_mp.h */

#ifdef HYPRE_MIXED_PRECISION
HYPRE_Int HYPRE_BoomerAMGSetup_mp(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_BoomerAMGSolve_mp(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_MPAMGPrecSetup_mp(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_MPAMGPrecSolve_mp(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_MPAMGSetup_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);

HYPRE_Int HYPRE_MPAMGSolve_mp(HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x);
HYPRE_Int HYPRE_MPAMGCreate_mp(HYPRE_Solver *solver);
HYPRE_Int HYPRE_MPAMGDestroy_mp(HYPRE_Solver solver);

#endif
