/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_LSI_Uzawa interface
 *
 *****************************************************************************/

#ifndef __HYPRE_UZAWA__
#define __HYPRE_UZAWA__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_LSI_UZAWA.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_UzawaCreate(MPI_Comm comm, HYPRE_Solver *solver);
extern int HYPRE_LSI_UzawaDestroy(HYPRE_Solver solver);
extern int HYPRE_LSI_UzawaSetMaxIterations(HYPRE_Solver solver, int iter);
extern int HYPRE_LSI_UzawaSetTolerance(HYPRE_Solver solver, double tol);
extern int HYPRE_LSI_UzawaSetParams(HYPRE_Solver solver, char *params);
extern int HYPRE_LSI_UzawaGetNumIterations(HYPRE_Solver solver, int *iter);
extern int HYPRE_LSI_UzawaSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b,HYPRE_ParVector x);
extern int HYPRE_LSI_UzawaSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

