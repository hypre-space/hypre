/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSR_SuperLU interface
 *
 *****************************************************************************/

#ifndef __HYPRE_PARCSR_SUPERLU__
#define __HYPRE_PARCSR_SUPERLU__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_ParCSR_SuperLUCreate(MPI_Comm comm, HYPRE_Solver *solver);
extern int HYPRE_ParCSR_SuperLUDestroy(HYPRE_Solver solver);
extern int HYPRE_ParCSR_SuperLUSetOutputLevel(HYPRE_Solver solver, int);
extern int HYPRE_ParCSR_SuperLUSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,HYPRE_ParVector x);
extern int HYPRE_ParCSR_SuperLUSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b, HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

