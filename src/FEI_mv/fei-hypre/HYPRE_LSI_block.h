/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_LSI_BlockP interface
 *
 *****************************************************************************/

#ifndef __HYPRE_BLOCKP__
#define __HYPRE_BLOCKP__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_LSI_blkprec.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_BlockPrecondCreate(MPI_Comm comm, HYPRE_Solver *solver);
extern int HYPRE_LSI_BlockPrecondDestroy(HYPRE_Solver solver);
extern int HYPRE_LSI_BlockPrecondSetLumpedMasses(HYPRE_Solver solver,
                                                 int,double *);
extern int HYPRE_LSI_BlockPrecondSetParams(HYPRE_Solver solver, char *params);
extern int HYPRE_LSI_BlockPrecondSetLookup(HYPRE_Solver solver, HYPRE_Lookup *);
extern int HYPRE_LSI_BlockPrecondSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector b,HYPRE_ParVector x);
extern int HYPRE_LSI_BlockPrecondSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector b, HYPRE_ParVector x);
extern int HYPRE_LSI_BlockPrecondSetA11Tolerance(HYPRE_Solver solver, double);

#ifdef __cplusplus
}
#endif

#endif

