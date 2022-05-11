/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_Schwarz interface
 *
 *****************************************************************************/

#ifndef __HYPRE_SCHWARZ__
#define __HYPRE_SCHWARZ__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_SchwarzCreate( MPI_Comm comm, HYPRE_Solver *solver );
extern int HYPRE_LSI_SchwarzDestroy( HYPRE_Solver solver );
extern int HYPRE_LSI_SchwarzSetBlockSize( HYPRE_Solver solver, int blksize);
extern int HYPRE_LSI_SchwarzSetNBlocks( HYPRE_Solver solver, int nblks);
extern int HYPRE_LSI_SchwarzSetILUTFillin( HYPRE_Solver solver, double fillin);
extern int HYPRE_LSI_SchwarzSetOutputLevel( HYPRE_Solver solver, int level);
extern int HYPRE_LSI_SchwarzSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,   HYPRE_ParVector x );
extern int HYPRE_LSI_SchwarzSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,   HYPRE_ParVector x );

#ifdef __cplusplus
}
#endif

#endif

