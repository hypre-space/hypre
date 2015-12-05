/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_LSI_DSuperLU interface
 *
 *****************************************************************************/

#ifndef __HYPRE_DSUPERLU__
#define __HYPRE_DSUPERLU__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "HYPRE.h"
#include "utilities/utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_DSuperLUCreate(MPI_Comm comm, HYPRE_Solver *solver);
extern int HYPRE_LSI_DSuperLUDestroy(HYPRE_Solver solver);
extern int HYPRE_LSI_DSuperLUSetOutputLevel(HYPRE_Solver solver, int);
extern int HYPRE_LSI_DSuperLUSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,HYPRE_ParVector x);
extern int HYPRE_LSI_DSuperLUSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

