/*BHEADER**********************************************************************
 * (c) 2004   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *********************************************************************EHEADER*/
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
#include "utilities/utilities.h"
#include "parcsr_mv/parcsr_mv.h"

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

