/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *********************************************************************EHEADER*/
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
#include "utilities/utilities.h"
#include "parcsr_ls/parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"
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

