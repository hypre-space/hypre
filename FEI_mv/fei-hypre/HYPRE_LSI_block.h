/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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

#include "utilities/utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_BlockPrecondCreate(MPI_Comm comm, HYPRE_Solver *solver);
extern int HYPRE_LSI_BlockPrecondDestroy(HYPRE_Solver *solver);
extern int HYPRE_LSI_BlockPrecondSetLumpedMasses(HYPRE_Solver *solver,int,double *);
extern int HYPRE_LSI_BlockPrecondSetSchemeBDiag(HYPRE_Solver *solver);
extern int HYPRE_LSI_BlockPrecondSetSchemeBTri(HYPRE_Solver *solver);
extern int HYPRE_LSI_BlockPrecondSetSchemeBInv(HYPRE_Solver *solver);
extern int HYPRE_LSI_BlockPrecondSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector b,HYPRE_ParVector x);
extern int HYPRE_LSI_BlockPrecondSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector b, HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

