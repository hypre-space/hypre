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
 * HYPRE_MLI interface
 *
 *****************************************************************************/

#ifndef __HYPRE_MLI__
#define __HYPRE_MLI__

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

extern int HYPRE_ParCSRMLI_Create( MPI_Comm, HYPRE_Solver * );
extern int HYPRE_ParCSRMLI_Destroy( HYPRE_Solver );
extern int HYPRE_ParCSRMLSetup( HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector,   HYPRE_ParVector );
extern int HYPRE_ParCSRMLSolve( HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector,   HYPRE_ParVector);
extern int HYPRE_ParCSRMLI_SetStrengthThreshold( HYPRE_Solver, double );
extern int HYPRE_ParCSRMLI_SetMethod( HYPRE_Solver, char * );
extern int HYPRE_ParCSRMLI_SetNumPDEs( HYPRE_Solver, int );
extern int HYPRE_ParCSRMLI_SetSmoother( HYPRE_Solver, int, int, int, char ** );
extern int HYPRE_ParCSRMLI_SetCoarseSolver( HYPRE_Solver, int, int, char ** );

#ifdef __cplusplus
}
#endif

#endif

