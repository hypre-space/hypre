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
 * HYPRE_LSI_MLI interface
 *
 *****************************************************************************/

#ifndef __HYPRE_LSI_MLI__
#define __HYPRE_LSI_MLI__

/******************************************************************************
 * system includes
 *---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/******************************************************************************
 * HYPRE internal libraries
 *---------------------------------------------------------------------------*/

#include "utilities/utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"

/******************************************************************************
 * Functions to access this data structure
 *---------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_MLICreate( MPI_Comm, HYPRE_Solver * );
extern int HYPRE_LSI_MLIDestroy( HYPRE_Solver );
extern int HYPRE_LSI_MLISetup( HYPRE_Solver, HYPRE_ParCSRMatrix,
                               HYPRE_ParVector,   HYPRE_ParVector );
extern int HYPRE_LSI_MLISolve( HYPRE_Solver, HYPRE_ParCSRMatrix,
                               HYPRE_ParVector,   HYPRE_ParVector);
extern int HYPRE_LSI_SetParams( HYPRE_Solver, char ** );
extern int HYPRE_LSI_MLISetStrengthThreshold( HYPRE_Solver, double );
extern int HYPRE_LSI_MLISetMethod( HYPRE_Solver, char * );
extern int HYPRE_LSI_MLISetNumPDEs( HYPRE_Solver, int );
extern int HYPRE_LSI_MLISetSmoother( HYPRE_Solver, int, int, int, char ** );
extern int HYPRE_LSI_MLISetCoarseSolver( HYPRE_Solver, int, int, char ** );

#ifdef __cplusplus
}
#endif

#endif

