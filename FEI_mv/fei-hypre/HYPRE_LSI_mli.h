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
#include "Lookup.h"

/******************************************************************************
 * Functions to access this data structure
 *---------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif

extern int  HYPRE_LSI_MLICreate( MPI_Comm, HYPRE_Solver * );
extern int  HYPRE_LSI_MLIDestroy( HYPRE_Solver );
extern int  HYPRE_LSI_MLISetup( HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector, HYPRE_ParVector );
extern int  HYPRE_LSI_MLISolve( HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector, HYPRE_ParVector);
extern int  HYPRE_LSI_MLISetParams( HYPRE_Solver, char * );
extern int  HYPRE_LSI_MLISetStrengthThreshold( HYPRE_Solver, double );
extern int  HYPRE_LSI_MLISetMethod( HYPRE_Solver, char * );
extern int  HYPRE_LSI_MLISetNumPDEs( HYPRE_Solver, int );
extern int  HYPRE_LSI_MLISetSmoother( HYPRE_Solver, int, int, int, char ** );
extern int  HYPRE_LSI_MLISetCoarseSolver( HYPRE_Solver, int, int, char ** );
extern int  HYPRE_LSI_MLISetNullSpace( HYPRE_Solver, int, int, double *, int );

extern void *HYPRE_LSI_MLIFEDataCreate( MPI_Comm );
extern int  HYPRE_LSI_MLIFEDataDestroy( void * );
extern int  HYPRE_LSI_MLIFEDataInit( void *, Lookup * );
extern int  HYPRE_LSI_MLIFEDataInitElemNodeList(void *, int, int, int*, int **);
extern int  HYPRE_LSI_MLIFEDataInitComplete( void * );
extern int  HYPRE_LSI_MLIFEDataLoadElemMatrix(void *, int, int, int *, double **);
extern int  HYPRE_LSI_MLIFEDataLoadNullSpaceInfo( void *, int );
extern int  HYPRE_LSI_MLIFEDataConstructNullSpace( void * );
extern int  HYPRE_LSI_MLIFEDataGetNullSpacePtr(void *, double **, int *, int *);
extern int  HYPRE_LSI_MLIFEDataWriteToFile( void *, char * );

#ifdef __cplusplus
}
#endif

#endif

