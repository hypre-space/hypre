/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.27 $
 ***********************************************************************EHEADER*/




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

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_FEI_includes.h"

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
                                HYPRE_ParVector,   HYPRE_ParVector );
extern int  HYPRE_LSI_MLISolve( HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector,   HYPRE_ParVector);
extern int  HYPRE_LSI_MLISetParams( HYPRE_Solver, char * );
extern int  HYPRE_LSI_MLICreateNodeEqnMap( HYPRE_Solver, int, int *, int *,
                                           int *procNRows );
extern int  HYPRE_LSI_MLIAdjustNodeEqnMap( HYPRE_Solver, int *, int * );
extern int  HYPRE_LSI_MLIGetNullSpace( HYPRE_Solver, int *, int *, double ** );
extern int  HYPRE_LSI_MLIAdjustNullSpace( HYPRE_Solver, int, int *,
                                          HYPRE_ParCSRMatrix );
extern int  HYPRE_LSI_MLISetFEData( HYPRE_Solver, void * );
extern int  HYPRE_LSI_MLISetSFEI( HYPRE_Solver, void * );

extern int  HYPRE_LSI_MLILoadNodalCoordinates( HYPRE_Solver, int, int, int *, 
                                   int, double *, int );
extern int  HYPRE_LSI_MLILoadMatrixScalings( HYPRE_Solver, int, double * );
extern int  HYPRE_LSI_MLILoadMaterialLabels( HYPRE_Solver, int, int * );

extern void *HYPRE_LSI_MLIFEDataCreate( MPI_Comm );
extern int  HYPRE_LSI_MLIFEDataDestroy( void * );
extern int  HYPRE_LSI_MLIFEDataInitFields( void *, int, int *, int * );
extern int  HYPRE_LSI_MLIFEDataInitElemBlock(void *, int, int, int, int *);
extern int  HYPRE_LSI_MLIFEDataInitElemNodeList(void *, int, int, int*);
extern int  HYPRE_LSI_MLIFEDataInitSharedNodes(void *, int, int *, int*, int **);
extern int  HYPRE_LSI_MLIFEDataInitComplete( void * );
extern int  HYPRE_LSI_MLIFEDataLoadElemMatrix(void *, int, int, int *, int,
                                              double **);
extern int  HYPRE_LSI_MLIFEDataWriteToFile( void *, char * );

extern void *HYPRE_LSI_MLISFEICreate( MPI_Comm );
extern int  HYPRE_LSI_MLISFEIDestroy( void * );
extern int  HYPRE_LSI_MLISFEILoadElemMatrices(void *, int, int, int *,
                                      double ***, int, int **);
extern int  HYPRE_LSI_MLISFEIAddNumElems(void *, int, int, int);

#ifdef __cplusplus
}
#endif

#endif

