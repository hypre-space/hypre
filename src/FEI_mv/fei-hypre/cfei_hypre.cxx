/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_FEI_includes.h"
#include "HYPRE_LinSysCore.h"
#include "cfei_hypre.h"

/******************************************************************************/
/* Create function for a HYPRE_LinSysCore object.                             */
/*----------------------------------------------------------------------------*/

LinearSystemCore *HYPRE_base_create( MPI_Comm comm ) 
{
   LinearSystemCore* linSys = new HYPRE_LinSysCore(comm);
   return (linSys);
}

/******************************************************************************/
/* Create function for a HYPRE_LinSysCore object.                             */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LinSysCore_create(LinSysCore** lsc, MPI_Comm comm) 
{
   HYPRE_LinSysCore* linSys = new HYPRE_LinSysCore(comm);

   if (linSys == NULL) return(1);

   *lsc = new LinSysCore;

   if (*lsc == NULL) return(1);

   (*lsc)->lsc_ = (void*)linSys;

   return(0);
}

/******************************************************************************/
/* Destroy function, to de-allocate a HYPRE_LinSysCore object.                */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LinSysCore_destroy(LinSysCore** lsc) 
{
   if (*lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)((*lsc)->lsc_);

   if (linSys == NULL) return(1);

   delete linSys;

   delete *lsc;
   *lsc = NULL;

   return(0);
}

/******************************************************************************/
/* function for loading to the matrix directly but with FEI equation mappings */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_BeginMappedMatrixLoad(LinSysCore* lsc) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->beginCreateMapFromSoln();

   return(0);
}

/******************************************************************************/
/* function for loading to the matrix directly but with FEI equation mappings */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_EndMappedMatrixLoad(LinSysCore* lsc) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->endCreateMapFromSoln();

   return(0);
}

/******************************************************************************/
/* function for loading to the matrix directly but with FEI equation mappings */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_MappedMatrixLoad(LinSysCore* lsc, int row,
                int col, double val) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->putIntoMappedMatrix(row, 1, &val, &col);

   return(0);
}

/******************************************************************************/
/* function for getting the version number                                    */
/*----------------------------------------------------------------------------*/

extern "C" char *HYPRE_LSC_GetVersion(LinSysCore* lsc)
{
   char *lscVersion;

   if (lsc == NULL) return(NULL);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(NULL);

   lscVersion = linSys->getVersion();

   return(lscVersion);
}

/******************************************************************************/
/* get the finite element grid object                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_GetFEDataObject(LinSysCore* lsc, void **object)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->getFEDataObject(object);

   return(0);
}

/******************************************************************************/
/* the parameter function (to set parameter values)                           */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_parameters(LinSysCore* lsc, int numParams, 
                                    char **params) 
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->parameters(numParams, params);

   return(0);
}

/******************************************************************************/
/* This function sets up the equation offset on each processor                */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_setGlobalOffsets(LinSysCore* lsc, int leng, 
                    int* nodeOffsets, int* eqnOffsets, int* blkEqnOffsets)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->setGlobalOffsets(leng, nodeOffsets, eqnOffsets, blkEqnOffsets);

   return(0);
}

/******************************************************************************/
/* set up the matrix sparsity pattern                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_setMatrixStructure(LinSysCore *lsc, int** ptColIndices, 
                     int* ptRowLengths, int** blkColIndices, int* blkRowLengths, 
                     int* ptRowsPerBlkRow)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->setMatrixStructure(ptColIndices, ptRowLengths, blkColIndices, 
                              blkRowLengths, ptRowsPerBlkRow);

   return(0);
}

/******************************************************************************/
/* reset the matrix but keep the sparsity pattern                             */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_resetMatrixAndVector(LinSysCore *lsc, double val)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->resetMatrixAndVector(val);

   return(0);
}

/******************************************************************************/
/* reset the matrix but keep the sparsity pattern                             */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_resetMatrix(LinSysCore *lsc, double val)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->resetMatrix(val);

   return(0);
}

/******************************************************************************/
/* reset the right hand side vector                                           */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_resetRHSVector(LinSysCore *lsc, double val)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->resetRHSVector(val);

   return(0);
}

/******************************************************************************/
/* load the matrix                                                            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_sumIntoSystemMatrix(LinSysCore *lsc, int numPtRows, 
                     const int* ptRows, int numPtCols, const int* ptCols, 
                     int numBlkRows, const int* blkRows, int numBlkCols, 
                     const int* blkCols, const double* const* values)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->sumIntoSystemMatrix(numPtRows, ptRows, numPtCols, ptCols, numBlkRows, 
                               blkRows, numBlkCols, blkCols, values);

   return(0);
}

/******************************************************************************/
/* load the right hand side vector                                            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_sumIntoRHSVector(LinSysCore *lsc, int num, 
                    const double* values, const int* indices)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->sumIntoRHSVector(num, values, indices);

   return(0);
}

/******************************************************************************/
/* matrix loading completed                                                   */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_matrixLoadComplete(LinSysCore *lsc)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->matrixLoadComplete();

   return(0);
}

/******************************************************************************/
/* enforce essential boundary condition                                       */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_enforceEssentialBC(LinSysCore *lsc, int* globalEqn,
                     double* alpha, double* gamma1, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->enforceEssentialBC(globalEqn, alpha, gamma1, leng);

   return(0);
}

/******************************************************************************/
/* enforce essential boundary condition (cross processor boundary)            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_enforceRemoteEssBCs(LinSysCore *lsc,int numEqns,
                    int* globalEqns, int** colIndices, int* colIndLen, 
                    double** coefs)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->enforceRemoteEssBCs(numEqns,globalEqns,colIndices,colIndLen,coefs);

   return(0);
}

/******************************************************************************/
/* enforce natural boundary condition                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_enforceOtherBC(LinSysCore *lsc, int* globalEqn, 
                     double* alpha, double* beta, double* gamma1, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->enforceOtherBC(globalEqn, alpha, beta, gamma1, leng);

   return(0);
}

/******************************************************************************/
/* put initial guess into HYPRE                                               */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_putInitialGuess(LinSysCore *lsc, const int* eqnNumbers, 
                                     const double* values, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->putInitialGuess(eqnNumbers, values, leng);

   return(0);
}

/******************************************************************************/
/* get the whole solution vector                                              */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_getSolution(LinSysCore *lsc, double *answers, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->getSolution(answers, leng);

   return(0);
}

/******************************************************************************/
/* get a solution entry                                                       */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_getSolnEntry(LinSysCore *lsc, int eqnNumber, 
                                      double *answer)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->getSolnEntry(eqnNumber, (*answer));

   return(0);
}

/******************************************************************************/
/* form and fetch the residual vector                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_formResidual(LinSysCore *lsc, double *values, int leng)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->formResidual(values, leng);

   return(0);
}

/******************************************************************************/
/* start iterating                                                            */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_launchSolver(LinSysCore *lsc, int *solveStatus, 
                                      int *iter)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if (linSys == NULL) return(1);

   linSys->launchSolver(*solveStatus, *iter);

   return(0);
}

/******************************************************************************/
/* begin initializing the field IDs                                           */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_FEDataInitFields(LinSysCore *lsc, int nFields,
                                          int *fieldSizes, int *fieldIDs)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if ( linSys == NULL ) return(1);

   linSys->FE_initFields(nFields, fieldSizes, fieldIDs);

   return(0);
}

/******************************************************************************/
/* begin initializing the element block                                       */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_FEDataInitElemBlock(LinSysCore *lsc, int nElems,
                         int nNodesPerElem, int numNodeFields, int *nodeFieldIDs)

{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if ( linSys == NULL ) return(1);

   linSys->FE_initElemBlock(nElems, nNodesPerElem, numNodeFields, nodeFieldIDs);

   return(0);
}

/******************************************************************************/
/* begin initializing element connectivity information                        */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_FEDataInitElemNodeList(LinSysCore *lsc, int elemID,
                                           int nNodesPerElem, int *elemConn)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if ( linSys == NULL ) return(1);

   linSys->FE_initElemNodeList(elemID,nNodesPerElem,elemConn);

   return(0);
}

/******************************************************************************/
/* begin initializing shared node information                                 */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_FEDataInitSharedNodes(LinSysCore *lsc, int nShared,
                         int *sharedIDs, int *sharedLengs, int **sharedProcs)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if ( linSys == NULL ) return(1);

   linSys->FE_initSharedNodes(nShared,sharedIDs,sharedLengs,sharedProcs);

   return(0);
}

/******************************************************************************/
/* initialization done                                                        */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_FEDataInitComplete(LinSysCore *lsc)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if ( linSys == NULL ) return(1);

   linSys->FE_initComplete();

   return(0);
}

/******************************************************************************/
/* begin initializing the element set                                         */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_LSC_FEDataLoadElemMatrix(LinSysCore *lsc, int elemID,
                    int nNodesPerElem, int *elemConn, int matDim, 
                    double **elemStiff)
{
   if (lsc == NULL) return(1);

   HYPRE_LinSysCore* linSys = (HYPRE_LinSysCore*)(lsc->lsc_);

   if ( linSys == NULL ) return(1);

   linSys->FE_loadElemMatrix(elemID,nNodesPerElem,elemConn,matDim,elemStiff);

   return(0);
}


