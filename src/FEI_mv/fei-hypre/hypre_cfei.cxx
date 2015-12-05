/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/


#include <string.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "LLNL_FEI_Impl.h"
#include "utilities/_hypre_utilities.h"
#include "hypre_cfei.h"

/******************************************************************************/
/* constructor                                                                */
/*----------------------------------------------------------------------------*/

extern "C" HYPRE_FEI_Impl *HYPRE_FEI_create( MPI_Comm comm ) 
{
   HYPRE_FEI_Impl *cfei;
   LLNL_FEI_Impl  *lfei;
   cfei = (HYPRE_FEI_Impl *) malloc(sizeof(HYPRE_FEI_Impl));
   lfei = new LLNL_FEI_Impl(comm);
   cfei->fei_ = (void *) lfei;
   return (cfei);
}

/******************************************************************************/
/* Destroy function                                                           */
/*----------------------------------------------------------------------------*/

extern "C" int HYPRE_FEI_destroy(HYPRE_FEI_Impl *fei) 
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   if (lfei != NULL) delete lfei;
   return(0);
}

/******************************************************************************/
/* function for setting algorithmic parameters                                */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_parameters(HYPRE_FEI_Impl *fei, int numParams, char **paramString)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->parameters(numParams, paramString);
   return(0);
}

/******************************************************************************/
/* set solve type                                                             */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_setSolveType(HYPRE_FEI_Impl *fei, int solveType)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->setSolveType(solveType);
   return(0);
}

/******************************************************************************/
/* initialize different fields                                                */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_initFields(HYPRE_FEI_Impl *fei, int numFields, int *fieldSizes,
                         int *fieldIDs)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initFields(numFields, fieldSizes, fieldIDs);
   return(0);
}

/******************************************************************************/
/* initialize element block                                                   */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_initElemBlock(HYPRE_FEI_Impl *fei, int elemBlockID, int numElements,
                            int numNodesPerElement, int *numFieldsPerNode,
                            int **nodalFieldIDs, int numElemDOFFieldsPerElement,
                            int *elemDOFFieldIDs, int interleaveStrategy)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initElemBlock(elemBlockID, numElements, numNodesPerElement, 
                       numFieldsPerNode, nodalFieldIDs, 
                       numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                       interleaveStrategy);
   return(0);
}

/******************************************************************************/
/* initialize element connectivity                                            */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_initElem(HYPRE_FEI_Impl *fei, int elemBlockID, int elemID,
                       int *elemConn)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initElem(elemBlockID, elemID, elemConn);
   return(0);
}

/******************************************************************************/
/* initialize shared nodes                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_initSharedNodes(HYPRE_FEI_Impl *fei, int nShared, int *sharedIDs,
                              int *sharedLeng, int **sharedProcs)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initSharedNodes(nShared, sharedIDs, sharedLeng, sharedProcs);
   return(0);
}

/******************************************************************************/
/* signal completion of initialization                                        */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_initComplete(HYPRE_FEI_Impl *fei)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initComplete();
   return(0);
}

/******************************************************************************/
/* reset the whole system                                                     */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_resetSystem(HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetSystem(s);
   return(0);
}

/******************************************************************************/
/* reset the matrix                                                           */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_resetMatrix(HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetMatrix(s);
   return(0);
}

/******************************************************************************/
/* reset the right hand side                                                  */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_resetRHSVector(HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetRHSVector(s);
   return(0);
}

/******************************************************************************/
/* reset the initial guess                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_resetInitialGuess(HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetInitialGuess(s);
   return(0);
}

/******************************************************************************/
/* load boundary condition                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_loadNodeBCs(HYPRE_FEI_Impl *fei, int nNodes, int *nodeIDs,
                          int fieldID, double **alpha, double **beta, 
                          double **gamma)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->loadNodeBCs(nNodes, nodeIDs, fieldID, alpha, beta, gamma);
   return(0);
}

/******************************************************************************/
/* submit element stiffness matrix (with right hand side)                     */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_sumInElem(HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                        int *elemConn, double **elemStiff, double *elemLoad, 
                        int elemFormat)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->sumInElem(elemBlock, elemID, elemConn, elemStiff, elemLoad, 
                   elemFormat);
   return(0);
}

/******************************************************************************/
/* submit element stiffness matrix                                            */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_sumInElemMatrix(HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                              int *elemConn, double **elemStiff, int elemFormat)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->sumInElemMatrix(elemBlock,elemID,elemConn,elemStiff,elemFormat);
   return(0);
}

/******************************************************************************/
/* submit element right hand side                                             */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_sumInElemRHS(HYPRE_FEI_Impl *fei, int elemBlock, int elemID,
                           int *elemConn, double *elemLoad)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->sumInElemRHS(elemBlock, elemID, elemConn, elemLoad);
   return(0);
}

/******************************************************************************/
/* signal completion of loading                                               */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_loadComplete(HYPRE_FEI_Impl *fei)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->loadComplete();
   return(0);
}

/******************************************************************************/
/* solve the linear system                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_solve(HYPRE_FEI_Impl *fei, int *status)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->solve(status);
   return(0);
}

/******************************************************************************/
/* get the iteration count                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_iterations(HYPRE_FEI_Impl *fei, int *iterTaken)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->iterations(iterTaken);
   return(0);
}

/******************************************************************************/
/* compute residual norm of the solution                                      */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_residualNorm(HYPRE_FEI_Impl *fei, int whichNorm, int numFields,
                           int* fieldIDs, double* norms)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->residualNorm(whichNorm, numFields, fieldIDs, norms);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (number of nodes)                            */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_getNumBlockActNodes(HYPRE_FEI_Impl *fei, int blockID, int *nNodes)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getNumBlockActNodes(blockID, nNodes);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (number of equations)                        */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_getNumBlockActEqns(HYPRE_FEI_Impl *fei, int blockID, int *nEqns)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getNumBlockActEqns(blockID, nEqns);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (node ID list)                               */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_getBlockNodeIDList(HYPRE_FEI_Impl *fei, int blockID, int numNodes,
                                 int *nodeIDList)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getBlockNodeIDList(blockID, numNodes, nodeIDList);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (actual solution)                            */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_getBlockNodeSolution(HYPRE_FEI_Impl *fei, int blockID, int numNodes,
                                   int *nodeIDList, int *solnOffsets, 
                                   double *solnValues)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getBlockNodeSolution(blockID, numNodes, nodeIDList, solnOffsets, 
                              solnValues);
   return(0);
}

/******************************************************************************/
/* initialze constraints                                                      */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_initCRMult(HYPRE_FEI_Impl *fei, int CRListLen, int *CRNodeList,
                         int *CRFieldList, int *CRID)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initCRMult(CRListLen, CRNodeList, CRFieldList, CRID);
   return(0);
}

/******************************************************************************/
/* load constraints                                                           */
/*----------------------------------------------------------------------------*/

extern "C"
int HYPRE_FEI_loadCRMult(HYPRE_FEI_Impl *fei, int CRID, int CRListLen, 
                         int *CRNodeList, int *CRFieldList, double *CRWeightList, 
                         double CRValue)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->loadCRMult(CRID, CRListLen, CRNodeList, CRFieldList, CRWeightList, 
                    CRValue);
   return(0);
}

