/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.3 $
 ***********************************************************************EHEADER*/


#ifndef _hypre_cfei_h_
#define _hypre_cfei_h_

struct HYPRE_FEI_struct {
   void* fei_;
};
typedef struct HYPRE_FEI_struct HYPRE_FEI_Impl;

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_FEI_Impl *HYPRE_FEI_create( MPI_Comm comm );
int HYPRE_FEI_destroy(HYPRE_FEI_Impl* fei);
int HYPRE_FEI_parameters(HYPRE_FEI_Impl *fei, int numParams, char **paramString);
int HYPRE_FEI_setSolveType(HYPRE_FEI_Impl *fei, int solveType);
int HYPRE_FEI_initFields(HYPRE_FEI_Impl *fei, int numFields, int *fieldSizes, 
                         int *fieldIDs);
int HYPRE_FEI_initElemBlock(HYPRE_FEI_Impl *fei, int elemBlockID, int numElements,
                            int numNodesPerElement, int *numFieldsPerNode,
                            int **nodalFieldIDs, int numElemDOFFieldsPerElement,
                            int *elemDOFFieldIDs, int interleaveStrategy);
int HYPRE_FEI_initElem(HYPRE_FEI_Impl *fei, int elemBlockID, int elemID, 
                            int *elemConn);
int HYPRE_FEI_initSharedNodes(HYPRE_FEI_Impl *fei, int nShared, int *sharedIDs, 
                              int *sharedLeng, int **sharedProcs);
int HYPRE_FEI_initComplete(HYPRE_FEI_Impl *fei);
int HYPRE_FEI_resetSystem(HYPRE_FEI_Impl *fei, double s);
int HYPRE_FEI_resetMatrix(HYPRE_FEI_Impl *fei, double s);
int HYPRE_FEI_resetRHSVector(HYPRE_FEI_Impl *fei, double s);
int HYPRE_FEI_resetInitialGuess(HYPRE_FEI_Impl *fei, double s);
int HYPRE_FEI_loadNodeBCs(HYPRE_FEI_Impl *fei, int nNodes, int *nodeIDs, 
                          int fieldID, double **alpha, double **beta, double **gamma);
int HYPRE_FEI_sumInElem(HYPRE_FEI_Impl *fei, int elemBlock, int elemID, int *elemConn,
                        double **elemStiff, double *elemLoad, int elemFormat);
int HYPRE_FEI_sumInElemMatrix(HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                              int* elemConn, double **elemStiffness, int elemFormat);
int HYPRE_FEI_sumInElemRHS(HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                           int *elemConn, double *elemLoad);
int HYPRE_FEI_loadComplete(HYPRE_FEI_Impl *fei);
int HYPRE_FEI_solve(HYPRE_FEI_Impl *fei, int *status);
int HYPRE_FEI_iterations(HYPRE_FEI_Impl *fei, int *iterTaken);
int HYPRE_FEI_residualNorm(HYPRE_FEI_Impl *fei, int whichNorm, int numFields, 
                           int* fieldIDs, double* norms);
int HYPRE_FEI_getNumBlockActNodes(HYPRE_FEI_Impl *fei, int blockID, int *nNodes);
int HYPRE_FEI_getNumBlockActEqns(HYPRE_FEI_Impl *fei, int blockID, int *nEqns);
int HYPRE_FEI_getBlockNodeIDList(HYPRE_FEI_Impl *fei, int blockID, int numNodes, 
                                 int *nodeIDList);
int HYPRE_FEI_getBlockNodeSolution(HYPRE_FEI_Impl *fei, int blockID, int numNodes, 
                                   int *nodeIDList, int *solnOffsets, double *solnValues);
int HYPRE_FEI_initCRMult(HYPRE_FEI_Impl *fei, int CRListLen, int *CRNodeList,
                         int *CRFieldList, int *CRID);
int HYPRE_FEI_loadCRMult(HYPRE_FEI_Impl *fei, int CRID, int CRListLen, int *CRNodeList,
                         int *CRFieldList, double *CRWeightList, double CRValue);

#ifdef __cplusplus
}
#endif

#endif

