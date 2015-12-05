/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/***************************************************************************
  Module:  LLNL_FEI_Fei.h
  Purpose: custom implementation of the FEI
 ***************************************************************************/

#ifndef _LLNL_FEI_FEI_H_
#define _LLNL_FEI_FEI_H_

#include "LLNL_FEI_Matrix.h"

/**************************************************************************
 definition of the class to capture the FEI information 
---------------------------------------------------------------------------*/

class LLNL_FEI_Elem_Block {
 
   int    blockID_;
   int    numElems_;
   int    nodeDOF_;
   int    *elemIDs_;
   int    **elemNodeLists_;
   int    *sortedIDs_;
   int    *sortedIDAux_;
   double **elemMatrices_;
   double **rhsVectors_;
   double **solnVectors_;
   int    nodesPerElem_;
   int    currElem_;
   double *tempX_;
   double *tempY_;

public :

   LLNL_FEI_Elem_Block(int blockID);
   ~LLNL_FEI_Elem_Block();
   int    getElemBlockID()     {return blockID_;}
   int    getNumElems()        {return numElems_;}
   int    getElemNumNodes()    {return nodesPerElem_;}
   int    getCurrentElem()     {return currElem_;}
   int    **getElemNodeLists() {return elemNodeLists_;}
   double **getElemMatrices()  {return elemMatrices_;}
   double **getRHSVectors()    {return rhsVectors_;}
   double **getSolnVectors()   {return solnVectors_;}

   int    initialize(int numElems, int nNodesPerElem, int nodeDOF);
   int    reset();
   int    resetRHSVectors(double s);
   int    resetSolnVectors(double s);

   int    loadElemInfo(int elemID, int *elemNodeList, double **elemStiff,   
                       double *elemRHS);
   int    loadElemMatrix(int elemID, int *elemNodeList, double **elemStiff);
   int    loadElemRHS(int elemID, double *elemRHS);

   int    checkLoadComplete();
};

/**************************************************************************
 definition of the class to capture the FEI information 
---------------------------------------------------------------------------*/

class LLNL_FEI_Fei 
{
   MPI_Comm mpiComm_;
   int      mypid_;
   int      outputLevel_;
   int      numBlocks_;
   LLNL_FEI_Elem_Block **elemBlocks_;

   int    numLocalNodes_;
   int    numExtNodes_;
   int    nodeDOF_;
   int    *nodeGlobalIDs_;
   int    *nodeExtNewGlobalIDs_;
   int    *globalNodeOffsets_;
   int    *globalCROffsets_;

   int    numCRMult_;
   int    CRListLen_;
   int    **CRNodeLists_;
   int    CRFieldID_;
   double **CRWeightLists_;
   double *CRValues_;

   int    numSharedNodes_;
   int    *sharedNodeIDs_;
   int    *sharedNodeNProcs_;
   int    **sharedNodeProcs_;

   int    nRecvs_;
   int    *recvLengs_;
   int    *recvProcs_;
   int    *recvProcIndices_;

   int    nSends_;
   int    *sendLengs_;
   int    *sendProcs_;
   int    *sendProcIndices_;

   int    numBCNodes_;
   int    *BCNodeIDs_;
   double **BCNodeAlpha_;
   double **BCNodeBeta_;
   double **BCNodeGamma_;

   LLNL_FEI_Matrix *matPtr_;
   double *solnVector_;
   double *rhsVector_;

   int    FLAG_LoadComplete_;
   double TimerLoad_;
   double TimerLoadStart_;
   double TimerSolve_;
   double TimerSolveStart_;

public :

   LLNL_FEI_Fei(MPI_Comm comm);
   ~LLNL_FEI_Fei();
   int  parameters(int numParams, char **paramString);

   int  initFields(int numFields, int *fieldSizes, int *fieldIDs);

   int  initElemBlock(int elemBlockID, int numElements, 
                      int numNodesPerElement, int *numFieldsPerNode, 
                      int **nodalFieldIDs, int numElemDOFFieldsPerElement, 
                      int *elemDOFFieldIDs, int interleaveStrategy);

   int  initElem(int elemBlockID, int elemID, int *elemConn) 
                      {(void) elemBlockID; (void) elemID; (void) elemConn;
                       return 0;}

   int  initSharedNodes(int nShared, int *sharedIDs, int *sharedLeng, 
                        int **sharedProcs);

   int  initComplete() {return 0;}

   int  resetSystem(double s);

   int  resetMatrix(double s);

   int  resetRHSVector(double s);

   int  resetInitialGuess(double s);

   int  loadNodeBCs(int nNodes, int *nodeIDs, int fieldID, double **alpha, 
                    double **beta, double **gamma);

   int  sumInElem(int elemBlock, int elemID, int *elemConn, 
                  double **elemStiff, double *elemLoad, int elemFormat);

   int  sumInElemMatrix(int elemBlock, int elemID, int* elemConn, 
                        double **elemStiffness, int elemFormat);

   int  sumInElemRHS(int elemBlock, int elemID, int *elemConn,
                     double *elemLoad);

   int  loadComplete();

   int  getNumBlockActNodes(int blockID, int *nNodes);
   int  getNumBlockActEqns(int blockID, int *nEqns);
   int  getBlockNodeIDList(int blockID, int numNodes, int *nodeIDList);
   int  getBlockNodeSolution(int blockID, int numNodes, int *nodeIDList,
                             int *solnOffsets, double *solnValues);

   int  initCRMult(int CRListLen,int *CRNodeList,int *CRFieldList,int *CRID);
   int  loadCRMult(int CRID, int CRListLen, int *CRNodeList, int *CRFieldList,
                   double *CRWeightList, double CRValue);

private:
   void assembleRHSVector();
   void assembleSolnVector();
   void buildGlobalMatrixVector();
   void IntSort(int *, int, int);
   void IntSort2a(int *, double *, int, int);
   void scatterDData(double *x);
   void gatherAddDData(double *x);
   void gatherIData(int *x, int *y);
   void gatherDData(double *x, double *y);
   void sortSharedNodes();
   void composeOrderedNodeIDList(int **nodeIDs, int **nodeIDAux, 
                                 int *totalNNodes, int *CRNNodes);
   void findSharedNodeProcs(int *nodeIDs, int *nodeIDAux, int totalNNodes, 
                            int CRNNodes, int **sharedNodePInfo) ;
   void findSharedNodeOwners( int *sharedNodePInfo );
   void setupCommPattern( int *sharedNodePInfo );
   void modifyCommPattern(int *nrecvs, int **recvlengs, int **recvprocs, 
                          int **recvindices, int *nsends, int **sendlengs, 
                          int **sendprocs, int **sendIndices);
   void fetchExtEqnList(int **eqnList);

public:
   void   getRHSVector(double **rhs) {(*rhs) = rhsVector_;}
   void   getSolnVector(double **soln) {(*soln) = solnVector_;}
   void   getMatrix(LLNL_FEI_Matrix **mat);
   void   disassembleSolnVector(double *);
   static void IntSort2(int *, int *, int, int);
};

#endif /* endif for _LLNL_FEI_FEI_H_ */

