/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***************************************************************************
  Module:  FEI_HYPRE_impl.h
  Purpose: local implementation of the FEI/LSC 
 ***************************************************************************/

#ifndef __FEI_HYPRE_IMPL_H__
#define __FEI_HYPRE_IMPL_H__

/**************************************************************************
 definition of the class to capture the FEI information 
---------------------------------------------------------------------------*/

class FEI_HYPRE_Elem_Block {
 
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

   FEI_HYPRE_Elem_Block(int blockID);
   ~FEI_HYPRE_Elem_Block();
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

class FEI_HYPRE_Impl 
{
   MPI_Comm mpiComm_;
   int      mypid_;
   int      outputLevel_;
   int      numBlocks_;
   FEI_HYPRE_Elem_Block **elemBlocks_;

   int    numLocalNodes_;
   int    numExtNodes_;
   int    nodeDOF_;
   int    *nodeGlobalIDs_;
   int    *nodeExtNewGlobalIDs_;
   int    *globalNodeOffsets_;

   int    numSharedNodes_;
   int    *sharedNodeIDs_;
   int    *sharedNodeNProcs_;
   int    **sharedNodeProcs_;

   int    nRecvs_;
   int    *recvLengs_;
   int    *recvProcs_;
   int    **recvProcIndices_;

   int    nSends_;
   int    *sendLengs_;
   int    *sendProcs_;
   int    **sendProcIndices_;

   int    solverID_;
   int    krylovMaxIterations_;
   double krylovTolerance_;
   int    krylovAbsRel_;
   int    krylovIterations_;
   double krylovResidualNorm_;
   int    gmresDim_;

   int    *diagIA_;
   int    *diagJA_;
   double *diagAA_;
   int    *offdIA_;
   int    *offdJA_;
   double *offdAA_;
   double *diagonal_;

   int    numBCNodes_;
   int    *BCNodeIDs_;
   double **BCNodeAlpha_;
   double **BCNodeBeta_;
   double **BCNodeGamma_;

   double *solnVector_;
   double *rhsVector_;

   int    FLAG_PrintMatrix_;
   int    FLAG_LoadComplete_;
   double TimerLoad_;
   double TimerLoadStart_;
   double TimerSolve_;
   double TimerSolveStart_;

public :

   FEI_HYPRE_Impl(MPI_Comm comm);
   ~FEI_HYPRE_Impl();
   int  parameters(int numParams, char **paramString);

   int  setSolveType(int solveType) {(void) solveType; return 0;}

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

   int  solve(int *status);

   int  iterations(int *iterTaken) {*iterTaken = krylovIterations_; return 0;}

   int  residualNorm(int whichNorm, int numFields, int* fieldIDs,
                     double* norms);

   int  getNumBlockActNodes(int blockID, int *nNodes);
   int  getNumBlockActEqns(int blockID, int *nEqns);
   int  getBlockNodeIDList(int blockID, int numNodes, int *nodeIDList);
   int  getBlockNodeSolution(int blockID, int numNodes, int *nodeIDList,
                             int *solnOffsets, double *solnValues);

private:
   void assembleRHSVector();
   void assembleSolnVector();
   void disassembleSolnVector();
   void buildGlobalMatrixVector();
   void matvec(double *x, double *y);
   int  solveUsingCG();
   int  solveUsingGMRES();
   int  solveUsingCGS();
   int  solveUsingBicgstab();
   int  solveUsingSuperLU();
   void IntSort(int *, int, int);
   void IntSort2a(int *, double *, int, int);
   void PVectorInterChange(double *x);
   void PVectorReverseChange(double *x);
   void printLinearSystem();

public:
   static void IntSort2(int *, int *, int, int);
};

#endif /* endif for __FEI_HYPRE_IMPL_H__ */

