/***************************************************************************
  Module:  LLNL_FEI_Impl.h
  Purpose: custom implementation of the FEI 
 ***************************************************************************/

#ifndef _LLNL_FEI_IMPL_H_
#define _LLNL_FEI_IMPL_H_

#include "HYPRE.h"
#include "utilities.h"
#include "LLNL_FEI_LSCore.h"
#include "LLNL_FEI_Fei.h"
#include "LLNL_FEI_Solver.h"
#include "LLNL_FEI_Matrix.h"

/**************************************************************************
 definition of the class to capture the FEI information 
---------------------------------------------------------------------------*/

class LLNL_FEI_Impl
{
   MPI_Comm        mpiComm_;
   LLNL_FEI_Fei    *feiPtr_;
   LLNL_FEI_Solver *solverPtr_;
   LLNL_FEI_Matrix *matPtr_;
   LLNL_FEI_LSCore *lscPtr_;
   int             solverLibID_;

public :

   LLNL_FEI_Impl(MPI_Comm comm);
   ~LLNL_FEI_Impl();
   int parameters(int numParams, char **paramString);

   int setSolveType(int solveType);

   int initFields(int numFields, int *fieldSizes, int *fieldIDs);

   int initElemBlock(int elemBlockID, int numElements, 
                   int numNodesPerElement, int *numFieldsPerNode, 
                   int **nodalFieldIDs, int numElemDOFFieldsPerElement, 
                   int *elemDOFFieldIDs, int interleaveStrategy);

   int initElem(int elemBlockID, int elemID, int *elemConn);

   int initSharedNodes(int nShared, int *sharedIDs, int *sharedLeng, 
                       int **sharedProcs);

   int initComplete();

   int resetSystem(double s);

   int resetMatrix(double s);

   int resetRHSVector(double s);

   int resetInitialGuess(double s);

   int loadNodeBCs(int nNodes, int *nodeIDs, int fieldID, 
                   double **alpha, double **beta, double **gamma);

   int sumInElem(int elemBlock, int elemID, int *elemConn, 
                   double **elemStiff, double *elemLoad, int elemFormat);

   int sumInElemMatrix(int elemBlock, int elemID, int* elemConn, 
                       double **elemStiffness, int elemFormat);

   int sumInElemRHS(int elemBlock, int elemID, int *elemConn,
                    double *elemLoad);

   int loadComplete();

   int solve(int *status);

   int iterations(int *iterTaken);

   int residualNorm(int whichNorm, int numFields, int* fieldIDs,
                    double* norms);

   int getNumBlockActNodes(int blockID, int *nNodes);

   int getNumBlockActEqns(int blockID, int *nEqns);

   int getBlockNodeIDList(int blockID, int numNodes, int *nodeIDList);

   int getBlockNodeSolution(int blockID, int numNodes, int *nodeIDList,
                            int *solnOffsets, double *solnValues);

   int initCRMult(int CRListLen,int *CRNodeList,int *CRFieldList,
                  int *CRID);

   int loadCRMult(int CRID, int CRListLen, int *CRNodeList, 
                  int *CRFieldList, double *CRWeightList, double CRValue);
};

#endif /* endif for _LLNL_FEI_IMPL_H_ */

