/**************************************************************************
  Module:  LLNL_FEI_Impl.cxx
  Purpose: custom implementation of the FEI
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
/*
#include "utilities.h"
#include "HYPRE.h"
#include "LLNL_FEI_Fei.h"
#include "LLNL_FEI_LSCore.h"
#include "LLNL_FEI_Solver.h"
#include "LLNL_FEI_Matrix.h"
*/
#include "LLNL_FEI_Impl.h"

/*-------------------------------------------------------------------------
 local defines 
 -------------------------------------------------------------------------*/

#define SOLVERLOCK 1024

/**************************************************************************
 LLNL_FEI_Impl is the top level finite element interface.  
 **************************************************************************/

/**************************************************************************
 Constructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Impl::LLNL_FEI_Impl( MPI_Comm comm )
{
   mpiComm_     = comm;
   feiPtr_      = new LLNL_FEI_Fei(comm);
   solverPtr_   = NULL;
   lscPtr_      = NULL;
   matPtr_      = NULL;
   solverLibID_ = 0;
}

/**************************************************************************
 destructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Impl::~LLNL_FEI_Impl()
{
   if (feiPtr_    != NULL) delete feiPtr_;
   if (solverPtr_ != NULL) delete solverPtr_;
   if (lscPtr_    != NULL) delete lscPtr_;
}

/**************************************************************************
 parameter function 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::parameters(int numParams, char **paramString)
{
   int  i, iOne=1;
   char param1[100], param2[100], *param3;
   SolverLib_t solver;

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(paramString[i], "%s", param1);
      if ( !strcmp( param1, "externalSolver" ) )
      {
         if ( (solverLibID_ & SOLVERLOCK) == 0 )
         {
            sscanf(paramString[i], "%s %s", param1, param2);
            if ( !strcmp( param2, "HYPRE" ) ) solverLibID_ = 1;
            else                              solverLibID_ = 0;
         }
      }
   }
   solverLibID_ |= SOLVERLOCK;
   if ( (solverLibID_ - SOLVERLOCK) > 0 ) 
   {
      if ( lscPtr_ != NULL ) delete lscPtr_;
      if ( solverPtr_ != NULL ) 
      {
         delete solverPtr_;
         solverPtr_ = NULL;
      }
      //if ( solverLibID_ == (SOLVERLOCK+1) ) solver = HYPRE;
      param3 = new char[30];
      strcpy( param3, "matrixNoOverlap" );
      feiPtr_->parameters(iOne,&param3);
      delete [] param3;
      solver = HYPRE;
      lscPtr_ = new LLNL_FEI_LSCore(solver);
   }
   else 
   {
      if ( solverPtr_ != NULL ) delete solverPtr_;
      if ( lscPtr_ != NULL ) 
      {
         delete lscPtr_;
         lscPtr_ = NULL;
      }
      solverPtr_ = new LLNL_FEI_Solver(mpiComm_);
   }
   feiPtr_->parameters(numParams,paramString);
   if (solverPtr_ != NULL) solverPtr_->parameters(numParams,paramString); 
   if (lscPtr_    != NULL) lscPtr_->parameters(numParams,paramString); 
   return 0;
}

/**************************************************************************
 set solver type 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::setSolveType(int solveType)
{ 
   (void) solveType; 
   return 0;
}

/**************************************************************************
 initialize the fields
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::initFields(int numFields, int *fieldSizes, int *fieldIDs)
{
   return feiPtr_->initFields(numFields,fieldSizes,fieldIDs);
}

/**************************************************************************
 initialize element blocks
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::initElemBlock(int elemBlockID, int numElements, 
                      int numNodesPerElement, int *numFieldsPerNode, 
                      int **nodalFieldIDs, int numElemDOFFieldsPerElement, 
                      int *elemDOFFieldIDs, int interleaveStrategy)
{
   return feiPtr_->initElemBlock(elemBlockID, numElements, 
                   numNodesPerElement, numFieldsPerNode, nodalFieldIDs, 
                   numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                   interleaveStrategy);
}

/**************************************************************************
 initialize an element
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::initElem(int elemBlockID, int elemID, int *elemConn) 
{
   (void) elemBlockID;
   (void) elemID;
   (void) elemConn;
   return 0;
}

/**************************************************************************
 initialize shared nodes
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::initSharedNodes(int nShared, int *sharedIDs, 
                                   int *sharedLeng, int **sharedProcs)
{
   return feiPtr_->initSharedNodes(nShared,sharedIDs,sharedLeng,
                                   sharedProcs);
}

/**************************************************************************
 initialize complete 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::initComplete()
{
   return feiPtr_->initComplete();
}

/**************************************************************************
 reset the matrix and vectors 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::resetSystem(double s)
{
   return feiPtr_->resetSystem(s);
}

/**************************************************************************
 reset the matrix 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::resetMatrix(double s)
{
   return feiPtr_->resetMatrix(s);
}

/**************************************************************************
 reset the rhs vector 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::resetRHSVector(double s)
{
   return feiPtr_->resetRHSVector(s);
}

/**************************************************************************
 reset the initial solution
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::resetInitialGuess(double s) 
{
   return feiPtr_->resetInitialGuess(s);
}

/**************************************************************************
 load boundary conditions
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::loadNodeBCs(int nNodes, int *nodeIDs, int fieldID, 
                             double **alpha, double **beta, double **gamma)
{
   return feiPtr_->loadNodeBCs(nNodes,nodeIDs,fieldID,alpha,beta,gamma);
}

/**************************************************************************
 put element matrices into global matrix
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::sumInElem(int elemBlock, int elemID, int *elemConn, 
                       double **elemStiff, double *elemLoad, int elemFormat)
{
   return feiPtr_->sumInElem(elemBlock,elemID,elemConn,elemStiff,elemLoad,
   elemFormat);
}

/**************************************************************************
 put element matrices into global matrix
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::sumInElemMatrix(int elemBlock, int elemID, int* elemConn, 
                                   double **elemStiffness, int elemFormat)
{
   return feiPtr_->sumInElemMatrix(elemBlock,elemID,elemConn,elemStiffness,
                                   elemFormat);
}

/**************************************************************************
 put element right hand side to global rhs
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::sumInElemRHS(int elemBlock, int elemID, int *elemConn,
                                double *elemLoad)
{
   return feiPtr_->sumInElemRHS(elemBlock,elemID,elemConn,elemLoad);
}

/**************************************************************************
 load complete
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::loadComplete()
{
   return feiPtr_->loadComplete();
}

/**************************************************************************
 get iteration count
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::iterations(int *iterTaken) 
{
   return solverPtr_->iterations(iterTaken);
}

/**************************************************************************
 get active nodes
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::getNumBlockActNodes(int blockID, int *nNodes)
{
   return feiPtr_->getNumBlockActNodes(blockID,nNodes);
}

/**************************************************************************
 get number of active equations
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::getNumBlockActEqns(int blockID, int *nEqns)
{
   return feiPtr_->getNumBlockActEqns(blockID,nEqns);
}

/**************************************************************************
 get node IDs
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::getBlockNodeIDList(int blockID, int numNodes, 
                                      int *nodeIDList)
{
   return feiPtr_->getBlockNodeIDList(blockID,numNodes,nodeIDList);
}

/**************************************************************************
 get solution
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::getBlockNodeSolution(int blockID, int numNodes, 
                      int *nodeIDList, int *solnOffsets, double *solnValues)
{
   return feiPtr_->getBlockNodeSolution(blockID,numNodes,nodeIDList,
                                        solnOffsets,solnValues);
}

/**************************************************************************
 initialize constraint relations
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::initCRMult(int CRListLen,int *CRNodeList,int *CRFieldList,
                              int *CRID)
{
   return feiPtr_->initCRMult(CRListLen,CRNodeList,CRFieldList,CRID);
}

/**************************************************************************
 load constraint relations
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::loadCRMult(int CRID, int CRListLen, int *CRNodeList, 
                   int *CRFieldList, double *CRWeightList, double CRValue)
{
   return feiPtr_->loadCRMult(CRID,CRListLen,CRNodeList,CRFieldList,
                              CRWeightList,CRValue);
}

/**************************************************************************
 solve 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::solve(int *status)
{
   double *rhsVector, *solnVector;

   if ((solverLibID_ & SOLVERLOCK) != 0) solverLibID_ -= SOLVERLOCK; 
   feiPtr_->getRHSVector(&rhsVector);
   feiPtr_->getSolnVector(&solnVector);
   feiPtr_->getMatrix(&matPtr_);
   if ( solverPtr_ != NULL )
   {
      solverPtr_->loadRHSVector(rhsVector);
      solverPtr_->loadSolnVector(solnVector);
      solverPtr_->loadMatrix(matPtr_);
      solverPtr_->solve(status);
   }
   else if (lscPtr_ != NULL)
   {
      int    localNRows, *diagIA, *diagJA, *indices, *offsets, rowSize;
      int    extNRows, *offdIA, *offdJA, *colMap, maxRowSize, *colInds;
      int    i, j, rowInd, one=1, iterations, status, mypid;
      double *diagAA, *offdAA, *colVals;
      char   format[20];

      MPI_Comm_rank(mpiComm_, &mypid);
      strcpy( format, "HYPRE" );
      matPtr_->getLocalMatrix(&localNRows, &diagIA, &diagJA, &diagAA);
      matPtr_->getExtMatrix(&extNRows, &offdIA, &offdJA, &offdAA, &colMap);
      offsets = matPtr_->getEqnOffsets();
      lscPtr_->setGlobalOffsets(localNRows, NULL, offsets, NULL);
      maxRowSize = 0;
      for ( i = 0; i < localNRows; i++ ) 
      {
         rowSize = diagIA[i+1] - diagIA[i];
         if (offdIA != NULL ) rowSize += offdIA[i+1] - offdIA[i];
         if (rowSize > maxRowSize) maxRowSize = rowSize;
      }
      if ( maxRowSize > 0 )
      {
         colInds = new int[maxRowSize];
         colVals = new double[maxRowSize];
      }
      for ( i = 0; i < localNRows; i++ ) 
      {
         rowSize = 0;
         for ( j = diagIA[i]; j < diagIA[i+1]; j++ ) 
         {
            colInds[rowSize] = diagJA[j] + offsets[mypid]; 
            colVals[rowSize++] = diagAA[j]; 
         }
         if ( offdIA != NULL )
         {
            for ( j = offdIA[i]; j < offdIA[i+1]; j++ ) 
            {
               colInds[rowSize] = colMap[offdJA[j]-localNRows]; 
               colVals[rowSize++] = offdAA[j]; 
            }
         }
         rowInd = offsets[mypid] + i;
         lscPtr_->putIntoSystemMatrix(one, &rowInd, rowSize, 
                      (const int *) colInds, (const double* const*) &colVals); 
      }
      if ( maxRowSize > 0 )
      {
         delete [] colInds;
         delete [] colVals;
      }
      if ( localNRows > 0 ) indices = new int[localNRows];
      for ( i = 0; i < localNRows; i++ ) indices[i] = i + offsets[mypid];
      lscPtr_->putIntoRHSVector(localNRows, (const double *) rhsVector,
                                (const int *) indices);
      lscPtr_->putInitialGuess((const int *) indices, 
                               (const double *) solnVector, localNRows);
      lscPtr_->matrixLoadComplete();
      lscPtr_->solve(&status,&iterations);
      lscPtr_->getSolution(solnVector, localNRows);
      if (localNRows > 0) delete [] indices;
   }
   feiPtr_->disassembleSolnVector();
   return 0;
}

/**************************************************************************
 residual norm calculation 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::residualNorm(int whichNorm, int numFields, int *fieldIDs, 
                              double *norms )
{
   (void) numFields;
   (void) fieldIDs;
   double *solnVec, *rhsVec;
   feiPtr_->getSolnVector(&solnVec);
   feiPtr_->getRHSVector(&rhsVec);
   matPtr_->residualNorm(whichNorm,solnVec,rhsVec,norms);
   return 0;
}


