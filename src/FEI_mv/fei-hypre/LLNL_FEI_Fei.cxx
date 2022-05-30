/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**************************************************************************
  Module:  LLNL_FEI_Fei.cxx
  Purpose: custom implementation of the FEI
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "LLNL_FEI_Fei.h"

/**************************************************************************
 **************************************************************************
 Each element block contains a number of elements of the same type (e.g. 
 hex or tet element).  For this implementation, all element block should
 have the same number of degree of freedom per node. 
 **************************************************************************/

/**************************************************************************
 Constructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Elem_Block::LLNL_FEI_Elem_Block( int blockID )
{
   blockID_       = blockID;
   currElem_      = 0;
   numElems_      = 0;
   nodesPerElem_  = 0;
   nodeDOF_       = 0;
   elemIDs_       = NULL;
   elemNodeLists_ = NULL;
   elemMatrices_  = NULL;
   rhsVectors_    = NULL;
   solnVectors_   = NULL;
   tempX_         = NULL;
   tempY_         = NULL;
   sortedIDs_     = NULL;
   sortedIDAux_   = NULL;
}

/**************************************************************************
 destructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Elem_Block::~LLNL_FEI_Elem_Block()
{
   int iE;

   if ( elemIDs_ != NULL ) delete [] elemIDs_;
   if ( elemNodeLists_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( elemNodeLists_[iE] != NULL ) 
            delete [] elemNodeLists_[iE];
      delete [] elemNodeLists_;
   }
   if ( elemMatrices_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( elemMatrices_[iE] != NULL ) 
            delete [] elemMatrices_[iE];
      delete [] elemMatrices_;
   }
   if ( rhsVectors_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( rhsVectors_[iE] != NULL ) 
            delete [] rhsVectors_[iE];
      delete [] rhsVectors_;
   }
   if ( solnVectors_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( solnVectors_[iE] != NULL ) 
            delete [] solnVectors_[iE];
      delete [] solnVectors_;
   }
   if ( sortedIDs_   != NULL ) delete [] sortedIDs_;
   if ( sortedIDAux_ != NULL ) delete [] sortedIDAux_;
   if ( tempX_ != NULL ) delete [] tempX_;
   if ( tempY_ != NULL ) delete [] tempY_;
}

/**************************************************************************
 initialization 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::initialize(int numElements, int numNodesPerElement,
                                   int dofPerNode)
{
   int iE;

   if ( elemIDs_ != NULL ) delete [] elemIDs_;
   if ( elemNodeLists_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( elemNodeLists_[iE] != NULL ) 
            delete [] elemNodeLists_[iE];
      delete [] elemNodeLists_;
   }
   if ( elemMatrices_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( elemMatrices_[iE] != NULL ) 
            delete [] elemMatrices_[iE];
      delete [] elemMatrices_;
   }
   if ( rhsVectors_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( rhsVectors_[iE] != NULL ) 
            delete [] rhsVectors_[iE];
      delete [] rhsVectors_;
   }
   if ( solnVectors_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         if ( solnVectors_[iE] != NULL ) 
            delete [] solnVectors_[iE];
      delete [] solnVectors_;
   }
   numElems_      = numElements;
   nodesPerElem_  = numNodesPerElement;
   nodeDOF_       = dofPerNode;
   currElem_      = 0;
   elemIDs_       = new int[numElems_];
   elemNodeLists_ = new int*[numElems_];
   for ( iE = 0; iE < numElems_; iE++ ) elemNodeLists_[iE] = NULL;
   elemMatrices_ = new double*[numElems_];
   for ( iE = 0; iE < numElems_; iE++ ) elemMatrices_[iE] = NULL;
   rhsVectors_ = new double*[numElems_];
   for ( iE = 0; iE < numElems_; iE++ ) rhsVectors_[iE] = NULL;
   solnVectors_ = new double*[numElems_];
   for ( iE = 0; iE < numElems_; iE++ ) solnVectors_[iE] = NULL;
   return 0;
}

/**************************************************************************
 reset the system for reloading (no reinitialization needed) 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::reset()
{
   int iE;

   if ( elemNodeLists_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
      {
         if ( elemNodeLists_[iE] != NULL ) 
            delete [] elemNodeLists_[iE];
         elemNodeLists_[iE] = NULL;
      }
   }
   if ( elemMatrices_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
      {
         if ( elemMatrices_[iE] != NULL ) 
            delete [] elemMatrices_[iE];
         elemMatrices_[iE] = NULL;
      }
   }
   if ( rhsVectors_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
      {
         if ( rhsVectors_[iE] != NULL ) 
            delete [] rhsVectors_[iE];
         rhsVectors_[iE] = NULL;
      }
   }
   currElem_ = 0;
   return 0;
}

/**************************************************************************
 reset the element load vectors
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::resetRHSVectors(double s)
{
   int iE, iD, matDim=nodesPerElem_*nodeDOF_;

   if ( rhsVectors_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         for ( iD = 0; iD < matDim; iD++ ) rhsVectors_[iE][iD] = s; 
   }
   currElem_ = 0;
   return 0;
}

/**************************************************************************
 reset the element solution vectors
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::resetSolnVectors(double s)
{
   int iE, iD, matDim=nodesPerElem_*nodeDOF_;

   if ( solnVectors_ != NULL )
   {
      for ( iE = 0; iE < numElems_; iE++ ) 
         for ( iD = 0; iD < matDim; iD++ ) solnVectors_[iE][iD] = s; 
   }
   currElem_ = 0;
   return 0;
}

/**************************************************************************
 load individual element information 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::loadElemInfo(int elemID, int *elemConn, 
                                     double **elemStiff, double *elemLoad)
{
   if ( currElem_ >= numElems_ )
   {
      printf("LLNL_FEI_Elem_Block::loadElemInfo ERROR : too many elements.\n");
      exit(1);
   }
#if 0
   printf("Loading element %d : ", elemID); 
   for ( int iN2 = 0; iN2 < nodesPerElem_; iN2++ )
      printf("%d ", elemConn[iN2]);
   printf("\n");
#endif
   elemNodeLists_[currElem_] = new int[nodesPerElem_];
   int matDim = nodesPerElem_ * nodeDOF_;
   elemMatrices_[currElem_]  = new double[matDim*matDim];
   rhsVectors_[currElem_]    = new double[matDim];
   if ( solnVectors_[currElem_] != NULL )
      delete [] solnVectors_[currElem_];
   solnVectors_[currElem_]   = new double[matDim];
   elemIDs_[currElem_] = elemID;
   for ( int iN = 0; iN < nodesPerElem_; iN++ )
      elemNodeLists_[currElem_][iN] = elemConn[iN];
   for ( int iM = 0; iM < matDim; iM++ )
      rhsVectors_[currElem_][iM] = elemLoad[iM];
   for ( int iM2 = 0; iM2 < matDim; iM2++ )
      solnVectors_[currElem_][iM2] = 0.0;
   for ( int iM3 = 0; iM3 < matDim; iM3++ )
      for ( int jM = 0; jM < matDim; jM++ )
         elemMatrices_[currElem_][jM*matDim+iM3] = elemStiff[iM3][jM];
   currElem_++;
   return 0;
}

/**************************************************************************
 load individual element matrix only 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::loadElemMatrix(int elemID, int *elemConn, 
                                        double **elemStiff)
{
   if ( currElem_ >= numElems_ )
   {
      printf("LLNL_FEI_Elem_Block::loadElemMatrix ERROR- too many elements.\n");
      exit(1);
   }
#if 0
   printf("Loading element %d : ", elemID); 
   for ( int iN = 0; iN < nodesPerElem_; iN++ )
      printf("%d ", elemConn[iN]);
   printf("\n");
#endif
   elemNodeLists_[currElem_] = new int[nodesPerElem_];
   int matDim = nodesPerElem_ * nodeDOF_;
   elemMatrices_[currElem_]  = new double[matDim*matDim];
   if ( solnVectors_[currElem_] != NULL )
       delete [] solnVectors_[currElem_];
   solnVectors_[currElem_]   = new double[matDim];
   elemIDs_[currElem_] = elemID;
   for ( int iN = 0; iN < nodesPerElem_; iN++ )
      elemNodeLists_[currElem_][iN] = elemConn[iN];
   for ( int iM2 = 0; iM2 < matDim; iM2++ )
      solnVectors_[currElem_][iM2] = 0.0;
   for ( int iM3 = 0; iM3 < matDim; iM3++ )
      for ( int jM = 0; jM < matDim; jM++ )
         elemMatrices_[currElem_][jM*matDim+iM3] = elemStiff[iM3][jM];
   currElem_++;
   return 0;
}

/**************************************************************************
 load individual load information 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::loadElemRHS(int elemID, double *elemLoad)
{
   int iD, iE, matDim=nodesPerElem_*nodeDOF_;

   if ( currElem_ >= numElems_ ) currElem_ = 0;
   if ( numElems_ > 0 && elemID != elemIDs_[currElem_] )
   {
      if ( sortedIDs_ == NULL )
      {
         sortedIDs_   = new int[numElems_];
         sortedIDAux_ = new int[numElems_];
         for ( iE = 0; iE < numElems_; iE++ ) sortedIDs_[iE] = elemIDs_[iE];
         for ( iE = 0; iE < numElems_; iE++ ) sortedIDAux_[iE] = iE;
         LLNL_FEI_Fei::IntSort2(sortedIDs_, sortedIDAux_, 0, numElems_-1);
      }
      currElem_ = hypre_BinarySearch(sortedIDs_,elemID,numElems_);
   }
   if ( rhsVectors_ == NULL ) 
   {
      rhsVectors_ = new double*[numElems_];
      for ( iE = 0; iE < numElems_; iE++ ) rhsVectors_[iE] = NULL;
   }
   if ( rhsVectors_[currElem_] == NULL )
      rhsVectors_[currElem_] = new double[matDim];

   for ( iD = 0; iD < matDim; iD++ ) rhsVectors_[currElem_][iD] = elemLoad[iD];
   currElem_++;
   return 0;
}

/**************************************************************************
 check to see if all element information has been loaded
 -------------------------------------------------------------------------*/
int LLNL_FEI_Elem_Block::checkLoadComplete()
{
   if ( currElem_ != numElems_ ) return 1;
   else
   {
      if ( tempX_ != NULL ) delete [] tempX_;
      if ( tempY_ != NULL ) delete [] tempY_;
      tempX_ = new double[nodesPerElem_*nodeDOF_];
      tempY_ = new double[nodesPerElem_*nodeDOF_];
   }
   return 0;
}

/**************************************************************************
 LLNL_FEI_Fei is the core linear system interface.  Each 
 instantiation supports multiple elememt blocks.
 **************************************************************************/

/**************************************************************************
 Constructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Fei::LLNL_FEI_Fei( MPI_Comm comm )
{
   mpiComm_     = comm;
   MPI_Comm_rank( comm, &mypid_ );
   numBlocks_   = 0;
   elemBlocks_  = NULL;
   outputLevel_ = 0;

   /* -----------------------------------------------------------------
    * node information
    * ----------------------------------------------------------------*/

   numLocalNodes_       = 0;
   numExtNodes_         = 0;
   nodeDOF_             = 1;
   nodeGlobalIDs_       = NULL;
   nodeExtNewGlobalIDs_ = NULL;
   globalNodeOffsets_   = NULL;
   globalCROffsets_     = NULL;

   numSharedNodes_      = 0;
   sharedNodeIDs_       = NULL;
   sharedNodeNProcs_    = NULL;
   sharedNodeProcs_     = NULL;

   numCRMult_           = 0;
   CRListLen_           = 0;
   CRNodeLists_         = NULL;
   CRFieldID_           = -1;
   CRWeightLists_       = NULL;
   CRValues_            = NULL;

   /* -----------------------------------------------------------------
    * communication information
    * ----------------------------------------------------------------*/

   nRecvs_          = 0;
   recvLengs_       = NULL;
   recvProcs_       = NULL;
   recvProcIndices_ = NULL;

   nSends_          = 0;
   sendLengs_       = NULL;
   sendProcs_       = NULL;
   sendProcIndices_ = NULL;

   /* -----------------------------------------------------------------
    * matrix and vector information
    * ----------------------------------------------------------------*/

   matPtr_     = new LLNL_FEI_Matrix(comm);
   solnVector_ = NULL;
   rhsVector_  = NULL;

   /* -----------------------------------------------------------------
    * node boundary condition information
    * ----------------------------------------------------------------*/

   numBCNodes_  = 0;
   BCNodeIDs_   = NULL;
   BCNodeAlpha_ = NULL;
   BCNodeBeta_  = NULL;
   BCNodeGamma_ = NULL;

   /* -----------------------------------------------------------------
    * others
    * ----------------------------------------------------------------*/

   FLAG_LoadComplete_ = 0;
   TimerLoad_         = 0.0;
   TimerLoadStart_    = 0.0;
}

/**************************************************************************
 destructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_Fei::~LLNL_FEI_Fei()
{
   int    iB;
   double internCode=1.0e35;

   if ( outputLevel_ > 2 ) printf("%4d : LLNL_FEI_Fei destructor\n", mypid_);
   resetSystem(internCode);
   if ( matPtr_ != NULL ) delete matPtr_;
   for ( iB = 0; iB < numBlocks_; iB++ )
      if ( elemBlocks_[iB] != NULL ) delete elemBlocks_[iB];
   if ( elemBlocks_ != NULL ) delete [] elemBlocks_;
   if ( solnVector_ != NULL ) delete [] solnVector_;
}

/**************************************************************************
 parameters function
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::parameters(int numParams, char **paramString)
{
   int  i, one=1;
   char param1[256];

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(paramString[i],"%s", param1);
      if ( !strcmp(param1, "outputLevel") )
      {
         sscanf(paramString[i],"%s %d", param1, &outputLevel_);
         if ( outputLevel_ < 0 ) outputLevel_ = 0;
         if ( outputLevel_ > 4 ) outputLevel_ = 4;
         matPtr_->parameters( one, &paramString[i] );
      }
      else if ( !strcmp(param1, "matrixNoOverlap") )
      {
         matPtr_->parameters( one, &paramString[i] );
      }
      else if ( !strcmp(param1, "setDebug") )
      {
         matPtr_->parameters( one, &paramString[i] );
      }
   }
   return 0;
}

/**************************************************************************
 initialize nodal degree of freedom 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::initFields(int numFields, int *fieldSizes, int *fieldIDs)
{
   (void) fieldIDs;
   if ( numFields != 1 )
   {
      printf("%4d : LLNL_FEI_Fei::initFields WARNING -  numFields != 1",
             mypid_);
      printf(" Take field 0.\n"); 
      nodeDOF_ = fieldSizes[0];
      return -1;
   }
   nodeDOF_ = fieldSizes[0];
   return 0;
}

/**************************************************************************
 set element and node information
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::initElemBlock(int elemBlockID, int numElements, 
                      int numNodesPerElement, int *numFieldsPerNode, 
                      int **nodalFieldIDs, int numElemDOFFieldsPerElement, 
                      int *elemDOFFieldIDs, int interleaveStrategy)
{
   (void) numFieldsPerNode;
   (void) nodalFieldIDs;
   (void) numElemDOFFieldsPerElement; 
   (void) elemDOFFieldIDs;
   (void) interleaveStrategy;
   if ( outputLevel_ > 2 ) 
   {
      printf("%4d : LLNL_FEI_Fei::initElemBlock begins... \n", mypid_);
      printf("               elemBlockID  = %d \n", elemBlockID);
      printf("               numElements  = %d \n", numElements);
      printf("               nodesPerElem = %d \n", numNodesPerElement);
      if ( outputLevel_ > 3 ) 
      {
         for ( int iN = 0; iN < numNodesPerElement; iN++ )
         {
            printf("               Node %d has fields : ", iN);
            for ( int iF = 0; iF < numFieldsPerNode[iN]; iF++ )
               printf("%d ", nodalFieldIDs[iN][iF]);
            printf("\n");
         }
         for ( int iE = 0; iE < numElemDOFFieldsPerElement; iE++ )
            printf("               Element field IDs %d = %d\n", iE, 
                   elemDOFFieldIDs[iE]);
      }
   }
   if ( numBlocks_ == 0 )
   {
      elemBlocks_    = new LLNL_FEI_Elem_Block*[1];
      elemBlocks_[0] = new LLNL_FEI_Elem_Block(elemBlockID);
      numBlocks_     = 1;
   }
   else
   {
      for ( int iB = 0; iB < numBlocks_; iB++ )
      {
         if ( elemBlocks_[iB]->getElemBlockID() == elemBlockID )
         {
            printf("%4d : LLNL_FEI_Fei::initElemBlock ERROR - ",mypid_);
            printf("repeated blockID\n");
            exit(1);
         }
      } 
      LLNL_FEI_Elem_Block **tempBlocks = elemBlocks_;
      numBlocks_++;
      elemBlocks_ = new LLNL_FEI_Elem_Block*[numBlocks_];
      for ( int iB2 = 0; iB2 < numBlocks_-1; iB2++ )
         elemBlocks_[iB2] = tempBlocks[iB2];
      elemBlocks_[numBlocks_-1] = new LLNL_FEI_Elem_Block(elemBlockID);
      delete [] tempBlocks;
   }
   elemBlocks_[numBlocks_-1]->initialize(numElements, numNodesPerElement,
                                         nodeDOF_); 
   FLAG_LoadComplete_= 0;
   if ( outputLevel_ > 2 ) 
      printf("%4d : LLNL_FEI_Fei::initElemBlock ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 initialize shared node information
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::initSharedNodes(int nShared, int *sharedIDs,
                                 int *sharedNProcs, int **sharedProcs)
{
   int iN, iP, newNumShared, *oldSharedIDs, *oldSharedNProcs;
   int **oldSharedProcs;

   if ( outputLevel_ > 2 ) 
      printf("%4d : LLNL_FEI_Fei::initSharedNodes begins... \n", mypid_);
   TimerLoadStart_ = MPI_Wtime();
   if ( numSharedNodes_ > 0 )
   {
      newNumShared = numSharedNodes_ + nShared;
      oldSharedIDs = sharedNodeIDs_;
      sharedNodeIDs_ = new int[newNumShared];
      for ( iN = 0; iN < numSharedNodes_; iN++ ) 
         sharedNodeIDs_[iN] = oldSharedIDs[iN];
      for ( iN = 0; iN < nShared; iN++ ) 
         sharedNodeIDs_[iN+numSharedNodes_] = sharedIDs[iN];
      oldSharedNProcs = sharedNodeNProcs_;
      sharedNodeNProcs_ = new int[newNumShared];
      for ( iN = 0; iN < numSharedNodes_; iN++ ) 
         sharedNodeNProcs_[iN] = oldSharedNProcs[iN];
      for ( iN = 0; iN < nShared; iN++ ) 
         sharedNodeNProcs_[iN+numSharedNodes_] = sharedNProcs[iN];
      oldSharedProcs = sharedNodeProcs_;
      sharedNodeProcs_ = new int*[newNumShared];
      for ( iN = 0; iN < numSharedNodes_; iN++ ) 
         sharedNodeProcs_[iN] = oldSharedProcs[iN];
      for ( iN = 0; iN < nShared; iN++ ) 
      {
         sharedNodeProcs_[iN+numSharedNodes_] = new int[sharedNProcs[iN]];
         for ( iP = 0; iP < sharedNProcs[iN]; iP++ ) 
            sharedNodeProcs_[iN+numSharedNodes_][iP] = sharedProcs[iN][iP];
      }
      numSharedNodes_ = newNumShared;
      delete [] oldSharedProcs;
      delete [] oldSharedNProcs;
      delete [] oldSharedIDs;
   }
   else
   {
      numSharedNodes_ = nShared;
      sharedNodeIDs_ = new int[nShared];
      for ( iN = 0; iN < nShared; iN++ ) 
         sharedNodeIDs_[iN] = sharedIDs[iN];
      sharedNodeNProcs_ = new int[nShared];
      for ( iN = 0; iN < nShared; iN++ ) 
         sharedNodeNProcs_[iN] = sharedNProcs[iN];
      sharedNodeProcs_ = new int*[nShared];
      for ( iN = 0; iN < nShared; iN++ ) 
      {
         sharedNodeProcs_[iN] = new int[sharedNProcs[iN]];
         for ( iP = 0; iP < sharedNProcs[iN]; iP++ ) 
            sharedNodeProcs_[iN][iP] = sharedProcs[iN][iP];
      }
   }
   TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
   if ( outputLevel_ > 2 ) 
      printf("%4d : LLNL_FEI_Fei::initSharedNodes ends. \n", mypid_);
   return 0;
}

/**************************************************************************
 reset the system
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::resetSystem(double s)
{
   (void) s;
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetSystem begins...\n", mypid_);

   resetMatrix(s);
   if ( rhsVector_ != NULL ) delete [] rhsVector_; 
   rhsVector_ = NULL;
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetSystem ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 reset the matrix
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::resetMatrix(double s)
{
   int iB, iD;
   double internCode=1.0e35;

   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetMatrix begins...\n", mypid_);
   for ( iB = 0; iB < numBlocks_; iB++ ) elemBlocks_[iB]->reset();
   numLocalNodes_ = 0;
   numExtNodes_   = 0;
   if ( nodeGlobalIDs_       != NULL ) delete [] nodeGlobalIDs_;
   if ( nodeExtNewGlobalIDs_ != NULL ) delete [] nodeExtNewGlobalIDs_;
   if ( globalNodeOffsets_   != NULL ) delete [] globalNodeOffsets_;
   if ( globalCROffsets_     != NULL ) delete [] globalCROffsets_;
   if ( recvLengs_           != NULL ) delete [] recvLengs_;
   if ( recvProcs_           != NULL ) delete [] recvProcs_;
   if ( recvProcIndices_     != NULL ) delete [] recvProcIndices_;
   if ( sendLengs_           != NULL ) delete [] sendLengs_;
   if ( sendProcs_           != NULL ) delete [] sendProcs_;
   if ( sendProcIndices_     != NULL ) delete [] sendProcIndices_;
   if ( matPtr_              != NULL ) delete matPtr_;
   if ( BCNodeAlpha_ != NULL ) 
   {
      for ( iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeAlpha_[iD];
      delete [] BCNodeAlpha_;
   }
   if ( BCNodeBeta_ != NULL ) 
   {
      for ( iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeBeta_[iD];
      delete [] BCNodeBeta_;
   }
   if ( BCNodeGamma_ != NULL ) 
   {
      for ( iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeGamma_[iD];
      delete [] BCNodeGamma_;
   }
   if ( BCNodeIDs_ != NULL ) delete [] BCNodeIDs_; 
   if ( s == internCode )
   {
      if ( CRNodeLists_ != NULL )
      {
         for ( iD = 0; iD < numCRMult_; iD++ )
            if ( CRNodeLists_[iD] != NULL ) delete [] CRNodeLists_[iD];
         delete [] CRNodeLists_;
      }
      if ( CRWeightLists_ != NULL )
      {
         for ( iD = 0; iD < numCRMult_; iD++ )
            if ( CRWeightLists_[iD] != NULL ) delete [] CRWeightLists_[iD];
         delete [] CRWeightLists_;
      }
      if ( CRValues_         != NULL ) delete [] CRValues_;
      numCRMult_           = 0;
      CRListLen_           = 0;
      CRNodeLists_         = NULL;
      CRWeightLists_       = NULL;
      CRValues_            = NULL;
   }
   if ( s == internCode )
   {
      if ( sharedNodeIDs_    != NULL ) delete [] sharedNodeIDs_;
      if ( sharedNodeNProcs_ != NULL ) delete [] sharedNodeNProcs_;
      if ( sharedNodeProcs_  != NULL )
      {
         for ( iD = 0; iD < numSharedNodes_; iD++ )
            if ( sharedNodeProcs_[iD] != NULL ) delete [] sharedNodeProcs_[iD];
         delete [] sharedNodeProcs_;
      }
      numSharedNodes_      = 0;
      sharedNodeIDs_       = NULL;
      sharedNodeNProcs_    = NULL;
      sharedNodeProcs_     = NULL;
   }
   nSends_              = 0;
   nRecvs_              = 0;
   nodeGlobalIDs_       = NULL;
   nodeExtNewGlobalIDs_ = NULL;
   globalNodeOffsets_   = NULL;
   globalCROffsets_     = NULL;
   recvLengs_           = NULL;
   recvProcs_           = NULL;
   recvProcIndices_     = NULL;
   sendLengs_           = NULL;
   sendProcs_           = NULL;
   sendProcIndices_     = NULL;
   matPtr_              = new LLNL_FEI_Matrix(mpiComm_);
   BCNodeIDs_           = NULL;
   BCNodeAlpha_         = NULL;
   BCNodeBeta_          = NULL;
   BCNodeGamma_         = NULL;
   numBCNodes_          = 0;
   TimerLoad_           = 0.0;
   TimerLoadStart_      = 0.0;
   TimerSolve_          = 0.0;
   TimerSolveStart_     = 0.0;
   FLAG_LoadComplete_   = 0;
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetMatrix ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 reset the rhs vector
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::resetRHSVector(double s)
{
   (void) s;
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetRHSVector begins...\n", mypid_);
   for ( int iB = 0; iB < numBlocks_; iB++ ) 
      elemBlocks_[iB]->resetRHSVectors(s);
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetRHSVector ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 reset the solution vector
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::resetInitialGuess(double s)
{
   (void) s;
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetInitialGuess begins...\n", mypid_);
   for ( int iB = 0; iB < numBlocks_; iB++ ) 
      elemBlocks_[iB]->resetSolnVectors(s);
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::resetInitialGuess ends (%e).\n", mypid_, s);
   return 0;
}

/**************************************************************************
 load node boundary conditions
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::loadNodeBCs(int numNodes, int *nodeIDs, int fieldID,
                             double **alpha, double **beta, double **gamma)
{
   int   iN, iD, oldNumBCNodes, *oldBCNodeIDs;
   double **oldBCAlpha, **oldBCBeta, **oldBCGamma;

   (void) fieldID;
   if ( outputLevel_ > 2 ) 
      printf("%4d : LLNL_FEI_Fei::loadNodeBCs begins...(%d)\n",mypid_,numNodes);
   TimerLoadStart_ = MPI_Wtime();
   if ( numNodes > 0 )
   {
      if ( numBCNodes_ == 0 )
      {
         numBCNodes_   = numNodes;
         BCNodeIDs_    = new int[numBCNodes_];
         BCNodeAlpha_  = new double*[numBCNodes_];
         BCNodeBeta_   = new double*[numBCNodes_];
         BCNodeGamma_  = new double*[numBCNodes_];
         for ( iN = 0; iN < numNodes; iN++ ) 
         {
            BCNodeIDs_[iN]   = nodeIDs[iN];
            BCNodeAlpha_[iN] = new double[nodeDOF_];
            BCNodeBeta_[iN]  = new double[nodeDOF_];
            BCNodeGamma_[iN] = new double[nodeDOF_];
            for ( iD = 0; iD < nodeDOF_; iD++ ) 
            {
               BCNodeAlpha_[iN][iD] = alpha[iN][iD];
               BCNodeBeta_[iN][iD]  = beta[iN][iD];
               BCNodeGamma_[iN][iD] = gamma[iN][iD];
            }
         }
      }
      else
      {
         oldNumBCNodes = numBCNodes_;
         oldBCNodeIDs  = BCNodeIDs_;
         oldBCAlpha    = BCNodeAlpha_;
         oldBCBeta     = BCNodeBeta_;
         oldBCGamma    = BCNodeGamma_;
         numBCNodes_   += numNodes;
         BCNodeIDs_    = new int[numBCNodes_];
         BCNodeAlpha_  = new double*[numBCNodes_];
         BCNodeBeta_   = new double*[numBCNodes_];
         BCNodeGamma_  = new double*[numBCNodes_];
         for ( iN = 0; iN < oldNumBCNodes; iN++ ) 
         {
            BCNodeIDs_[iN]   = oldBCNodeIDs[iN];
            BCNodeAlpha_[iN] = oldBCAlpha[iN];
            BCNodeBeta_[iN]  = oldBCBeta[iN];
            BCNodeGamma_[iN] = oldBCGamma[iN];
         }   
         delete [] oldBCNodeIDs;
         delete [] oldBCAlpha;
         delete [] oldBCBeta;
         delete [] oldBCGamma;
         for ( iN = 0; iN < numNodes; iN++ ) 
         {
            BCNodeIDs_[oldNumBCNodes+iN]   = nodeIDs[iN];
            BCNodeAlpha_[oldNumBCNodes+iN] = new double[nodeDOF_];
            BCNodeBeta_[oldNumBCNodes+iN]  = new double[nodeDOF_];
            BCNodeGamma_[oldNumBCNodes+iN] = new double[nodeDOF_];
            for ( iD = 0; iD < nodeDOF_; iD++ ) 
            {
               BCNodeAlpha_[oldNumBCNodes+iN][iD] = alpha[iN][iD];
               BCNodeBeta_[oldNumBCNodes+iN][iD]  = beta[iN][iD];
               BCNodeGamma_[oldNumBCNodes+iN][iD] = gamma[iN][iD];
            }
         }
      }
   }
   TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
   if ( outputLevel_ > 2 ) 
      printf("%4d : LLNL_FEI_Fei::loadNodeBCs ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 load element connectivities, stiffness matrices, and element load 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::sumInElem(int elemBlockID, int elemID, int *elemConn,
                            double **elemStiff, double *elemLoad, 
                            int elemFormat)
{
   int iB=0;

   (void) elemFormat;
   if ( numBlocks_ > 1 )
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == elemBlockID ) break;
   }
#if DEBUG
   if ( iB == numBlocks_ )
   {
      printf("%4d : LLNL_FEI_Fei::sumInElem ERROR - ", mypid_);
      printf("blockID invalid (%d).\n", iB);
      exit(1);
   }
#endif
#if DEBUG
   if ( outputLevel_ > 3 && elemBlocks_[iB]->getCurrentElem()==0 ) 
      printf("%4d : LLNL_FEI_Fei::sumInElem begins... \n", mypid_); 
#endif
   if ( elemBlocks_[iB]->getCurrentElem()==0 ) TimerLoadStart_ = MPI_Wtime();
   elemBlocks_[iB]->loadElemInfo(elemID, elemConn, elemStiff, elemLoad);
   if ( elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
#if DEBUG
   if ( outputLevel_ > 3 && 
        elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      printf("%4d : LLNL_FEI_Fei::sumInElem ends. \n", mypid_); 
#endif
   return 0;
}

/**************************************************************************
 load element connectivities and stiffness matrices
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::sumInElemMatrix(int elemBlockID, int elemID, int *elemConn,
                           double **elemStiff, int elemFormat)
{
   int iB=0;

   (void) elemFormat;
   if ( numBlocks_ > 1 )
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == elemBlockID ) break;
   }
#if DEBUG
   if ( iB == numBlocks_ )
   {
      printf("%4d : LLNL_FEI_Fei::sumInElemMatrix ERROR - ", mypid_);
      printf("blockID invalid (%d).\n", iB);
      exit(1);
   }
#endif
#if DEBUG
   if ( outputLevel_ > 3 && elemBlocks_[iB]->getCurrentElem()==0 ) 
      printf("%4d : LLNL_FEI_Fei::sumInElemMatrix begins... \n", mypid_); 
#endif
   if ( elemBlocks_[iB]->getCurrentElem()==0 ) TimerLoadStart_ = MPI_Wtime();
   elemBlocks_[iB]->loadElemMatrix(elemID, elemConn, elemStiff);
   if ( elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
#if DEBUG
   if ( outputLevel_ > 3 && 
        elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      printf("%4d : LLNL_FEI_Fei::sumInElemMatrix ends. \n", mypid_); 
#endif
   return 0;
}

/**************************************************************************
 load element load
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::sumInElemRHS(int elemBlockID, int elemID, int *elemConn,
                              double *elemLoad)
{
   int iB=0;

   (void) elemConn;
   if ( numBlocks_ > 1 )
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == elemBlockID ) break;
   }
#if DEBUG
   if ( iB == numBlocks_ )
   {
      printf("%4d : LLNL_FEI_Fei::sumInElemRHS ERROR - ", mypid_);
      printf("blockID invalid (%d).\n", iB);
      exit(1);
   }
#endif
   elemBlocks_[iB]->loadElemRHS(elemID, elemLoad);
   return 0;
}

/**************************************************************************
 assemble matrix information 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::loadComplete()
{
   int   nprocs, iB, iP, iN, iE, ierr, nodeRegister;
   int   totalNNodes, nElems, elemNNodes, **elemNodeList, nodeNumber;
   int   *nodeIDs, *nodeIDAux, localNNodes, *nodeIDAux2;
   int   CRNNodes, *sharedNodePInfo, *gatherBuf1, *gatherBuf2;

   /* -----------------------------------------------------------------
    * get machine information
    * ----------------------------------------------------------------*/

   if ( outputLevel_ > 2 ) 
      printf("%4d : LLNL_FEI_Fei::loadComplete begins.... \n", mypid_);
   TimerLoadStart_ = MPI_Wtime();
   MPI_Comm_size( mpiComm_, &nprocs );

   /* -----------------------------------------------------------------
    * check that element stiffness matrices, connectivities, and rhs
    * have been loaded, and create solution vectors.
    * ----------------------------------------------------------------*/

   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      ierr = elemBlocks_[iB]->checkLoadComplete();
      hypre_assert( !ierr );
   }

   /* -----------------------------------------------------------------
    * sort the shared nodes
    * ----------------------------------------------------------------*/

   sortSharedNodes();

   /* -----------------------------------------------------------------
    * obtain an ordered array of maybe repeated node IDs
    * ----------------------------------------------------------------*/

   composeOrderedNodeIDList(&nodeIDs, &nodeIDAux, &totalNNodes,
                            &CRNNodes);

   /* -----------------------------------------------------------------
    * based on shared node and CRMult information, construct an array
    * of to which processor each shared node belongs (nodeIDAux and
    * sharedNodePInfo)
    * ----------------------------------------------------------------*/

   findSharedNodeProcs(nodeIDs, nodeIDAux, totalNNodes, CRNNodes,
                       &sharedNodePInfo);

   /* -----------------------------------------------------------------
    * tally the number of local nodes (internal and external) 
    * ----------------------------------------------------------------*/

   localNNodes = numLocalNodes_ = 0;
   for ( iN = 1; iN < totalNNodes; iN++ )
   {
      if ( nodeIDs[iN] != nodeIDs[iN-1] ) 
      {
         localNNodes++;
         if ( nodeIDAux[iN] >= 0 ) numLocalNodes_++; 
      }
   }
   if ( totalNNodes > 0 ) localNNodes++;
   if ( totalNNodes > 0 && nodeIDAux[0] >= 0 ) numLocalNodes_++;
   if ( outputLevel_ > 2 )
   {
      printf("%4d : LLNL_FEI_Fei::loadComplete - nLocalNodes = %d\n",
             mypid_, numLocalNodes_);
      printf("%4d : LLNL_FEI_Fei::loadComplete - numExtNodes = %d\n",
             mypid_, localNNodes-numLocalNodes_);
      printf("%4d : LLNL_FEI_Fei::loadComplete - numCRMult   = %d\n",
             mypid_, numCRMult_);
   }

   /* -----------------------------------------------------------------
    * construct global node list, starting with nodes that belong
    * to local processor 
    * ----------------------------------------------------------------*/

   numExtNodes_ = localNNodes - numLocalNodes_;
   nodeGlobalIDs_ = new int[localNNodes];
   nodeNumber     = 0;
   nodeRegister   = -1;
   for ( iN = 0; iN < totalNNodes; iN++ )
   {
      if ( nodeIDAux[iN] >= 0 && nodeIDs[iN] != nodeRegister )
      {
         nodeRegister = nodeIDs[iN];
         nodeGlobalIDs_[nodeNumber++] = nodeIDs[iN];
         nodeIDs[iN] = nodeNumber - 1;
      }
      else if ( nodeIDAux[iN] >= 0 ) nodeIDs[iN] = nodeNumber - 1;
   }

   nodeRegister   = -1;
   for ( iN = 0; iN < totalNNodes; iN++ )
   {
      if ( nodeIDAux[iN] < 0 && nodeIDs[iN] != nodeRegister )
      {
         nodeRegister = nodeIDs[iN];
         nodeGlobalIDs_[nodeNumber++] = nodeIDs[iN];
         nodeIDs[iN] = nodeNumber - 1;
      }
      else if ( nodeIDAux[iN] < 0 ) nodeIDs[iN] = nodeNumber - 1;
   }

   /* -----------------------------------------------------------------
    * diagnostics
    * ----------------------------------------------------------------*/

#if 0
   {
      int  mypid, iNN, iNN2;
      char fname[30];
      FILE *fp;
      MPI_Comm_rank( mpiComm_, &mypid );
      sprintf(fname, "nodeID.%d", mypid);
      fp = fopen( fname, "w" );
      fprintf(fp, "%9d %9d\n", numLocalNodes_, numExtNodes_);
      for ( iNN = 0; iNN < numLocalNodes_+numExtNodes_; iNN++ )
         fprintf(fp, "%9d %9d\n", iNN, nodeGlobalIDs_[iNN]);
      fprintf(fp, "\n ---- shared nodes ---- \n");
      for ( iNN = 0; iNN < numSharedNodes_; iNN++ )
         for ( iNN2 = 0; iNN2 < sharedNodeNProcs_[iNN]; iNN2++ )
            fprintf(fp, "%9d %9d %9d %9d\n", iNN,sharedNodeIDs_[iNN],
      sharedNodePInfo[iNN],sharedNodeProcs_[iNN][iNN2]);
      fprintf(fp, "\n ---- CR ---- \n");
      for ( iNN = 0; iNN < numCRMult_; iNN++ )
      {
         fprintf(fp, "%9d : ", iNN);
         for ( iNN2 = 0; iNN2 < CRListLen_; iNN2++ )
            fprintf(fp, "%9d ", CRNodeLists_[iNN][iNN2]);
         fprintf(fp, "\n");
         for ( iNN2 = 0; iNN2 < CRListLen_; iNN2++ )
            fprintf(fp, "    %9d : %16.8e %16.8e %16.8e\n",
               CRNodeLists_[iNN][iNN2], CRWeightLists_[iNN][iNN2*nodeDOF_],
               CRWeightLists_[iNN][iNN2*nodeDOF_+1],
               CRWeightLists_[iNN][iNN2*nodeDOF_+2]);
      }
      fclose(fp);
   }
#endif

   /* -----------------------------------------------------------------
    * rewrite the element connectivities with local node numbers 
    * ----------------------------------------------------------------*/

   if ( totalNNodes > 0 ) nodeIDAux2 = new int[totalNNodes];
   for ( iN = 0; iN < totalNNodes; iN++ )
      if ( nodeIDAux[iN] < 0 ) nodeIDAux[iN] = - nodeIDAux[iN] - 1;
   for ( iN = 0; iN < totalNNodes; iN++ )
   {
      if ( nodeIDAux[iN] >= 0 && nodeIDAux[iN] < totalNNodes )
         nodeIDAux2[nodeIDAux[iN]] = nodeIDs[iN];
      else
      {
         printf("%4d : LLNL_FEI_Fei::loadComplete ERROR(2)\n",mypid_);
         exit(1);
      }
   }
   totalNNodes = 0;
   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      nElems       = elemBlocks_[iB]->getNumElems();
      elemNNodes   = elemBlocks_[iB]->getElemNumNodes();
      elemNodeList = elemBlocks_[iB]->getElemNodeLists();
      for ( iE = 0; iE < nElems; iE++ )
      {
         for ( iN = 0; iN < elemNNodes; iN++ )
            elemNodeList[iE][iN] = nodeIDAux2[totalNNodes++];
      }
   }
   if ( totalNNodes > 0 ) 
   {
      delete [] nodeIDAux;
      delete [] nodeIDAux2;
      delete [] nodeIDs;
   }

   /* -----------------------------------------------------------------
    * diagnostics
    * ----------------------------------------------------------------*/

#if 0
   {
      int  mypid, iD, iNN, iNN2;
      double **elemMats;
      char fname[30];
      FILE *fp;
      MPI_Comm_rank( mpiComm_, &mypid );
      sprintf(fname, "elemNode.%d", mypid);
      fp = fopen( fname, "w" );
      for ( iB = 0; iB < numBlocks_; iB++ )
      {
         nElems       = elemBlocks_[iB]->getNumElems();
         elemNNodes   = elemBlocks_[iB]->getElemNumNodes();
         elemNodeList = elemBlocks_[iB]->getElemNodeLists();
         elemMats     = elemBlocks_[iB]->getElemMatrices();
         for ( iE = 0; iE < nElems; iE++ )
         {
            fprintf(fp, "Element %7d : ", iE);
            for ( iN = 0; iN < elemNNodes; iN++ )
               fprintf(fp, "%7d ", nodeGlobalIDs_[elemNodeList[iE][iN]]);
            fprintf(fp, "\n");
            for ( iN = 0; iN < elemNNodes; iN++ )
            {
               for ( iD = 0; iD < nodeDOF_; iD++ )
                  fprintf(fp, "%16.8e ", elemMats[iE][iN*nodeDOF_+iD]);
               fprintf(fp, "\n");
            }
         }
      }
      fclose(fp);
   }
#endif

   /* -----------------------------------------------------------------
    * get global node offset information (globalNodeOffsets_)
    * ----------------------------------------------------------------*/

   if ( globalNodeOffsets_ != NULL ) delete [] globalNodeOffsets_;
   if ( globalCROffsets_   != NULL ) delete [] globalCROffsets_;
   globalNodeOffsets_ = new int[nprocs+1];
   globalCROffsets_ = new int[nprocs+1];
   gatherBuf1 = new int[2];
   gatherBuf2 = new int[2*nprocs];
   gatherBuf1[0] = numLocalNodes_;
   gatherBuf1[1] = numCRMult_;
   MPI_Allgather(gatherBuf1, 2, MPI_INT, gatherBuf2, 2, MPI_INT, mpiComm_);
   for (iP = 0; iP < nprocs; iP++) globalNodeOffsets_[iP] = gatherBuf2[2*iP]; 
   for (iP = 0; iP < nprocs; iP++) globalCROffsets_[iP] = gatherBuf2[2*iP+1];
   for ( iP = nprocs; iP > 0; iP-- ) 
      globalNodeOffsets_[iP] = globalNodeOffsets_[iP-1];
   globalNodeOffsets_[0] = 0;
   for ( iP = 1; iP <= nprocs; iP++ ) 
      globalNodeOffsets_[iP] += globalNodeOffsets_[iP-1];
   for ( iP = nprocs; iP > 0; iP-- ) 
      globalCROffsets_[iP] = globalCROffsets_[iP-1];
   globalCROffsets_[0] = 0;
   for ( iP = 1; iP <= nprocs; iP++ ) 
      globalCROffsets_[iP] += globalCROffsets_[iP-1];
   delete [] gatherBuf1;
   delete [] gatherBuf2;

   /* -----------------------------------------------------------------
    * next construct communication pattern 
    * ----------------------------------------------------------------*/

   setupCommPattern( sharedNodePInfo );
   if ( sharedNodePInfo != NULL ) delete [] sharedNodePInfo;

   /* -----------------------------------------------------------------
    * construct the global matrix and diagonal
    * ----------------------------------------------------------------*/

   buildGlobalMatrixVector();
   matPtr_->setComplete();
   TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
   FLAG_LoadComplete_ = 1;
   if ( outputLevel_ > 2 ) 
      printf("%4d : LLNL_FEI_Fei::loadComplete ends. \n", mypid_);
   return 0;
}

/**************************************************************************
 get number of distinct node in a given block
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::getNumBlockActNodes(int blockID, int *numNodes)
{
   int localNNodes, iB, iE, iN, totalNNodes, nElems;
   int elemNNodes, **elemNodeLists, *nodeIDs;  

   if ( numBlocks_ == 1 ) 
   {
      (*numNodes) = numLocalNodes_ + numExtNodes_;
      if ( outputLevel_ > 2 ) 
      {
         printf("%4d : LLNL_FEI_Fei::getNumBlockActNodes blockID = %d.\n", 
                mypid_, blockID);
         printf("%4d : LLNL_FEI_Fei::getNumBlockActNodes numNodes = %d\n", 
                mypid_, (*numNodes));
      }
      return 0;
   }
   else
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == blockID ) break;
      if ( iB >= numBlocks_ )
      {
         printf("%4d : LLNL_FEI_Fei::getNumBlockActNodes ERROR -",mypid_);
         printf(" invalid blockID\n");
         exit(1);
      } 
      totalNNodes = numLocalNodes_ + numExtNodes_; 
      nodeIDs     = new int[totalNNodes];
      for ( iN = 0; iN < totalNNodes; iN++ ) nodeIDs[iN] = 0;
      nElems      = elemBlocks_[iB]->getNumElems();
      elemNNodes  = elemBlocks_[iB]->getElemNumNodes();
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      for ( iE = 0; iE < nElems; iE++ )
         for ( iN = 0; iN < elemNNodes; iN++ )
            nodeIDs[elemNodeLists[iE][iN]] = 1;
      localNNodes = 0;
      for ( iN = 0; iN < totalNNodes; iN++ ) 
         if ( nodeIDs[iN] == 1 ) localNNodes++;
      delete [] nodeIDs;
      (*numNodes) = localNNodes;

      if ( outputLevel_ > 2 ) 
      {
         printf("%4d : LLNL_FEI_Fei::getNumBlockActNodes blockID = %d.\n", 
                mypid_, blockID);
         printf("%4d : LLNL_FEI_Fei::getNumBlockActNodes numNodes = %d\n", 
                mypid_, (*numNodes));
      }
   }
   return 0;
}

/**************************************************************************
 get number of distinct equations in a given block
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::getNumBlockActEqns(int blockID, int *numEqns)
{
   int numNodes;

   getNumBlockActNodes(blockID, &numNodes);
   (*numEqns) = numNodes * nodeDOF_;
   if ( outputLevel_ > 2 ) 
   {
      printf("%4d : LLNL_FEI_Fei::getNumBlockActEqns blockID = %d\n", 
             mypid_, blockID);
      printf("%4d : LLNL_FEI_Fei::getNumBlockActEqns numEqns = %d\n", 
             mypid_, (*numEqns));
   }
   return 0;
}

/**************************************************************************
 get a node list in a given block
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::getBlockNodeIDList(int blockID,int numNodes,int *nodeList)
{
   int localNNodes, iB, iE, iN, totalNNodes, nElems;
   int elemNNodes, **elemNodeLists, *nodeIDs;  

   if ( outputLevel_ > 2 ) 
   {
      printf("%4d : LLNL_FEI_Fei::getBlockNodeIDList blockID  = %d\n", 
             mypid_, blockID);
      printf("%4d : LLNL_FEI_Fei::getBlockNodeIDList numNodes = %d\n", 
             mypid_, numNodes);
   }
   if ( numBlocks_ == 1 ) 
   {
      localNNodes = numLocalNodes_ + numExtNodes_;
      if ( localNNodes != numNodes )
      {
         printf("%4d : LLNL_FEI_Fei::getBlockNodeIDList ERROR - nNodes",mypid_);
         printf(" mismatch.\n");
         exit(1);
      }
      for ( iN = 0; iN < localNNodes; iN++ )
         nodeList[iN] = nodeGlobalIDs_[iN]; 
      return 0;
   }
   else
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == blockID ) break;
      if ( iB >= numBlocks_ )
      {
         printf("%4d : LLNL_FEI_Fei::getBlockNodeIDList ERROR -",mypid_);
         printf(" invalid blockID.\n");
         exit(1);
      } 
      totalNNodes = numLocalNodes_ + numExtNodes_; 
      nodeIDs     = new int[totalNNodes];
      for ( iN = 0; iN < totalNNodes; iN++ ) nodeIDs[iN] = 0;
      nElems      = elemBlocks_[iB]->getNumElems();
      elemNNodes  = elemBlocks_[iB]->getElemNumNodes();
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      for ( iE = 0; iE < nElems; iE++ )
         for ( iN = 0; iN < elemNNodes; iN++ )
            nodeIDs[elemNodeLists[iE][iN]] = 1;
      localNNodes = 0;
      for ( iN = 0; iN < totalNNodes; iN++ ) 
         if ( nodeIDs[iN] == 1 ) nodeList[localNNodes++] = nodeGlobalIDs_[iN];
      if ( localNNodes != numNodes )
      {
         printf("%4d : LLNL_FEI_Fei::getBlockNodeIDList ERROR -",mypid_);
         printf(" nNodes mismatch (%d,%d).\n", localNNodes, numNodes);
         exit(1);
      }
      delete [] nodeIDs;
   }
   return 0;
}

/**************************************************************************
 get solution 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::getBlockNodeSolution(int blockID,int numNodes,
                         int *nodeList, int *nodeOffsets, double *solnValues)
{
   int    iB, iE, iN, iD, totalNNodes, *nodeIDs, index, offset;
   int    nElems, elemNNodes, **elemNodeLists, nodeID, localNNodes;
   double *dataBuf, **solnVecs;

   (void) nodeList;
   if ( outputLevel_ > 2 ) 
   {
      printf("%4d : LLNL_FEI_Fei::getBlockNodeSolution blockID  = %d\n", 
             mypid_, blockID);
      printf("%4d : LLNL_FEI_Fei::getBlockNodeSolution numNodes = %d\n", 
             mypid_, numNodes);
   }
   if ( numBlocks_ == 1 ) 
   {
      for ( iN = 0; iN < numNodes; iN++ )
      {
         offset = iN * nodeDOF_;
         nodeOffsets[iN] = offset;
         if ( numCRMult_ > 0 )
         {
            if ( nodeList[iN] == nodeGlobalIDs_[iN] )
            {
               if ( iN >= numLocalNodes_) offset += numCRMult_;
            }
            else
            {
               index = -1;
               if ( numLocalNodes_ > 0 )
                  index = hypre_BinarySearch(nodeGlobalIDs_,nodeList[iN],                                                    numLocalNodes_);
               if ( index < 0 ) offset += numCRMult_;
            }
         }
         for ( iD = 0; iD < nodeDOF_; iD++ )
            solnValues[iN*nodeDOF_+iD] = solnVector_[offset+iD];
      }
   }
   else
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == blockID ) break;
      if ( iB >= numBlocks_ )
      {
         printf("%4d : LLNL_FEI_Fei::getBlockNodeSolution ERROR -",mypid_);
         printf(" invalid blockID.\n");
         exit(1);
      } 
      totalNNodes = numLocalNodes_ + numExtNodes_; 
      nodeIDs     = new int[totalNNodes];
      dataBuf     = new double[totalNNodes*nodeDOF_];
      for ( iN = 0; iN < totalNNodes; iN++ ) nodeIDs[iN] = 0;
      nElems        = elemBlocks_[iB]->getNumElems();
      elemNNodes    = elemBlocks_[iB]->getElemNumNodes();
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      solnVecs      = elemBlocks_[iB]->getSolnVectors();
      for ( iE = 0; iE < nElems; iE++ )
      {
         for ( iN = 0; iN < elemNNodes; iN++ )
         {
            nodeID = elemNodeLists[iE][iN];
            nodeIDs[nodeID] = 1;
            for ( iD = 0; iD < nodeDOF_; iD++ )
               dataBuf[nodeID*nodeDOF_+iD] = solnVecs[iE][iN*nodeDOF_+iD];
         }
      }
      localNNodes = 0;
      for ( iN = 0; iN < totalNNodes; iN++ ) 
      {
         nodeID = nodeIDs[iN];
         if ( nodeID == 1 ) 
         {
            nodeOffsets[localNNodes] = localNNodes * nodeDOF_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
               solnValues[localNNodes*nodeDOF_+iD] = dataBuf[iN*nodeDOF_+iD];
            localNNodes++;
         }
      }
      delete [] nodeIDs;
      delete [] dataBuf;
   }
   return 0;
}

/**************************************************************************
 initialize Lagrange Multipliers 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::initCRMult(int CRListLen,int *CRNodeList,int *CRFieldList,
                             int *CRID)
{
   (void) CRNodeList;
   (void) CRFieldList;
   if ( outputLevel_ > 3 ) 
      printf("%4d : LLNL_FEI_Fei::initCRMult begins...\n", mypid_);
   if ( numCRMult_ == 0 ) CRListLen_ = CRListLen;
   else
   {
      if ( CRListLen != CRListLen_ )
      {
         printf("%4d : LLNL_FEI_Fei::initCRMult ERROR : inconsistent lengths\n",
                mypid_);
         printf("%4d : LLNL_FEI_Fei::initCRMult lengths = %d %d\n",
                mypid_, CRListLen, CRListLen_);
         exit(1);
      }
   }
   (*CRID) = numCRMult_++;
   if ( outputLevel_ > 3 ) 
      printf("%4d : LLNL_FEI_Fei::initCRMult ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 load Lagrange Multipliers 
 -------------------------------------------------------------------------*/
int LLNL_FEI_Fei::loadCRMult(int CRID, int CRListLen, int *CRNodeList, 
                            int *CRFieldList, double *CRWeightList, 
                            double CRValue)
{
   (void) CRFieldList;
   int iD, iD2;

   if ( outputLevel_ > 3 ) 
      printf("%4d : LLNL_FEI_Fei::loadCRMult begins...\n", mypid_);
   if ( CRNodeLists_ == NULL )
   {
      if ( numCRMult_ > 0 && CRListLen_ > 0 )
      {
         CRNodeLists_ = new int*[numCRMult_];
         for ( iD = 0; iD < numCRMult_; iD++ ) 
         {
            CRNodeLists_[iD] = new int[CRListLen_];
            for ( iD2 = 0; iD2 < CRListLen_; iD2++ ) 
               CRNodeLists_[iD][iD2] = -1;
         }
         CRWeightLists_ = new double*[numCRMult_];
         for ( iD = 0; iD < numCRMult_; iD++ ) 
            CRWeightLists_[iD] = new double[CRListLen_*nodeDOF_];
         CRValues_ = new double[numCRMult_];
      }
   }
   if ( CRID < 0 || CRID >= numCRMult_ ) 
   {
      printf("%4d : LLNL_FEI_Fei::loadCRMult ERROR : invalid ID = %d (%d)\n",
                mypid_, CRID, numCRMult_);
      exit(1);
   }
   if ( CRListLen != CRListLen_ ) 
   {
      printf("%4d : LLNL_FEI_Fei::loadCRMult ERROR : inconsistent lengths\n",
             mypid_);
      printf("%4d : LLNL_FEI_Fei::loadCRMult lengths = %d %d\n", mypid_, 
             CRListLen, CRListLen_);
      exit(1);
   }
   for ( iD = 0; iD < CRListLen_; iD++ ) 
   {
      CRNodeLists_[CRID][iD] = CRNodeList[iD];
      for ( iD2 = 0; iD2 < nodeDOF_; iD2++ ) 
         CRWeightLists_[CRID][iD*nodeDOF_+iD2] = CRWeightList[iD*nodeDOF_+iD2];
   }
   CRValues_[CRID] = CRValue;
   if ( outputLevel_ > 3 ) 
      printf("%4d : LLNL_FEI_Fei::loadCRMult ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 build global stiffness matrix
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::buildGlobalMatrixVector()
{
   int    matDim, *diagCounts=NULL, nElems, elemNNodes, **elemNodeLists=NULL;
   int    iB, iD, iE, iN, offset, iD2, iD3, iN2, *elemNodeList=NULL, diagNNZ; 
   int    offdNNZ, *offdCounts=NULL, rowIndBase, rowInd, colIndBase, colInd;
   int    iP, nprocs, iCount, index, iBegin, *TdiagIA=NULL, *TdiagJA=NULL;
   int    *ToffdIA=NULL, *ToffdJA=NULL, elemNExt, elemNLocal, nodeID, length;
   int    diagOffset, offdOffset, nLocal, rowInd2, *globalEqnOffsets; 
   int    *diagIA, *diagJA, *offdIA, *offdJA, *extEqnList, *crOffsets;
   int    nRecvs, *recvLengs, *recvProcs, *recvProcIndices, *flags;
   int    nSends, *sendLengs, *sendProcs, *sendProcIndices, BCcnt, *iArray;
   double **elemMats=NULL, *elemMat=NULL, *TdiagAA=NULL, *ToffdAA=NULL;
   double alpha, beta, gamma, dtemp, *diagAA, *offdAA, *diagonal, *dvec; 
   double **tArray;

   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::buildGlobalMatrixVector begins..\n",mypid_);

   /* -----------------------------------------------------------------
    * assemble the right hand side vector
    * -----------------------------------------------------------------*/

   assembleRHSVector();

   /* -----------------------------------------------------------------
    * count the number of nonzeros per row (diagCounts, offdCounts)
    * -----------------------------------------------------------------*/

   matDim = ( numLocalNodes_ + numExtNodes_) * nodeDOF_ + numCRMult_;
   nLocal = numLocalNodes_ * nodeDOF_;
   if ( matDim > 0 )
   {
      diagCounts = new int[matDim];
      offdCounts = new int[matDim];
   }
   for ( iD = 0; iD < matDim; iD++ ) diagCounts[iD] = offdCounts[iD] = 0;

   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      nElems        = elemBlocks_[iB]->getNumElems();
      elemNNodes    = elemBlocks_[iB]->getElemNumNodes();
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      elemMats      = elemBlocks_[iB]->getElemMatrices();
      for ( iE = 0; iE < nElems; iE++ )
      {
         elemMat      = elemMats[iE];
         elemNodeList = elemNodeLists[iE];
         elemNExt     = elemNLocal = 0;
         for ( iN = 0; iN < elemNNodes; iN++ )
         {
            if ( elemNodeList[iN] < numLocalNodes_ ) elemNLocal++;
            else                                     elemNExt++;
         }
         for ( iN = 0; iN < elemNNodes; iN++ )
         {
            rowInd = elemNodeList[iN] * nodeDOF_;
            if ( rowInd >= nLocal ) rowInd += numCRMult_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
            {
               diagCounts[rowInd+iD] += elemNLocal * nodeDOF_;
               offdCounts[rowInd+iD] += elemNExt * nodeDOF_;
            }
         }
      }
   }  

   for ( iD = 0; iD < numCRMult_; iD++ )
   {
      for ( iD2 = 0; iD2 < CRListLen_; iD2++ )
      {
         nodeID = CRNodeLists_[iD][iD2];
         index  = -1;
         if ( numLocalNodes_ > 0 )
            index = hypre_BinarySearch(nodeGlobalIDs_,nodeID,numLocalNodes_);
         /* if CR nodes is in my local node collection */
         if ( index >= 0 )
         {
            for ( iD3 = 0; iD3 < nodeDOF_; iD3++ )
            {
               dtemp = CRWeightLists_[iD][iD2*nodeDOF_+iD3];
               if ( dtemp != 0.0 )
               {
                  diagCounts[nLocal+iD]++; 
                  diagCounts[index*nodeDOF_+iD3]++; 
               }
            }
         }
         /* if CR nodes is not in my local node collection */
         else
         {
            index = -1;
            if ( numExtNodes_ > 0 )
               index = hypre_BinarySearch(&nodeGlobalIDs_[numLocalNodes_],
                                          nodeID, numExtNodes_);
            /* but if CR nodes is in my remote node collection */
            if ( index >= 0 )
            {
               for ( iD3 = 0; iD3 < nodeDOF_; iD3++ )
               {
                  dtemp = CRWeightLists_[iD][iD2*nodeDOF_+iD3];
                  if ( dtemp != 0.0 )
                  {
                     offdCounts[nLocal+iD]++; 
                     rowInd = (index+numLocalNodes_)*nodeDOF_+numCRMult_+iD3;
                     diagCounts[rowInd]++;
                  }
               }
            }
            else
            {
               printf("%4d : LLNL_FEI_Fei::buildGlobalMatrixVector ERROR(1).\n",
                      mypid_);
               exit(1);
            }
         }
      }
   }

   /* -----------------------------------------------------------------
    * allocate the CSR matrix storage space 
    * -----------------------------------------------------------------*/

   diagNNZ = offdNNZ = 0;
   for ( iD = 0; iD < matDim; iD++ ) 
   {
      diagNNZ += diagCounts[iD];
      offdNNZ += offdCounts[iD];
   }
   if ( diagNNZ > 0 ) 
   {
      TdiagIA = new int[matDim+1];
      TdiagJA = new int[diagNNZ];
      TdiagAA = new double[diagNNZ];
   }
   if ( offdNNZ > 0 ) 
   {
      ToffdIA = new int[matDim+1];
      ToffdJA = new int[offdNNZ];
      ToffdAA = new double[offdNNZ];
   }

   /* -----------------------------------------------------------------
    * get ready for loading up the CSR matrix 
    * -----------------------------------------------------------------*/

   offset = 0;
   for ( iD = 0; iD < matDim; iD++ )
   {
      TdiagIA[iD] = offset;
      offset += diagCounts[iD];
   }
   offset = 0;
   if ( offdNNZ > 0 ) 
   {
      for ( iD = 0; iD < matDim; iD++ )
      {
         ToffdIA[iD] = offset;
         offset += offdCounts[iD];
      }
   }

   /* -----------------------------------------------------------------
    * load the CSR matrix 
    * -----------------------------------------------------------------*/

   /* first load the element stiffness matrices */

   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      nElems        = elemBlocks_[iB]->getNumElems();
      elemNNodes    = elemBlocks_[iB]->getElemNumNodes();
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      elemMats      = elemBlocks_[iB]->getElemMatrices();
      if ( nodeDOF_ == 1 )
      {
         for ( iE = 0; iE < nElems; iE++ )
         {
            elemMat      = elemMats[iE];
            elemNodeList = elemNodeLists[iE];
            offset       = 0;
            for ( iN = 0; iN < elemNNodes; iN++ )
            {
               colInd = elemNodeList[iN];
               if ( colInd >= nLocal ) 
               {
                  colInd += numCRMult_;
                  for ( iN2 = 0; iN2 < elemNNodes; iN2++ )
                  {
                     rowInd = elemNodeList[iN2];
                     if ( rowInd >= nLocal ) rowInd += numCRMult_;
                     if ( *elemMat != 0.0 ) 
                     {
                        index = ToffdIA[rowInd]++;
                        ToffdJA[index] = colInd;
                        ToffdAA[index] = *elemMat;
                     }
                     elemMat++;
                  }
               }
               else
               {
                  for ( iN2 = 0; iN2 < elemNNodes; iN2++ )
                  {
                     rowInd = elemNodeList[iN2];
                     if ( rowInd >= nLocal ) rowInd += numCRMult_;
                     if ( *elemMat != 0.0 ) 
                     {
                        index = TdiagIA[rowInd]++;
                        TdiagJA[index] = colInd;
                        TdiagAA[index] = *elemMat;
                     }
                     elemMat++;
                  }
               }
            }
            delete [] elemMats[iE];
            elemMats[iE] = NULL;
         }
      }
      else
      {
         for ( iE = 0; iE < nElems; iE++ )
         {
            elemMat      = elemMats[iE];
            elemNodeList = elemNodeLists[iE];
            offset       = 0;
            for ( iN = 0; iN < elemNNodes; iN++ )
            {
               colIndBase = elemNodeList[iN] * nodeDOF_;
               for ( iD = 0; iD < nodeDOF_; iD++ )
               {
                  colInd = colIndBase + iD;
                  if ( colInd >= nLocal ) colInd += numCRMult_;
                  for ( iN2 = 0; iN2 < elemNNodes; iN2++ )
                  {
                     rowIndBase = elemNodeList[iN2] * nodeDOF_;
                     for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
                     {
                        rowInd = rowIndBase + iD2;
                        if ( rowInd >= nLocal ) rowInd += numCRMult_;
                        if ( elemMat[offset] != 0.0 ) 
                        {
                           if ( colInd >= nLocal ) 
                           {
                              index = ToffdIA[rowInd]++;
                              ToffdJA[index] = colInd;
                              ToffdAA[index] = elemMat[offset];
                           }
                           else
                           {
                              index = TdiagIA[rowInd]++;
                              TdiagJA[index] = colInd;
                              TdiagAA[index] = elemMat[offset];
                           }
                        }
                        offset++;
                     }
                  }
               }
            }
            delete [] elemMats[iE];
            elemMats[iE] = NULL;
         }
      }
   }

   /* then load the constraint matrix A21 */

   for ( iD = 0; iD < numCRMult_; iD++ )
   {
      rowInd = nLocal + iD;
      for ( iD2 = 0; iD2 < CRListLen_; iD2++ )
      {
         nodeID = CRNodeLists_[iD][iD2];
         index = hypre_BinarySearch(nodeGlobalIDs_,nodeID,numLocalNodes_);
         if ( index >= 0 )
         {
            for ( iD3 = 0; iD3 < nodeDOF_; iD3++ )
            {
               dtemp = CRWeightLists_[iD][iD2*nodeDOF_+iD3];
               if ( dtemp != 0.0 )
               {
                  offset = TdiagIA[rowInd]++;
                  colInd = index * nodeDOF_ + iD3;
                  TdiagJA[offset] = colInd;
                  TdiagAA[offset] = dtemp;
                  offset = TdiagIA[colInd]++;
                  TdiagJA[offset] = rowInd;
                  TdiagAA[offset] = dtemp;
               }
            }
         }
         else
         {
            if ( numExtNodes_ <= 0 )
            {
               printf("%4d : LLNL_FEI_Fei::buildGlobalMatrixVector ERROR(2).\n",
                      mypid_);
               exit(1);
            }
            index = hypre_BinarySearch(&nodeGlobalIDs_[numLocalNodes_],nodeID,
                                       numExtNodes_);
            for ( iD3 = 0; iD3 < nodeDOF_; iD3++ )
            {
               dtemp = CRWeightLists_[iD][iD2*nodeDOF_+iD3];
               if ( dtemp != 0.0 )
               {
                  offset = ToffdIA[rowInd]++;
                  colInd = (index + numLocalNodes_)*nodeDOF_+numCRMult_+iD3;
                  ToffdJA[offset] = colInd;
                  ToffdAA[offset] = dtemp;
                  offset = TdiagIA[colInd]++;
                  TdiagJA[offset] = rowInd;
                  TdiagAA[offset] = dtemp;
               }
            }
         }
      }
   }

   /* -----------------------------------------------------------------
    * compress the matrix (take out redundant columns)
    * -----------------------------------------------------------------*/

   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::buildGlobalMatrixVector mid phase\n",mypid_);
   offset = 0;
   for ( iD = 0; iD < matDim; iD++ )
   {
      iCount = TdiagIA[iD] - offset;
      TdiagIA[iD] = offset;
      offset += diagCounts[iD];
      diagCounts[iD] = iCount;
   }
   if ( offdNNZ > 0 )
   {
      offset = 0;
      for ( iD = 0; iD < matDim; iD++ )
      {
         iCount = ToffdIA[iD] - offset;
         ToffdIA[iD] = offset;
         offset += offdCounts[iD];
         offdCounts[iD] = iCount;
      }
   }
   for ( iD = 0; iD < matDim; iD++ )
   {
      if ( diagCounts[iD] > 0 ) 
      {
         iBegin = TdiagIA[iD];
         iCount = diagCounts[iD];
         index  = iBegin;
         if ( iCount > 0 )
            IntSort2a(&(TdiagJA[iBegin]),&(TdiagAA[iBegin]),0,iCount-1);
         for ( iD2 = iBegin+1; iD2 < iBegin+iCount; iD2++ )
         {
            if ( TdiagJA[iD2] == TdiagJA[index] ) 
               TdiagAA[index] += TdiagAA[iD2];
            else
            {
               if (TdiagAA[index] != 0.0 ) index++;
               TdiagJA[index] = TdiagJA[iD2];
               TdiagAA[index] = TdiagAA[iD2];
            }
         }
         if ( iCount > 0 && TdiagAA[index] != 0.0 ) index++;
         diagCounts[iD] = index - iBegin;
      }
      if ( offdCounts[iD] > 0 ) 
      {
         iBegin = ToffdIA[iD];
         iCount = offdCounts[iD];
         index  = iBegin;
         if ( iCount > 0 )
            IntSort2a(&(ToffdJA[iBegin]),&(ToffdAA[iBegin]),0,iCount-1);
         for ( iD2 = iBegin+1; iD2 < iBegin+iCount; iD2++ )
         {
            if ( ToffdJA[iD2] == ToffdJA[index] ) 
               ToffdAA[index] += ToffdAA[iD2];
            else
            {
               if (ToffdAA[index] != 0.0 ) index++;
               ToffdJA[index] = ToffdJA[iD2];
               ToffdAA[index] = ToffdAA[iD2];
            }
         }
         if ( iCount > 0 && ToffdAA[index] != 0.0 ) index++;
         offdCounts[iD] = index - iBegin;
      }
   }

   /* -----------------------------------------------------------------
    * obtain off-processor boundary conditions
    * -----------------------------------------------------------------*/

   length = (numLocalNodes_ + numExtNodes_) * nodeDOF_ * 3;

   /* part 1: if BC is declared on off-processors, get them in */ 
   if (length > 0 )
   {
      dvec = new double[length];
      for (iD = 0; iD < length; iD++) dvec[iD] = 0.0;
      for ( iN = 0; iN < numBCNodes_; iN++ )
      {
         nodeID = BCNodeIDs_[iN];
         index  = -1;
         if (numExtNodes_ > 0)
            index = hypre_BinarySearch(&nodeGlobalIDs_[numLocalNodes_],nodeID,
                                       numExtNodes_);
         if (index >= 0) 
            dvec[(numLocalNodes_+index)*nodeDOF_] = 1.0;
      }
      gatherAddDData( dvec );
      BCcnt = 0;
      for (iD = 0; iD < numLocalNodes_; iD++)
         if  (dvec[iD*nodeDOF_] == 1.0) BCcnt++;
      if (BCcnt > 0)
      {
         iArray = BCNodeIDs_;
         BCNodeIDs_ = new int[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeIDs_[iN] = iArray[iN];
         delete [] iArray;
         offset = numBCNodes_;
         for (iD = 0; iD < numLocalNodes_; iD++)
            if  (dvec[iD*nodeDOF_] == 1.0) 
               BCNodeIDs_[offset++] = iD;
         tArray = BCNodeAlpha_;
         BCNodeAlpha_ = new double*[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeAlpha_[iN] = tArray[iN];
         for (iN = 0; iN < BCcnt; iN++) 
         {
            BCNodeAlpha_[numBCNodes_+iN] = new double[nodeDOF_];
            for (iD = 0; iD < nodeDOF_; iD++)
               BCNodeAlpha_[numBCNodes_+iN][iD] = 0.0;
         }
         delete [] tArray;
         tArray = BCNodeBeta_;
         BCNodeBeta_ = new double*[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeBeta_[iN] = tArray[iN];
         for (iN = 0; iN < BCcnt; iN++) 
         {
            BCNodeBeta_[numBCNodes_+iN] = new double[nodeDOF_];
            for (iD = 0; iD < nodeDOF_; iD++)
               BCNodeBeta_[numBCNodes_+iN][iD] = 0.0;
         }
         delete [] tArray;
         tArray = BCNodeGamma_;
         BCNodeGamma_ = new double*[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeGamma_[iN] = tArray[iN];
         for (iN = 0; iN < BCcnt; iN++) 
         {
            BCNodeGamma_[numBCNodes_+iN] = new double[nodeDOF_];
            for (iD = 0; iD < nodeDOF_; iD++)
               BCNodeGamma_[numBCNodes_+iN][iD] = 0.0;
         }
         delete [] tArray;
      }
      for (iD = 0; iD < length; iD++) dvec[iD] = 0.0;
      for ( iN = 0; iN < numBCNodes_; iN++ )
      {
         nodeID = BCNodeIDs_[iN];
         index  = -1;
         if (numExtNodes_ > 0)
            index = hypre_BinarySearch(&nodeGlobalIDs_[numLocalNodes_],nodeID,
                                       numExtNodes_);
         if (index >= 0)
         {
            for (iD = index*nodeDOF_*3; iD < index*nodeDOF_*3+nodeDOF_; iD++)
               dvec[numLocalNodes_*nodeDOF_*3+iD] =
                  BCNodeAlpha_[iN][iD%nodeDOF_]; 
            for (iD = index*nodeDOF_*3+nodeDOF_; 
                 iD < index*nodeDOF_*3+nodeDOF_*2; iD++)
               dvec[numLocalNodes_*nodeDOF_*3+iD] =
                  BCNodeBeta_[iN][iD%nodeDOF_]; 
            for (iD = index*nodeDOF_*3+nodeDOF_*2; 
                 iD < (index+1)*nodeDOF_*3; iD++)
               dvec[numLocalNodes_*nodeDOF_*3+iD] =
                  BCNodeGamma_[iN][iD%nodeDOF_]; 
         }
      }
      iD = nodeDOF_;
      nodeDOF_ *= 3;
      gatherAddDData( dvec );
      nodeDOF_ = iD;
      for (iN = numBCNodes_; iN < numBCNodes_+BCcnt; iN++) 
      {
         nodeID = BCNodeIDs_[iN];
         for (iD = 0; iD < nodeDOF_; iD++) 
            BCNodeAlpha_[iN][iD] = dvec[nodeID*nodeDOF_*3+iD]; 
         for (iD = 0; iD < nodeDOF_; iD++) 
            BCNodeBeta_[iN][iD] = dvec[nodeID*nodeDOF_*3+nodeDOF_+iD]; 
         for (iD = 0; iD < nodeDOF_; iD++) 
            BCNodeGamma_[iN][iD] = dvec[nodeID*nodeDOF_*3+nodeDOF_*2+iD]; 
         BCNodeIDs_[iN] = nodeGlobalIDs_[nodeID];
      }
      numBCNodes_ += BCcnt;
      delete [] dvec;
   }

   /* part 1: if BC is declared on-processors, get them out */ 
   if (length > 0 )
   {
      dvec = new double[length];
      for (iD = 0; iD < length; iD++) dvec[iD] = 0.0;
      for ( iN = 0; iN < numBCNodes_; iN++ )
      {
         nodeID = BCNodeIDs_[iN];
         index = hypre_BinarySearch(nodeGlobalIDs_,nodeID,numLocalNodes_);
         if (index >= 0) 
            dvec[index*nodeDOF_] = 1.0;
      }
      scatterDData( dvec );
      BCcnt = 0;
      for (iD = numLocalNodes_; iD < numLocalNodes_+numExtNodes_; iD++)
         if  (dvec[iD*nodeDOF_] != 0.0) BCcnt++;
      if (BCcnt > 0)
      {
         iArray = BCNodeIDs_;
         BCNodeIDs_ = new int[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeIDs_[iN] = iArray[iN];
         delete [] iArray;
         offset = numBCNodes_;
         for (iD = numLocalNodes_; iD < numLocalNodes_+numExtNodes_; iD++)
            if  (dvec[iD*nodeDOF_] == 1.0) 
               BCNodeIDs_[offset++] = iD;
         tArray = BCNodeAlpha_;
         BCNodeAlpha_ = new double*[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeAlpha_[iN] = tArray[iN];
         for (iN = 0; iN < BCcnt; iN++) 
         {
            BCNodeAlpha_[numBCNodes_+iN] = new double[nodeDOF_];
            for (iD = 0; iD < nodeDOF_; iD++)
               BCNodeAlpha_[numBCNodes_+iN][iD] = 0.0;
         }
         delete [] tArray;
         tArray = BCNodeBeta_;
         BCNodeBeta_ = new double*[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeBeta_[iN] = tArray[iN];
         for (iN = 0; iN < BCcnt; iN++) 
         {
            BCNodeBeta_[numBCNodes_+iN] = new double[nodeDOF_];
            for (iD = 0; iD < nodeDOF_; iD++)
               BCNodeBeta_[numBCNodes_+iN][iD] = 0.0;
         }
         delete [] tArray;
         tArray = BCNodeGamma_;
         BCNodeGamma_ = new double*[numBCNodes_+BCcnt];
         for (iN = 0; iN < numBCNodes_; iN++) BCNodeGamma_[iN] = tArray[iN];
         for (iN = 0; iN < BCcnt; iN++) 
         {
            BCNodeGamma_[numBCNodes_+iN] = new double[nodeDOF_];
            for (iD = 0; iD < nodeDOF_; iD++)
               BCNodeGamma_[numBCNodes_+iN][iD] = 0.0;
         }
         delete [] tArray;
      }
      for (iD = 0; iD < length; iD++) dvec[iD] = 0.0;
      for ( iN = 0; iN < numBCNodes_; iN++ )
      {
         nodeID = BCNodeIDs_[iN];
         index = hypre_BinarySearch(nodeGlobalIDs_,nodeID,numLocalNodes_);
         if (index >= 0) 
         {
            for (iD = index*nodeDOF_*3; iD < index*nodeDOF_*3+nodeDOF_; iD++)
               dvec[iD] = BCNodeAlpha_[iN][iD%nodeDOF_]; 
            for (iD = index*nodeDOF_*3+nodeDOF_; 
                 iD < index*nodeDOF_*3+nodeDOF_*2; iD++)
               dvec[iD] = BCNodeBeta_[iN][iD%nodeDOF_]; 
            for (iD = index*nodeDOF_*3+nodeDOF_*2; 
                 iD < (index+1)*nodeDOF_*3; iD++)
               dvec[iD] = BCNodeGamma_[iN][iD%nodeDOF_]; 
         }
      }
      iD = nodeDOF_;
      nodeDOF_ *= 3;
      scatterDData( dvec );
      nodeDOF_ = iD;
      for (iN = numBCNodes_; iN < numBCNodes_+BCcnt; iN++) 
      {
         nodeID = BCNodeIDs_[iN];
         for (iD = 0; iD < nodeDOF_; iD++) 
            BCNodeAlpha_[iN][iD] = dvec[nodeID*nodeDOF_*3+iD]; 
         for (iD = 0; iD < nodeDOF_; iD++) 
            BCNodeBeta_[iN][iD] = dvec[nodeID*nodeDOF_*3+nodeDOF_+iD]; 
         for (iD = 0; iD < nodeDOF_; iD++) 
            BCNodeGamma_[iN][iD] = dvec[nodeID*nodeDOF_*3+nodeDOF_*2+iD]; 
         BCNodeIDs_[iN] = nodeGlobalIDs_[nodeID];
      }
      numBCNodes_ += BCcnt;
      delete [] dvec;
   }

   /* -----------------------------------------------------------------
    * impose boundary conditions
    * -----------------------------------------------------------------*/

   for (iD = nLocal; iD < matDim; iD++) rhsVector_[iD] = 0.0;
   flags = NULL;
   if (numLocalNodes_+numExtNodes_ > 0)
      flags = new int[(numLocalNodes_+numExtNodes_)*nodeDOF_];
   for (iD = 0; iD < (numLocalNodes_+numExtNodes_)*nodeDOF_; iD++) 
      flags[iD] = 0;

   for ( iN = 0; iN < numBCNodes_; iN++ )
   {
      nodeID = BCNodeIDs_[iN];
      index  = -1;
      if (numLocalNodes_ > 0)
         index = hypre_BinarySearch(nodeGlobalIDs_,nodeID,numLocalNodes_);
      if (index >= 0)
      {
         for (iD = index*nodeDOF_; iD < (index+1)*nodeDOF_; iD++)
         {
            if (flags[iD] == 0)
            {
               alpha = BCNodeAlpha_[iN][iD%nodeDOF_]; 
               beta  = BCNodeBeta_[iN][iD%nodeDOF_]; 
               gamma = BCNodeGamma_[iN][iD%nodeDOF_]; 
               if (beta == 0.0 && alpha != 0.0)
               {
                  flags[iD] = 1;
                  for (iD2=TdiagIA[iD];iD2<TdiagIA[iD]+diagCounts[iD];iD2++)
                  {
                     rowInd = TdiagJA[iD2];
                     if (rowInd != iD && rowInd >= 0)
                     {
                        for (iD3 = TdiagIA[rowInd];
                             iD3<TdiagIA[rowInd]+diagCounts[rowInd]; iD3++)
                        {
                           if (TdiagJA[iD3] == iD && TdiagAA[iD3] != 0.0)
                           {
                              rhsVector_[rowInd] -= (gamma/alpha*TdiagAA[iD3]); 
                              TdiagAA[iD3] = 0.0;
                              break;
                           }
                        }
                     }
                  }
                  TdiagJA[TdiagIA[iD]] = iD;
                  TdiagAA[TdiagIA[iD]] = 1.0;
                  for (iD2=TdiagIA[iD]+1; iD2<TdiagIA[iD]+diagCounts[iD]; iD2++)
                  {
                     TdiagJA[iD2] = -1;
                     TdiagAA[iD2] = 0.0;
                  }
                  if (ToffdIA != NULL)
                  {
                     for (iD2=ToffdIA[iD];iD2<ToffdIA[iD]+offdCounts[iD];iD2++)
                     {
                        rowInd = ToffdJA[iD2];
                        if (rowInd != iD && rowInd >= 0)
                        {
                           for (iD3 = TdiagIA[rowInd];
                                iD3<TdiagIA[rowInd]+diagCounts[rowInd]; iD3++)
                           {
                              if (TdiagJA[iD3] == iD && TdiagAA[iD3] != 0.0)
                              {
                                 rhsVector_[rowInd]-=(gamma/alpha*TdiagAA[iD3]);
                                 TdiagAA[iD3] = 0.0;
                                 break;
                              }
                           }
                        }
                     }
                     for (iD2=ToffdIA[iD];iD2<ToffdIA[iD]+offdCounts[iD];iD2++)
                     {
                        ToffdJA[iD2] = -1;
                        ToffdAA[iD2] = 0.0;
                     }
                  }
                  rhsVector_[iD] = gamma / alpha;
               }
               else if (beta != 0.0)
               {
                  flags[iD] = 1;
                  for (iD2=TdiagIA[iD]; iD2<TdiagIA[iD]+diagCounts[iD]; iD2++)
                  {
                     rowInd = TdiagJA[iD2];
                     if (rowInd == iD)
                     {
                        TdiagAA[iD2] += alpha / beta;
                        break;
                     }
                  }
                  rhsVector_[iD] += gamma / beta;
               }
            }
         }
      }
      else
      {
         if (numExtNodes_ <= 0)
         {
            printf("%4d : LLNL_FEI_Fei::buildGlobalMatrixVector ERROR(2).\n",
                   mypid_);
            exit(1);
         }
         index = hypre_BinarySearch(&nodeGlobalIDs_[numLocalNodes_],nodeID,
                                    numExtNodes_);
         if (index < 0)
         {
            printf("ERROR : BC node ID not local.\n");
            exit(1);
         }
         index += numLocalNodes_;
         for (iD = index*nodeDOF_; iD < (index+1)*nodeDOF_; iD++)
         {
            if (flags[iD] == 0)
            {
               alpha  = BCNodeAlpha_[iN][iD%nodeDOF_]; 
               beta   = BCNodeBeta_[iN][iD%nodeDOF_]; 
               gamma  = BCNodeGamma_[iN][iD%nodeDOF_]; 
               if (beta == 0.0 && alpha != 0.0)
               {
                  flags[iD] = 1;
                  rowInd = iD + numCRMult_;
                  if (numExtNodes_ > 0)
                  {
                     for (iD2=TdiagIA[rowInd]; 
                          iD2<TdiagIA[rowInd]+diagCounts[rowInd];iD2++)
                     {
                        rowInd2 = TdiagJA[iD2];
                        if (rowInd2 >= 0)
                        {
                           for (iD3 = ToffdIA[rowInd2];
                                iD3<ToffdIA[rowInd2]+offdCounts[rowInd2];iD3++)
                           {
                              if (ToffdJA[iD3] == rowInd && ToffdAA[iD3] != 0.0)
                              {
                                 rhsVector_[rowInd2]-=(gamma/alpha*ToffdAA[iD3]);
                                 ToffdAA[iD3] = 0.0;
                                 break;
                              }
                           }
                        }
                     }
                     for (iD2=ToffdIA[rowInd]; 
                          iD2<ToffdIA[rowInd]+offdCounts[rowInd]; iD2++)
                     {
                        rowInd2 = ToffdJA[iD2];
                        if (rowInd2 != rowInd && rowInd2 >= 0)
                        {
                           for (iD3 = ToffdIA[rowInd2];
                                iD3<ToffdIA[rowInd2]+offdCounts[rowInd2]; iD3++)
                           {
                              if (ToffdJA[iD3] == rowInd && ToffdAA[iD3] != 0.0)
                              {
                                 rhsVector_[rowInd2]-=(gamma/alpha*ToffdAA[iD3]);
                                 ToffdAA[iD3] = 0.0;
                                 break;
                              }
                           }
                        }
                     }
                  }
                  for (iD2=TdiagIA[rowInd]; 
                       iD2<TdiagIA[rowInd]+diagCounts[rowInd]; iD2++)
                  {
                     TdiagJA[iD2] = -1;
                     TdiagAA[iD2] = 0.0;
                  }
                  if ( ToffdIA != NULL )
                  {
                     for (iD2=ToffdIA[rowInd]; 
                          iD2<ToffdIA[rowInd]+offdCounts[rowInd]; iD2++)
                     {
                        ToffdJA[iD2] = -1;
                        ToffdAA[iD2] = 0.0;
                     }
                  }
                  rhsVector_[rowInd] = 0.0;
               }
            }
         }
      }
   }
   if (flags != NULL) delete [] flags;
   gatherAddDData(rhsVector_);
   for (iD = nLocal; iD < matDim; iD++) rhsVector_[iD] = 0.0;

   /* -----------------------------------------------------------------
    * recompute the sparsity structure of the compressed matrix
    * allocate and load the final CSR matrix 
    * -----------------------------------------------------------------*/

   diagNNZ = 0;
   for ( iD = 0; iD < matDim; iD++ ) 
   {
      for ( iD2 = TdiagIA[iD]; iD2 < TdiagIA[iD]+diagCounts[iD]; iD2++ ) 
         if ( TdiagAA[iD2] != 0.0 ) diagNNZ++;
   }
   if ( offdNNZ > 0 )
   {
      offdNNZ = 0;
      for ( iD = 0; iD < matDim; iD++ ) 
         for ( iD2 = ToffdIA[iD]; iD2 < ToffdIA[iD]+offdCounts[iD]; iD2++ ) 
            if ( ToffdAA[iD2] != 0.0 ) offdNNZ++;
   }
   if ( diagNNZ > 0 ) 
   {
      diagIA = new int[matDim+1];
      diagJA = new int[diagNNZ];
      diagAA = new double[diagNNZ];
      diagonal = new double[matDim];
      diagIA[0] = 0;
   }
   else
   {
      diagIA = NULL;
      diagJA = NULL;
      diagAA = NULL;
      diagonal = NULL;
   }
   if ( offdNNZ > 0 ) 
   {
      offdIA = new int[matDim+1];
      offdJA = new int[offdNNZ];
      offdAA = new double[offdNNZ];
      offdIA[0] = 0;
   }
   else
   {
      offdIA = NULL;
      offdJA = NULL;
      offdAA = NULL;
   }
   diagOffset = offdOffset = 0;
   for ( iD = 0; iD < matDim; iD++ ) 
   {
      iCount = diagCounts[iD];
      index  = TdiagIA[iD];
      diagonal[iD] = 0.0;
      for ( iD2 = 0; iD2 < iCount; iD2++ ) 
      {
         if ( TdiagJA[index] == iD ) 
         {
            if ( TdiagAA[index] != 0.0 ) diagonal[iD] = TdiagAA[index];
         }
         if ( TdiagJA[index] >= 0 && TdiagAA[index] != 0.0 ) 
         {
            diagJA[diagOffset] = TdiagJA[index];
            diagAA[diagOffset++] = TdiagAA[index];
         }
         index++; 
      }
      diagIA[iD+1] = diagOffset;
      if ( offdNNZ > 0 ) 
      {
         iCount = offdCounts[iD];
         index  = ToffdIA[iD];
         for ( iD2 = 0; iD2 < iCount; iD2++ ) 
         {
            if ( ToffdJA[index] == iD ) 
            {
               if ( ToffdAA[index] != 0.0 ) diagonal[iD] = ToffdAA[index];
            }
            if ( ToffdJA[index] >= 0 && ToffdAA[index] != 0.0 ) 
            {
               offdJA[offdOffset] = ToffdJA[index];
               offdAA[offdOffset++] = ToffdAA[index];
            }
            index++; 
         }
         offdIA[iD+1] = offdOffset;
      }
   }

   /* -----------------------------------------------------------------
    * fix up diagonal entries in light of parallel processing
    * -----------------------------------------------------------------*/

   gatherAddDData( diagonal );
   for ( iD = 0; iD < matDim; iD++ ) 
   {
      if ( diagonal[iD] == 0.0 ) diagonal[iD] = 1.0;
      else                       diagonal[iD] = 1.0 / diagonal[iD];
   }

   /* -----------------------------------------------------------------
    * assemble initial guess vector
    * -----------------------------------------------------------------*/

   assembleSolnVector();

   /* -----------------------------------------------------------------
    * create matrix class
    * -----------------------------------------------------------------*/

   fetchExtEqnList(&extEqnList);
   MPI_Comm_size(mpiComm_, &nprocs);
   globalEqnOffsets = new int[nprocs+1];
   crOffsets = new int[nprocs+1];
   for ( iP = 0; iP <= nprocs; iP++ )
   {
      globalEqnOffsets[iP] = globalNodeOffsets_[iP] * nodeDOF_ +
                             globalCROffsets_[iP];
      crOffsets[iP] = globalCROffsets_[iP];
   }
   matPtr_->setMatrix(numLocalNodes_*nodeDOF_+numCRMult_, diagIA,
                      diagJA, diagAA, numExtNodes_*nodeDOF_, extEqnList,
                      offdIA, offdJA, offdAA, diagonal, globalEqnOffsets,
                      crOffsets);
   modifyCommPattern(&nRecvs,&recvLengs,&recvProcs,&recvProcIndices,
                     &nSends,&sendLengs,&sendProcs,&sendProcIndices);
   matPtr_->setCommPattern(nRecvs, recvLengs, recvProcs, recvProcIndices,
                           nSends, sendLengs, sendProcs, sendProcIndices);

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/
      
   if ( matDim > 0 )
   {
      delete [] diagCounts;
      delete [] offdCounts;
   }
   if ( diagNNZ > 0 ) 
   {
      delete [] TdiagIA;
      delete [] TdiagJA;
      delete [] TdiagAA;
   }
   if ( offdNNZ > 0 ) 
   {
      delete [] ToffdIA;
      delete [] ToffdJA;
      delete [] ToffdAA;
   }
   if ( outputLevel_ > 2 )
      printf("%4d : LLNL_FEI_Fei::buildGlobalMatrixVector ends. \n",mypid_);
}

/**************************************************************************
 form right hand side vector from element load vectors 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::assembleRHSVector()
{
   int    iB, iE, iN, iD, **elemNodeLists, numElems, elemNumNodes;
   int    eqnIndex1, eqnIndex2, matDim, nLocal;
   double **rhsVectors;

   if ( rhsVector_ != NULL ) delete [] rhsVector_;
   matDim = (numLocalNodes_ + numExtNodes_) * nodeDOF_ + numCRMult_;
   nLocal = numLocalNodes_ * nodeDOF_;
   rhsVector_ = new double[matDim];
   for ( iD = 0; iD < matDim; iD++ ) rhsVector_[iD] = 0.0;
   for ( iD = nLocal; iD < nLocal+numCRMult_; iD++ ) 
      rhsVector_[iD] = CRValues_[iD-nLocal];

   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      rhsVectors    = elemBlocks_[iB]->getRHSVectors();
      numElems      = elemBlocks_[iB]->getNumElems();
      elemNumNodes  = elemBlocks_[iB]->getElemNumNodes();
      for ( iE = 0; iE < numElems; iE++ )
      {
         for ( iN = 0; iN < elemNumNodes; iN++ )
         {
            eqnIndex1 = elemNodeLists[iE][iN] * nodeDOF_;
            if ( eqnIndex1 >= nLocal ) eqnIndex1 += numCRMult_;
            eqnIndex2 = iN * nodeDOF_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
               rhsVector_[eqnIndex1+iD]  += rhsVectors[iE][eqnIndex2+iD];
         }
      }
   }
   gatherAddDData( rhsVector_ );
   scatterDData( rhsVector_ );
}

/**************************************************************************
 form solution vector 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::assembleSolnVector()
{
   int    iB, iE, iN, iD, **elemNodeLists, numElems, elemNumNodes;
   int    eqnIndex1, eqnIndex2, matDim, nLocal;
   double **solnVectors;

   matDim = (numLocalNodes_ + numExtNodes_) * nodeDOF_ + numCRMult_;
   nLocal = numLocalNodes_ * nodeDOF_;
   if ( solnVector_ == NULL ) solnVector_ = new double[matDim];
   for ( iD = 0; iD < matDim; iD++ ) solnVector_[iD] = 0.0;
   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      solnVectors   = elemBlocks_[iB]->getSolnVectors();
      numElems      = elemBlocks_[iB]->getNumElems();
      elemNumNodes  = elemBlocks_[iB]->getElemNumNodes();
      for ( iE = 0; iE < numElems; iE++ )
      {
         for ( iN = 0; iN < elemNumNodes; iN++ )
         {
            eqnIndex1 = elemNodeLists[iE][iN] * nodeDOF_;
            if ( eqnIndex1 >= nLocal ) eqnIndex1 += numCRMult_;
            eqnIndex2 = iN * nodeDOF_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
               solnVector_[eqnIndex1+iD] += solnVectors[iE][eqnIndex2+iD];
         }
      }
   }
   gatherAddDData( solnVector_ );
   scatterDData( solnVector_ );
}

/**************************************************************************
 distribute solution vector to element solution vectors
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::disassembleSolnVector(double *solns)
{
   int    iB, iE, iN, iD, **elemNodeLists, numElems, elemNumNodes;
   int    eqnIndex1, eqnIndex2, nLocal;
   double **solnVectors;

   nLocal = numLocalNodes_ * nodeDOF_;
   for ( iD = 0; iD < nLocal; iD++ ) solnVector_[iD] = solns[iD];
   scatterDData( solnVector_ );
   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      elemNodeLists = elemBlocks_[iB]->getElemNodeLists();
      solnVectors   = elemBlocks_[iB]->getSolnVectors();
      numElems      = elemBlocks_[iB]->getNumElems();
      elemNumNodes  = elemBlocks_[iB]->getElemNumNodes();
      for ( iE = 0; iE < numElems; iE++ )
      {
         for ( iN = 0; iN < elemNumNodes; iN++ )
         {
            eqnIndex1 = elemNodeLists[iE][iN] * nodeDOF_;
            if ( eqnIndex1 >= nLocal ) eqnIndex1 += numCRMult_;
            eqnIndex2 = iN * nodeDOF_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
               solnVectors[iE][eqnIndex2+iD] = solnVector_[eqnIndex1+iD]; 
         }
      }
   }
}

/**************************************************************************
 sort an integer array
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::IntSort(int *ilist, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   IntSort(ilist, left, last-1);
   IntSort(ilist, last+1, right);
}

/**************************************************************************
 sort an integer array and an auxiliary array
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::IntSort2(int *ilist, int *ilist2, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   itemp        = ilist2[left];
   ilist2[left] = ilist2[mid];
   ilist2[mid]  = itemp;
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
         itemp        = ilist2[last];
         ilist2[last] = ilist2[i];
         ilist2[i]    = itemp;
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   itemp        = ilist2[left];
   ilist2[left] = ilist2[last];
   ilist2[last] = itemp;
   IntSort2(ilist, ilist2, left, last-1);
   IntSort2(ilist, ilist2, last+1, right);
}

/**************************************************************************
 sort an integer array with an auxiliary double array
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::IntSort2a(int *ilist,double *dlist,int left,int right)
{
   int    mid, i, itemp, last, end2, isort, *ilist2, *ilist3;
   double dtemp, *dlist2, *dlist3;

   if (left >= right) return;
   mid         = (left + right) / 2;
   itemp       = ilist[left];
   ilist[left] = ilist[mid];
   ilist[mid]  = itemp;
   dtemp       = dlist[left];
   dlist[left] = dlist[mid];
   dlist[mid]  = dtemp;
   last        = left;
   isort       = ilist[left];
   ilist2      = &(ilist[last]);
   dlist2      = &(dlist[last]);
   ilist3      = &(ilist[left+1]);
   dlist3      = &(dlist[left+1]);
   end2        = right + 1;
   for (i = left+1; i < end2; i++)
   {
      if ( *ilist3 < isort )
      {
         last++;
         ilist2++; dlist2++;
         itemp   = *ilist2;
         *ilist2 = *ilist3;
         *ilist3 = itemp;
         dtemp   = *dlist2;
         *dlist2 = *dlist3;
         *dlist3 = dtemp;
      }
      ilist3++; dlist3++;
   }
   itemp       = ilist[left];
   ilist[left] = ilist[last];
   ilist[last] = itemp;
   dtemp       = dlist[left];
   dlist[left] = dlist[last];
   dlist[last] = dtemp;
   IntSort2a(ilist, dlist, left, last-1);
   IntSort2a(ilist, dlist, last+1, right);
}

/**************************************************************************
 exchange extended vectors between processors
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::scatterDData( double *dvec )
{
   int         iD, iD2, iP, ind1, ind2, offset;
   double      *dRecvBufs, *dSendBufs;
   MPI_Request *requests;
   MPI_Status  status;

   if ( nRecvs_ > 0 ) 
   {
      offset = 0;
      for ( iP = 0; iP < nRecvs_; iP++ ) offset += recvLengs_[iP];
      dRecvBufs = new double[offset*nodeDOF_];
      requests  = new MPI_Request[nRecvs_];
   }
   if ( nSends_ > 0 ) 
   {
      offset = 0;
      for ( iP = 0; iP < nSends_; iP++ ) offset += sendLengs_[iP];
      dSendBufs = new double[offset*nodeDOF_];
      offset = 0;
      for ( iP = 0; iP < nSends_; iP++ ) 
      {
         for ( iD = 0; iD < sendLengs_[iP]; iD++ )
         {
            ind1 = sendProcIndices_[offset+iD] * nodeDOF_;
            ind2 = iD * nodeDOF_;
            for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
               dSendBufs[offset*nodeDOF_+ind2+iD2] = dvec[ind1+iD2]; 
         }
         offset += sendLengs_[iP];
      }
   }
   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      MPI_Irecv( &dRecvBufs[offset], recvLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                 recvProcs_[iP], 40343, mpiComm_, &requests[iP]);
      offset += recvLengs_[iP] * nodeDOF_;
   }
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      MPI_Send( &dSendBufs[offset], sendLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                sendProcs_[iP], 40343, mpiComm_);
      offset += sendLengs_[iP] * nodeDOF_;
   }
   for ( iP = 0; iP < nRecvs_; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nRecvs_ > 0 ) delete [] requests;
   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      for ( iD = 0; iD < recvLengs_[iP]; iD++ )
      {
         ind1 = recvProcIndices_[offset+iD] * nodeDOF_ + numCRMult_;
         ind2 = iD * nodeDOF_;
         for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
            dvec[ind1+iD2] = dRecvBufs[offset*nodeDOF_+ind2+iD2]; 
      }
      offset += recvLengs_[iP];
   }
   if ( nRecvs_ > 0 ) delete [] dRecvBufs;
   if ( nSends_ > 0 ) delete [] dSendBufs;
}

/**************************************************************************
 exchange data between processors 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::gatherAddDData( double *dvec )
{
   int         iD, iD2, iP, ind1, ind2, offset;
   double      *dRecvBufs, *dSendBufs;
   MPI_Request *requests;
   MPI_Status  status;

   if ( nSends_ > 0 ) 
   {
      offset = 0;
      for ( iP = 0; iP < nSends_; iP++ ) offset += sendLengs_[iP];
      dRecvBufs = new double[offset*nodeDOF_];
      requests  = new MPI_Request[nSends_];
   }
   if ( nRecvs_ > 0 ) 
   {
      offset = 0;
      for ( iP = 0; iP < nRecvs_; iP++ ) offset += recvLengs_[iP];
      dSendBufs = new double[offset*nodeDOF_];
      offset = 0;
      for ( iP = 0; iP < nRecvs_; iP++ ) 
      {
         for ( iD = 0; iD < recvLengs_[iP]; iD++ )
         {
            ind1 = recvProcIndices_[offset+iD] * nodeDOF_ + numCRMult_;
            ind2 = iD * nodeDOF_;
            for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
               dSendBufs[offset*nodeDOF_+ind2+iD2] = dvec[ind1+iD2]; 
         }
         offset += recvLengs_[iP];
      }
   }
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      MPI_Irecv( &dRecvBufs[offset], sendLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                 sendProcs_[iP], 40342, mpiComm_, &requests[iP]);
      offset += sendLengs_[iP] * nodeDOF_;
   }
   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      MPI_Send( &dSendBufs[offset], recvLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                recvProcs_[iP], 40342, mpiComm_);
      offset += recvLengs_[iP] * nodeDOF_;
   }
   for ( iP = 0; iP < nSends_; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nSends_ > 0 ) delete [] requests;
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      for ( iD = 0; iD < sendLengs_[iP]; iD++ )
      {
         ind1 = sendProcIndices_[offset+iD] * nodeDOF_;
         ind2 = iD * nodeDOF_;
         for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
            dvec[ind1+iD2] += dRecvBufs[offset*nodeDOF_+ind2+iD2]; 
      }
      offset += sendLengs_[iP];
   }
   if ( nSends_ > 0 ) delete [] dRecvBufs;
   if ( nRecvs_ > 0 ) delete [] dSendBufs;
}

/**************************************************************************
 exchange data between processors 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::gatherIData( int *iSendBuf, int *iRecvBuf )
{
   int         iP, msgid=40342, offset, length;
   MPI_Request *requests;
   MPI_Status  status;

   if ( nSends_ > 0 ) requests  = new MPI_Request[nSends_];
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      length = sendLengs_[iP] * nodeDOF_;
      MPI_Irecv( &iRecvBuf[offset], length, MPI_INT, sendProcs_[iP], 
                 msgid, mpiComm_, &requests[iP]);
      offset += length;
   }
   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      length = recvLengs_[iP] * nodeDOF_;
      MPI_Send( &iSendBuf[offset], length, MPI_INT, recvProcs_[iP], 
                msgid, mpiComm_);
      offset += length;
   }
   for ( iP = 0; iP < nSends_; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nSends_ > 0 ) delete [] requests;
}

/**************************************************************************
 exchange data between processors 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::gatherDData( double *dSendBuf, double *dRecvBuf )
{
   int         iP, msgid=40343, offset, length;
   MPI_Request *requests;
   MPI_Status  status;

   if ( nSends_ > 0 ) requests = new MPI_Request[nSends_];
   offset = 0;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      length = sendLengs_[iP] * nodeDOF_;
      MPI_Irecv( &dRecvBuf[offset], length, MPI_DOUBLE, sendProcs_[iP], 
                 msgid, mpiComm_, &requests[iP]);
      offset += length;
   }
   offset = 0;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      length = recvLengs_[iP] * nodeDOF_;
      MPI_Send( &dSendBuf[offset], length, MPI_DOUBLE, recvProcs_[iP], 
                msgid, mpiComm_);
      offset += length;
   }
   for ( iP = 0; iP < nSends_; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nSends_ > 0 ) delete [] requests;
}

/**************************************************************************
 sort the shared nodes
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::sortSharedNodes()
{
   int *nodeIDs, *nodeIDAux, **sharedNodeProcAux, iN, index, iP; 

   /* -----------------------------------------------------------------
    * sort the shared nodes
    * ----------------------------------------------------------------*/

   if ( numSharedNodes_ > 0 )
   {
      nodeIDs   = new int[numSharedNodes_];
      nodeIDAux = new int[numSharedNodes_];
      sharedNodeProcAux = new int*[numSharedNodes_];
      for ( iN = 0; iN < numSharedNodes_; iN++ ) nodeIDs[iN] = iN;
      IntSort2(sharedNodeIDs_, nodeIDs, 0, numSharedNodes_-1);
      for ( iN = 0; iN < numSharedNodes_; iN++ ) 
      {
         sharedNodeProcAux[iN] = sharedNodeProcs_[iN]; 
         nodeIDAux[iN] = sharedNodeNProcs_[iN];
      }
      for ( iN = 0; iN < numSharedNodes_; iN++ ) 
      {
         index = nodeIDs[iN];
         sharedNodeProcs_[iN] = sharedNodeProcAux[index]; 
         sharedNodeNProcs_[iN] = nodeIDAux[index];
      }
      delete [] sharedNodeProcAux;
      delete [] nodeIDAux;
      delete [] nodeIDs;
      index = 0;
      for ( iN = 1; iN < numSharedNodes_; iN++ ) 
      {
         if ( sharedNodeIDs_[iN] == sharedNodeIDs_[index] )
         {
            nodeIDAux = sharedNodeProcs_[index];
            sharedNodeProcs_[index] = 
               new int[sharedNodeNProcs_[index]+sharedNodeNProcs_[iN]];
            for ( iP = 0; iP < sharedNodeNProcs_[index]; iP++ ) 
               sharedNodeProcs_[index][iP] = nodeIDAux[iP]; 
            for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ ) 
               sharedNodeProcs_[index][sharedNodeNProcs_[index]+iP] = 
                                       sharedNodeProcs_[iN][iP];

            sharedNodeNProcs_[index] += sharedNodeNProcs_[iN];
            delete [] nodeIDAux;
            delete [] sharedNodeProcs_[iN];
         }
         else
         {
            index++;
            sharedNodeIDs_[index] = sharedNodeIDs_[iN];
            sharedNodeProcs_[index] = sharedNodeProcs_[iN];
            sharedNodeNProcs_[index] = sharedNodeNProcs_[iN];
         }
      }
      if ( numSharedNodes_ > 0 ) numSharedNodes_ = index + 1;
      for ( iN = 0; iN < numSharedNodes_; iN++ ) 
      {
         IntSort(sharedNodeProcs_[iN], 0, sharedNodeNProcs_[iN]-1);
         index = 0;
         for ( iP = 1; iP < sharedNodeNProcs_[iN]; iP++ ) 
            if (sharedNodeProcs_[iN][iP] != sharedNodeProcs_[iN][index])
               sharedNodeProcs_[iN][++index] = sharedNodeProcs_[iN][iP];
         sharedNodeNProcs_[iN] = index + 1;
      }
   }
}

/**************************************************************************
 * obtain an ordered array of distinct node IDs
 * (nodeGlobalIDs_, numLocalNodes_)
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::composeOrderedNodeIDList(int **nodeIDs_out, 
                           int **nodeIDAux_out, int *totalNNodes_out,
                           int *CRNNodes_out)
{
   int NNodes, iB, nElems, elemNNodes, *nodeIDs=NULL, **elemNodeList=NULL;
   int iE, iN, iD, *nodeIDAux=NULL, CRNNodes, totalNNodes;

   /* ------------------------------------------------------- */
   /* count the total number of nodes (ID can be repeated)    */
   /* ------------------------------------------------------- */

   NNodes = 0;
   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      nElems     = elemBlocks_[iB]->getNumElems();
      elemNNodes = elemBlocks_[iB]->getElemNumNodes();
      NNodes    += nElems * elemNNodes;
   }
   CRNNodes = numCRMult_ * CRListLen_;
   totalNNodes = NNodes + CRNNodes;

   /* ------------------------------------------------------- */
   /* allocate space and fetch the global node numbers        */
   /* ------------------------------------------------------- */

   if ( totalNNodes > 0 ) nodeIDs = new int[totalNNodes];
   totalNNodes = 0;
   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      nElems       = elemBlocks_[iB]->getNumElems();
      elemNNodes   = elemBlocks_[iB]->getElemNumNodes();
      elemNodeList = elemBlocks_[iB]->getElemNodeLists();
      for ( iE = 0; iE < nElems; iE++ )
         for ( iN = 0; iN < elemNNodes; iN++ )
            nodeIDs[totalNNodes++] = elemNodeList[iE][iN];
   }
   for ( iN = 0; iN < numCRMult_; iN++ )
      for ( iD = 0; iD < CRListLen_; iD++ )
         nodeIDs[totalNNodes++] = CRNodeLists_[iN][iD];
   
   /* -- sort the global node numbers (ordering in nodeIDAux) -- */

   if ( totalNNodes > 0 ) nodeIDAux = new int[totalNNodes];
   for ( iN = 0; iN < totalNNodes; iN++ ) nodeIDAux[iN] = iN;
   IntSort2(nodeIDs, nodeIDAux, 0, totalNNodes-1);
   (*nodeIDs_out) = nodeIDs;
   (*nodeIDAux_out) = nodeIDAux;
   (*totalNNodes_out) = totalNNodes;
   (*CRNNodes_out) = CRNNodes;
}

/**************************************************************************
 * find which processor each shared node belongs to
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::findSharedNodeProcs(int *nodeIDs, int *nodeIDAux, 
                            int totalNNodes, int CRNNodes,
                            int **sharedNodePInfo_out) 
{
   int *sharedNodePInfo, iN, minProc, NNodes;
   int nprocs, index, *sharedNodePInds, iN2;

   MPI_Comm_size( mpiComm_, &nprocs);
   
   if ( numSharedNodes_ == 0 ) 
   {
      (*sharedNodePInfo_out) = NULL;
      return;
   }
   sharedNodePInfo = new int[numSharedNodes_];
   sharedNodePInds = new int[numSharedNodes_];
   NNodes = totalNNodes - CRNNodes;
   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      /* --- search for processor with lowest rank to be owner --- */

      index = hypre_BinarySearch(nodeIDs,sharedNodeIDs_[iN],totalNNodes);
      sharedNodePInds[iN] = -1;
#if 0
      /* Charles: 3/5/07 : sorted already, no need to do it again */
      minProc = nprocs;
      for (int iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
      {
         int pindex = sharedNodeProcs_[iN][iP];
         if ( pindex < minProc )
            minProc = pindex;
      }
#else
      minProc = sharedNodeProcs_[iN][0];
#endif
      /* pind = -ve if my processor doesn't own nor need the shared node
       * pind = 0:nprocs-1 if my processor owns or needs the shared node
         pind >= nprocs if my processor needs but does not own it */
      if ( index >= 0 )
      {
         /* this segment is needed for comparison with NNodes */
         for ( iN2 = index-1; iN2 >= 0; iN2-- )
         {
            if ( nodeIDs[iN2] != nodeIDs[index] ) break;
            if ( nodeIDAux[iN2] < nodeIDAux[index] ) index = iN2;
         }    
         for ( iN2 = index+1; iN2 < totalNNodes; iN2++ )
         {
            if ( nodeIDs[iN2] != nodeIDs[index] ) break;
            if ( nodeIDAux[iN2] < nodeIDAux[index] ) index = iN2;
         }    
         sharedNodePInds[iN] = index;
         if (nodeIDAux[index] < NNodes && mypid_ < minProc) minProc = mypid_;
         if (nodeIDAux[index] >= NNodes) minProc += nprocs;
         sharedNodePInfo[iN] = minProc;
      }
      else sharedNodePInfo[iN] = - minProc - 1;
   }
   findSharedNodeOwners( sharedNodePInfo );

   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      if ( sharedNodePInfo[iN] != mypid_ )
      {
         index = sharedNodePInds[iN];
         if ( index >= 0 && nodeIDAux[index] >= 0 )
         {
            for ( iN2 = index-1; iN2 >= 0; iN2-- )
            {
               if ( nodeIDs[iN2] == nodeIDs[index] ) 
                  nodeIDAux[iN2] = - nodeIDAux[iN2] - 1;
               else break;
            }
            for ( iN2 = index+1; iN2 < totalNNodes; iN2++ )
            {
               if ( nodeIDs[iN2] == nodeIDs[index] ) 
                  nodeIDAux[iN2] = - nodeIDAux[iN2] - 1;
               else break;
            }
            nodeIDAux[index] = - nodeIDAux[index] - 1;
         }
      }
   }
   delete [] sharedNodePInds;
   (*sharedNodePInfo_out) = sharedNodePInfo;
}

/**************************************************************************
 * obtain an ordered array of distinct node IDs
 * (nodeGlobalIDs_, numLocalNodes_)
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::findSharedNodeOwners( int *sharedNodePInfo )
{
   int nComm, nCommAux, *commProcs, *commLengs, nprocs, iN, iP; 
   int **sbuffer, **rbuffer, pindex, sindex, index, minProc;
   MPI_Request *requests;
   MPI_Status  status;

   /* --- find number of distinct processors (nComm, commProcs) --- */

   MPI_Comm_size( mpiComm_, &nprocs );
   nComm = 0;
   for ( iN = 0; iN < numSharedNodes_; iN++ )
      for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
         if ( sharedNodeProcs_[iN][iP] != mypid_ ) nComm++;
   if ( nComm > 0 ) commProcs = new int[nComm];
   nComm = 0;
   for ( iN = 0; iN < numSharedNodes_; iN++ )
      for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
         if ( sharedNodeProcs_[iN][iP] != mypid_ ) 
            commProcs[nComm++] = sharedNodeProcs_[iN][iP];
   if ( nComm > 0 ) IntSort(commProcs, 0, nComm-1);
   nCommAux = nComm;
   nComm = 0;
   for ( iP = 1; iP < nCommAux; iP++ )
   {
      if ( commProcs[iP] != commProcs[nComm] )
         commProcs[++nComm] = commProcs[iP];
   }
   if ( nCommAux > 0 ) nComm++;

   /* --- allocate buffers (commLengs, sbuffer, rbuffer) --- */

   if ( nComm > 0 ) commLengs = new int[nComm];
   for ( iP = 0; iP < nComm; iP++ ) commLengs[iP] = 0;
   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
      {
         pindex = sharedNodeProcs_[iN][iP];
         if ( pindex != mypid_ ) 
         {
            sindex = hypre_BinarySearch(commProcs,pindex,nComm);
            commLengs[sindex]++;
         } 
      } 
   } 
   if ( nComm > 0 ) sbuffer = new int*[nComm];
   if ( nComm > 0 ) rbuffer = new int*[nComm];
   for ( iP = 0; iP < nComm; iP++ )
   {
      sbuffer[iP] = new int[commLengs[iP]];
      rbuffer[iP] = new int[commLengs[iP]];
      commLengs[iP] = 0; 
   }

   /* --- fill buffers --- */

   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
      {
         pindex = sharedNodeProcs_[iN][iP];
         if ( pindex != mypid_ ) 
         {
            sindex = hypre_BinarySearch(commProcs,pindex,nComm);
            sbuffer[sindex][commLengs[sindex]++] = sharedNodePInfo[iN];
         } 
      } 
   } 

   /* --- communicate --- */

   if ( nComm > 0 ) requests = new MPI_Request[nComm];
   for ( iP = 0; iP < nComm; iP++ )
      MPI_Irecv( rbuffer[iP], commLengs[iP], MPI_INT,
                 commProcs[iP], 89034, mpiComm_, &requests[iP]);
   for ( iP = 0; iP < nComm; iP++ )
      MPI_Send( sbuffer[iP], commLengs[iP], MPI_INT,
                commProcs[iP], 89034, mpiComm_);
   for ( iP = 0; iP < nComm; iP++ ) MPI_Wait( &requests[iP], &status );
   if ( nComm > 0 ) delete [] requests;

   /* --- update --- */

   for ( iP = 0; iP < nComm; iP++ ) commLengs[iP] = 0; 
   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      /* first take out all processors which disavow themselves */
      for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
      {
         pindex = sharedNodeProcs_[iN][iP];
         if ( pindex != mypid_ ) 
         {
            sindex = hypre_BinarySearch(commProcs,pindex,nComm);
            index = rbuffer[sindex][commLengs[sindex]++];
            if ( index < 0 ) 
               sharedNodeProcs_[iN][iP] = - sharedNodeProcs_[iN][iP] - 1;
            else if ( index >= nprocs ) 
               sharedNodeProcs_[iN][iP] += nprocs; 
         } 
      } 
      /* now choose one from the rest of the processors */
      minProc = nprocs;
      for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
      {
         pindex = sharedNodeProcs_[iN][iP];
         if ( pindex >= nprocs ) sharedNodeProcs_[iN][iP] -= nprocs; 
         else if ( mypid_ != pindex && pindex >= 0 ) 
         {
            if ( pindex < minProc ) minProc = pindex;
         } 
      } 
      index = sharedNodePInfo[iN];
      if (index >= 0 && index < nprocs && mypid_ < minProc) 
         minProc = mypid_;
      sharedNodePInfo[iN] = minProc;
   } 

   /* --- clean up --- */

   if ( nComm > 0 ) delete [] commProcs;
   if ( nComm > 0 ) delete [] commLengs;
   for ( iP = 0; iP < nComm; iP++ )
   {
      delete [] sbuffer[iP];
      delete [] rbuffer[iP];
   }
   if ( nComm > 0 ) delete [] sbuffer;
   if ( nComm > 0 ) delete [] rbuffer;
   return;
}

/**************************************************************************
 * setupCommPattern 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::setupCommPattern( int *sharedNodePInfo )
{
   int   iP, iN, iN2, index, index2, *ownerProcs, *sndrcvReg, *pArrayAux;
   int   pCounts, nRecv, *recvLengs, *recvProcs, *recvBuf;
   int   nSend, *sendLengs, *sendProcs, *sendBuf, nodeOffset, pindex;
   int   nodeID, offset, *tLengs;
   MPI_Request *requests;
   MPI_Status  status;

   /* allocate temporary variables */

   ownerProcs = NULL;
   pArrayAux  = NULL;
   sndrcvReg  = NULL;
   if ( numExtNodes_    > 0 ) ownerProcs = new int[numExtNodes_];
   if ( numExtNodes_    > 0 ) pArrayAux  = new int[numExtNodes_];
   if ( numSharedNodes_ > 0 ) sndrcvReg  = new int[numSharedNodes_];
   nodeOffset = globalNodeOffsets_[mypid_];

   /* for all shared nodes, see if they are recv or send nodes */

   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      nodeID = sharedNodeIDs_[iN];
      index  = -1;
      if ( numExtNodes_ > 0 )
         index = hypre_BinarySearch(&(nodeGlobalIDs_[numLocalNodes_]),nodeID,
                                    numExtNodes_);
      if ( index >= 0 )
      {
         sndrcvReg[iN] = 1; // recv
         ownerProcs[index] = sharedNodePInfo[iN];
         pArrayAux[index] = sharedNodePInfo[iN];
      }
      else 
      {
         index2 = hypre_BinarySearch(nodeGlobalIDs_,nodeID,numLocalNodes_);
         if ( index2 >= 0 ) sndrcvReg[iN] = 0; // send
         else               sndrcvReg[iN] = -1;
      }
   }

   /* based on the pArrayAux and ownerProcs arrays, compose */
   /* receive information (nRecv, recvLengs, recvProcs) */

   nRecv      = 0;
   recvProcs  = NULL;
   recvLengs  = NULL;
   recvBuf    = NULL;
   if (numExtNodes_ > 0) IntSort(pArrayAux, 0, numExtNodes_-1);
   if ( numExtNodes_ > 0 ) nRecv = 1;
   for ( iN = 1; iN < numExtNodes_; iN++ )
      if ( pArrayAux[iN] != pArrayAux[nRecv-1] )
         pArrayAux[nRecv++] = pArrayAux[iN];
   if ( nRecv > 0 )
   {
      recvProcs = new int[nRecv];
      for ( iP = 0; iP < nRecv; iP++ ) recvProcs[iP] = pArrayAux[iP];
      recvLengs = new int[nRecv];
      for ( iP = 0; iP < nRecv; iP++ ) recvLengs[iP] = 0;
      for ( iN = 0; iN < numSharedNodes_; iN++ )
      {
         if ( sndrcvReg[iN] == 1 ) 
         {
            nodeID = sharedNodeIDs_[iN];
            index = hypre_BinarySearch(&(nodeGlobalIDs_[numLocalNodes_]),
                                       nodeID,numExtNodes_);
            index = hypre_BinarySearch(recvProcs,ownerProcs[index],nRecv);
            recvLengs[index]++;
         }
      }
      offset = 0;
      for ( iP = 0; iP < nRecv; iP++ ) offset += recvLengs[iP];
      recvBuf = new int[offset];
   }
   delete [] pArrayAux;

   /* compose send information (nSend, sendLengs, sendProcs) */

   pCounts = 0;
   for ( iN = 0; iN < numSharedNodes_; iN++ )
      if ( sndrcvReg[iN] == 0 ) pCounts += sharedNodeNProcs_[iN];
   if ( pCounts > 0 ) pArrayAux = new int[pCounts];
   pCounts = 0;
   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      if ( sndrcvReg[iN] == 0 )
      {
         for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
         {
            pindex = sharedNodeProcs_[iN][iP];
            if ( pindex >= 0 && pindex != mypid_ ) 
               pArrayAux[pCounts++] = pindex;
         }
      }
   }
   nSend     = 0;
   sendProcs = NULL;
   sendLengs = NULL;

   if ( pCounts > 0 )
   {
      IntSort( pArrayAux, 0, pCounts-1 );
      nSend = 1;
      for ( iP = 1; iP < pCounts; iP++ )
         if ( pArrayAux[iP] !=  pArrayAux[nSend-1] )
            pArrayAux[nSend++] = pArrayAux[iP];
      if ( nSend > 0 ) sendProcs = new int[nSend];
      for ( iP = 0; iP < nSend; iP++ ) sendProcs[iP] = pArrayAux[iP];
      sendLengs = new int[nSend];
      for ( iP = 0; iP < nSend; iP++ ) sendLengs[iP] = 0;
      for ( iN = 0; iN < numSharedNodes_; iN++ )
      {
         if ( sndrcvReg[iN] == 0 )
         {
            for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
            {
               pindex = sharedNodeProcs_[iN][iP];
               if ( pindex >= 0 && pindex != mypid_ )
               {
                  index = sharedNodeProcs_[iN][iP];
                  index = hypre_BinarySearch(sendProcs,index,nSend);
                  sendLengs[index]++;
               }
            }
         }
      }
      offset = 0;
      for ( iP = 0; iP < nSend; iP++ ) offset += sendLengs[iP];
      if ( offset > 0 ) sendBuf = new int[offset];
      tLengs = new int[nSend+1];
      tLengs[0] = 0;
      for (iP = 1; iP < nSend; iP++) tLengs[iP] = tLengs[iP-1]+sendLengs[iP-1];
      for ( iP = 0; iP < nSend; iP++ ) sendLengs[iP] = 0;
      for ( iN = 0; iN < numSharedNodes_; iN++ )
      {
         if ( sndrcvReg[iN] == 0 )
         {
            for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
            {
               pindex = sharedNodeProcs_[iN][iP];
               if ( pindex >= 0 && pindex != mypid_ )
               {
                  index  = hypre_BinarySearch(sendProcs,pindex,nSend);
                  index2 = hypre_BinarySearch(nodeGlobalIDs_,
                                        sharedNodeIDs_[iN],numLocalNodes_);
                  sendBuf[tLengs[index]+sendLengs[index]] = nodeOffset + index2;
                  sendLengs[index]++;
               }
            }
         }
      }
      delete [] tLengs;
   }
   if ( pCounts > 0 ) delete [] pArrayAux;

   /* -- exchange the (NEW) global node indices -- */

   if ( nRecv > 0 ) requests = new MPI_Request[nRecv];
   offset = 0;
   for ( iP = 0; iP < nRecv; iP++ )
   {
      MPI_Irecv( &recvBuf[offset], recvLengs[iP], MPI_INT,
                 recvProcs[iP], 183, mpiComm_, &requests[iP]);
      offset += recvLengs[iP];
   }
   offset = 0;
   for ( iP = 0; iP < nSend; iP++ )
   {
      MPI_Send( &sendBuf[offset], sendLengs[iP], MPI_INT,
                sendProcs[iP], 183, mpiComm_);
      offset += sendLengs[iP];
   }
   for ( iP = 0; iP < nRecv; iP++ ) MPI_Wait( &requests[iP], &status );
   if ( nRecv > 0 ) delete [] requests;

   /* -- fix the send index array -- */

   offset = 0;
   for (iP = 0; iP < nSend; iP++) 
   {
      for ( iN = 0; iN < sendLengs[iP]; iN++ )
         sendBuf[offset+iN] -= nodeOffset;
      offset += sendLengs[iP];
   }

   /* -- based on the recv information, construct recv index array -- */

   if ( numExtNodes_ > 0 ) nodeExtNewGlobalIDs_ = new int[numExtNodes_];
   tLengs = new int[nRecv+1];
   tLengs[0] = 0;
   for ( iP = 1; iP < nRecv; iP++ ) tLengs[iP] = tLengs[iP-1] + recvLengs[iP-1];
   for ( iP = 0; iP < nRecv; iP++ ) recvLengs[iP] = 0;
   for ( iN = 0; iN < numExtNodes_; iN++ )
   {
      index = hypre_BinarySearch(recvProcs, ownerProcs[iN], nRecv);
      iN2 = recvBuf[tLengs[index]+recvLengs[index]];
      nodeExtNewGlobalIDs_[iN] = iN2;
      recvBuf[tLengs[index]+recvLengs[index]] = iN + numLocalNodes_;
      recvLengs[index]++;
   }
   delete [] tLengs;
   if ( numSharedNodes_ > 0 ) delete [] sndrcvReg;
   if ( numExtNodes_    > 0 ) delete [] ownerProcs;

   /* -- construct the receive communication pattern -- */

   nRecvs_ = nRecv;
   if ( nRecv > 0 ) 
   {
      recvProcs_ = recvProcs;
      recvLengs_ = recvLengs;
   }
   else recvProcs_ = recvLengs_ = NULL;
   if ( nRecv > 0 ) recvProcIndices_ = recvBuf;
   else             recvProcIndices_ = NULL;
   
   /* -- construct the send communication pattern -- */

   nSends_ = nSend;
   if ( nSend > 0 ) 
   {
      sendLengs_ = sendLengs;
      sendProcs_ = sendProcs;
   } 
   else sendLengs_ = sendProcs_ = NULL;
   if ( nSend > 0 ) sendProcIndices_ = sendBuf;
   else             sendProcIndices_ = NULL;
}

/**************************************************************************
 fetch the matrix 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::getMatrix(LLNL_FEI_Matrix **mat)
{
   if ( FLAG_LoadComplete_ == 0 ) loadComplete();
   (*mat) = matPtr_;
}

/**************************************************************************
 get the off processor column indices 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::fetchExtEqnList(int **eqnList)
{
   int iN, iD, iP, index, offset;

   (*eqnList) = NULL;
   if ( numExtNodes_ == 0 ) return;
   (*eqnList) = new int[numExtNodes_ * nodeDOF_];
   if ( globalCROffsets_ == NULL )
   {
      for ( iN = 0; iN < numExtNodes_; iN++ )
         for ( iD = 0; iD < nodeDOF_; iD++ )
            (*eqnList)[iN*nodeDOF_+iD] = nodeExtNewGlobalIDs_[iN]*nodeDOF_+
                                         iD;
   }
   else
   {
      offset = 0;
      for ( iP = 0; iP < nRecvs_; iP++ )
      {
         for ( iN = 0; iN < recvLengs_[iP]; iN++ )
         {
            index = recvProcIndices_[offset+iN] - numLocalNodes_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
            {
               (*eqnList)[index*nodeDOF_+iD] = 
                       nodeExtNewGlobalIDs_[index] * nodeDOF_ + iD +
                       globalCROffsets_[recvProcs_[iP]];
            }
         }
         offset += recvLengs_[iP];
      }
   }
}

/**************************************************************************
 get the off processor column indices 
 -------------------------------------------------------------------------*/
void LLNL_FEI_Fei::modifyCommPattern(int *nrecvs, int **recvlengs, 
                        int **recvprocs, int **recvindices, int *nsends,
                        int **sendlengs, int **sendprocs, int **sendindices)
{
   int iP, iD, iD2, offset, index, count; 
   int nRecvs=0, *recvLengs=NULL, *recvProcs=NULL, *recvIndices=NULL;
   int nSends=0, *sendLengs=NULL, *sendProcs=NULL, *sendIndices=NULL;

   if ( nRecvs_ > 0 )
   {
      nRecvs = nRecvs_;
      recvLengs = new int[nRecvs];
      recvProcs = new int[nRecvs];
      offset = 0;
      for ( iP = 0; iP < nRecvs_; iP++ ) offset += recvLengs_[iP];
      recvIndices = new int[offset*nodeDOF_];

      offset = 0;
      for ( iP = 0; iP < nRecvs_; iP++ )
      {
         recvLengs[iP] = recvLengs_[iP] * nodeDOF_;
         recvProcs[iP] = recvProcs_[iP];
         for ( iD = 0; iD < recvLengs_[iP]; iD++ )
         {
            index = iD * nodeDOF_;
            for ( iD2 = 0; iD2 < nodeDOF_; iD2++)
               recvIndices[offset*nodeDOF_+index+iD2] = 
                  recvProcIndices_[offset+iD] * nodeDOF_ + iD2 + numCRMult_;
         }
         offset += recvLengs_[iP];
      }
   }
   if ( nSends_ > 0 )
   {
      nSends = nSends_;
      sendLengs = new int[nSends];
      sendProcs = new int[nSends];
      offset = 0;
      for ( iP = 0; iP < nSends_; iP++ ) offset += sendLengs_[iP];
      sendIndices = new int[offset*nodeDOF_];
      offset = 0;
      for ( iP = 0; iP < nSends_; iP++ )
      {
         sendLengs[iP] = sendLengs_[iP] * nodeDOF_;
         sendProcs[iP] = sendProcs_[iP];
         for ( iD = 0; iD < sendLengs_[iP]; iD++)
         {
            count = iD * nodeDOF_;
            for ( iD2 = 0; iD2 < nodeDOF_; iD2++)
               sendIndices[offset*nodeDOF_+count+iD2] = 
                  sendProcIndices_[offset+iD] * nodeDOF_ + iD2;
         }
         offset += sendLengs_[iP];
      }
   }
   (*nrecvs) = nRecvs;
   (*recvlengs) = recvLengs;
   (*recvprocs) = recvProcs;
   (*recvindices) = recvIndices;
   (*nsends) = nSends;
   (*sendlengs) = sendLengs;
   (*sendprocs) = sendProcs;
   (*sendindices) = sendIndices;
}

