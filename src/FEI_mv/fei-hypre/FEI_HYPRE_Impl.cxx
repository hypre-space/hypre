/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.18 $
 ***********************************************************************EHEADER*/




/**************************************************************************
  Module:  FEI_HYPRE_Impl.cpp
  Purpose: custom (local) implementation of the FEI/LSC
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#if HAVE_SUPERLU_20
#include "dsp_defs.h"
#include "superlu_util.h"
#endif
#if HAVE_SUPERLU
#include "SRC/slu_ddefs.h"
#include "SRC/slu_util.h"
#endif

/*-------------------------------------------------------------------------
 MPI definitions 
 -------------------------------------------------------------------------*/

#include "FEI_HYPRE_include.h"
#include "_hypre_utilities.h"
#include "HYPRE.h"

/*-------------------------------------------------------------------------
 local defines and external functions
 -------------------------------------------------------------------------*/

#include "FEI_HYPRE_Impl.h"

extern "C"
{
  int HYPRE_LSI_Search(int *, int, int);
}

#define habs(x) ((x > 0) ? x : -(x))

/**************************************************************************
 **************************************************************************
 Each element block contains a number of elements of the same type (e.g. 
 hex or tet element).  For this implementation, all element block should
 have the same number of degree of freedom per node. 
 **************************************************************************/

/**************************************************************************
 Constructor 
 -------------------------------------------------------------------------*/
FEI_HYPRE_Elem_Block::FEI_HYPRE_Elem_Block( int blockID )
{
   blockID_           = blockID;
   currElem_          = 0;
   numElems_          = 0;
   nodesPerElem_      = 0;
   nodeDOF_           = 0;
   elemIDs_           = NULL;
   elemNodeLists_     = NULL;
   elemMatrices_      = NULL;
   rhsVectors_        = NULL;
   solnVectors_       = NULL;
   tempX_             = NULL;
   tempY_             = NULL;
   sortedIDs_         = NULL;
   sortedIDAux_       = NULL;
}

/**************************************************************************
 destructor 
 -------------------------------------------------------------------------*/
FEI_HYPRE_Elem_Block::~FEI_HYPRE_Elem_Block()
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
int FEI_HYPRE_Elem_Block::initialize(int numElements, int numNodesPerElement,
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
int FEI_HYPRE_Elem_Block::reset()
{
   if ( elemNodeLists_ != NULL )
   {
      for ( int iE = 0; iE < numElems_; iE++ ) 
      {
         if ( elemNodeLists_[iE] != NULL ) 
            delete [] elemNodeLists_[iE];
         elemNodeLists_[iE] = NULL;
      }
   }
   if ( elemMatrices_ != NULL )
   {
      for ( int iE = 0; iE < numElems_; iE++ ) 
      {
         if ( elemMatrices_[iE] != NULL ) 
            delete [] elemMatrices_[iE];
         elemMatrices_[iE] = NULL;
      }
   }
   if ( rhsVectors_ != NULL )
   {
      for ( int iE = 0; iE < numElems_; iE++ ) 
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
int FEI_HYPRE_Elem_Block::resetRHSVectors(double s)
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
int FEI_HYPRE_Elem_Block::resetSolnVectors(double s)
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
int FEI_HYPRE_Elem_Block::loadElemInfo(int elemID, int *elemConn, 
                                     double **elemStiff, double *elemLoad)
{
   if ( currElem_ >= numElems_ )
   {
      printf("FEI_HYPRE_Elem_Block::loadElemInfo ERROR : too many elements.\n");
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
int FEI_HYPRE_Elem_Block::loadElemMatrix(int elemID, int *elemConn, 
                                       double **elemStiff)
{
   if ( currElem_ >= numElems_ )
   {
      printf("FEI_HYPRE_Elem_Block::loadElemMatrix ERROR:too many elements.\n");
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
int FEI_HYPRE_Elem_Block::loadElemRHS(int elemID, double *elemLoad)
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
         FEI_HYPRE_Impl::IntSort2(sortedIDs_, sortedIDAux_, 0, numElems_-1);
      }
      currElem_ = HYPRE_LSI_Search(sortedIDs_, elemID, numElems_);
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
int FEI_HYPRE_Elem_Block::checkLoadComplete()
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
 FEI_HYPRE_Impl is the core linear system interface.  Each 
 instantiation supports multiple elememt blocks.
 **************************************************************************/

/**************************************************************************
 Constructor 
 -------------------------------------------------------------------------*/
FEI_HYPRE_Impl::FEI_HYPRE_Impl( MPI_Comm comm )
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

   numSharedNodes_      = 0;
   sharedNodeIDs_       = NULL;
   sharedNodeNProcs_    = NULL;
   sharedNodeProcs_     = NULL;

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
    * solver information
    * ----------------------------------------------------------------*/

   solverID_            = 0;
   krylovMaxIterations_ = 1000;
   krylovAbsRel_        = 0;       /* 0 - relative norm */
   krylovTolerance_     = 1.0e-6;
   krylovIterations_    = 0;
   krylovResidualNorm_  = 0.0;
   gmresDim_            = 20;

   /* -----------------------------------------------------------------
    * matrix and vector information
    * ----------------------------------------------------------------*/

   diagIA_     = NULL;
   diagJA_     = NULL;
   diagAA_     = NULL;
   offdIA_     = NULL;
   offdJA_     = NULL;
   offdAA_     = NULL;
   diagonal_   = NULL;
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

   FLAG_PrintMatrix_  = 0;
   FLAG_LoadComplete_ = 0;
   TimerLoad_         = 0.0;
   TimerLoadStart_    = 0.0;
   TimerSolve_        = 0.0;
   TimerSolveStart_   = 0.0;
}

/**************************************************************************
 destructor 
 -------------------------------------------------------------------------*/
FEI_HYPRE_Impl::~FEI_HYPRE_Impl()
{
   if ( outputLevel_ > 0 ) printf("%4d : FEI_HYPRE_Impl destructor\n", mypid_);
   for ( int iB = 0; iB < numBlocks_; iB++ )
      if ( elemBlocks_[iB] != NULL ) delete elemBlocks_[iB];
   if ( nodeGlobalIDs_       != NULL ) delete [] nodeGlobalIDs_;
   if ( nodeExtNewGlobalIDs_ != NULL ) delete [] nodeExtNewGlobalIDs_;
   if ( globalNodeOffsets_   != NULL ) delete [] globalNodeOffsets_;
   if ( recvLengs_           != NULL ) delete [] recvLengs_;
   if ( recvProcs_           != NULL ) delete [] recvProcs_;
   if ( recvProcIndices_     != NULL ) 
   {
      for (int iP = 0; iP < nRecvs_; iP++) delete [] recvProcIndices_[iP];
      delete [] recvProcIndices_;
   }
   if ( sendLengs_           != NULL ) delete [] sendLengs_;
   if ( sendProcs_           != NULL ) delete [] sendProcs_;
   if ( sendProcIndices_     != NULL ) 
   {
      for (int iP = 0; iP < nSends_; iP++) delete [] sendProcIndices_[iP];
      delete [] sendProcIndices_;
   }
   if ( diagIA_              != NULL ) delete [] diagIA_;
   if ( diagJA_              != NULL ) delete [] diagJA_;
   if ( diagAA_              != NULL ) delete [] diagAA_;
   if ( offdIA_              != NULL ) delete [] offdIA_;
   if ( offdJA_              != NULL ) delete [] offdJA_;
   if ( offdAA_              != NULL ) delete [] offdAA_;
   if ( diagonal_            != NULL ) delete [] diagonal_;
   if ( solnVector_          != NULL ) delete [] solnVector_;
   if ( rhsVector_           != NULL ) delete [] rhsVector_;
   if ( BCNodeIDs_           != NULL ) delete [] BCNodeIDs_;
   if ( BCNodeAlpha_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeAlpha_[iD];
      delete [] BCNodeAlpha_;
   }
   if ( BCNodeBeta_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeBeta_[iD];
      delete [] BCNodeBeta_;
   }
   if ( BCNodeGamma_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeGamma_[iD];
      delete [] BCNodeGamma_;
   }
}

/**************************************************************************
 parameters function
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::parameters(int numParams, char **paramString)
{
   int  i, olevel;
   char param[256], param1[256];
#if HAVE_SUPERLU
   int  nprocs;
#endif

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(paramString[i],"%s", param1);
      if ( ! strcmp(param1, "outputLevel") )
      {
         sscanf(paramString[i],"%s %d", param1, &olevel);
         outputLevel_ = olevel;
         if ( olevel < 0 ) outputLevel_ = 0;
         if ( olevel > 4 ) outputLevel_ = 4;
      }
      else if ( ! strcmp(param1, "setDebug") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if ( ! strcmp(param, "printMat") ) FLAG_PrintMatrix_ = 1;
      }
      else if ( ! strcmp(param1, "gmresDim") )
      {
         sscanf(paramString[i],"%s %d", param1, &gmresDim_);
         if ( gmresDim_ < 0 ) gmresDim_ = 10;
      }
      else if ( ! strcmp(param1, "maxIterations") )
      {
         sscanf(paramString[i],"%s %d", param1, &krylovMaxIterations_);
         if ( krylovMaxIterations_ <= 0 ) krylovMaxIterations_ = 1;
      }
      else if ( ! strcmp(param1, "tolerance") )
      {
         sscanf(paramString[i],"%s %lg", param1, &krylovTolerance_);
         if ( krylovTolerance_ >= 1.0 || krylovTolerance_ <= 0.0 )
            krylovTolerance_ = 1.0e-6;
      }
      else if ( ! strcmp(param1, "stopCrit") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if      ( ! strcmp(param, "absolute") ) krylovAbsRel_ = 1;
         else if ( ! strcmp(param, "relative") ) krylovAbsRel_ = 0;
         else                                  krylovAbsRel_ = 0;
      }
      else if ( ! strcmp(param1, "solver") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if      ( ! strcmp(param, "cg") )      solverID_ = 0;
         else if ( ! strcmp(param, "gmres") )   solverID_ = 1;
         else if ( ! strcmp(param, "cgs") )     solverID_ = 2;
         else if ( ! strcmp(param, "bicgstab")) solverID_ = 3;
#if HAVE_SUPERLU
         else if ( ! strcmp(param, "superlu") ) 
         {
            MPI_Comm_size( mpiComm_, &nprocs );
            if ( nprocs == 1 ) solverID_ = 4;
            else
            {
               printf("FEI_HYPRE_Impl WARNING : SuperLU not supported on ");
               printf("more than 1 proc.  Use GMRES instead.\n");
               solverID_ = 1;
            }
         }
#endif
         else                                 solverID_ = 1;
      }
      else if ( ! strcmp(param1, "preconditioner") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if ( (! ! strcmp(param, "diag")) && (! ! strcmp(param, "diagonal")) )
            printf("FEI_HYPRE_Impl::parameters - invalid preconditioner.\n");
      }
   }
   return 0;
}

/**************************************************************************
 initialize nodal degree of freedom 
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::initFields(int numFields, int *fieldSizes, int *fieldIDs)
{
   (void) fieldIDs;
   if ( numFields != 1 )
   {
      printf("%4d : FEI_HYPRE_Impl::initFields WARNING -  numFields != 1.",
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
int FEI_HYPRE_Impl::initElemBlock(int elemBlockID, int numElements, 
                      int numNodesPerElement, int *numFieldsPerNode, 
                      int **nodalFieldIDs, int numElemDOFFieldsPerElement, 
                      int *elemDOFFieldIDs, int interleaveStrategy)
{
   (void) numFieldsPerNode;
   (void) nodalFieldIDs;
   (void) numElemDOFFieldsPerElement; 
   (void) elemDOFFieldIDs;
   (void) interleaveStrategy;
   if ( outputLevel_ >= 2 ) 
   {
      printf("%4d : FEI_HYPRE_Impl::initElemBlock begins... \n", mypid_);
      printf("               elemBlockID  = %d \n", elemBlockID);
      printf("               numElements  = %d \n", numElements);
      printf("               nodesPerElem = %d \n", numNodesPerElement);
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
   if ( numBlocks_ == 0 )
   {
      elemBlocks_    = new FEI_HYPRE_Elem_Block*[1];
      elemBlocks_[0] = new FEI_HYPRE_Elem_Block(elemBlockID);
      numBlocks_     = 1;
   }
   else
   {
      for ( int iB = 0; iB < numBlocks_; iB++ )
      {
         if ( elemBlocks_[iB]->getElemBlockID() == elemBlockID )
         {
            printf("%4d : FEI_HYPRE_Impl::initElemBlock ERROR - ",mypid_);
            printf("repeated blockID\n");
            exit(1);
         }
      } 
      FEI_HYPRE_Elem_Block **tempBlocks = elemBlocks_;
      numBlocks_++;
      elemBlocks_ = new FEI_HYPRE_Elem_Block*[numBlocks_];
      for ( int iB2 = 0; iB2 < numBlocks_-1; iB2++ )
         elemBlocks_[iB2] = tempBlocks[iB2];
      elemBlocks_[numBlocks_-1] = new FEI_HYPRE_Elem_Block(elemBlockID);
   }
   elemBlocks_[numBlocks_-1]->initialize(numElements, numNodesPerElement,
                                         nodeDOF_); 
   FLAG_LoadComplete_= 0;
   if ( outputLevel_ >= 2 ) 
      printf("%4d : FEI_HYPRE_Impl::initElemBlock ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 initialize shared node information
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::initSharedNodes(int nShared, int *sharedIDs,
                                 int *sharedNProcs, int **sharedProcs)
{
   int iN, iP, newNumShared, *oldSharedIDs, *oldSharedNProcs;
   int **oldSharedProcs;

   if ( outputLevel_ >= 2 ) 
      printf("%4d : FEI_HYPRE_Impl::initSharedNodes begins... \n", mypid_);
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
   if ( outputLevel_ >= 2 ) 
      printf("%4d : FEI_HYPRE_Impl::initSharedNodes ends. \n", mypid_);
   return 0;
}

/**************************************************************************
 reset the system
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::resetSystem(double s)
{
   (void) s;
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetSystem begins...\n", mypid_);
   for ( int iB = 0; iB < numBlocks_; iB++ ) elemBlocks_[iB]->reset();
   numLocalNodes_ = 0;
   numExtNodes_   = 0;
   if ( nodeGlobalIDs_       != NULL ) delete [] nodeGlobalIDs_;
   if ( nodeExtNewGlobalIDs_ != NULL ) delete [] nodeExtNewGlobalIDs_;
   if ( globalNodeOffsets_   != NULL ) delete [] globalNodeOffsets_;
   if ( recvLengs_           != NULL ) delete [] recvLengs_;
   if ( recvProcs_           != NULL ) delete [] recvProcs_;
   if ( recvProcIndices_     != NULL ) 
   {
      for (int iP = 0; iP < nRecvs_; iP++) delete [] recvProcIndices_[iP];
      delete [] recvProcIndices_;
   }
   if ( sendLengs_           != NULL ) delete [] sendLengs_;
   if ( sendProcs_           != NULL ) delete [] sendProcs_;
   if ( sendProcIndices_     != NULL ) 
   {
      for (int iP = 0; iP < nSends_; iP++) delete [] sendProcIndices_[iP];
      delete [] sendProcIndices_;
   }
   if ( diagIA_              != NULL ) delete [] diagIA_;
   if ( diagJA_              != NULL ) delete [] diagJA_;
   if ( diagAA_              != NULL ) delete [] diagAA_;
   if ( offdIA_              != NULL ) delete [] offdIA_;
   if ( offdJA_              != NULL ) delete [] offdJA_;
   if ( offdAA_              != NULL ) delete [] offdAA_;
   if ( diagonal_            != NULL ) delete [] diagonal_;
   if ( BCNodeAlpha_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeAlpha_[iD];
      delete [] BCNodeAlpha_;
   }
   if ( BCNodeBeta_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeBeta_[iD];
      delete [] BCNodeBeta_;
   }
   if ( BCNodeGamma_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeGamma_[iD];
      delete [] BCNodeGamma_;
   }
   if ( BCNodeIDs_ != NULL ) delete [] BCNodeIDs_; 
   if ( rhsVector_ != NULL ) delete [] rhsVector_; 
   nSends_              = 0;
   nRecvs_              = 0;
   nodeGlobalIDs_       = NULL;
   nodeExtNewGlobalIDs_ = NULL;
   globalNodeOffsets_   = NULL;
   recvLengs_           = NULL;
   recvProcs_           = NULL;
   recvProcIndices_     = NULL;
   sendLengs_           = NULL;
   sendProcs_           = NULL;
   sendProcIndices_     = NULL;
   diagIA_              = NULL;
   diagJA_              = NULL;
   diagAA_              = NULL;
   offdIA_              = NULL;
   offdJA_              = NULL;
   offdAA_              = NULL;
   diagonal_            = NULL;
   BCNodeIDs_           = NULL;
   BCNodeAlpha_         = NULL;
   BCNodeBeta_          = NULL;
   BCNodeGamma_         = NULL;
   rhsVector_           = NULL;
   numBCNodes_          = 0;
   TimerLoad_           = 0.0;
   TimerLoadStart_      = 0.0;
   TimerSolve_          = 0.0;
   TimerSolveStart_     = 0.0;
   FLAG_LoadComplete_   = 0;
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetSystem ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 reset the matrix
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::resetMatrix(double s)
{
   (void) s;
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetMatrix begins...\n", mypid_);
   for ( int iB = 0; iB < numBlocks_; iB++ ) elemBlocks_[iB]->reset();
   numLocalNodes_ = 0;
   numExtNodes_   = 0;
   if ( nodeGlobalIDs_       != NULL ) delete [] nodeGlobalIDs_;
   if ( nodeExtNewGlobalIDs_ != NULL ) delete [] nodeExtNewGlobalIDs_;
   if ( globalNodeOffsets_   != NULL ) delete [] globalNodeOffsets_;
   if ( recvLengs_           != NULL ) delete [] recvLengs_;
   if ( recvProcs_           != NULL ) delete [] recvProcs_;
   if ( recvProcIndices_     != NULL ) 
   {
      for (int iP = 0; iP < nRecvs_; iP++) delete [] recvProcIndices_[iP];
      delete [] recvProcIndices_;
   }
   if ( sendLengs_           != NULL ) delete [] sendLengs_;
   if ( sendProcs_           != NULL ) delete [] sendProcs_;
   if ( sendProcIndices_     != NULL ) 
   {
      for (int iP = 0; iP < nSends_; iP++) delete [] sendProcIndices_[iP];
      delete [] sendProcIndices_;
   }
   if ( diagIA_              != NULL ) delete [] diagIA_;
   if ( diagJA_              != NULL ) delete [] diagJA_;
   if ( diagAA_              != NULL ) delete [] diagAA_;
   if ( offdIA_              != NULL ) delete [] offdIA_;
   if ( offdJA_              != NULL ) delete [] offdJA_;
   if ( offdAA_              != NULL ) delete [] offdAA_;
   if ( diagonal_            != NULL ) delete [] diagonal_;
   if ( BCNodeAlpha_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeAlpha_[iD];
      delete [] BCNodeAlpha_;
   }
   if ( BCNodeBeta_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeBeta_[iD];
      delete [] BCNodeBeta_;
   }
   if ( BCNodeGamma_ != NULL ) 
   {
      for ( int iD = 0; iD < numBCNodes_; iD++ ) delete [] BCNodeGamma_[iD];
      delete [] BCNodeGamma_;
   }
   if ( BCNodeIDs_ != NULL ) delete [] BCNodeIDs_; 
   nSends_              = 0;
   nRecvs_              = 0;
   nodeGlobalIDs_       = NULL;
   nodeExtNewGlobalIDs_ = NULL;
   globalNodeOffsets_   = NULL;
   recvLengs_           = NULL;
   recvProcs_           = NULL;
   recvProcIndices_     = NULL;
   sendLengs_           = NULL;
   sendProcs_           = NULL;
   sendProcIndices_     = NULL;
   diagIA_              = NULL;
   diagJA_              = NULL;
   diagAA_              = NULL;
   offdIA_              = NULL;
   offdJA_              = NULL;
   offdAA_              = NULL;
   diagonal_            = NULL;
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
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetMatrix ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 reset the rhs vector
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::resetRHSVector(double s)
{
   (void) s;
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetRHSVector begins...\n", mypid_);
   for ( int iB = 0; iB < numBlocks_; iB++ ) 
      elemBlocks_[iB]->resetRHSVectors(s);
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetRHSVector ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 reset the solution vector
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::resetInitialGuess(double s)
{
   (void) s;
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetInitialGuess begins...\n", mypid_);
   for ( int iB = 0; iB < numBlocks_; iB++ ) 
      elemBlocks_[iB]->resetSolnVectors(s);
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::resetInitialGuess ends (%e).\n", mypid_, s);
   return 0;
}

/**************************************************************************
 load node boundary conditions
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::loadNodeBCs(int numNodes, int *nodeIDs, int fieldID,
                             double **alpha, double **beta, double **gamma1)
{
   int   iN, iD, oldNumBCNodes, *oldBCNodeIDs;
   double **oldBCAlpha, **oldBCBeta, **oldBCGamma;

   (void) fieldID;
   if ( outputLevel_ >= 2 ) 
      printf("%4d : FEI_HYPRE_Impl::loadNodeBCs begins...(%d)\n",mypid_,numNodes);
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
               BCNodeGamma_[iN][iD] = gamma1[iN][iD];
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
               BCNodeGamma_[oldNumBCNodes+iN][iD] = gamma1[iN][iD];
            }
         }
      }
   }
   TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
   if ( outputLevel_ >= 2 ) 
      printf("%4d : FEI_HYPRE_Impl::loadNodeBCs ends.\n", mypid_);
   return 0;
}

/**************************************************************************
 load element connectivities, stiffness matrices, and element load 
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::sumInElem(int elemBlockID, int elemID, int *elemConn,
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
#ifdef HAVE_DEBUG
   if ( iB == numBlocks_ )
   {
      printf("%4d : FEI_HYPRE_Impl::sumInElem ERROR - ", mypid_);
      printf("blockID invalid (%d).\n", iB);
      exit(1);
   }
#endif
#ifdef HAVE_DEBUG
   if ( outputLevel_ > 0 && elemBlocks_[iB]->getCurrentElem()==0 ) 
      printf("%4d : FEI_HYPRE_Impl::sumInElem begins... \n", mypid_); 
#endif
   if ( elemBlocks_[iB]->getCurrentElem()==0 ) TimerLoadStart_ = MPI_Wtime();
   elemBlocks_[iB]->loadElemInfo(elemID, elemConn, elemStiff, elemLoad);
   if ( elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
#ifdef HAVE_DEBUG
   if ( outputLevel_ > 0 && 
        elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      printf("%4d : FEI_HYPRE_Impl::sumInElem ends. \n", mypid_); 
#endif
   return 0;
}

/**************************************************************************
 load element connectivities and stiffness matrices
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::sumInElemMatrix(int elemBlockID, int elemID, int *elemConn,
                           double **elemStiff, int elemFormat)
{
   int iB=0;

   (void) elemFormat;
   if ( numBlocks_ > 1 )
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == elemBlockID ) break;
   }
#ifdef HAVE_DEBUG
   if ( iB == numBlocks_ )
   {
      printf("%4d : FEI_HYPRE_Impl::sumInElemMatrix ERROR - ", mypid_);
      printf("blockID invalid (%d).\n", iB);
      exit(1);
   }
#endif
#ifdef HAVE_DEBUG
   if ( outputLevel_ > 0 && elemBlocks_[iB]->getCurrentElem()==0 ) 
      printf("%4d : FEI_HYPRE_Impl::sumInElemMatrix begins... \n", mypid_); 
#endif
   if ( elemBlocks_[iB]->getCurrentElem()==0 ) TimerLoadStart_ = MPI_Wtime();
   elemBlocks_[iB]->loadElemMatrix(elemID, elemConn, elemStiff);
   if ( elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
#ifdef HAVE_DEBUG
   if ( outputLevel_ > 0 && 
        elemBlocks_[iB]->getCurrentElem()==elemBlocks_[iB]->getNumElems() ) 
      printf("%4d : FEI_HYPRE_Impl::sumInElemMatrix ends. \n", mypid_); 
#endif
   return 0;
}

/**************************************************************************
 load element load
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::sumInElemRHS(int elemBlockID, int elemID, int *elemConn,
                              double *elemLoad)
{
   int iB=0;

   (void) elemConn;
   if ( numBlocks_ > 1 )
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == elemBlockID ) break;
   }
#ifdef HAVE_DEBUG
   if ( iB == numBlocks_ )
   {
      printf("%4d : FEI_HYPRE_Impl::sumInElemRHS ERROR - ", mypid_);
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
int FEI_HYPRE_Impl::loadComplete()
{
   int   nprocs, iB, iP, iN, iN2, iE, ierr, index, index2, nodeRegister;
   int   totalNNodes, nElems, elemNNodes, **elemNodeList, nodeNumber;
   int   *nodeIDs, *nodeIDAux, localNNodes, minProc;
   int   *ownerProcs, *nodeIDAux2, *sndrcvReg, *pArrayAux, pnum, pCounts;
   int   nRecv, *recvLengs, *recvProcs, **recvBuf;
   int   nSend, *sendLengs, *sendProcs, **sendBuf, nodeOffset;
   int   **sharedNodeProcAux;
   MPI_Request *request;
   MPI_Status  status;

   /* -----------------------------------------------------------------
    * get machine information
    * ----------------------------------------------------------------*/

   if ( outputLevel_ >= 2 ) 
      printf("%4d : FEI_HYPRE_Impl::loadComplete begins.... \n", mypid_);
   TimerLoadStart_ = MPI_Wtime();
   MPI_Comm_size( mpiComm_, &nprocs );

   /* -----------------------------------------------------------------
    * check that element stiffness matrices, connectivities, and rhs
    * have been loaded, and create solution vectors.
    * ----------------------------------------------------------------*/

   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      ierr = elemBlocks_[iB]->checkLoadComplete();
      assert( !ierr );
   }

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

   /* -----------------------------------------------------------------
    * obtain an ordered array of distinct node IDs
    * (nodeGlobalIDs_, numLocalNodes_, numExtNodes_)
    * ----------------------------------------------------------------*/

   /* -- count the total number of nodes (can be repeated) -- */

   totalNNodes = 0;
   for ( iB = 0; iB < numBlocks_; iB++ )
   {
      nElems       = elemBlocks_[iB]->getNumElems();
      elemNNodes   = elemBlocks_[iB]->getElemNumNodes();
      totalNNodes += nElems * elemNNodes;
   }

   /* -- allocate space and fetch the global node numbers -- */

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

   /* -- sort the global node numbers (ordering in nodeIDAux) -- */

   if ( totalNNodes > 0 ) nodeIDAux = new int[totalNNodes];
   for ( iN = 0; iN < totalNNodes; iN++ ) nodeIDAux[iN] = iN;
   IntSort2(nodeIDs, nodeIDAux, 0, totalNNodes-1);

   /* -- identify the external nodes (nodeIDAux) -- */

   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      minProc = mypid_;
      for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
         if ( sharedNodeProcs_[iN][iP] < minProc )
            minProc = sharedNodeProcs_[iN][iP];
      if ( minProc >=  mypid_ ) continue;
      index = HYPRE_LSI_Search(nodeIDs,sharedNodeIDs_[iN],totalNNodes);
      if ( nodeIDAux[index] >= 0 )
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

   /* -- tally the number of local nodes (internal and external) -- */

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
   if ( outputLevel_ >= 2 )
   {
      printf("%4d : FEI_HYPRE_Impl::loadComplete - nLocalNodes = %d\n",
             mypid_, numLocalNodes_);
      printf("%4d : FEI_HYPRE_Impl::loadComplete - numExtNodes = %d\n",
             mypid_, localNNodes-numLocalNodes_);
   }

   /* -- construct global node list, starting with nodes -- */
   /* -- that belongs to local processor                 -- */

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
    * rewrite the element connectivities with local node numbers 
    * ----------------------------------------------------------------*/

   if ( totalNNodes > 0 ) nodeIDAux2 = new int[totalNNodes];
   for ( iN = 0; iN < totalNNodes; iN++ )
      if ( nodeIDAux[iN] < 0 ) nodeIDAux[iN] = - nodeIDAux[iN] - 1;
   for ( iN = 0; iN < totalNNodes; iN++ )
      nodeIDAux2[nodeIDAux[iN]] = nodeIDs[iN];

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
    * get global node offset information (globalNodeOffsets_)
    * ----------------------------------------------------------------*/

   globalNodeOffsets_ = new int[nprocs+1];
   MPI_Allgather(&numLocalNodes_, 1, MPI_INT, globalNodeOffsets_, 1,
                 MPI_INT, mpiComm_);
   for ( iP = nprocs; iP > 0; iP-- ) 
      globalNodeOffsets_[iP] = globalNodeOffsets_[iP-1];
   globalNodeOffsets_[0] = 0;
   for ( iP = 1; iP <= nprocs; iP++ ) 
      globalNodeOffsets_[iP] += globalNodeOffsets_[iP-1];
   nodeOffset = globalNodeOffsets_[mypid_];

   /* -----------------------------------------------------------------
    * next construct communication pattern 
    * ----------------------------------------------------------------*/

   /* -- create an aux array for holding mapped external node IDs -- */

   ownerProcs = NULL;
   pArrayAux  = NULL;
   sndrcvReg  = NULL;
   if ( numExtNodes_    > 0 ) ownerProcs = new int[numExtNodes_];
   if ( numExtNodes_    > 0 ) pArrayAux  = new int[numExtNodes_];
   if ( numSharedNodes_ > 0 ) sndrcvReg = new int[numSharedNodes_];

   /* -- for all shared nodes, see if they are recv or send nodes -- */
   /* -- ( sndrcvReg[numSharedNodes]; and pArrayAux and           -- */
   /* -- ownerProcs[numExtNodes])                                 -- */

   localNNodes = numLocalNodes_;
   for ( iN = 0; iN < numSharedNodes_; iN++ )
   {
      index = HYPRE_LSI_Search(&(nodeGlobalIDs_[numLocalNodes_]),
                           sharedNodeIDs_[iN], numExtNodes_);
      if ( index >= 0 )
      {
         sndrcvReg[iN] = 1; // recv
         pnum  = mypid_;
         for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
            if (sharedNodeProcs_[iN][iP] < pnum) 
               pnum = sharedNodeProcs_[iN][iP];
         ownerProcs[index] = pnum;
         pArrayAux[index] = pnum;
      }
      else sndrcvReg[iN] = 0; // send
   }

   /* -- based on the pArrayAux and ownerProcs arrays, compose     -- */
   /* -- receive information (nRecv, recvLengs, recvProcs)         -- */

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
            index = HYPRE_LSI_Search(&(nodeGlobalIDs_[numLocalNodes_]),
                                 sharedNodeIDs_[iN], numExtNodes_);
            index = HYPRE_LSI_Search(recvProcs,ownerProcs[index],nRecv);
            recvLengs[index]++;
         }
      }
      recvBuf = new int*[nRecv];
      for ( iP = 0; iP < nRecv; iP++ ) recvBuf[iP] = new int[recvLengs[iP]];
   }
   delete [] pArrayAux;

   /* -- compose send information (nSend, sendLengs, sendProcs) -- */

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
            if ( sharedNodeProcs_[iN][iP] != mypid_ )
               pArrayAux[pCounts++] = sharedNodeProcs_[iN][iP];
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
               if ( sharedNodeProcs_[iN][iP] != mypid_ )
               {
                  index = sharedNodeProcs_[iN][iP];
                  index = HYPRE_LSI_Search(sendProcs,index,nSend);
                  sendLengs[index]++;
               }
            }
         }
      }
      if ( nSend > 0 ) sendBuf = new int*[nSend];
      for ( iP = 0; iP < nSend; iP++ ) 
      {
         sendBuf[iP]   = new int[sendLengs[iP]];
         sendLengs[iP] = 0;
      }
      for ( iN = 0; iN < numSharedNodes_; iN++ )
      {
         if ( sndrcvReg[iN] == 0 )
         {
            for ( iP = 0; iP < sharedNodeNProcs_[iN]; iP++ )
            {
               if ( sharedNodeProcs_[iN][iP] != mypid_ )
               {
                  index  = HYPRE_LSI_Search(sendProcs,
                                        sharedNodeProcs_[iN][iP], nSend);
                  index2 = HYPRE_LSI_Search(nodeGlobalIDs_,
                                        sharedNodeIDs_[iN], numLocalNodes_);
                  sendBuf[index][sendLengs[index]++] = nodeOffset + index2;
               }
            }
         }
      }
   }
   if ( pCounts > 0 ) delete [] pArrayAux;

   /* -- exchange the (NEW) global node indices -- */

   if ( nRecv > 0 ) request = new MPI_Request[nRecv];
   for ( iP = 0; iP < nRecv; iP++ )
      MPI_Irecv( recvBuf[iP], recvLengs[iP], MPI_INT,
                 recvProcs[iP], 183, mpiComm_, &request[iP]);
   for ( iP = 0; iP < nSend; iP++ )
      MPI_Send( sendBuf[iP], sendLengs[iP], MPI_INT,
                sendProcs[iP], 183, mpiComm_);
   for ( iP = 0; iP < nRecv; iP++ ) MPI_Wait( &request[iP], &status );
   if ( nRecv > 0 ) delete [] request;

   /* -- fix the send index array -- */

   for (iP = 0; iP < nSend; iP++) 
      for ( iN = 0; iN < sendLengs[iP]; iN++ )
         sendBuf[iP][iN] -= nodeOffset;

   /* -- based on the recv information, construct recv index array -- */

   if ( numExtNodes_ > 0 ) nodeExtNewGlobalIDs_ = new int[numExtNodes_];
   for ( iP = 0; iP < nRecv; iP++ ) recvLengs[iP] = 0;
   for ( iN = 0; iN < numExtNodes_; iN++ )
   {
      index  = HYPRE_LSI_Search(recvProcs, ownerProcs[iN], nRecv);
      iN2 = recvBuf[index][recvLengs[index]];
      nodeExtNewGlobalIDs_[iN] = iN2;
      recvBuf[index][recvLengs[index]++] = iN + numLocalNodes_;
   }
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

   /* -----------------------------------------------------------------
    * construct the global matrix and diagonal
    * ----------------------------------------------------------------*/

   buildGlobalMatrixVector();
   TimerLoad_ += MPI_Wtime() - TimerLoadStart_;
   if ( FLAG_PrintMatrix_ > 0 ) printLinearSystem();
   FLAG_LoadComplete_ = 1;
   if ( outputLevel_ >= 2 ) 
      printf("%4d : FEI_HYPRE_Impl::loadComplete ends. \n", mypid_);
   return 0;
}

/**************************************************************************
 solve the linear system
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::solve(int *status)
{
   int    nprocs;
   double dArray[2], dArray2[2];

   if ( FLAG_LoadComplete_ == 0 ) loadComplete();
   MPI_Comm_size( mpiComm_, &nprocs );
   if ( outputLevel_ >= 1 && mypid_ == 0 )
      printf("\t**************************************************\n");
   switch (solverID_)
   {
      case 0 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tFEI_HYPRE CG with diagonal preconditioning\n");
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingCG();
               break;
      case 1 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tFEI_HYPRE GMRES with diagonal preconditioning\n");
                  printf("\t\tGMRES dimension = %d\n", gmresDim_);
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingGMRES();
               break;
      case 2 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tFEI_HYPRE CGS with diagonal preconditioning\n");
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingCGS();
               break;
      case 3 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tFEI_HYPRE Bicgstab with diagonal preconditioning\n");
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingBicgstab();
               break;
      case 4 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tFEI_HYPRE direct link to SuperLU \n");
               }
               (*status) = solveUsingSuperLU();
               break;
   }
   TimerSolve_ = MPI_Wtime() - TimerSolveStart_;
   dArray[0] = TimerLoad_;
   dArray[1] = TimerSolve_;
   MPI_Allreduce(dArray,dArray2,2,MPI_DOUBLE,MPI_SUM,mpiComm_);
   if ( outputLevel_ >= 1 && mypid_ == 0 )
   {
      printf("\tFEI_HYPRE local solver : number of iterations = %d\n",
             krylovIterations_);
      printf("\tFEI_HYPRE local solver : final residual norm  = %e\n",
             krylovResidualNorm_);
      printf("\tFEI_HYPRE local FEI    : average load  time   = %e\n",
             dArray2[0]/(double) nprocs);
      printf("\tFEI_HYPRE local FEI    : average solve time   = %e\n",
             dArray2[1]/(double) nprocs);
      printf("\t**************************************************\n");
   }
   return (*status);
}

/**************************************************************************
 form residual norm
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::residualNorm(int whichNorm, int numFields, int* fieldIDs,
                              double* norms)
{
   int    localNRows, extNRows, totalNRows, irow;
   double *rVec, rnorm, dtemp;

   (void) numFields;
   (void) fieldIDs;

   if ( solnVector_ == NULL || rhsVector_ == NULL ) return 1;
   if (whichNorm < 0 || whichNorm > 2) return(-1);
   if ( FLAG_LoadComplete_ == 0 ) loadComplete();

   localNRows = numLocalNodes_ * nodeDOF_;
   extNRows   = numExtNodes_ * nodeDOF_;
   totalNRows = localNRows + extNRows;
   rVec       = new double[totalNRows];
   matvec( solnVector_, rVec ); 
   for ( irow = 0; irow < localNRows; irow++ ) 
      rVec[irow] = rhsVector_[irow] - rVec[irow];

   switch(whichNorm) 
   {
      case 0:
           rnorm = 0.0;
           for ( irow = 0; irow < localNRows; irow++ ) 
           {
              dtemp = fabs( rVec[irow] );
              if ( dtemp > rnorm ) rnorm = dtemp;
           }
           MPI_Allreduce(&rnorm, &dtemp, 1, MPI_DOUBLE, MPI_MAX, mpiComm_);
           (*norms) = dtemp;
           break;
      case 1:
           rnorm = 0.0;
           for ( irow = 0; irow < localNRows; irow++ ) 
              rnorm += fabs( rVec[irow] );
           MPI_Allreduce(&rnorm, &dtemp, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
           (*norms) = dtemp;
           break;
      case 2:
           rnorm = 0.0;
           for ( irow = 0; irow < localNRows; irow++ ) 
              rnorm += rVec[irow] * rVec[irow];
           MPI_Allreduce(&rnorm, &dtemp, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
           (*norms) = sqrt(dtemp);
           break;
   }
   delete [] rVec;
   return 0;
}

/**************************************************************************
 get number of distinct node in a given block
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::getNumBlockActNodes(int blockID, int *numNodes)
{
   int localNNodes, iB, iE, iN, totalNNodes, nElems;
   int elemNNodes, **elemNodeLists, *nodeIDs;  

   if ( numBlocks_ == 1 ) 
   {
      (*numNodes) = numLocalNodes_ + numExtNodes_;
      if ( outputLevel_ >= 2 ) 
      {
         printf("%4d : FEI_HYPRE_Impl::getNumBlockActNodes blockID = %d.\n", 
                mypid_, blockID);
         printf("%4d : FEI_HYPRE_Impl::getNumBlockActNodes numNodes = %d\n", 
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
         printf("%4d : FEI_HYPRE_Impl::getNumBlockActNodes ERROR -",mypid_);
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

      if ( outputLevel_ >= 2 ) 
      {
         printf("%4d : FEI_HYPRE_Impl::getNumBlockActNodes blockID = %d.\n", 
                mypid_, blockID);
         printf("%4d : FEI_HYPRE_Impl::getNumBlockActNodes numNodes = %d\n", 
                mypid_, (*numNodes));
      }
   }
   return 0;
}

/**************************************************************************
 get number of distinct equations in a given block
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::getNumBlockActEqns(int blockID, int *numEqns)
{
   int numNodes;

   getNumBlockActNodes(blockID, &numNodes);
   (*numEqns) = numNodes * nodeDOF_;
   if ( outputLevel_ >= 2 ) 
   {
      printf("%4d : FEI_HYPRE_Impl::getNumBlockActEqns blockID = %d\n", 
             mypid_, blockID);
      printf("%4d : FEI_HYPRE_Impl::getNumBlockActEqns numEqns = %d\n", 
             mypid_, (*numEqns));
   }
   return 0;
}

/**************************************************************************
 get a node list in a given block
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::getBlockNodeIDList(int blockID,int numNodes,int *nodeList)
{
   int localNNodes, iB, iE, iN, totalNNodes, nElems;
   int elemNNodes, **elemNodeLists, *nodeIDs;  

   if ( outputLevel_ >= 2 ) 
   {
      printf("%4d : FEI_HYPRE_Impl::getBlockNodeIDList blockID  = %d\n", 
             mypid_, blockID);
      printf("%4d : FEI_HYPRE_Impl::getBlockNodeIDList numNodes = %d\n", 
             mypid_, numNodes);
   }
   if ( numBlocks_ == 1 ) 
   {
      localNNodes = numLocalNodes_ + numExtNodes_;
      if ( localNNodes != numNodes )
      {
         printf("%4d : FEI_HYPRE_Impl::getBlockNodeIDList ERROR - nNodes",mypid_);
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
         printf("%4d : FEI_HYPRE_Impl::getBlockNodeIDList ERROR -",mypid_);
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
         printf("%4d : FEI_HYPRE_Impl::getBlockNodeIDList ERROR -",mypid_);
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
int FEI_HYPRE_Impl::getBlockNodeSolution(int blockID,int numNodes,
                         int *nodeList, int *nodeOffsets, double *solnValues)
{
   int    iB, iE, iN, iD, totalNNodes, *nodeIDs;
   int    nElems, elemNNodes, **elemNodeLists, nodeID, localNNodes;
   double *dataBuf, **solnVecs;

   (void) nodeList;
   if ( outputLevel_ >= 2 ) 
   {
      printf("%4d : FEI_HYPRE_Impl::getBlockNodeSolution blockID  = %d\n", 
             mypid_, blockID);
      printf("%4d : FEI_HYPRE_Impl::getBlockNodeSolution numNodes = %d\n", 
             mypid_, numNodes);
   }
   if ( numBlocks_ == 1 ) 
   {
      for ( iN = 0; iN < numNodes; iN++ )
      {
         nodeOffsets[iN] = iN * nodeDOF_;
         for ( iD = 0; iD < nodeDOF_; iD++ )
            solnValues[iN*nodeDOF_+iD] = solnVector_[iN*nodeDOF_+iD];
      }
   }
   else
   {
      for ( iB = 0; iB < numBlocks_; iB++ )
         if ( elemBlocks_[iB]->getElemBlockID() == blockID ) break;
      if ( iB >= numBlocks_ )
      {
         printf("%4d : FEI_HYPRE_Impl::getBlockNodeSolution ERROR -",mypid_);
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
 build global stiffness matrix
 -------------------------------------------------------------------------*/
void FEI_HYPRE_Impl::buildGlobalMatrixVector()
{
   int    matDim, *diagCounts=NULL, nElems, elemNNodes, **elemNodeLists=NULL;
   int    iB, iD, iE, iN, offset, iD2, iD3, iN2, *elemNodeList=NULL, diagNNZ; 
   int    offdNNZ, *offdCounts=NULL, rowIndBase, rowInd, colIndBase, colInd;
   int    bound, iCount, index, iBegin, *TdiagIA=NULL, *TdiagJA=NULL;
   int    *ToffdIA=NULL, *ToffdJA=NULL, elemNExt, elemNLocal, nodeID;
   int    diagOffset, offdOffset; 
   double **elemMats=NULL, *elemMat=NULL, *TdiagAA=NULL, *ToffdAA=NULL;
   double alpha, beta, gamma1;

   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::buildGlobalMatrixVector begins..\n",mypid_);

   /* -----------------------------------------------------------------
    * assemble the right hand side vector
    * -----------------------------------------------------------------*/

   assembleRHSVector();

   /* -----------------------------------------------------------------
    * count the number of nonzeros per row (diagCounts, offdCounts)
    * -----------------------------------------------------------------*/

   matDim = ( numLocalNodes_ + numExtNodes_) * nodeDOF_;
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
            for ( iD = 0; iD < nodeDOF_; iD++ )
            {
               diagCounts[rowInd+iD] += elemNLocal * nodeDOF_;
               offdCounts[rowInd+iD] += elemNExt * nodeDOF_;
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

   bound = numLocalNodes_ * nodeDOF_;
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
               if ( colInd >= bound ) 
               {
                  for ( iN2 = 0; iN2 < elemNNodes; iN2++ )
                  {
                     rowInd = elemNodeList[iN2];
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
                  for ( iN2 = 0; iN2 < elemNNodes; iN2++ )
                  {
                     rowIndBase = elemNodeList[iN2] * nodeDOF_;
                     for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
                     {
                        rowInd = rowIndBase + iD2;
                        if ( elemMat[offset] != 0.0 ) 
                        {
                           if ( colInd >= bound ) 
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

   /* -----------------------------------------------------------------
    * compress the matrix (take out redundant columns)
    * -----------------------------------------------------------------*/

   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::buildGlobalMatrixVector mid phase\n",mypid_);
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
    * impose boundary conditions
    * -----------------------------------------------------------------*/

   for ( iD = bound; iD < matDim; iD++ ) rhsVector_[iD] = 0.0;
   for ( iN = 0; iN < numBCNodes_; iN++ )
   {
      nodeID = BCNodeIDs_[iN];
      index = HYPRE_LSI_Search(nodeGlobalIDs_, nodeID, numLocalNodes_);
      if ( index >= 0 )
      {
         for ( iD = index*nodeDOF_; iD < (index+1)*nodeDOF_; iD++ )
         {
            alpha = BCNodeAlpha_[iN][iD%nodeDOF_]; 
            beta  = BCNodeBeta_[iN][iD%nodeDOF_]; 
            gamma1= BCNodeGamma_[iN][iD%nodeDOF_]; 
            if ( beta == 0.0 && alpha != 0.0 )
            {
               for (iD2=TdiagIA[iD]; iD2<TdiagIA[iD]+diagCounts[iD]; iD2++)
               {
                  rowInd = TdiagJA[iD2];
                  if ( rowInd != iD && rowInd >= 0 )
                  {
                     for (iD3 = TdiagIA[rowInd];
                          iD3<TdiagIA[rowInd]+diagCounts[rowInd]; iD3++)
                     {
                        if ( TdiagJA[iD3] == iD && TdiagAA[iD3] != 0.0 )
                        {
                           rhsVector_[rowInd] -= (gamma1/alpha*TdiagAA[iD3]); 
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
               if ( ToffdIA != NULL )
               {
                  for (iD2=ToffdIA[iD]; iD2<ToffdIA[iD]+offdCounts[iD]; iD2++)
                  {
                     rowInd = ToffdJA[iD2];
                     if ( rowInd != iD && rowInd >= 0 )
                     {
                        for (iD3 = TdiagIA[rowInd];
                             iD3<TdiagIA[rowInd]+diagCounts[rowInd]; iD3++)
                        {
                           if ( TdiagJA[iD3] == iD && TdiagAA[iD3] != 0.0 )
                           {
                              rhsVector_[rowInd] -= (gamma1/alpha*TdiagAA[iD3]);
                              TdiagAA[iD3] = 0.0;
                              break;
                           }
                        }
                     }
                  }
                  for (iD2=ToffdIA[iD]; iD2<ToffdIA[iD]+offdCounts[iD]; iD2++)
                  {
                     ToffdJA[iD2] = -1;
                     ToffdAA[iD2] = 0.0;
                  }
               }
               rhsVector_[iD] = gamma1 / alpha;
            }
            else if ( beta != 0.0 )
            {
               for (iD2=TdiagIA[iD]; iD2<TdiagIA[iD]+diagCounts[iD]; iD2++)
               {
                  rowInd = TdiagJA[iD2];
                  if ( rowInd == iD )
                  {
                     TdiagAA[iD2] += alpha / beta;
                     break;
                  }
               }
               rhsVector_[iD] += gamma1 / beta;
            }
         }
      }
      else
      {
         index = HYPRE_LSI_Search(&nodeGlobalIDs_[numLocalNodes_], nodeID,
                                  numExtNodes_);
         if ( index < 0 )
         {
            printf("ERROR : BC node ID not local.\n");
            exit(1);
         }
         index += numLocalNodes_;
         for ( iD = index*nodeDOF_; iD < (index+1)*nodeDOF_; iD++ )
         {
            alpha = BCNodeAlpha_[iN][iD%nodeDOF_]; 
            beta  = BCNodeBeta_[iN][iD%nodeDOF_]; 
            gamma1= BCNodeGamma_[iN][iD%nodeDOF_]; 
            if ( beta == 0.0 && alpha != 0.0 )
            {
               if ( numExtNodes_ > 0 )
               {
                  for (iD2=TdiagIA[iD]; iD2<TdiagIA[iD]+diagCounts[iD]; iD2++)
                  {
                     rowInd = TdiagJA[iD2];
                     if ( rowInd >= 0 )
                     {
                        for (iD3 = ToffdIA[rowInd];
                             iD3<ToffdIA[rowInd]+offdCounts[rowInd]; iD3++)
                        {
                           if ( ToffdJA[iD3] == iD && ToffdAA[iD3] != 0.0 )
                           {
                              rhsVector_[rowInd] -= (gamma1/alpha*ToffdAA[iD3]);
                              ToffdAA[iD3] = 0.0;
                              break;
                           }
                        }
                     }
                  }
                  for (iD2=ToffdIA[iD]; iD2<ToffdIA[iD]+offdCounts[iD]; iD2++)
                  {
                     rowInd = ToffdJA[iD2];
                     if ( rowInd != iD && rowInd >= 0 )
                     {
                        for (iD3 = ToffdIA[rowInd];
                             iD3<ToffdIA[rowInd]+offdCounts[rowInd]; iD3++)
                        {
                           if ( ToffdJA[iD3] == iD && ToffdAA[iD3] != 0.0 )
                           {
                              rhsVector_[rowInd] -= (gamma1/alpha*ToffdAA[iD3]);
                              ToffdAA[iD3] = 0.0;
                              break;
                           }
                        }
                     }
                  }
               }
               for (iD2=TdiagIA[iD]; iD2<TdiagIA[iD]+diagCounts[iD]; iD2++)
               {
                  TdiagJA[iD2] = -1;
                  TdiagAA[iD2] = 0.0;
               }
               if ( ToffdIA != NULL )
               {
                  for (iD2=ToffdIA[iD]; iD2<ToffdIA[iD]+offdCounts[iD]; iD2++)
                  {
                     ToffdJA[iD2] = -1;
                     ToffdAA[iD2] = 0.0;
                  }
               }
               rhsVector_[iD] = 0.0;
            }
         }
      }
   }
   PVectorReverseChange( rhsVector_ );
   for ( iD = bound; iD < matDim; iD++ ) rhsVector_[iD] = 0.0;

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
      diagIA_ = new int[matDim+1];
      diagJA_ = new int[diagNNZ];
      diagAA_ = new double[diagNNZ];
      diagonal_ = new double[matDim];
      diagIA_[0] = 0;
   }
   if ( offdNNZ > 0 ) 
   {
      offdIA_ = new int[matDim+1];
      offdJA_ = new int[offdNNZ];
      offdAA_ = new double[offdNNZ];
      offdIA_[0] = 0;
   }
   diagOffset = offdOffset = 0;
   for ( iD = 0; iD < matDim; iD++ ) 
   {
      iCount = diagCounts[iD];
      index  = TdiagIA[iD];
      diagonal_[iD] = 0.0;
      for ( iD2 = 0; iD2 < iCount; iD2++ ) 
      {
         if ( TdiagJA[index] == iD ) 
         {
            if ( TdiagAA[index] != 0.0 ) diagonal_[iD] = TdiagAA[index];
         }
         if ( TdiagJA[index] >= 0 && TdiagAA[index] != 0.0 ) 
         {
            diagJA_[diagOffset] = TdiagJA[index];
            diagAA_[diagOffset++] = TdiagAA[index];
         }
         index++; 
      }
      diagIA_[iD+1] = diagOffset;
      if ( offdNNZ > 0 ) 
      {
         iCount = offdCounts[iD];
         index  = ToffdIA[iD];
         for ( iD2 = 0; iD2 < iCount; iD2++ ) 
         {
            if ( ToffdJA[index] == iD ) 
            {
               if ( ToffdAA[index] != 0.0 ) diagonal_[iD] = ToffdAA[index];
            }
            if ( ToffdJA[index] >= 0 && ToffdAA[index] != 0.0 ) 
            {
               offdJA_[offdOffset] = ToffdJA[index];
               offdAA_[offdOffset++] = ToffdAA[index];
            }
            index++; 
         }
         offdIA_[iD+1] = offdOffset;
      }
   }

   /* -----------------------------------------------------------------
    * fix up diagonal entries in light of parallel processing
    * -----------------------------------------------------------------*/

   PVectorReverseChange( diagonal_ );
   for ( iD = 0; iD < numLocalNodes_*nodeDOF_; iD++ ) 
   {
      if ( diagonal_[iD] == 0.0 ) diagonal_[iD] = 1.0;
      else                        diagonal_[iD] = 1.0 / diagonal_[iD];
   }

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
   if ( outputLevel_ >= 2 )
      printf("%4d : FEI_HYPRE_Impl::buildGlobalMatrixVector ends. \n",mypid_);
}

/**************************************************************************
 solve linear system using conjugate gradient
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::solveUsingCG()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    numTrials, innerIteration;
   double alpha, beta, rho=0.0, rhom1, rnorm0, rnorm, sigma, eps1;
   double *rVec, *pVec, *apVec, *zVec, dArray[2], dArray2[2];

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows = numLocalNodes_ * nodeDOF_;
   extNRows   = numExtNodes_ * nodeDOF_;
   totalNRows = localNRows + extNRows;
   rVec       = new double[totalNRows];
 
   /* -----------------------------------------------------------------
    * assemble the initial guess vector
    * -----------------------------------------------------------------*/

   assembleSolnVector();

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/
 
   matvec( solnVector_, rVec ); 
   for ( irow = 0; irow < localNRows; irow++ ) 
      rVec[irow] = rhsVector_[irow] - rVec[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ ) 
   {
      rnorm0 += (rVec[irow] * rVec[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[1]);
   rnorm  = sqrt(dArray2[0]);
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tFEI_HYPRE_Impl initial rnorm = %e (%e)\n",rnorm,rnorm0);
   if ( rnorm0 == 0.0 ) 
   {
      delete [] rVec;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   iter = 0;
   numTrials  = 0;
   pVec       = new double[totalNRows];
   apVec      = new double[totalNRows];
   zVec       = new double[totalNRows];
   for ( irow = 0; irow < localNRows; irow++ ) pVec[irow] = 0.0;
   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   if ( rnorm < eps1 ) converged = 1;

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   while ( converged == 0 && numTrials < 2 ) 
   {
      innerIteration = 0;
      while ( rnorm >= eps1 && iter < krylovMaxIterations_ ) 
      {
         iter++;
         innerIteration++;
         if ( innerIteration == 1 )
         {
            if ( diagonal_ != NULL )
               for (irow = 0; irow < localNRows; irow++) 
                  zVec[irow] = rVec[irow] * diagonal_[irow];
            else
               for (irow = 0; irow < localNRows; irow++)
                  zVec[irow] = rVec[irow];

            rhom1 = rho;
            rho   = 0.0;
            for ( irow = 0; irow < localNRows; irow++ ) 
               rho += rVec[irow] * zVec[irow];
            dArray[0] = rho;
            MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
            rho  = dArray2[0];
            beta = 0.0;
         }
         else beta = rho / rhom1;
         for ( irow = 0; irow < localNRows; irow++ ) 
            pVec[irow] = zVec[irow] + beta * pVec[irow];
         matvec( pVec, apVec ); 
         sigma = 0.0;
         for ( irow = 0; irow < localNRows; irow++ ) 
            sigma += pVec[irow] * apVec[irow];
         dArray[0] = sigma;
         MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         sigma  = dArray2[0];
         alpha  = rho / sigma; 
         for ( irow = 0; irow < localNRows; irow++ ) 
         {
            solnVector_[irow] += alpha * pVec[irow];
            rVec[irow] -= alpha * apVec[irow];
         }
         rnorm = 0.0;
         for ( irow = 0; irow < localNRows; irow++ ) 
            rnorm += rVec[irow] * rVec[irow];
         dArray[0] = rnorm;

         if ( diagonal_ != NULL )
            for (irow = 0; irow < localNRows; irow++) 
               zVec[irow] = rVec[irow] * diagonal_[irow];
         else
            for (irow = 0; irow < localNRows; irow++) zVec[irow] = rVec[irow];

         rhom1 = rho;
         rho   = 0.0;
         for ( irow = 0; irow < localNRows; irow++ ) 
            rho += rVec[irow] * zVec[irow];
         dArray[1] = rho;
         MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
         rho = dArray2[1]; 
         rnorm = sqrt( dArray2[0] );
         if ( outputLevel_ >= 2 && iter % 1 == 0 && mypid_ == 0 )
            printf("\tFEI_HYPRE_Impl : iteration %d - rnorm = %e (%e)\n",
                   iter, rnorm, eps1);
      }
      matvec( solnVector_, rVec ); 
      for ( irow = 0; irow < localNRows; irow++ ) 
         rVec[irow] = rhsVector_[irow] - rVec[irow]; 
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ ) 
         rnorm += rVec[irow] * rVec[irow];
      dArray[0] = rnorm;
      MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt( dArray2[0] );
      if ( outputLevel_ >= 2 && mypid_ == 0 )
         printf("\tFEI_HYPRE_Impl actual rnorm = %e \n",rnorm);
      if ( (rnorm < eps1 || rnorm < 1.0e-16) || 
            iter >= krylovMaxIterations_ ) converged = 1;
      numTrials++;
   }

   disassembleSolnVector();

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   delete [] rVec;
   delete [] pVec;
   delete [] apVec;
   delete [] zVec;
   return (1-converged);
}

/**************************************************************************
 solve linear system using GMRES
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::solveUsingGMRES()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    innerIterations, iV, iV2, kStep, kp1, jV;
   double rnorm0, rnorm, eps1, epsmac=1.0e-16, gam;
   double **kVectors, **HH, *RS, *C, *S, *dArray, *dArray2;
   double *tVector, *tVector2, *v1, *v2, dtemp;

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows = numLocalNodes_ * nodeDOF_;
   extNRows   = numExtNodes_ * nodeDOF_;
   totalNRows = localNRows + extNRows;
   kVectors   = new double*[gmresDim_+2];
   for (iV = 0; iV <= gmresDim_+1; iV++) kVectors[iV] = new double[totalNRows];
   dArray  = new double[gmresDim_+1];
   dArray2 = new double[gmresDim_+1];
 
   /* -----------------------------------------------------------------
    * assemble the initial guess vector
    * -----------------------------------------------------------------*/

   assembleSolnVector();

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/
 
   tVector = kVectors[1];
   matvec( solnVector_, tVector ); 
   for ( irow = 0; irow < localNRows; irow++ ) 
      tVector[irow] = rhsVector_[irow] - tVector[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ ) 
   {
      rnorm0 += (tVector[irow] * tVector[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[0]);
   rnorm  = sqrt(dArray2[1]);
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tFEI_HYPRE_Impl initial rnorm = %e (%e)\n",
             rnorm, rnorm0);
   if ( rnorm0 < 1.0e-20 ) 
   {
      for (iV = 0; iV <= gmresDim_+1; iV++) delete [] kVectors[iV];
      delete [] kVectors;
      delete [] dArray;
      delete [] dArray2;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   HH = new double*[gmresDim_+2];
   for (iV=1; iV<=gmresDim_+1; iV++) HH[iV] = new double[gmresDim_+2];
   RS      = new double[gmresDim_+2];
   S       = new double[gmresDim_+1];
   C       = new double[gmresDim_+1];

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   iter = 0;

   while ( rnorm >= eps1 && iter < krylovMaxIterations_ ) 
   {
      dtemp = 1.0 / rnorm;
      tVector = kVectors[1];
      for (irow = 0; irow < localNRows; irow++) tVector[irow] *= dtemp;
      RS[1] = rnorm;
      innerIterations = 0;

      while ( innerIterations < gmresDim_ && rnorm >= eps1 && 
              iter < krylovMaxIterations_ ) 
      {
         innerIterations++;
         iter++;
         kStep = innerIterations;
         kp1   = innerIterations + 1;
         v1   = kVectors[kStep];
         v2   = kVectors[0];
         if ( diagonal_ != NULL )
            for (irow = 0; irow < localNRows; irow++) 
               v2[irow] = v1[irow] * diagonal_[irow];
         else
            for (irow = 0; irow < localNRows; irow++) v2[irow] = v1[irow];

         matvec( kVectors[0], kVectors[kp1] ); 

#if 0
         tVector = kVectors[kp1];
         for ( iV = 1; iV <= kStep; iV++ ) 
         {
            dtemp = 0.0;
            tVector2 = kVectors[iV];
            for ( irow = 0; irow < localNRows; irow++ ) 
               dtemp += tVector2[irow] * tVector[irow];
            dArray[iV-1] = dtemp;
         }
         MPI_Allreduce(dArray, dArray2, kStep, MPI_DOUBLE, MPI_SUM, 
                       mpiComm_);

         tVector  = kVectors[kp1];
         for ( iV = 1; iV <= kStep; iV++ ) 
         {
            dtemp = dArray2[iV-1];
            HH[iV][kStep] = dtemp;  
            tVector2 = kVectors[iV];
            for ( irow = 0; irow < localNRows; irow++ ) 
               tVector[irow] -= dtemp * tVector2[irow];
         }
#else
         tVector = kVectors[kp1];
         for ( iV = 1; iV <= kStep; iV++ ) 
         {
            dtemp = 0.0;
            tVector2 = kVectors[iV];
            for ( irow = 0; irow < localNRows; irow++ ) 
               dtemp += tVector2[irow] * tVector[irow];
            dArray[0] = dtemp;
            MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
            dtemp = dArray2[0];
            HH[iV][kStep] = dtemp;  
            for ( irow = 0; irow < localNRows; irow++ ) 
               tVector[irow] -= dtemp * tVector2[irow];
         }
#endif
         dtemp = 0.0;
         for ( irow = 0; irow < localNRows; irow++ ) 
            dtemp += tVector[irow] * tVector[irow];
         MPI_Allreduce(&dtemp, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         dtemp = sqrt(dArray2[0]);
         HH[kp1][kStep] = dtemp;
         if ( dtemp != 0.0 ) 
         {
            dtemp = 1.0 / dtemp;
            for (irow = 0; irow < localNRows; irow++) tVector[irow] *= dtemp;
         }
         for ( iV = 2; iV <= kStep; iV++ ) 
         {
            dtemp = HH[iV-1][kStep];
            HH[iV-1][kStep] =  C[iV-1] * dtemp + S[iV-1] * HH[iV][kStep];
            HH[iV][kStep]   = -S[iV-1] * dtemp + C[iV-1] * HH[iV][kStep];
         }
         gam = sqrt(HH[kStep][kStep]*HH[kStep][kStep]+
                    HH[kp1][kStep]*HH[kp1][kStep]);
         if ( gam == 0.0 ) gam = epsmac;
         C[kStep]  = HH[kStep][kStep] / gam;
         S[kStep]  = HH[kp1][kStep] / gam;
         RS[kp1]   = -S[kStep] * RS[kStep];
         RS[kStep] = C[kStep] * RS[kStep];
         HH[kStep][kStep] = C[kStep] * HH[kStep][kStep] + 
                            S[kStep] * HH[kp1][kStep];
         rnorm = habs(RS[kp1]);
         if ( outputLevel_ >= 2 && mypid_ == 0 )
            printf("\tFEI_HYPRE_Impl : iteration %d - rnorm = %e\n",
                   iter, rnorm);
      }
      RS[kStep] = RS[kStep] / HH[kStep][kStep];
      for ( iV = 2; iV <= kStep; iV++ ) 
      {
         iV2 = kStep - iV + 1;
         dtemp = RS[iV2];
         for ( jV = iV2+1; jV <= kStep; jV++ ) 
            dtemp = dtemp - HH[iV2][jV] * RS[jV];
         RS[iV2] = dtemp / HH[iV2][iV2];
      }
      tVector = kVectors[1];
      dtemp   = RS[1];
      for ( irow = 0; irow < localNRows; irow++ ) tVector[irow] *= dtemp;
      for ( iV = 2; iV <= kStep; iV++ ) 
      {
         dtemp = RS[iV];
         tVector2 = kVectors[iV];
         for ( irow = 0; irow < localNRows; irow++ ) 
            tVector[irow] += dtemp * tVector2[irow];
      }
      tVector = kVectors[1];
      if ( diagonal_ != NULL )
      {
         for (irow = 0; irow < localNRows; irow++) 
            tVector[irow] *= diagonal_[irow];
      }
      for (irow = 0; irow < localNRows; irow++) 
         solnVector_[irow] += tVector[irow];
      matvec( solnVector_, tVector ); 
      for ( irow = 0; irow < localNRows; irow++ ) 
         tVector[irow] = rhsVector_[irow] - tVector[irow];
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ ) 
         rnorm += (tVector[irow] * tVector[irow]);
      MPI_Allreduce(&rnorm, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt(dArray2[0]);
   }
   if ( rnorm < eps1 ) converged = 1; 
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tFEI_HYPRE_Impl : final rnorm = %e\n", rnorm);

   disassembleSolnVector();

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   for (iV = 0; iV <= gmresDim_+1; iV++) delete [] kVectors[iV];
   delete [] kVectors;
   for ( iV =1; iV <= gmresDim_+1; iV++ ) delete [] HH[iV];
   delete [] HH;
   delete [] RS;
   delete [] S;
   delete [] C;
   delete [] dArray;
   delete [] dArray2;
   return (1-converged);
}

/**************************************************************************
 solve linear system using CGS 
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::solveUsingCGS()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    numTrials, innerIteration;
   double *rVec, *rhVec, *vVec, *pVec, *qVec, *uVec, *tVec;
   double rho1, rho2, sigma, alpha, dtemp, dtemp2, rnorm, rnorm0;
   double beta, beta2, eps1, dArray[2], dArray2[2];

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows = numLocalNodes_ * nodeDOF_;
   extNRows   = numExtNodes_ * nodeDOF_;
   totalNRows = localNRows + extNRows;
   rVec       = new double[totalNRows];
 
   /* -----------------------------------------------------------------
    * assemble the initial guess vector
    * -----------------------------------------------------------------*/

   assembleSolnVector();

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/
 
   matvec( solnVector_, rVec ); 
   for ( irow = 0; irow < localNRows; irow++ ) 
      rVec[irow] = rhsVector_[irow] - rVec[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ ) 
   {
      rnorm0 += (rVec[irow] * rVec[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[1]);
   rnorm  = sqrt(dArray2[0]);
   if ( outputLevel_ >= 1 && mypid_ == 0 )
      printf("\tFEI_HYPRE_Impl initial rnorm = %e (%e)\n",rnorm,rnorm0);
   if ( rnorm0 == 0.0 ) 
   {
      delete [] rVec;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   rhVec = new double[totalNRows];
   vVec  = new double[totalNRows];
   pVec  = new double[totalNRows];
   qVec  = new double[totalNRows];
   uVec  = new double[totalNRows];
   tVec  = new double[totalNRows];
   for (irow = 0; irow < localNRows; irow++) rhVec[irow] = rVec[irow];
   for (irow = 0; irow < totalNRows; irow++) pVec[irow] = qVec[irow] = 0.0;
   rho2 = rnorm * rnorm;
   beta = rho2;
   iter = 0;
   numTrials  = 0;
   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   if ( rnorm < eps1 )  converged = 1;

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   while ( converged == 0 && numTrials < 2 ) 
   {
      innerIteration = 0;
      while ( rnorm >= eps1 && iter < krylovMaxIterations_ ) 
      {
         iter++;
         innerIteration++;
         rho1 = rho2;
         beta2 = beta * beta;
         for (irow = 0; irow < totalNRows; irow++) 
         {
            tVec[irow] = beta * qVec[irow];
            uVec[irow] = rVec[irow] + tVec[irow];
            pVec[irow] = uVec[irow] + tVec[irow] + beta2 * pVec[irow];
         }
         if ( diagonal_ != NULL )
         {
            for (irow = 0; irow < localNRows; irow++) 
               tVec[irow] = pVec[irow] * diagonal_[irow];
         }
         else
            for (irow = 0; irow < localNRows; irow++) tVec[irow] = pVec[irow];

         matvec( tVec, vVec ); 
         sigma = 0.0;
         for ( irow = 0; irow < localNRows; irow++ ) 
            sigma += (rhVec[irow] * vVec[irow]);
         MPI_Allreduce(&sigma, dArray, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         sigma = dArray[0];
         alpha = rho1 / sigma;

         for (irow = 0; irow < totalNRows; irow++) 
         {
            qVec[irow] = uVec[irow] - alpha * vVec[irow];
            uVec[irow] += qVec[irow];
         }
         if ( diagonal_ != NULL )
         {
            for (irow = 0; irow < localNRows; irow++) 
            {
               tVec[irow] = uVec[irow] * diagonal_[irow];
               solnVector_[irow] += alpha * uVec[irow] * diagonal_[irow];
            }
         }
         else
         {
            for (irow = 0; irow < localNRows; irow++) 
            {
               tVec[irow] = uVec[irow];
               solnVector_[irow] += alpha * uVec[irow];
            }
         }
         matvec( tVec, vVec ); 

         for (irow = 0; irow < totalNRows; irow++) 
            rVec[irow] -= alpha * vVec[irow];

         dtemp = dtemp2 = 0.0;
         for ( irow = 0; irow < localNRows; irow++ ) 
         {
            dtemp  += (rVec[irow] * rhVec[irow]);
            dtemp2 += (rVec[irow] * rVec[irow]);
         }
         dArray[0] = dtemp;
         dArray[1] = dtemp2;
         MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
         rho2 = dArray2[0];
         beta = rho2 / rho1;
         rnorm = sqrt(dArray2[1]);
         if ( outputLevel_ >= 1 && iter % 1 == 0 && mypid_ == 0 )
            printf("\tFEI_HYPRE_Impl : iteration %d - rnorm = %e (%e)\n",
                   iter, rnorm, eps1);
      }
      matvec( solnVector_, rVec ); 
      for ( irow = 0; irow < localNRows; irow++ ) 
         rVec[irow] = rhsVector_[irow] - rVec[irow]; 
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ ) 
         rnorm += rVec[irow] * rVec[irow];
      MPI_Allreduce(&rnorm, dArray, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt( dArray[0] );
      if ( outputLevel_ >= 2 && mypid_ == 0 )
         printf("\tFEI_HYPRE_Impl actual rnorm = %e \n",rnorm);
      if ( rnorm < eps1 || iter >= krylovMaxIterations_ ) break;
      numTrials++;
   }
   if ( rnorm < eps1 ) converged = 1;

   disassembleSolnVector();

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   delete [] rVec;
   delete [] rhVec;
   delete [] pVec;
   delete [] qVec;
   delete [] uVec;
   delete [] tVec;
   return (1-converged);
}

/**************************************************************************
 solve linear system using Bicgstab 
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::solveUsingBicgstab()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    iM, jM, numTrials, innerIteration, blen=2, vecByteSize;
   double *rVec, *rhVec, *xhVec, *tVec, **utVec, **rtVec;
   double rho, rho1, alpha, dtemp, dtemp2, rnorm, rnorm0;
   double beta, omega, gamma1, eps1, dArray[2], dArray2[2];
   double *sigma, *gammap, *gammanp, *gammapp, **mat, **tau;

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows  = numLocalNodes_ * nodeDOF_;
   extNRows    = numExtNodes_ * nodeDOF_;
   totalNRows  = localNRows + extNRows;
   rVec        = new double[totalNRows];
   vecByteSize = localNRows * sizeof(double);
 
   /* -----------------------------------------------------------------
    * assemble the initial guess vector
    * -----------------------------------------------------------------*/

   assembleSolnVector();

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/
 
   matvec( solnVector_, rVec ); 
   for ( irow = 0; irow < localNRows; irow++ ) 
      rVec[irow] = rhsVector_[irow] - rVec[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ ) 
   {
      rnorm0 += (rVec[irow] * rVec[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[1]);
   rnorm  = sqrt(dArray2[0]);
   if ( outputLevel_ >= 1 && mypid_ == 0 )
      printf("\tFEI_HYPRE_Impl initial rnorm = %e (%e)\n",rnorm,rnorm0);
   if ( rnorm0 == 0.0 ) 
   {
      delete [] rVec;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   if ( rnorm < eps1 )  converged = 1;

   sigma   = new double[blen+1];
   gammap  = new double[blen+1];
   gammanp = new double[blen+1];
   gammapp = new double[blen+1];
   mat     = new double*[blen+1];
   tau     = new double*[blen+1];
   for ( iM = 1; iM <= blen; iM++ ) 
   {
      mat[iM] = new double[blen+1];
      tau[iM] = new double[blen+1];
   }
   rhVec = new double[totalNRows];
   xhVec = new double[totalNRows];
   tVec  = new double[totalNRows];
   utVec = new double*[blen+2];
   rtVec = new double*[blen+2];
   for ( iM = 0; iM < blen+2; iM++ ) 
   {
      utVec[iM] = new double[totalNRows];
      rtVec[iM] = new double[totalNRows];
   }
   iter = 0;
   numTrials  = 0;

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   while ( converged == 0 && numTrials < 2 ) 
   {
      innerIteration = 0;
      memcpy( rhVec, rVec, vecByteSize );
      memcpy( rtVec[0], rVec, vecByteSize );
      memcpy( xhVec, solnVector_, vecByteSize );
      memset( utVec[0], 0, vecByteSize );
      omega = rho = 1.0;
      alpha = 0.0;
      while ( rnorm >= eps1 && iter < krylovMaxIterations_ ) 
      {
         iter += blen;
         innerIteration += blen;
         memcpy( utVec[1], utVec[0], vecByteSize );
         memcpy( rtVec[1], rtVec[0], vecByteSize );
         rho = -omega * rho;
         for ( iM = 0; iM < blen; iM++ )
         {
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ ) 
               dtemp += (rhVec[irow] * rtVec[iM+1][irow]);
            MPI_Allreduce(&dtemp, &rho1, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
            beta = alpha * rho1 / rho;
            rho   = rho1;
            dtemp = -beta;
            for ( jM = 0; jM <= iM; jM++ ) 
               for ( irow = 0; irow < localNRows; irow++ ) 
                  utVec[jM+1][irow] = dtemp * utVec[jM+1][irow] + 
                                      rtVec[jM+1][irow]; 
            if ( diagonal_ != NULL )
            {
               for (irow = 0; irow < localNRows; irow++) 
                  tVec[irow] = utVec[iM+1][irow] * diagonal_[irow];
            }
            else
            {
               memcpy( tVec, utVec[iM+1], vecByteSize );
            }
            matvec( tVec, utVec[iM+2] ); 
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ ) 
               dtemp += (rhVec[irow] * utVec[iM+2][irow]);
            MPI_Allreduce(&dtemp, &gamma1, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);

            alpha = rho / gamma1; 
            for ( jM = 0; jM <= iM; jM++ ) 
               for ( irow = 0; irow < localNRows; irow++ ) 
                  rtVec[jM+1][irow] -= alpha * utVec[jM+2][irow]; 

            if ( diagonal_ != NULL )
            {
               for (irow = 0; irow < localNRows; irow++) 
                  tVec[irow] = rtVec[iM+1][irow] * diagonal_[irow];
            }
            else
            {
               memcpy( tVec, rtVec[iM+1], vecByteSize );
            }
            matvec( tVec, rtVec[iM+2] ); 
            for (irow = 0; irow < localNRows; irow++) 
               xhVec[irow] += alpha * utVec[1][irow];
         }
         for ( iM = 1; iM <= blen; iM++ )
            for ( jM = 1; jM <= blen; jM++ ) mat[iM][jM] = 0.0;
         for ( iM = 1; iM <= blen; iM++ )
         {
            for ( jM = 1; jM <= iM-1; jM++ ) 
            {
               dtemp = 0.0;
               for ( irow = 0; irow < localNRows; irow++ ) 
                  dtemp += (rtVec[jM+1][irow] * rtVec[iM+1][irow]);
               MPI_Allreduce(&dtemp, &dtemp2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
               tau[jM][iM] = dtemp2 / sigma[jM];
               mat[jM][iM] = tau[jM][iM] * sigma[jM];
               dtemp = -tau[jM][iM];
               for (irow = 0; irow < localNRows; irow++) 
                  rtVec[iM+1][irow] += dtemp * rtVec[jM+1][irow];
            }
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ ) 
               dtemp += (rtVec[iM+1][irow] * rtVec[iM+1][irow]);
            dArray[0] = dtemp;
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ ) 
               dtemp += (rtVec[1][irow] * rtVec[iM+1][irow]);
            dArray[1] = dtemp;
            MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
            sigma[iM] = dArray2[0];
            mat[iM][iM] = sigma[iM];
            gammap[iM] = dArray2[1] / sigma[iM];
         }
         gammanp[blen] = gammap[blen];
         omega = gammanp[blen];
         for ( iM = blen-1; iM >= 1; iM-- ) 
         {
           gammanp[iM] = gammap[iM];
           for (jM=iM+1; jM<=blen; jM++)
             gammanp[iM] = gammanp[iM] - tau[iM][jM] * gammanp[jM];
         }
         for (iM=1; iM<=blen-1; iM++) 
         {
            gammapp[iM] = gammanp[iM+1];
            for (jM=iM+1; jM<=blen-1; jM++)
               gammapp[iM] = gammapp[iM] + tau[iM][jM] * gammanp[jM+1];
         }
         dtemp = gammanp[1];
         for (irow = 0; irow < localNRows; irow++) 
            xhVec[irow] += dtemp * rtVec[1][irow];
         dtemp = - gammap[blen];
         for (irow = 0; irow < localNRows; irow++) 
            rtVec[1][irow] += dtemp * rtVec[blen+1][irow];
         dtemp = - gammanp[blen];
         for (irow = 0; irow < localNRows; irow++) 
            utVec[1][irow] += dtemp * utVec[blen+1][irow];
         for (iM=1; iM<=blen-1; iM++) 
         {
            dtemp = - gammanp[iM];
            for (irow = 0; irow < localNRows; irow++) 
               utVec[1][irow] += dtemp * utVec[iM+1][irow];
            dtemp = gammapp[iM];
            for (irow = 0; irow < localNRows; irow++) 
               xhVec[irow] += dtemp * rtVec[iM+1][irow];
            dtemp = - gammap[iM];
            for (irow = 0; irow < localNRows; irow++) 
               rtVec[1][irow] += dtemp * rtVec[iM+1][irow];
         }
         memcpy( utVec[0], utVec[1], vecByteSize );
         memcpy( rtVec[0], rtVec[1], vecByteSize );
         memcpy( solnVector_, xhVec, vecByteSize );
         dtemp = 0.0;
         for ( irow = 0; irow < localNRows; irow++ ) 
            dtemp += (rtVec[1][irow] * rtVec[1][irow]);
         MPI_Allreduce(&dtemp, &rnorm, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         rnorm = sqrt( rnorm );
         if ( outputLevel_ >= 1 && iter % 1 == 0 && mypid_ == 0 )
            printf("\tFEI_HYPRE_Impl : iteration %d - rnorm = %e (%e)\n",
                   iter, rnorm, eps1);
      }

      if ( diagonal_ != NULL )
      {
         for (irow = 0; irow < localNRows; irow++) 
            solnVector_[irow] *= diagonal_[irow];
      }
      matvec( solnVector_, rVec ); 
      for ( irow = 0; irow < localNRows; irow++ ) 
         rVec[irow] = rhsVector_[irow] - rVec[irow]; 
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ ) 
         rnorm += rVec[irow] * rVec[irow];
      MPI_Allreduce(&rnorm, dArray, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt( dArray[0] );
      if ( outputLevel_ >= 2 && mypid_ == 0 )
         printf("\tFEI_HYPRE_Impl actual rnorm = %e \n",rnorm);
      if ( rnorm < eps1 || iter >= krylovMaxIterations_ ) break;
      numTrials++;
   }
   if ( rnorm < eps1 ) converged = 1;

   disassembleSolnVector();

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   delete [] sigma;
   delete [] gammap;
   delete [] gammanp;
   delete [] gammapp;
   for ( iM = 1; iM <= blen; iM++ ) 
   {
      delete [] mat[iM];
      delete [] tau[iM];
   }
   delete [] mat;
   delete [] tau;
   delete [] rVec;
   delete [] rhVec;
   delete [] xhVec;
   delete [] tVec;
   for ( iM = 0; iM < blen+2; iM++ ) 
   {
      delete [] utVec[iM];
      delete [] rtVec[iM];
   }
   delete [] utVec;
   delete [] rtVec;

   return (1-converged);
}

/**************************************************************************
 solve linear system using SuperLU 
 -------------------------------------------------------------------------*/
int FEI_HYPRE_Impl::solveUsingSuperLU()
{
#if HAVE_SUPERLU
   int    localNRows, localNnz, *countArray, irow, jcol, *cscIA, *cscJA;
   int    colNum, index, *etree, permcSpec, lwork, panelSize, relax, info;
   int    *permC, *permR;
   double *cscAA, diagPivotThresh, dropTol, *rVec, rnorm;
   superlu_options_t slu_options;
   SuperLUStat_t     slu_stat;
   trans_t           trans;
   SuperMatrix superLU_Amat;
   SuperMatrix superLU_Lmat;
   SuperMatrix superLU_Umat;
   SuperMatrix AC;
   SuperMatrix B;

   /* ---------------------------------------------------------------
    * conversion from CSR to CSC
    * -------------------------------------------------------------*/
   
   localNRows = numLocalNodes_ * nodeDOF_;
   countArray = new int[localNRows];
   for ( irow = 0; irow < localNRows; irow++ ) countArray[irow] = 0;
   for ( irow = 0; irow < localNRows; irow++ )
      for ( jcol = diagIA_[irow]; jcol < diagIA_[irow+1]; jcol++ )
         countArray[diagJA_[jcol]]++;
   localNnz = diagIA_[localNRows];
   cscJA = (int *)    malloc( (localNRows+1) * sizeof(int) );
   cscIA = (int *)    malloc( localNnz * sizeof(int) );
   cscAA = (double *) malloc( localNnz * sizeof(double) );
   cscJA[0] = 0;
   localNnz = 0;
   for ( jcol = 1; jcol <= localNRows; jcol++ )
   {
      localNnz += countArray[jcol-1];
      cscJA[jcol] = localNnz;
   }
   for ( irow = 0; irow < localNRows; irow++ )
   {
      for ( jcol = diagIA_[irow]; jcol < diagIA_[irow+1]; jcol++ )
      {
         colNum = diagJA_[jcol];
         index  = cscJA[colNum]++;
         cscIA[index] = irow;
         cscAA[index] = diagAA_[jcol];
      }
   }
   cscJA[0] = 0;
   localNnz = 0;
   for ( jcol = 1; jcol <= localNRows; jcol++ )
   {
      localNnz += countArray[jcol-1];
      cscJA[jcol] = localNnz;
   }
   delete [] countArray;

   /* ---------------------------------------------------------------
    * make SuperMatrix
    * -------------------------------------------------------------*/

   dCreate_CompCol_Matrix(&superLU_Amat, localNRows, localNRows, 
                          cscJA[localNRows], cscAA, cscIA, cscJA, SLU_NC, 
                          SLU_D, SLU_GE);
   etree     = new int[localNRows];
   permC     = new int[localNRows];
   permR     = new int[localNRows];
   permcSpec = 0;
   get_perm_c(permcSpec, &superLU_Amat, permC);
   slu_options.Fact = DOFACT;
   slu_options.SymmetricMode = NO;
   sp_preorder(&slu_options, &superLU_Amat, permC, etree, &AC);
   diagPivotThresh = 1.0;
   dropTol = 0.0;
   panelSize = sp_ienv(1);
   relax = sp_ienv(2);
   StatInit(&slu_stat);
   lwork = 0;
   slu_options.ColPerm = MY_PERMC;
   slu_options.Fact = DOFACT;
   slu_options.DiagPivotThresh = diagPivotThresh;

   dgstrf(&slu_options, &AC, dropTol, relax, panelSize,
          etree, NULL, lwork, permC, permR, &superLU_Lmat,
          &superLU_Umat, &slu_stat, &info);

   Destroy_CompCol_Permuted(&AC);
   Destroy_CompCol_Matrix(&superLU_Amat);
   delete [] etree;

   /* -------------------------------------------------------------
    * create a SuperLU dense matrix from right hand side
    * -----------------------------------------------------------*/

   solnVector_ = new double[localNRows];
   for ( irow = 0; irow < localNRows; irow++ ) 
      solnVector_[irow] = rhsVector_[irow];
   dCreate_Dense_Matrix(&B, localNRows, 1, solnVector_, localNRows, 
                        SLU_DN, SLU_D, SLU_GE);

   /* -------------------------------------------------------------
    * solve the problem
    * -----------------------------------------------------------*/

   trans = NOTRANS;
   dgstrs (trans, &superLU_Lmat, &superLU_Umat, permC, permR, &B, 
           &slu_stat, &info);
   rVec = new double[localNRows];
   matvec( solnVector_, rVec ); 
   for ( irow = 0; irow < localNRows; irow++ ) 
      rVec[irow] = rhsVector_[irow] - rVec[irow]; 
   rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ ) 
      rnorm += rVec[irow] * rVec[irow];
   rnorm = sqrt( rnorm );
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tFEI_HYPRE_Impl rnorm = %e \n",rnorm);

   disassembleSolnVector();
   krylovIterations_   = 1;
   krylovResidualNorm_ = rnorm;

   /* -------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------*/

   Destroy_SuperMatrix_Store(&B);
   delete [] rVec;
   if ( permR != NULL )
   {
      Destroy_SuperNode_Matrix(&superLU_Lmat);
      Destroy_CompCol_Matrix(&superLU_Umat);
   }
   delete [] permR;
   delete [] permC;
   StatFree(&slu_stat);
   return (info);
#else
   return (1);
#endif
}
 
/**************************************************************************
 matrix vector multiply
 -------------------------------------------------------------------------*/
void FEI_HYPRE_Impl::matvec(double *xvec, double *yvec)
{
   /* -----------------------------------------------------------------
    * exchange vector information between processors
    * -----------------------------------------------------------------*/

   PVectorInterChange( xvec );

   /* -----------------------------------------------------------------
    * in case global stiffness matrix has been composed, use it
    * -----------------------------------------------------------------*/

   if ( diagIA_ != NULL )
   {
      int matDim = ( numLocalNodes_ + numExtNodes_ ) * nodeDOF_;
      double ddata;
      for ( int iD = 0; iD < matDim; iD++ ) 
      {
         ddata = 0.0;
         for ( int iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ ) 
           ddata += diagAA_[iD2] * xvec[diagJA_[iD2]];
         yvec[iD] = ddata;
      }
   }

   /* -----------------------------------------------------------------
    * in case global stiffness matrix has been composed, use it
    * -----------------------------------------------------------------*/

   if ( offdIA_ != NULL )
   {
      int matDim = ( numLocalNodes_ + numExtNodes_ ) * nodeDOF_;
      double ddata;
      for ( int iD = 0; iD < matDim; iD++ ) 
      {
         ddata = 0.0;
         for ( int iD2 = offdIA_[iD]; iD2 < offdIA_[iD+1]; iD2++ ) 
           ddata += offdAA_[iD2] * xvec[offdJA_[iD2]];
         yvec[iD] += ddata;
      }
   }

   /* -----------------------------------------------------------------
    * exchange vector information between processors
    * -----------------------------------------------------------------*/

   PVectorReverseChange( yvec );
}

/**************************************************************************
 form right hand side vector from element load vectors 
 -------------------------------------------------------------------------*/
void FEI_HYPRE_Impl::assembleRHSVector()
{
   int    iB, iE, iN, iD, **elemNodeLists, numElems, elemNumNodes;
   int    eqnIndex1, eqnIndex2, matDim;
   double **rhsVectors;

   if ( rhsVector_ != NULL ) delete [] rhsVector_;
   matDim = (numLocalNodes_ + numExtNodes_) * nodeDOF_;
   rhsVector_ = new double[matDim];
   for ( iD = 0; iD < matDim; iD++ ) rhsVector_[iD] = 0.0;

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
            eqnIndex2 = iN * nodeDOF_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
               rhsVector_[eqnIndex1+iD]  += rhsVectors[iE][eqnIndex2+iD];
         }
      }
   }
   PVectorReverseChange( rhsVector_ );
   PVectorInterChange( rhsVector_ );
}

/**************************************************************************
 form solution vector 
 -------------------------------------------------------------------------*/
void FEI_HYPRE_Impl::assembleSolnVector()
{
   int    iB, iE, iN, iD, **elemNodeLists, numElems, elemNumNodes;
   int    eqnIndex1, eqnIndex2, matDim;
   double **solnVectors;

   matDim = (numLocalNodes_ + numExtNodes_) * nodeDOF_;
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
            eqnIndex2 = iN * nodeDOF_;
            for ( iD = 0; iD < nodeDOF_; iD++ )
               solnVector_[eqnIndex1+iD] += solnVectors[iE][eqnIndex2+iD];
         }
      }
   }
   PVectorReverseChange( solnVector_ );
   PVectorInterChange( solnVector_ );
}

/**************************************************************************
 distribute solution vector to element solution vectors
 -------------------------------------------------------------------------*/
void FEI_HYPRE_Impl::disassembleSolnVector()
{
   int    iB, iE, iN, iD, **elemNodeLists, numElems, elemNumNodes;
   int    eqnIndex1, eqnIndex2;
   double **solnVectors;

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
void FEI_HYPRE_Impl::IntSort(int *ilist, int left, int right)
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
void FEI_HYPRE_Impl::IntSort2(int *ilist, int *ilist2, int left, int right)
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
void FEI_HYPRE_Impl::IntSort2a(int *ilist,double *dlist,int left,int right)
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
void FEI_HYPRE_Impl::PVectorInterChange( double *dvec )
{
   int         iD, iD2, iP, ind1, ind2;
   double      **dRecvBufs, **dSendBufs;
   MPI_Request *requests;
   MPI_Status  status;

   if ( nRecvs_ > 0 ) 
   {
      dRecvBufs = new double*[nRecvs_];
      requests  = new MPI_Request[nRecvs_];
      for ( iP = 0; iP < nRecvs_; iP++ ) 
         dRecvBufs[iP] = new double[recvLengs_[iP]*nodeDOF_];
   }
   if ( nSends_ > 0 ) 
   {
      dSendBufs = new double*[nSends_];
      for ( iP = 0; iP < nSends_; iP++ ) 
      {
         dSendBufs[iP] = new double[sendLengs_[iP]*nodeDOF_];
         for ( iD = 0; iD < sendLengs_[iP]; iD++ )
         {
            ind1 = sendProcIndices_[iP][iD] * nodeDOF_;
            ind2 = iD * nodeDOF_;
            for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
               dSendBufs[iP][ind2+iD2] = dvec[ind1+iD2]; 
         }
      }
   }
   for ( iP = 0; iP < nRecvs_; iP++ )
      MPI_Irecv( dRecvBufs[iP], recvLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                 recvProcs_[iP], 40343, mpiComm_, &requests[iP]);
   for ( iP = 0; iP < nSends_; iP++ )
      MPI_Send( dSendBufs[iP], sendLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                sendProcs_[iP], 40343, mpiComm_);
   for ( iP = 0; iP < nRecvs_; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nRecvs_ > 0 ) delete [] requests;
   for ( iP = 0; iP < nRecvs_; iP++ )
   {
      for ( iD = 0; iD < recvLengs_[iP]; iD++ )
      {
         ind1 = recvProcIndices_[iP][iD] * nodeDOF_;
         ind2 = iD * nodeDOF_;
         for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
            dvec[ind1+iD2] = dRecvBufs[iP][ind2+iD2]; 
      }
      delete [] dRecvBufs[iP];
   }
   if ( nRecvs_ > 0 ) delete [] dRecvBufs;
   if ( nSends_ > 0 ) 
   {
      for ( iP = 0; iP < nSends_; iP++ ) delete [] dSendBufs[iP];
      delete [] dSendBufs;
   }
}

/**************************************************************************
 compress overlapped vector
 -------------------------------------------------------------------------*/
void FEI_HYPRE_Impl::PVectorReverseChange( double *dvec )
{
   int         iD, iD2, iP, ind1, ind2;
   double      **dRecvBufs, **dSendBufs;
   MPI_Request *requests;
   MPI_Status  status;

   if ( nSends_ > 0 ) 
   {
      dRecvBufs = new double*[nSends_];
      requests  = new MPI_Request[nSends_];
      for ( iP = 0; iP < nSends_; iP++ ) 
         dRecvBufs[iP] = new double[sendLengs_[iP]*nodeDOF_];
   }
   if ( nRecvs_ > 0 ) 
   {
      dSendBufs = new double*[nRecvs_];
      for ( iP = 0; iP < nRecvs_; iP++ ) 
      {
         dSendBufs[iP] = new double[recvLengs_[iP]*nodeDOF_];
         for ( iD = 0; iD < recvLengs_[iP]; iD++ )
         {
            ind1 = recvProcIndices_[iP][iD] * nodeDOF_;
            ind2 = iD * nodeDOF_;
            for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
               dSendBufs[iP][ind2+iD2] = dvec[ind1+iD2]; 
         }
      }
   }
   for ( iP = 0; iP < nSends_; iP++ )
      MPI_Irecv( dRecvBufs[iP], sendLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                 sendProcs_[iP], 40342, mpiComm_, &requests[iP]);
   for ( iP = 0; iP < nRecvs_; iP++ )
      MPI_Send( dSendBufs[iP], recvLengs_[iP]*nodeDOF_, MPI_DOUBLE,
                recvProcs_[iP], 40342, mpiComm_);
   for ( iP = 0; iP < nSends_; iP++ ) MPI_Wait( &requests[iP], &status );

   if ( nSends_ > 0 ) delete [] requests;
   for ( iP = 0; iP < nSends_; iP++ )
   {
      for ( iD = 0; iD < sendLengs_[iP]; iD++ )
      {
         ind1 = sendProcIndices_[iP][iD] * nodeDOF_;
         ind2 = iD * nodeDOF_;
         for ( iD2 = 0; iD2 < nodeDOF_; iD2++ )
            dvec[ind1+iD2] += dRecvBufs[iP][ind2+iD2]; 
      }
      delete [] dRecvBufs[iP];
   }
   if ( nSends_ > 0 ) delete [] dRecvBufs;
   if ( nRecvs_ > 0 ) 
   {
      for ( iP = 0; iP < nRecvs_; iP++ ) delete [] dSendBufs[iP];
      delete [] dSendBufs;
   }
}

/**************************************************************************
 print matrix and right hand side vector to a file
 -------------------------------------------------------------------------*/
void FEI_HYPRE_Impl::printLinearSystem()
{
   int  iD, iD2, offset, iBegin, iEnd, totalNNZ;
   char filename[20];
   FILE *fp;

   sprintf(filename, "mat.%d", mypid_);
   fp       = fopen(filename, "w");
   offset   = globalNodeOffsets_[mypid_];
   iEnd     = numLocalNodes_ * nodeDOF_;
   totalNNZ = diagIA_[iEnd];
   if ( offdIA_ != NULL ) totalNNZ += offdIA_[iEnd];
   fprintf(fp, "%6d  %7d \n", iEnd, totalNNZ);
   for ( iD = 0; iD < iEnd; iD++ )
   {
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         if ( diagJA_[iD2] == iD )
            fprintf(fp,"%6d  %6d  %25.16e \n", iD+1+offset, 
                    diagJA_[iD2]+1+offset, diagAA_[iD2]);
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         if ( diagJA_[iD2] != iD )
            fprintf(fp,"%6d  %6d  %25.16e \n", iD+1+offset, 
                    diagJA_[iD2]+1+offset, diagAA_[iD2]);
      if ( offdIA_ != NULL )
      {
         for ( iD2 = offdIA_[iD]; iD2 < offdIA_[iD+1]; iD2++ )
            fprintf(fp,"%6d  %6d  %25.16e \n", iD+1+offset, 
                    nodeExtNewGlobalIDs_[offdJA_[iD2]-iEnd]+1,offdAA_[iD2]);
      }
   }
   iBegin = numLocalNodes_ * nodeDOF_;
   iEnd   = (numLocalNodes_ + numExtNodes_ ) * nodeDOF_;
   for ( iD = iBegin; iD < iEnd; iD++ )
   {
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         if ( diagJA_[iD2] == iD )
            fprintf(fp,"%6d  %6d  %25.16e \n",nodeExtNewGlobalIDs_[iD-iBegin]+1,
                    diagJA_[iD2]+1+offset, diagAA_[iD]);
      for ( iD2 = diagIA_[iD]; iD2 < diagIA_[iD+1]; iD2++ )
         if ( diagJA_[iD2] != iD )
            fprintf(fp,"%6d  %6d  %25.16e \n",nodeExtNewGlobalIDs_[iD-iBegin]+1,
                    diagJA_[iD2]+1+offset, diagAA_[iD]);
      if ( offdIA_ != NULL )
      {
         for ( iD2 = offdIA_[iD]; iD2 < offdIA_[iD+1]; iD2++ )
            fprintf(fp,"%6d  %6d  %25.16e \n",nodeExtNewGlobalIDs_[iD-iBegin]+1,
                    nodeExtNewGlobalIDs_[offdJA_[iD2]-iBegin]+1, offdAA_[iD2]);
      }
   }
   fclose(fp);

   sprintf(filename, "rhs.%d", mypid_);
   fp      = fopen(filename, "w");
   iEnd    = numLocalNodes_ * nodeDOF_;
   fprintf(fp, "%6d \n", iEnd);
   for ( iD = 0; iD < iEnd; iD++ )
   {
      fprintf(fp,"%6d  %25.16e \n", iD+1+offset, rhsVector_[iD]); 
   }
   iBegin = numLocalNodes_ * nodeDOF_;
   iEnd   = (numLocalNodes_ + numExtNodes_ ) * nodeDOF_;
   for ( iD = iBegin; iD < iEnd; iD++ )
   {
      fprintf(fp,"%8d  %25.16e\n",nodeExtNewGlobalIDs_[iD-iBegin]+1,
              rhsVector_[iD]);
   }
   fclose(fp);
}

