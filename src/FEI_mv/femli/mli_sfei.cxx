/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**************************************************************************
 **************************************************************************
 * MLI_SFEI Class functions (simplified FEI)
 **************************************************************************
 **************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "mli_sfei.h"

/**************************************************************************
 * constructor 
 *-----------------------------------------------------------------------*/

MLI_SFEI::MLI_SFEI(MPI_Comm mpiComm)
{
   mpiComm_          = mpiComm;
   outputLevel_      = 1;
   maxElemBlocks_    = 0;
   nElemBlocks_      = 0;
   blkNumElems_      = NULL;
   blkElemNEqns_     = NULL;
   blkNodeDofs_      = NULL;
   blkElemEqnLists_  = NULL;
   blkElemStiffness_ = NULL;
   // the following variable is added to counter the fact that
   // the Sandia FEI called addNumElems starting with blkID = 0
   // while it called loadElemBlock starting with the actual blkID
   blkIDBase_        = -1;
}

//*************************************************************************
// destructor 
//-------------------------------------------------------------------------

MLI_SFEI::~MLI_SFEI()
{
   int i, j;
   if ( blkElemEqnLists_ != NULL )
   {
      for ( i = 0; i < nElemBlocks_; i++ ) 
      {
         for ( j = 0; j < blkNumElems_[i]; j++ ) 
            if ( blkElemEqnLists_[i][j] != NULL ) 
               delete [] blkElemEqnLists_[i][j];
         delete [] blkElemEqnLists_[i];
      }
      delete [] blkElemEqnLists_;
   }
   if ( blkElemStiffness_ != NULL )
   {
      for ( i = 0; i < nElemBlocks_; i++ ) 
      {
         for ( j = 0; j < blkNumElems_[i]; j++ ) 
            if ( blkElemStiffness_[i][j] != NULL ) 
               delete [] blkElemStiffness_[i][j];
         delete [] blkElemStiffness_[i];
      }
      delete [] blkElemStiffness_;
   }
   if ( blkNumElems_   != NULL ) delete [] blkNumElems_;
   if ( blkElemNEqns_  != NULL ) delete [] blkElemNEqns_;
   if ( blkNodeDofs_   != NULL ) delete [] blkNodeDofs_;
}

//*************************************************************************
// set diagnostics output level
//-------------------------------------------------------------------------

int MLI_SFEI::setOutputLevel(int level)
{
   if ( level < 0 ) 
   {
      printf("MLI_SFEI::setOutputLevel ERROR - level should be >= 0.\n");
      return 0;
   }
   outputLevel_ = level;
   return 1;
}

//*************************************************************************
// free up stiffness matrices because it is not going to be used any more
//-------------------------------------------------------------------------

int MLI_SFEI::freeStiffnessMatrices()
{
   int i, j;
   if ( blkElemStiffness_ != NULL )
   {
      for ( i = 0; i < nElemBlocks_; i++ ) 
      {
         for ( j = 0; j < blkNumElems_[i]; j++ ) 
            if ( blkElemStiffness_[i][j] != NULL ) 
               delete [] blkElemStiffness_[i][j];
         delete [] blkElemStiffness_[i];
      }
      delete [] blkElemStiffness_;
   }
   blkElemStiffness_ = NULL;
   blkIDBase_        = -1;
   return 0;
}

//*************************************************************************
// accumulate number of element information
//-------------------------------------------------------------------------

int MLI_SFEI::addNumElems(int elemBlk, int nElems, int nNodesPerElem)
{
   int iB, *tempBlkNumElems, *tempBlkElemNEqns, *tempBlkNodeDofs;

   if ( elemBlk != nElemBlocks_ && elemBlk != (nElemBlocks_-1) )
   {
      printf("MLI_SFEI::addNumElems ERROR : elemBlk %d(%d) invalid\n",
             elemBlk,nElemBlocks_);
      return -1;
   }
   if ( blkNumElems_ == NULL )
   {
      maxElemBlocks_ = 20;
      nElemBlocks_   = 0;
      blkNumElems_   = new int[maxElemBlocks_];
      blkElemNEqns_  = new int[maxElemBlocks_];
      blkNodeDofs_   = new int[maxElemBlocks_];
      for ( iB = 0; iB < maxElemBlocks_; iB++ ) 
      {
         blkNumElems_[iB]  = 0;
         blkElemNEqns_[iB] = 0;
         blkNodeDofs_[iB]  = 0;
      }
   }
   if ( elemBlk >= nElemBlocks_ )
   {
      if ( nElemBlocks_ >= maxElemBlocks_ )
      {
         tempBlkNumElems  = blkNumElems_;
         tempBlkElemNEqns = blkElemNEqns_;
         tempBlkNodeDofs  = blkNodeDofs_;
         maxElemBlocks_ += 10;
         blkNumElems_   = new int[maxElemBlocks_];
         blkElemNEqns_  = new int[maxElemBlocks_];
         blkNodeDofs_   = new int[maxElemBlocks_];
         for ( iB = 0; iB < nElemBlocks_; iB++ )
         {
            blkNumElems_[iB]  = tempBlkNumElems[iB];
            blkElemNEqns_[iB] = tempBlkElemNEqns[iB];
            blkNodeDofs_[iB]  = tempBlkNodeDofs[iB];
         }
      }
      blkNumElems_[elemBlk] = nElems;
      blkElemNEqns_[elemBlk] = nNodesPerElem;
   }
   else if ( elemBlk >= 0 ) blkNumElems_[elemBlk] += nElems;
   if ( elemBlk == nElemBlocks_ ) nElemBlocks_++;
   return 0;
}

//*************************************************************************
// initialize the element connectivities
//-------------------------------------------------------------------------

int MLI_SFEI::loadElemBlock(int blkID, int nElems, const int* elemIDs,
                     const double *const *const *stiff,
                     int nEqnsPerElem, const int *const * eqnIndices)
{
   (void) elemIDs;
   int    iB, iE, iN, iN2, count, currElem, matSize, *nodeList, elemBlk;
   double *stiffMat;

   if (blkIDBase_ == -1) blkIDBase_ = blkID;
   elemBlk = blkID - blkIDBase_;
   if (nElemBlocks_ <= 0) return 0;
   if (elemBlk < 0 || elemBlk >= nElemBlocks_)
   {
      printf("MLI_SFEI::loadElemBlock ERROR : elemBlk %d invalid\n",elemBlk);
      return -1;
   }
   if (blkElemEqnLists_ == NULL)
   {
      for (iB = 0; iB < nElemBlocks_; iB++)
      {
         if (blkNumElems_[iB] <= 0)
         {
            printf("MLI_SFEI::addNumElems ERROR : some elemBlk has 0 elems\n");
            return -1;
         }
      }
      blkElemEqnLists_  = new int**[nElemBlocks_];
      blkElemStiffness_ = new double**[nElemBlocks_];
      for (iB = 0; iB < nElemBlocks_; iB++)
      {
         blkElemEqnLists_[iB]  = new int*[blkNumElems_[iB]];
         blkElemStiffness_[iB] = new double*[blkNumElems_[iB]];
         for (iE = 0; iE < blkNumElems_[iB]; iE++)
         {
            blkElemEqnLists_[iB][iE]  = NULL;
            blkElemStiffness_[iB][iE] = NULL;
         }
         blkNumElems_[iB] = 0;
      }
   }
   if (nEqnsPerElem != blkElemNEqns_[elemBlk] && 
        blkElemNEqns_[elemBlk] != 0)
      blkNodeDofs_[elemBlk] = nEqnsPerElem / blkElemNEqns_[elemBlk];

   blkElemNEqns_[elemBlk] = nEqnsPerElem;
   currElem = blkNumElems_[elemBlk];
   matSize = nEqnsPerElem * nEqnsPerElem;
   
   for (iE = 0; iE < nElems; iE++)
   {
      blkElemEqnLists_[elemBlk][currElem] = new int[nEqnsPerElem];
      nodeList = blkElemEqnLists_[elemBlk][currElem];
      for (iN = 0; iN < nEqnsPerElem; iN++)
         nodeList[iN] = eqnIndices[iE][iN];
      blkElemStiffness_[elemBlk][currElem] = new double[matSize];
      stiffMat = blkElemStiffness_[elemBlk][currElem];
      count = 0;
      for (iN = 0; iN < nEqnsPerElem; iN++)
         for (iN2 = 0; iN2 < nEqnsPerElem; iN2++)
            stiffMat[count++] = stiff[iE][iN2][iN];
      currElem++;
   }
   blkNumElems_[elemBlk] = currElem;
   
   return 0;
}

//*************************************************************************
// get block number of elements 
//-------------------------------------------------------------------------

int MLI_SFEI::getBlockNumElems(int blkID)
{
   if (blkID < 0 || blkID >= nElemBlocks_)
   {
      printf("MLI_SFEI::getBlockNumElems ERROR - invalid blkID.\n");
      return -1;
   }
   return blkNumElems_[blkID];
}

//*************************************************************************
// get block number of nodes per element
//-------------------------------------------------------------------------

int MLI_SFEI::getBlockElemNEqns(int blkID)
{
   if (blkID < 0 || blkID >= nElemBlocks_) 
   {
      printf("MLI_SFEI::getBlockElemNEqns ERROR - invalid blkID.\n");
      return -1;
   }
   return blkElemNEqns_[blkID];
}

//*************************************************************************
// get element block nodelists 
//-------------------------------------------------------------------------

int **MLI_SFEI::getBlockElemEqnLists(int blkID)
{
   if (blkID < 0 || blkID >= nElemBlocks_)
   {
      printf("MLI_SFEI::getBlockElemEqnLists ERROR - invalid blkID.\n");
      return NULL;
   }
   return blkElemEqnLists_[blkID];
}

//*************************************************************************
// get block element stiffness matrices 
//-------------------------------------------------------------------------

double **MLI_SFEI::getBlockElemStiffness(int blkID)
{
   if (blkID < 0 || blkID >= nElemBlocks_)
   {
      printf("MLI_SFEI::getBlockElemStiffness ERROR - invalid blkID.\n");
      return NULL;
   }
   return blkElemStiffness_[blkID];
}

