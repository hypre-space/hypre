/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLISFEI_H__
#define __MLISFEI_H__

#include "_hypre_utilities.h"
#include "mli_febase.h"

/****************************************************************************/ 
/* data structures for Finite element grid information                      */
/*--------------------------------------------------------------------------*/

class MLI_SFEI : public MLI_FEBase
{
   MPI_Comm mpiComm_;
   int      outputLevel_;
   int      nElemBlocks_;
   int      maxElemBlocks_;
   int      *blkNumElems_;
   int      *blkElemNEqns_;
   int      *blkNodeDofs_;
   int      ***blkElemEqnLists_;
   double   ***blkElemStiffness_;
   int      blkIDBase_;

public :

   MLI_SFEI(MPI_Comm comm);

   ~MLI_SFEI();

   int setOutputLevel(int level);

   int freeStiffnessMatrices();

   int addNumElems(int elemBlk, int nElems, int nNodesPerElem);

   int loadElemBlock(int elemBlk, int nElems, const int* elemIDs,
                     const double *const *const *stiff,
                     int nEqnsPerElem, const int *const * eqnIndices);

   int    getNumElemBlocks() {return nElemBlocks_;}
   int    getBlockNumElems(int iD);
   int    getBlockElemNEqns(int iD);
   int    **getBlockElemEqnLists(int iD);
   double **getBlockElemStiffness(int iD);
};

#endif

