/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





#ifndef __MLISFEI_H__
#define __MLISFEI_H__

#include "utilities/_hypre_utilities.h"
#include "fedata/mli_febase.h"

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

