/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 1.5 $
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

