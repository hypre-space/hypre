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





#ifndef __MLI_SOLVER_BJACOBI_H__
#define __MLI_SOLVER_BJACOBI_H__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "solver/mli_solver_seqsuperlu.h"

/******************************************************************************
 * data structure for the BJacobi relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_BJacobi : public MLI_Solver
{
   MLI_Matrix *Amat_;
   int        nSweeps_;
   double     *relaxWeights_;
   int        useOverlap_;
   int        nBlocks_;
   int        blockSize_;
   int        *blockLengths_;
   int        maxBlkLeng_;
   int        zeroInitialGuess_;
   int        offNRows_;
   int        *offRowIndices_;
   int        *offRowLengths_;
   int        *offCols_;
   double     *offVals_;
   MLI_Solver_SeqSuperLU **blockSolvers_;
   int        blkScheme_;
   double     **esslMatrices_;

public :

   MLI_Solver_BJacobi(char *name);
   ~MLI_Solver_BJacobi();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams( char *paramString, int argc, char **argv);

   int composeOverlappedMatrix();
   int buildBlocks();
   int adjustOffColIndices();
   int cleanBlocks();
};

#endif

