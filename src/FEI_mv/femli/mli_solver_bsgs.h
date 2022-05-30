/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_BSGS_H__
#define __MLI_SOLVER_BSGS_H__

#include <stdio.h>
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"
#include "mli_solver_seqsuperlu.h"

/******************************************************************************
 * data structure for the BSGS relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_BSGS : public MLI_Solver
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
   int        scheme_;
   int        numColors_;
   int        myColor_;
#ifdef HAVE_ESSL
   double     **esslMatrices_;
#endif

public :

   MLI_Solver_BSGS(char *name);
   ~MLI_Solver_BSGS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams( char *paramString, int argc, char **argv);

   int composeOverlappedMatrix();
   int buildBlocks();
   int adjustOffColIndices();
   int cleanBlocks();
   int doProcColoring();
};

#endif

