/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_BSGS_H__
#define __MLI_SOLVER_BSGS_H__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "solver/mli_solver_seqsuperlu.h"

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

