/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_BSGSH__
#define __MLI_SOLVER_BSGSH__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the BSGS relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_BSGS : public MLI_Solver
{
   MLI_Matrix *Amat_;
   int        nBlocks_;
   int        *blockLengths_;
   int        **blockIndices_;
   double     ***blockInverses_;
   int        zeroInitialGuess_;
   int        nSweeps_;
   double     *relaxWeights_;
   int        useOverlap_;
   int        offNRows_;
   int        *offRowIndices_;
   int        *offRowLengths_;
   int        *offCols_;
   double     *offVals_;

public :

   MLI_Solver_BSGS();
   ~MLI_Solver_BSGS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams( char *param_string, int argc, char **argv);

   int setNBlocks(int nblocks);
   int composeOverlappedMatrix();
   int buildBlocks();
   int adjustOffColIndices();
};

#endif

