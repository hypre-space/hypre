/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_SCHWARZH__
#define __MLI_SOLVER_SCHWARZH__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Schwarz relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_Schwarz : public MLI_Solver
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

public :

   MLI_Solver_Schwarz();
   ~MLI_Solver_Schwarz();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams( char *param_string, int argc, char **argv);

   int setNBlocks(int nblocks);
   int composedOverlappedMatrix(int *off_nrows, int **off_rows, 
                 int **off_row_lengths, int **off_cols, double **off_vals);
   int buildBlocks(int off_nrows, int *off_rows, int *off_row_lengths, 
                   int *off_cols, double *off_vals);
};

#endif

