/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVERSCHWARZH__
#define __MLI_SOLVERSCHWARZH__

#include <stdio.h>
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the Schwarz relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_SolverSchwarz : public MLI_Solver
{
   MLI_Matrix *Amat;
   int        nblocks;
   int        *block_lengths;
   int        **block_indices;
   double     **block_inverses;

public :

   MLI_SolverSchwarz();
   ~MLI_SolverSchwarz();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setNBlocks(int nblocks);
   int composedOverlappedMatrix(void *A_in, int *off_nrows, 
                 int **off_row_lengths, int **off_cols, double **off_vals);
};

#endif

