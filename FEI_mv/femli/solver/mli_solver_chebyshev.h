/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_CHEBYSHEV_H__
#define __MLI_SOLVER_CHEBYSHEV_H__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Chebyshev smoother
 *---------------------------------------------------------------------------*/

class MLI_Solver_Chebyshev : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   MLI_Vector  *rVec_;
   MLI_Vector  *zVec_;
   MLI_Vector  *pVec_;
   double      *diagonal_;
   int         degree_;
   int         zeroInitialGuess_;
   double      maxEigen_;
   double      minEigen_;

public :

   MLI_Solver_Chebyshev(char *name);
   ~MLI_Solver_Chebyshev();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams( char *paramString, int argc, char **argv );
};

#endif

