/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_SGS_H__
#define __MLI_SOLVER_SGS_H__

#include <stdio.h>
#include "parcsr_mv/parcsr_mv.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the symmetric Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_SGS : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         zeroInitialGuess_;
   int         nSweeps_;
   double      *relaxWeights_;
   int         myColor_;
   int         numColors_;
   int         scheme_;

public :

   MLI_Solver_SGS(char *name);
   ~MLI_Solver_SGS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams(char *param, int argc, char **argv);
   int setParams(int nTimes, double *relaxWeights);

   int doProcColoring();
};

#endif

