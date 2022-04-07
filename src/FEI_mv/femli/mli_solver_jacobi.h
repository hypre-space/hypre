/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_JACOBI_H__
#define __MLI_SOLVER_JACOBI_H__

#include <stdio.h>
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the Damped Jacobi relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_Jacobi : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   double      *relaxWeights_;
   double      *diagonal_;
   double      maxEigen_;
   MLI_Vector  *auxVec_;
   MLI_Vector  *auxVec2_;
   MLI_Vector  *auxVec3_;
   int         zeroInitialGuess_;
   int         numFpts_;
   int         *FptList_;
   int         ownAmat_;
   int         modifiedD_;

public :

   MLI_Solver_Jacobi(char *name);
   ~MLI_Solver_Jacobi();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *paramString, int argc, char **argv );
   int setParams( int ntimes, double *relax_weights );
   int getParams( char *paramString, int *argc, char **argv );
};

#endif

