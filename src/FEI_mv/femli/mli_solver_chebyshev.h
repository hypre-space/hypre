/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_CHEBYSHEV_H__
#define __MLI_SOLVER_CHEBYSHEV_H__

#include <stdio.h>
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

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

