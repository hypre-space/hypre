/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_GS_H__
#define __MLI_SOLVER_GS_H__

#include <stdio.h>
#include "_hypre_parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_GS : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   double      *relaxWeights_;
   int         zeroInitialGuess_;

public :

   MLI_Solver_GS(char *name);
   ~MLI_Solver_GS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams(char *paramString, int argc, char **argv);
   int setParams(int ntimes, double *relax_weights);
};

#endif

