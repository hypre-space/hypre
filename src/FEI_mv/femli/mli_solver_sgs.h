/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_SGS_H__
#define __MLI_SOLVER_SGS_H__

#include <stdio.h>
#include "_hypre_parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

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
   int         printRNorm_;
   int         findOmega_;
   int         omegaNumIncr_;
   double      omegaIncrement_;

public :

   MLI_Solver_SGS(char *name);
   ~MLI_Solver_SGS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams(char *param, int argc, char **argv);
   int setParams(int nTimes, double *relaxWeights);

   int doProcColoring();
   int findOmega();
};

#endif

