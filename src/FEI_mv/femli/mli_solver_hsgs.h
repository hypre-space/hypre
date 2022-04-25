/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_HSGS_H__
#define __MLI_SOLVER_HSGS_H__

#include <stdio.h>
#include "_hypre_parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the symmetric Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_HSGS : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   // The private field below appears to be unused
   // int         printRNorm_;

   int         calcOmega_;
   double      relaxWeights_;
   double      relaxOmega_;
   MLI_Vector  *mliVec_;

public :

   MLI_Solver_HSGS(char *name);
   ~MLI_Solver_HSGS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *param, int argc, char **argv);
   int calcOmega();
};

#endif

