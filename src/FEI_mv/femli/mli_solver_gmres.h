/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_GMRES_H__
#define __MLI_SOLVER_GMRES_H__

#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the GMRES scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_GMRES : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         maxIterations_;
   double      tolerance_;
   int         KDim_;
   MLI_Vector  *rVec_;
   MLI_Vector  **pVec_;
   MLI_Vector  **zVec_;
   MLI_Solver  *baseSolver_;
   int         baseMethod_;

public :

   MLI_Solver_GMRES(char *name);
   ~MLI_Solver_GMRES();

   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *paramString, int argc, char **argv );
};

#endif

