/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_AMG_H__
#define __MLI_SOLVER_AMG_H__

#include <stdio.h>
#include "HYPRE_config.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for BoomerAMG smoother
 *---------------------------------------------------------------------------*/

class MLI_Solver_AMG : public MLI_Solver
{
   MLI_Matrix   *Amat_;
   HYPRE_Solver precond_;

public :

   MLI_Solver_AMG(char *name);
   ~MLI_Solver_AMG();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
};

#endif

