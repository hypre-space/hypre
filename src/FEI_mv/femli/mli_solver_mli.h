/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVER_MLI_H__
#define __MLI_SOLVER_MLI_H__

#include <stdio.h>
#include "HYPRE_config.h"
#include "_hypre_utilities.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"
#include "mli.h"

/******************************************************************************
 * data structure for the MLI smoothed aggregation smoother
 *---------------------------------------------------------------------------*/

class MLI_Solver_MLI : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   MLI         *mli_;

public :

   MLI_Solver_MLI(char *name);
   ~MLI_Solver_MLI();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
};

#endif

