/*BHEADER**********************************************************************
 * (c) 2005   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_AMG_H__
#define __MLI_SOLVER_AMG_H__

#include <stdio.h>
#include "HYPRE_config.h"
#include "utilities/utilities.h"
#include "parcsr_ls/parcsr_ls.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

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

