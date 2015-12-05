/*BHEADER**********************************************************************
 * (c) 2004   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_MLI_H__
#define __MLI_SOLVER_MLI_H__

#include <stdio.h>
#include "HYPRE_config.h"
#include "utilities/utilities.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "base/mli.h"

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

