/*BHEADER**********************************************************************
 * (c) 2003   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_HSGS_H__
#define __MLI_SOLVER_HSGS_H__

#include <stdio.h>
#include "parcsr_mv/parcsr_mv.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the symmetric Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_HSGS : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   int         printRNorm_;
   int         calcOmega_;
   double      *relaxWeights_;
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

