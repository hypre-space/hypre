/*BHEADER**********************************************************************
 * (c) 2003   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_GMRES_H__
#define __MLI_SOLVER_GMRES_H__

#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

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

