/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_JACOBI__
#define __MLI_SOLVER_JACOBI__

#include <stdio.h>
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "../solver/mli_solver.h"

/******************************************************************************
 * data structure for the Damped Jacobi relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_Jacobi : public MLI_Solver
{
   MLI_Matrix  *Amat;
   int         nsweeps;
   double      *relax_weights;

public :

   MLI_Solver_Jacobi();
   ~MLI_Solver_Jacobi();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *param_string, int argc, char **argv );
   int setParams( int ntimes, double *relax_weights );
};

#endif

