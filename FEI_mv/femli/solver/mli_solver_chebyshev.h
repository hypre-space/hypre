/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_CHEBYSHEV__
#define __MLI_SOLVER_CHEBYSHEV__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Chebyshev smoother
 *---------------------------------------------------------------------------*/

class MLI_Solver_Chebyshev : public MLI_Solver
{
   MLI_Matrix  *Amat;
   MLI_Vector  *mli_Vtemp;
   MLI_Vector  *mli_Wtemp;
   MLI_Vector  *mli_Ytemp;
   double      max_eigen;
   int         degree; /* degree for current level */

public :

   MLI_Solver_ChebyShev();
   ~MLI_Solver_ChebyShev();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *param_string, int argc, char **argv );
   int setParams( double max_eigen );
};

#endif

