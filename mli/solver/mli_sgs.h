/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVERSGSH__
#define __MLI_SOLVERSGSH__

#include <stdio.h>
#include "parcsr_mv/parcsr_mv.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the symmetric Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_SolverSGS : public MLI_Solver
{
   MLI_Matrix  *Amat;
   int         nsweeps;
   double      *relax_weights;

public :

   MLI_SolverSGS();
   ~MLI_SolverSGS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams(char *param, int argc, char **argv);
   int setParams(int ntimes, double *relax_weights);
};

#endif

