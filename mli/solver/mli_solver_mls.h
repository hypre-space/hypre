/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_MLS__
#define __MLI_SOLVER_MLS__

#include <stdio.h>
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "../solver/mli_solver.h"

/******************************************************************************
 * data structure for the MLS smoother by Brezina S = (I - omega A)^2 A
 * where omega = 2 / max_eigen of A)
 *---------------------------------------------------------------------------*/

class MLI_Solver_MLS : public MLI_Solver
{
   MLI_Matrix  *Amat;
   MLI_Vector  *mli_Vtemp;
   MLI_Vector  *mli_Wtemp;
   MLI_Vector  *mli_Ytemp;
   double      max_eigen;
   int         mlsDeg; /* degree for current level */
   double      mlsBoost;
   double      mlsOver;
   double      mlsOm[5];
   double      mlsOm2;
   double      mlsCf[5];

public :

   MLI_Solver_MLS();
   ~MLI_Solver_MLS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *param_string, int argc, char **argv );
   int setParams( double max_eigen );
};

#endif

