/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVERPARASAILSH__
#define __MLI_SOLVERPARASAILSH__

#include <stdio.h>
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "mli_solver.h"
#ifdef __cplusplus
extern "C" {
#endif
#ifdef PARASAILS
#include "distributed_ls/ParaSails/Matrix.h"
#include "distributed_ls/ParaSails/ParaSails.h"
#endif
#ifdef __cplusplus
}
#endif

/******************************************************************************
 * data structure for the ParaSails relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_SolverParaSails : public MLI_Solver
{
   MLI_Matrix *Amat;
#ifdef PARASAILS
   ParaSails  *ps;
#endif
   int        nlevels;
   int        symmetric;
   double     threshold;
   int        num_levels;
   double     filter;
   int        loadbal;
   int        factorized;
   int        transpose;

public :

   MLI_SolverParaSails();
   ~MLI_SolverParaSails();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int applyParaSails(MLI_Vector *f, MLI_Vector *u);
   int applyParaSailsTrans(MLI_Vector *f, MLI_Vector *u);

   int setParams(char *param_string, int argc, char **argv);
   int setNumLevels( int nlevels );
   int setSymmetric();
   int setUnSymmetric();
   int setThreshold( double thresh );
   int setFilter( double filter );
   int setLoadBal();
   int setFactorized();
   int setTranspose();
};

#endif

