/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __MLI_SOLVERPARASAILS_H__
#define __MLI_SOLVERPARASAILS_H__

#define MLI_PARASAILS

#include <stdio.h>
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef MLI_PARASAILS
#include "ParaSails/Matrix.h"
#include "ParaSails/ParaSails.h"
#endif
#ifdef __cplusplus
}
#endif

/******************************************************************************
 * data structure for the ParaSails relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_ParaSails : public MLI_Solver
{
   MLI_Matrix *Amat_;
#ifdef MLI_PARASAILS
   ParaSails  *ps_;
#endif
   int        nlevels_;
   int        symmetric_;
   double     threshold_;
   double     filter_;
   int        loadbal_;
   int        transpose_;
   double     correction_;
   int        zeroInitialGuess_;
   int        numFpts_;
   int        *fpList_;
   int        ownAmat_;
   MLI_Vector *auxVec2_;
   MLI_Vector *auxVec3_;

public :

   MLI_Solver_ParaSails(char *name);
   ~MLI_Solver_ParaSails();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *paramString, int argc, char **argv);

   int applyParaSails(MLI_Vector *f, MLI_Vector *u);
   int applyParaSailsTrans(MLI_Vector *f, MLI_Vector *u);

   int setNumLevels( int nlevels );
   int setSymmetric();
   int setUnSymmetric();
   int setThreshold( double thresh );
   int setFilter( double filter );
   int setLoadBal();
   int setTranspose();
   int setUnderCorrection(double);
};

#endif

