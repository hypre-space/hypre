/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





#ifndef __MLI_SOLVERPARASAILS_H__
#define __MLI_SOLVERPARASAILS_H__

#define MLI_PARASAILS

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef MLI_PARASAILS
#include "distributed_ls/ParaSails/Matrix.h"
#include "distributed_ls/ParaSails/ParaSails.h"
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

