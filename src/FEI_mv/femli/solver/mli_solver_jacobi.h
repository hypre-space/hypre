/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.8 $
 ***********************************************************************EHEADER*/





#ifndef __MLI_SOLVER_JACOBI_H__
#define __MLI_SOLVER_JACOBI_H__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Damped Jacobi relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_Jacobi : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   double      *relaxWeights_;
   double      *diagonal_;
   double      maxEigen_;
   MLI_Vector  *auxVec_;
   MLI_Vector  *auxVec2_;
   MLI_Vector  *auxVec3_;
   int         zeroInitialGuess_;
   int         numFpts_;
   int         *FptList_;
   int         ownAmat_;
   int         modifiedD_;

public :

   MLI_Solver_Jacobi(char *name);
   ~MLI_Solver_Jacobi();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *paramString, int argc, char **argv );
   int setParams( int ntimes, double *relax_weights );
   int getParams( char *paramString, int *argc, char **argv );
};

#endif

