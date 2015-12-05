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





#ifndef __MLI_SOLVER_CHEBYSHEV_H__
#define __MLI_SOLVER_CHEBYSHEV_H__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Chebyshev smoother
 *---------------------------------------------------------------------------*/

class MLI_Solver_Chebyshev : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   MLI_Vector  *rVec_;
   MLI_Vector  *zVec_;
   MLI_Vector  *pVec_;
   double      *diagonal_;
   int         degree_;
   int         zeroInitialGuess_;
   double      maxEigen_;
   double      minEigen_;

public :

   MLI_Solver_Chebyshev(char *name);
   ~MLI_Solver_Chebyshev();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams( char *paramString, int argc, char **argv );
};

#endif

