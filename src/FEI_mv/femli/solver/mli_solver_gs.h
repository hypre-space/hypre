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





#ifndef __MLI_SOLVER_GS_H__
#define __MLI_SOLVER_GS_H__

#include <stdio.h>
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_GS : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   double      *relaxWeights_;
   int         zeroInitialGuess_;

public :

   MLI_Solver_GS(char *name);
   ~MLI_Solver_GS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams(char *paramString, int argc, char **argv);
   int setParams(int ntimes, double *relax_weights);
};

#endif

