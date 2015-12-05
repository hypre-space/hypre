/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_HSGS_H__
#define __MLI_SOLVER_HSGS_H__

#include <stdio.h>
#include "_hypre_parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the symmetric Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_HSGS : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   // The private field below appears to be unused
   // int         printRNorm_;

   int         calcOmega_;
   double      relaxWeights_;
   double      relaxOmega_;
   MLI_Vector  *mliVec_;

public :

   MLI_Solver_HSGS(char *name);
   ~MLI_Solver_HSGS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *param, int argc, char **argv);
   int calcOmega();
};

#endif

