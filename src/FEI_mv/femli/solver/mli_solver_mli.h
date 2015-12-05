/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.6 $
 ***********************************************************************EHEADER*/





#ifndef __MLI_SOLVER_MLI_H__
#define __MLI_SOLVER_MLI_H__

#include <stdio.h>
#include "HYPRE_config.h"
#include "utilities/_hypre_utilities.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "base/mli.h"

/******************************************************************************
 * data structure for the MLI smoothed aggregation smoother
 *---------------------------------------------------------------------------*/

class MLI_Solver_MLI : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   MLI         *mli_;

public :

   MLI_Solver_MLI(char *name);
   ~MLI_Solver_MLI();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
};

#endif

