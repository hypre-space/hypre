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





#ifndef __MLI_SOLVER_HSCHWARZ_H__
#define __MLI_SOLVER_HSCHWARZ_H__

#include <stdio.h>
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Schwarz relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_HSchwarz : public MLI_Solver
{
   MLI_Matrix   *Amat_;
   int          nSweeps_;
   int          printRNorm_;
   int          blkSize_;
   double       relaxWeight_;
   MLI_Vector   *mliVec_;
   HYPRE_Solver smoother_;

public :

   MLI_Solver_HSchwarz(char *name);
   ~MLI_Solver_HSchwarz();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *param, int argc, char **argv);
   int calcOmega();
};

#endif

