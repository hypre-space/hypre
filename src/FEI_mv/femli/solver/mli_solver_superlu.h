/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.10 $
 ***********************************************************************EHEADER*/

#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_SUPERLU_H__
#define __MLI_SOLVER_SUPERLU_H__

#include <stdio.h>
#include "SRC/slu_ddefs.h"
#include "SRC/slu_util.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the SuperLU solution scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_SuperLU : public MLI_Solver
{
   MLI_Matrix   *mliAmat_;
   int          factorized_;
   int          *permR_;
   int          *permC_;
   SuperMatrix  superLU_Amat;
   SuperMatrix  superLU_Lmat;
   SuperMatrix  superLU_Umat;

public :

   MLI_Solver_SuperLU(char *name);
   ~MLI_Solver_SuperLU();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *paramString, int argc, char **argv) {return -1;}
};

#endif

#else
   int bogus;
#endif

