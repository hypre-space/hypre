/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_SUPERLU_H__
#define __MLI_SOLVER_SUPERLU_H__

#include <stdio.h>
#include "slu_ddefs.h"
#include "slu_util.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

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

