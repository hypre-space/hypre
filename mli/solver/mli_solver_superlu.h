/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_SUPERLUH__
#define __MLI_SOLVER_SUPERLUH__

#include <stdio.h>
#include "dsp_defs.h"
#include "util.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_SuperLU : public MLI_Solver
{
   MLI_Matrix   *mli_Amat;
   int          factorized;
   int          *perm_r;
   int          *perm_c;
   SuperMatrix  superLU_Amat;
   SuperMatrix  superLU_Lmat;
   SuperMatrix  superLU_Umat;

public :

   MLI_Solver_SuperLU();
   ~MLI_Solver_SuperLU();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *param_string, int argc, char **argv) {return -1;}
};

#endif

#else
   int bogus;
#endif

