/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifdef SUPERLU

#ifndef __MLI_SOLVERSUPERLUH__
#define __MLI_SOLVERSUPERLUH__

#include <stdio.h>
#include "dsp_defs.h"
#include "util.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the Gauss Seidel relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_SolverSuperLU : public MLI_Solver
{
   MLI_Matrix   *mli_Amat;
   int          factorized;
   int          *perm_r;
   int          *perm_c;
   SuperMatrix  superLU_Amat;
   SuperMatrix  superLU_Lmat;
   SuperMatrix  superLU_Umat;

public :

   MLI_SolverSuperLU();
   ~MLI_SolverSuperLU();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *param_string, int argc, char **argv) {return -1;}
};

#endif

#else
   int bogus;
#endif

