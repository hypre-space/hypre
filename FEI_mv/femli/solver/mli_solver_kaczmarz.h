/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifndef __MLI_SOLVER_KACZMARZ_H__
#define __MLI_SOLVER_KACZMARZ_H__

#include <stdio.h>
#include "parcsr_mv/parcsr_mv.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Kaczmarz relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_Kaczmarz : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   double      *AsqDiag_;
   int         zeroInitialGuess_;

public :

   MLI_Solver_Kaczmarz(char *name);
   ~MLI_Solver_Kaczmarz();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams(char *paramString, int argc, char **argv);
};

#endif

