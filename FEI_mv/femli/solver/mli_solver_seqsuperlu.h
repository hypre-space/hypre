/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_SEQSUPERLU_H__
#define __MLI_SOLVER_SEQSUPERLU_H__

#include <stdio.h>
#include "dsp_defs.h"
#include "util.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the sequential SuperLU solution scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_SeqSuperLU : public MLI_Solver
{
   MLI_Matrix   *mliAmat_;
   int          factorized_;
   int          *permR_;
   int          *permC_;
   int          localNRows_;
   SuperMatrix  superLU_Amat;
   SuperMatrix  superLU_Lmat;
   SuperMatrix  superLU_Umat;

public :

   MLI_Solver_SeqSuperLU(char *name);
   ~MLI_Solver_SeqSuperLU();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *paramString, int argc, char **argv) {return -1;}
};

#endif

#else
   int bogus;
#endif

