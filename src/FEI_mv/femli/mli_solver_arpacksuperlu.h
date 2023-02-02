/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_ARPACKSUPERLU_H__
#define __MLI_SOLVER_ARPACKSUPERLU_H__

#include <stdio.h>
#include "mli_vector.h"
#include "mli_solver.h"
#include "mli_matrix.h"

/******************************************************************************
 * data structure for the SuperLU in ARPACK shift-and-invert procedure
 *---------------------------------------------------------------------------*/

class MLI_Solver_ARPACKSuperLU : public MLI_Solver
{
   MLI_Matrix *Amat_;
   int        nRecvs_;
   int        *recvLengs_;
   int        *recvProcs_;
   int        nSends_;
   int        *sendLengs_;
   int        *sendProcs_;
   int        *sendMap_;
   int        nSendMap_;
   int        nNodes_;
   int        *ANodeEqnList_;
   int        *SNodeEqnList_;
   int        blockSize_;

public :

   MLI_Solver_ARPACKSuperLU(char *name);
   ~MLI_Solver_ARPACKSuperLU();
   int setup(MLI_Matrix *mat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *paramString, int argc, char **argv);
};

#endif

#else
   int bogus;
#endif

