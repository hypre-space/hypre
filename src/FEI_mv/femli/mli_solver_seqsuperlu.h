/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef MLI_SUPERLU

#ifndef __MLI_SOLVER_SEQSUPERLU_H__
#define __MLI_SOLVER_SEQSUPERLU_H__

#include <stdio.h>
#include "slu_ddefs.h"
#include "slu_util.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_solver.h"

/******************************************************************************
 * data structure for the sequential SuperLU solution scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_SeqSuperLU : public MLI_Solver
{
   MLI_Matrix   *mliAmat_;
   int          factorized_;
   int          **permRs_;
   int          **permCs_;
   int          localNRows_;
   SuperMatrix  superLU_Lmats[100];
   SuperMatrix  superLU_Umats[100];
   int          nSubProblems_;
   int          **subProblemRowIndices_;
   int          *subProblemRowSizes_;
   int          numColors_;
   int          *myColors_;
   int          nRecvs_;
   int          *recvProcs_;
   int          *recvLengs_;
   int          nSends_;
   int          *sendProcs_;
   int          *sendLengs_;
   MPI_Comm     AComm_;
   MLI_Matrix   *PSmat_;
   MLI_Vector   *PSvec_;

public :

   MLI_Solver_SeqSuperLU(char *name);
   ~MLI_Solver_SeqSuperLU();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);
   int setParams(char *paramString, int argc, char **argv);
   int setupBlockColoring();
};

#endif

#else
   int bogus;
#endif

