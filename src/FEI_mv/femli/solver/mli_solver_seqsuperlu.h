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

#ifndef __MLI_SOLVER_SEQSUPERLU_H__
#define __MLI_SOLVER_SEQSUPERLU_H__

#include <stdio.h>
#include "SRC/slu_ddefs.h"
#include "SRC/slu_util.h"
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

