/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.8 $
 ***********************************************************************EHEADER*/





#ifndef __MLI_SOLVER_CG_H__
#define __MLI_SOLVER_CG_H__

#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the CG scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_CG : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         maxIterations_;
   double      tolerance_;
   int         zeroInitialGuess_;
   MLI_Vector  *rVec_;
   MLI_Vector  *zVec_;
   MLI_Vector  *pVec_;
   MLI_Vector  *apVec_;
   MLI_Solver  *baseSolver_;
   int         baseMethod_;
   MLI_Matrix  *PSmat_;
   MLI_Vector  *PSvec_;
   int	       nRecvs_;
   int	       *recvProcs_;
   int	       *recvLengs_;
   int	       nSends_;
   int	       *sendProcs_;
   int	       *sendLengs_;
   MPI_Comm    AComm_;
   int         *iluI_;
   int         *iluJ_;
   int         *iluD_;
   double      *iluA_;

public :

   MLI_Solver_CG(char *name);
   ~MLI_Solver_CG();

   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *paramString, int argc, char **argv );
   int iluDecomposition();
   int iluSolve(double *, double *);
};

#endif

