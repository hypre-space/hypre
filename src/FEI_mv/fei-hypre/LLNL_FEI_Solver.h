/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***************************************************************************
  Module:  LLNL_FEI_Solver.h
  Purpose: custom implementation of the FEI/Solver 
 ***************************************************************************/

#ifndef _LLNL_FEI_SOLVER_H_
#define _LLNL_FEI_SOLVER_H_

#include "LLNL_FEI_Matrix.h"

/**************************************************************************
 definition of the class to capture the FEI/Solver information 
---------------------------------------------------------------------------*/

class LLNL_FEI_Solver
{
   MPI_Comm mpiComm_;
   int      mypid_;
   int      outputLevel_;
   LLNL_FEI_Matrix *matPtr_;

   int    solverID_;
   int    krylovMaxIterations_;
   double krylovTolerance_;
   int    krylovAbsRel_;
   int    krylovIterations_;
   double krylovResidualNorm_;
   int    gmresDim_;

   double *solnVector_;
   double *rhsVector_;

   double TimerSolve_;
   double TimerSolveStart_;

public :

   LLNL_FEI_Solver(MPI_Comm comm);
   ~LLNL_FEI_Solver() {}
   int parameters(int numParams, char **paramString);

   int solve(int *status);

   int iterations(int *iterTaken) {*iterTaken=krylovIterations_; return 0;}

   int getResidualNorm(double *rnorm) {*rnorm=krylovResidualNorm_; return 0;}

   int getSolveTime(double *stime) {*stime=TimerSolve_; return 0;}

   int loadMatrix(LLNL_FEI_Matrix *mat) {matPtr_ = mat; return 0;}

   int loadSolnVector(double *soln) {solnVector_ = soln ; return 0;}
   int loadRHSVector(double *rhs) {rhsVector_ = rhs; return 0;}

private:
   int  solveUsingCG();
   int  solveUsingGMRES();
   int  solveUsingCGS();
   int  solveUsingBicgstab();
   int  solveUsingSuperLU();
};

#endif /* endif for _LLNL_FEI_SOLVER_H_ */

