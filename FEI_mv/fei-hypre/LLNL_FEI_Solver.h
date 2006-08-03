/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

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

