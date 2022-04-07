/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "mli_solver_amg.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_AMG::MLI_Solver_AMG(char *name) : MLI_Solver(name)
{
   Amat_ = NULL;
   precond_ = NULL;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_AMG::~MLI_Solver_AMG()
{
   Amat_ = NULL;
   if (precond_ != NULL) HYPRE_BoomerAMGDestroy(precond_);
   precond_ = NULL;
}

/******************************************************************************
 * set up the solver
 *---------------------------------------------------------------------------*/

int MLI_Solver_AMG::setup(MLI_Matrix *mat)
{
   int                i, *nSweeps, *rTypes;
   double             *relaxWt, *relaxOmega;
   hypre_ParCSRMatrix *hypreA;

   Amat_  = mat;
   hypreA = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   HYPRE_BoomerAMGCreate(&precond_);
   HYPRE_BoomerAMGSetMaxIter(precond_, 1);
   HYPRE_BoomerAMGSetCycleType(precond_, 1);
   HYPRE_BoomerAMGSetMaxLevels(precond_, 25);
   HYPRE_BoomerAMGSetMeasureType(precond_, 0);
   HYPRE_BoomerAMGSetDebugFlag(precond_, 0);
   HYPRE_BoomerAMGSetPrintLevel(precond_, 1);
   HYPRE_BoomerAMGSetCoarsenType(precond_, 0);
   HYPRE_BoomerAMGSetStrongThreshold(precond_, 0.8);
   nSweeps = hypre_TAlloc(int, 4 , HYPRE_MEMORY_HOST);
   for (i = 0; i < 4; i++) nSweeps[i] = 1;
   HYPRE_BoomerAMGSetNumGridSweeps(precond_, nSweeps);
   rTypes = hypre_TAlloc(int, 4 , HYPRE_MEMORY_HOST);
   for (i = 0; i < 4; i++) rTypes[i] = 6;
   relaxWt = hypre_TAlloc(double, 25 , HYPRE_MEMORY_HOST);
   for (i = 0; i < 25; i++) relaxWt[i] = 1.0;
   HYPRE_BoomerAMGSetRelaxWeight(precond_, relaxWt);
   relaxOmega = hypre_TAlloc(double, 25 , HYPRE_MEMORY_HOST);
   for (i = 0; i < 25; i++) relaxOmega[i] = 1.0;
   HYPRE_BoomerAMGSetOmega(precond_, relaxOmega);
   HYPRE_BoomerAMGSetup(precond_, (HYPRE_ParCSRMatrix) hypreA, 
         (HYPRE_ParVector) NULL, (HYPRE_ParVector) NULL);
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_AMG::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   if (precond_ == NULL || Amat_ == NULL)
   {
      printf("MLI_Solver_AMG::solve ERROR - setup not called\n");
      exit(1);
   }
   HYPRE_ParCSRMatrix hypreA = (HYPRE_ParCSRMatrix) Amat_->getMatrix();
   HYPRE_ParVector f = (HYPRE_ParVector) fIn->getVector();
   HYPRE_ParVector u = (HYPRE_ParVector) uIn->getVector();
   HYPRE_BoomerAMGSolve(precond_, hypreA, f, u);
   return 0;
}

