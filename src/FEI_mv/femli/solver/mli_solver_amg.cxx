/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/




#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "solver/mli_solver_amg.h"

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
   nSweeps = (int *) malloc(4 * sizeof(int));
   for (i = 0; i < 4; i++) nSweeps[i] = 1;
   HYPRE_BoomerAMGSetNumGridSweeps(precond_, nSweeps);
   rTypes = (int *) malloc(4 * sizeof(int));
   for (i = 0; i < 4; i++) rTypes[i] = 6;
   relaxWt = (double *) malloc(25 * sizeof(double));
   for (i = 0; i < 25; i++) relaxWt[i] = 1.0;
   HYPRE_BoomerAMGSetRelaxWeight(precond_, relaxWt);
   relaxOmega = (double *) malloc(25 * sizeof(double));
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

