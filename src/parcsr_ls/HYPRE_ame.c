/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.3 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMECreate
 *--------------------------------------------------------------------------*/

int HYPRE_AMECreate(HYPRE_Solver *esolver)
{
   *esolver = (HYPRE_Solver) hypre_AMECreate();
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_AMEDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_AMEDestroy(HYPRE_Solver esolver)
{
   return hypre_AMEDestroy((void *) esolver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetup
 *--------------------------------------------------------------------------*/

int HYPRE_AMESetup (HYPRE_Solver esolver)
{
   return hypre_AMESetup((void *) esolver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESolve
 *--------------------------------------------------------------------------*/

int HYPRE_AMESolve (HYPRE_Solver esolver)
{
   return hypre_AMESolve((void *) esolver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetAMSSolver
 *--------------------------------------------------------------------------*/

int HYPRE_AMESetAMSSolver(HYPRE_Solver esolver,
                          HYPRE_Solver ams_solver)
{
   return hypre_AMESetAMSSolver((void *) esolver,
                                (void *) ams_solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetMassMatrix
 *--------------------------------------------------------------------------*/

int HYPRE_AMESetMassMatrix(HYPRE_Solver esolver,
                           HYPRE_ParCSRMatrix M)
{
   return hypre_AMESetMassMatrix((void *) esolver,
                                 (hypre_ParCSRMatrix *) M);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetBlockSize
 *--------------------------------------------------------------------------*/

int HYPRE_AMESetBlockSize(HYPRE_Solver esolver,
                          int block_size)
{
   return hypre_AMESetBlockSize((void *) esolver, block_size);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetMaxIter
 *--------------------------------------------------------------------------*/

int HYPRE_AMESetMaxIter(HYPRE_Solver esolver,
                        int maxit)
{
   return hypre_AMESetMaxIter((void *) esolver, maxit);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetTol
 *--------------------------------------------------------------------------*/

int HYPRE_AMESetTol(HYPRE_Solver esolver,
                    double tol)
{
   return hypre_AMESetTol((void *) esolver, tol);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetPrintLevel
 *--------------------------------------------------------------------------*/

int HYPRE_AMESetPrintLevel(HYPRE_Solver esolver,
                           int print_level)
{
   return hypre_AMESetPrintLevel((void *) esolver, print_level);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMEGetEigenvalues
 *--------------------------------------------------------------------------*/

int HYPRE_AMEGetEigenvalues(HYPRE_Solver esolver,
                            double **eigenvalues)
{
   return hypre_AMEGetEigenvalues((void *) esolver, eigenvalues);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMEGetEigenvectors
 *--------------------------------------------------------------------------*/

int HYPRE_AMEGetEigenvectors(HYPRE_Solver esolver,
                             HYPRE_ParVector **eigenvectors)
{
   return hypre_AMEGetEigenvectors((void *) esolver,
                                   eigenvectors);
}
