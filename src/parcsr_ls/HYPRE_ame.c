/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMECreate
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMECreate(HYPRE_Solver *esolver)
{
   *esolver = (HYPRE_Solver) hypre_AMECreate();
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_AMEDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMEDestroy(HYPRE_Solver esolver)
{
   return hypre_AMEDestroy((void *) esolver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetup
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetup (HYPRE_Solver esolver)
{
   return hypre_AMESetup((void *) esolver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESolve
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESolve (HYPRE_Solver esolver)
{
   return hypre_AMESolve((void *) esolver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetAMSSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetAMSSolver(HYPRE_Solver esolver,
                                HYPRE_Solver ams_solver)
{
   return hypre_AMESetAMSSolver((void *) esolver,
                                (void *) ams_solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetMassMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetMassMatrix(HYPRE_Solver esolver,
                                 HYPRE_ParCSRMatrix M)
{
   return hypre_AMESetMassMatrix((void *) esolver,
                                 (hypre_ParCSRMatrix *) M);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetBlockSize
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetBlockSize(HYPRE_Solver esolver,
                                HYPRE_Int block_size)
{
   return hypre_AMESetBlockSize((void *) esolver, block_size);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetMaxIter(HYPRE_Solver esolver,
                              HYPRE_Int maxit)
{
   return hypre_AMESetMaxIter((void *) esolver, maxit);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetMaxPCGIter
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetMaxPCGIter(HYPRE_Solver esolver,
                                 HYPRE_Int maxit)
{
   return hypre_AMESetMaxPCGIter((void *) esolver, maxit);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetTol(HYPRE_Solver esolver,
                          HYPRE_Real tol)
{
   return hypre_AMESetTol((void *) esolver, tol);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetRTol
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetRTol(HYPRE_Solver esolver,
                           HYPRE_Real tol)
{
   return hypre_AMESetRTol((void *) esolver, tol);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMESetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetPrintLevel(HYPRE_Solver esolver,
                                 HYPRE_Int print_level)
{
   return hypre_AMESetPrintLevel((void *) esolver, print_level);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMEGetEigenvalues
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMEGetEigenvalues(HYPRE_Solver esolver,
                                  HYPRE_Real **eigenvalues)
{
   return hypre_AMEGetEigenvalues((void *) esolver, eigenvalues);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMEGetEigenvectors
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMEGetEigenvectors(HYPRE_Solver esolver,
                                   HYPRE_ParVector **eigenvectors)
{
   return hypre_AMEGetEigenvectors((void *) esolver,
                                   eigenvectors);
}
