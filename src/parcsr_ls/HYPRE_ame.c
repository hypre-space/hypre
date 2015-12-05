/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/





#include "headers.h"

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
 * HYPRE_AMESetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMESetTol(HYPRE_Solver esolver,
                    double tol)
{
   return hypre_AMESetTol((void *) esolver, tol);
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
                            double **eigenvalues)
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
