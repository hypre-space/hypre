/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGCreate( MPI_Comm             comm,
                        HYPRE_SStructSolver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_SStructKrylovCAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGDestroy( HYPRE_SStructSolver solver )
{
   return ( hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetup( HYPRE_SStructSolver solver,
                       HYPRE_SStructMatrix A,
                       HYPRE_SStructVector b,
                       HYPRE_SStructVector x )
{
   return ( HYPRE_PCGSetup( (HYPRE_Solver) solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSolve( HYPRE_SStructSolver solver,
                       HYPRE_SStructMatrix A,
                       HYPRE_SStructVector b,
                       HYPRE_SStructVector x )
{
   return ( HYPRE_PCGSolve( (HYPRE_Solver) solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetTol( HYPRE_SStructSolver solver,
                        HYPRE_Real          tol )
{
   return ( HYPRE_PCGSetTol( (HYPRE_Solver) solver, tol ) );
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetAbsoluteTol( HYPRE_SStructSolver solver,
                                HYPRE_Real          tol )
{
   return ( HYPRE_PCGSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetMaxIter( HYPRE_SStructSolver solver,
                            HYPRE_Int           max_iter )
{
   return ( HYPRE_PCGSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetTwoNorm( HYPRE_SStructSolver solver,
                            HYPRE_Int           two_norm )
{
   return ( HYPRE_PCGSetTwoNorm( (HYPRE_Solver) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetRelChange( HYPRE_SStructSolver solver,
                              HYPRE_Int           rel_change )
{
   return ( HYPRE_PCGSetRelChange( (HYPRE_Solver) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetPrecond( HYPRE_SStructSolver          solver,
                            HYPRE_PtrToSStructSolverFcn  precond,
                            HYPRE_PtrToSStructSolverFcn  precond_setup,
                            void                        *precond_data )
{
   return ( HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) precond,
                                 (HYPRE_PtrToSolverFcn) precond_setup,
                                 (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetLogging( HYPRE_SStructSolver solver,
                            HYPRE_Int           logging )
{
   return ( HYPRE_PCGSetLogging( (HYPRE_Solver) solver, logging ) );
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGSetPrintLevel( HYPRE_SStructSolver solver,
                               HYPRE_Int           level )
{
   return ( HYPRE_PCGSetPrintLevel( (HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGGetNumIterations( HYPRE_SStructSolver  solver,
                                  HYPRE_Int           *num_iterations )
{
   return ( HYPRE_PCGGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                              HYPRE_Real          *norm )
{
   return ( HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructPCGGetResidual( HYPRE_SStructSolver  solver,
                             void              **residual )
{
   return ( HYPRE_PCGGetResidual( (HYPRE_Solver) solver, residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructDiagScaleSetup( HYPRE_SStructSolver solver,
                             HYPRE_SStructMatrix A,
                             HYPRE_SStructVector y,
                             HYPRE_SStructVector x      )
{

   return ( HYPRE_StructDiagScaleSetup( (HYPRE_StructSolver) solver,
                                        (HYPRE_StructMatrix) A,
                                        (HYPRE_StructVector) y,
                                        (HYPRE_StructVector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructDiagScale( HYPRE_SStructSolver solver,
                        HYPRE_SStructMatrix A,
                        HYPRE_SStructVector y,
                        HYPRE_SStructVector x      )
{
   HYPRE_Int                nparts = hypre_SStructMatrixNParts(A);

   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_StructMatrix      *sA;
   hypre_StructVector      *sx;
   hypre_StructVector      *sy;

   HYPRE_Int part, vi;
   HYPRE_Int nvars;

   for (part = 0; part < nparts; part++)
   {
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      py = hypre_SStructVectorPVector(y, part);
      nvars = hypre_SStructPMatrixNVars(pA);
      for (vi = 0; vi < nvars; vi++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = hypre_SStructPVectorSVector(px, vi);
         sy = hypre_SStructPVectorSVector(py, vi);

         HYPRE_StructDiagScale( (HYPRE_StructSolver) solver,
                                (HYPRE_StructMatrix) sA,
                                (HYPRE_StructVector) sy,
                                (HYPRE_StructVector) sx );
      }
   }

   return hypre_error_flag;
}
