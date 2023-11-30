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
HYPRE_SStructFlexGMRESCreate( MPI_Comm             comm,
                              HYPRE_SStructSolver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_FlexGMRESFunctions * fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_SStructKrylovCAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovCreateVectorArray,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_FlexGMRESCreate( fgmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESDestroy( HYPRE_SStructSolver solver )
{
   return ( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetup( HYPRE_SStructSolver solver,
                             HYPRE_SStructMatrix A,
                             HYPRE_SStructVector b,
                             HYPRE_SStructVector x )
{
   return ( HYPRE_FlexGMRESSetup( (HYPRE_Solver) solver,
                                  (HYPRE_Matrix) A,
                                  (HYPRE_Vector) b,
                                  (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSolve( HYPRE_SStructSolver solver,
                             HYPRE_SStructMatrix A,
                             HYPRE_SStructVector b,
                             HYPRE_SStructVector x )
{
   return ( HYPRE_FlexGMRESSolve( (HYPRE_Solver) solver,
                                  (HYPRE_Matrix) A,
                                  (HYPRE_Vector) b,
                                  (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetKDim( HYPRE_SStructSolver solver,
                               HYPRE_Int           k_dim )
{
   return ( HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetTol( HYPRE_SStructSolver solver,
                              HYPRE_Real          tol )
{
   return ( HYPRE_FlexGMRESSetTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetAbsoluteTol( HYPRE_SStructSolver solver,
                                      HYPRE_Real          tol )
{
   return ( HYPRE_FlexGMRESSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetMinIter( HYPRE_SStructSolver solver,
                                  HYPRE_Int           min_iter )
{
   return ( HYPRE_FlexGMRESSetMinIter( (HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetMaxIter( HYPRE_SStructSolver solver,
                                  HYPRE_Int           max_iter )
{
   return ( HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetPrecond( HYPRE_SStructSolver          solver,
                                  HYPRE_PtrToSStructSolverFcn  precond,
                                  HYPRE_PtrToSStructSolverFcn  precond_setup,
                                  void *          precond_data )
{
   return ( HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) precond,
                                       (HYPRE_PtrToSolverFcn) precond_setup,
                                       (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetLogging( HYPRE_SStructSolver solver,
                                  HYPRE_Int           logging )
{
   return ( HYPRE_FlexGMRESSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetPrintLevel( HYPRE_SStructSolver solver,
                                     HYPRE_Int           level )
{
   return ( HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESGetNumIterations( HYPRE_SStructSolver  solver,
                                        HYPRE_Int           *num_iterations )
{
   return ( HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver) solver,
                                             num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                    HYPRE_Real          *norm )
{
   return ( HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver,
                                                         norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESGetResidual( HYPRE_SStructSolver  solver,
                                   void              **residual )
{
   return ( HYPRE_FlexGMRESGetResidual( (HYPRE_Solver) solver, residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


HYPRE_Int HYPRE_SStructFlexGMRESSetModifyPC( HYPRE_SStructSolver  solver,
                                             HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( HYPRE_FlexGMRESSetModifyPC( (HYPRE_Solver) solver,
                                        (HYPRE_PtrToModifyPCFcn) modify_pc));

}
