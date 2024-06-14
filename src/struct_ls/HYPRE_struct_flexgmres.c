/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_FlexGMRESFunctions * fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_StructKrylovCAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovCreateVectorArray,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (HYPRE_StructSolver) hypre_FlexGMRESCreate( fgmres_functions ) );

   return hypre_error_flag;
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESDestroy( HYPRE_StructSolver solver )
{
   return ( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetup( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x      )
{
   return ( HYPRE_FlexGMRESSetup( (HYPRE_Solver) solver,
                                  (HYPRE_Matrix) A,
                                  (HYPRE_Vector) b,
                                  (HYPRE_Vector) x ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSolve( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x      )
{
   return ( HYPRE_FlexGMRESSolve( (HYPRE_Solver) solver,
                                  (HYPRE_Matrix) A,
                                  (HYPRE_Vector) b,
                                  (HYPRE_Vector) x ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetTol( HYPRE_StructSolver solver,
                             HYPRE_Real         tol    )
{
   return ( HYPRE_FlexGMRESSetTol( (HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetAbsoluteTol( HYPRE_StructSolver solver,
                                     HYPRE_Real         atol    )
{
   return ( HYPRE_FlexGMRESSetAbsoluteTol( (HYPRE_Solver) solver, atol ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetMaxIter( HYPRE_StructSolver solver,
                                 HYPRE_Int          max_iter )
{
   return ( HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetKDim( HYPRE_StructSolver solver,
                              HYPRE_Int          k_dim )
{
   return ( HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetPrecond( HYPRE_StructSolver         solver,
                                 HYPRE_PtrToStructSolverFcn precond,
                                 HYPRE_PtrToStructSolverFcn precond_setup,
                                 HYPRE_StructSolver         precond_solver )
{
   return ( HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) precond,
                                       (HYPRE_PtrToSolverFcn) precond_setup,
                                       (HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetLogging( HYPRE_StructSolver solver,
                                 HYPRE_Int          logging )
{
   return ( HYPRE_FlexGMRESSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESSetPrintLevel( HYPRE_StructSolver solver,
                                    HYPRE_Int          print_level )
{
   return ( HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESGetNumIterations( HYPRE_StructSolver  solver,
                                       HYPRE_Int          *num_iterations )
{
   return ( HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver) solver,
                                             num_iterations ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                   HYPRE_Real         *norm   )
{
   return ( HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver,
                                                         norm ) );
}

/*==========================================================================*/

HYPRE_Int HYPRE_StructFlexGMRESSetModifyPC( HYPRE_StructSolver  solver,
                                            HYPRE_PtrToModifyPCFcn modify_pc)
{
   return ( HYPRE_FlexGMRESSetModifyPC( (HYPRE_Solver) solver,
                                        (HYPRE_PtrToModifyPCFcn) modify_pc));
}
