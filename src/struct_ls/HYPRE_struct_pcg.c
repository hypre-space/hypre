/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   /* The function names with a PCG in them are in
      struct_ls/pcg_struct.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
      These names should be called, e.g., hypre_struct_Free, to reduce the
      chance of name conflicts. */
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_StructKrylovCAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (HYPRE_StructSolver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGDestroy( HYPRE_StructSolver solver )
{
   return ( hypre_PCGDestroy( (void *) solver ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return ( HYPRE_PCGSetup( (HYPRE_Solver) solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return ( HYPRE_PCGSolve( (HYPRE_Solver) solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetTol( HYPRE_StructSolver solver,
                       HYPRE_Real         tol    )
{
   return ( HYPRE_PCGSetTol( (HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetAbsoluteTol( HYPRE_StructSolver solver,
                               HYPRE_Real         tol    )
{
   return ( HYPRE_PCGSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetMaxIter( HYPRE_StructSolver solver,
                           HYPRE_Int          max_iter )
{
   return ( HYPRE_PCGSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetTwoNorm( HYPRE_StructSolver solver,
                           HYPRE_Int          two_norm )
{
   return ( HYPRE_PCGSetTwoNorm( (HYPRE_Solver) solver, two_norm ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetRelChange( HYPRE_StructSolver solver,
                             HYPRE_Int          rel_change )
{
   return ( HYPRE_PCGSetRelChange( (HYPRE_Solver) solver, rel_change ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetPrecond( HYPRE_StructSolver         solver,
                           HYPRE_PtrToStructSolverFcn precond,
                           HYPRE_PtrToStructSolverFcn precond_setup,
                           HYPRE_StructSolver         precond_solver )
{
   return ( HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) precond,
                                 (HYPRE_PtrToSolverFcn) precond_setup,
                                 (HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetLogging( HYPRE_StructSolver solver,
                           HYPRE_Int          logging )
{
   return ( HYPRE_PCGSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGSetPrintLevel( HYPRE_StructSolver solver,
                              HYPRE_Int      print_level )
{
   return ( HYPRE_PCGSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGGetNumIterations( HYPRE_StructSolver  solver,
                                 HYPRE_Int          *num_iterations )
{
   return ( HYPRE_PCGGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructPCGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             HYPRE_Real         *norm   )
{
   return ( HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructDiagScaleSetup( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector y,
                            HYPRE_StructVector x      )
{
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(y);
   HYPRE_UNUSED_VAR(x);

   return hypre_error_flag;
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructDiagScale( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix HA,
                       HYPRE_StructVector Hy,
                       HYPRE_StructVector Hx      )
{
   HYPRE_UNUSED_VAR(solver);

   hypre_StructMatrix   *A = (hypre_StructMatrix *) HA;
   hypre_StructVector   *y = (hypre_StructVector *) Hy;
   hypre_StructVector   *x = (hypre_StructVector *) Hx;

   hypre_BoxArray       *boxes;
   hypre_Box            *box;

   hypre_Box            *A_data_box;
   hypre_Box            *y_data_box;
   hypre_Box            *x_data_box;

   HYPRE_Real           *Ap;
   HYPRE_Real           *yp;
   HYPRE_Real           *xp;

   hypre_Index           index;
   hypre_IndexRef        start;
   hypre_Index           stride;
   hypre_Index           loop_size;

   HYPRE_Int             i;

   /* x = D^{-1} y */
   hypre_SetIndex(stride, 1);
   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      hypre_SetIndex(index, 0);
      Ap = hypre_StructMatrixExtractPointerByIndex(A, i, index);
      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      start  = hypre_BoxIMin(box);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,yp,Ap)
      hypre_BoxLoop3Begin(hypre_StructVectorNDim(Hx), loop_size,
                          A_data_box, start, stride, Ai,
                          x_data_box, start, stride, xi,
                          y_data_box, start, stride, yi);
      {
         xp[xi] = yp[yi] / Ap[Ai];
      }
      hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}
