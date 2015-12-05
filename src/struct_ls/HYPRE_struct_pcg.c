/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*==========================================================================*/
/** Creates a new PCG solver object.

{\bf Input files:}
headers.h

@return Error code.

@param comm [IN]
  MPI communicator
@param solver [OUT]
  solver structure

@see HYPRE_StructPCGDestroy */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   /* The function names with a PCG in them are in
      struct_ls/pcg_struct.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
      These names should be called, e.g., hypre_struct_Free, to reduce the
      chance of name conflicts. */
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (HYPRE_StructSolver) hypre_PCGCreate( pcg_functions ) );

   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroys a PCG solver object.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure

@see HYPRE_StructPCGCreate */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructPCGDestroy( HYPRE_StructSolver solver )
{
   return( hypre_PCGDestroy( (void *) solver ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Precomputes tasks necessary for doing the solve.  This routine
ensures that the setup for the preconditioner is also called.

NOTE: This is supposed to be an optional call, but currently is required.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param A [IN]
  coefficient matrix
@param b [IN]
  right-hand-side vector
@param x [IN]
  unknown vector

@see HYPRE_StructPCGSolve */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructPCGSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_PCGSetup( (HYPRE_Solver) solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Performs the PCG linear solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param A [IN]
  coefficient matrix
@param b [IN]
  right-hand-side vector
@param x [IN]
  unknown vector

@see HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructPCGSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( HYPRE_PCGSolve( (HYPRE_Solver) solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the stopping tolerance.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param tol [IN]
  PCG solver tolerance

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_PCGSetTol( (HYPRE_Solver) solver, tol ) );
}
/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the absolute stopping tolerance.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param tol [IN]
  PCG solver tolerance

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup   */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetAbsoluteTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( HYPRE_PCGSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the maximum number of iterations.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param max_iter [IN]
  PCG solver maximum number of iterations

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetMaxIter( HYPRE_StructSolver solver,
                           HYPRE_Int          max_iter )
{
   return( HYPRE_PCGSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the type of norm to use in the stopping criteria.
If parameter two\_norm is set to 0, the preconditioner norm is used.
If set to 1, the two-norm is used.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param two_norm [IN]
  boolean indicating whether or not to use the two-norm

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetTwoNorm( HYPRE_StructSolver solver,
                           HYPRE_Int          two_norm )
{
   return( HYPRE_PCGSetTwoNorm( (HYPRE_Solver) solver, two_norm ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set whether or not to do an additional relative change
stopping test.  If parameter rel\_change is set to 0, no additional
stopping test is done.  If set to 1, the additional test is done.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param rel_change [IN]
  boolean indicating whether or not to do relative change test

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetRelChange( HYPRE_StructSolver solver,
                             HYPRE_Int          rel_change )
{
   return( HYPRE_PCGSetRelChange( (HYPRE_Solver) solver, rel_change ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Sets the precondioner to use in PCG.  The Default is no
preconditioner, i.e. the solver is just conjugate gradients (CG).

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param precond [IN]
  pointer to the preconditioner solve function
@param precond_setup [IN]
  pointer to the preconditioner setup function
@param precond_solver [IN/OUT]
  preconditioner solver structure

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup*/
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetPrecond( HYPRE_StructSolver         solver,
                           HYPRE_PtrToStructSolverFcn precond,
                           HYPRE_PtrToStructSolverFcn precond_setup,
                           HYPRE_StructSolver         precond_solver )
{
   return( HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                (HYPRE_PtrToSolverFcn) precond,
                                (HYPRE_PtrToSolverFcn) precond_setup,
                                (HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Set the type of logging to do.  Currently, if parameter
logging is set to 0, no logging is done.  If set to 1, the norms and
relative norms for each iteration are saved.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param logging [IN]
  integer indicating what type of logging to do

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetLogging( HYPRE_StructSolver solver,
                           HYPRE_Int          logging )
{
   return( HYPRE_PCGSetLogging( (HYPRE_Solver) solver, logging ) );
}
/*==========================================================================*/
/*==========================================================================*/
/** (Optional)  Currently, if parameter print_level is set to 0, no printing 
is allowed.  If set to 1, printing takes place.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param logging [IN]
  integer allowing printing to take place

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGSetPrintLevel( HYPRE_StructSolver solver,
                               HYPRE_Int      print_level )
{
   return( HYPRE_PCGSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
}
/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Gets the number of iterations done in the solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN]
  solver structure
@param num_iterations [OUT]
  number of iterations

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGGetNumIterations( HYPRE_StructSolver  solver,
                                 HYPRE_Int          *num_iterations )
{
   return( HYPRE_PCGGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** (Optional) Gets the final relative residual norm for the solve.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN]
  solver structure
@param norm [OUT]
  final relative residual norm

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPCGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Setup routine for diagonally scaling a vector.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param A [IN]
  coefficient matrix
@param b [IN]
  right-hand-side vector
@param x [IN]
  unknown vector

@see HYPRE_StructDiagScale */
/*--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructDiagScaleSetup( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector y,
                            HYPRE_StructVector x      )
{
   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Diagonally scale a vector.

{\bf Input files:}
headers.h

@return Error code.

@param solver [IN/OUT]
  solver structure
@param A [IN]
  coefficient matrix
@param b [IN]
  right-hand-side vector
@param x [IN]
  unknown vector

@see HYPRE_StructDiagScaleSetup */
/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------
 * HYPRE_StructDiagScale
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructDiagScale( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix HA,
                       HYPRE_StructVector Hy,
                       HYPRE_StructVector Hx      )
{
   hypre_StructMatrix   *A = (hypre_StructMatrix *) HA;
   hypre_StructVector   *y = (hypre_StructVector *) Hy;
   hypre_StructVector   *x = (hypre_StructVector *) Hx;

   hypre_BoxArray       *boxes;
   hypre_Box            *box;

   hypre_Box            *A_data_box;
   hypre_Box            *y_data_box;
   hypre_Box            *x_data_box;
                     
   double               *Ap;
   double               *yp;
   double               *xp;
                       
   HYPRE_Int             Ai;
   HYPRE_Int             yi;
   HYPRE_Int             xi;
                     
   hypre_Index           index;
   hypre_IndexRef        start;
   hypre_Index           stride;
   hypre_Index           loop_size;
                     
   HYPRE_Int             i;
   HYPRE_Int             loopi, loopj, loopk;

   HYPRE_Int             ierr = 0;
  
   /* x = D^{-1} y */
   hypre_SetIndex(stride, 1, 1, 1);
   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         hypre_SetIndex(index, 0, 0, 0);
         Ap = hypre_StructMatrixExtractPointerByIndex(A, i, index);
         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         start  = hypre_BoxIMin(box);

         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop3Begin(loop_size,
                             A_data_box, start, stride, Ai,
                             x_data_box, start, stride, xi,
                             y_data_box, start, stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi,xi,Ai
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, yi)
            {
               xp[xi] = yp[yi] / Ap[Ai];
            }
         hypre_BoxLoop3End(Ai, xi, yi);
      }

   return ierr;
}

