/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_StructJacobi interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiCreate( MPI_Comm            comm,
                          HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_JacobiCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructJacobiDestroy( HYPRE_StructSolver solver )
{
   return( hypre_JacobiDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructJacobiSetup( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_JacobiSetup( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructJacobiSolve( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_JacobiSolve( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSetTol( HYPRE_StructSolver solver,
                          double             tol    )
{
   return( hypre_JacobiSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetTol( HYPRE_StructSolver solver,
                          double           * tol    )
{
   return( hypre_JacobiGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSetMaxIter( HYPRE_StructSolver solver,
                              HYPRE_Int          max_iter  )
{
   return( hypre_JacobiSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetMaxIter( HYPRE_StructSolver solver,
                              HYPRE_Int        * max_iter  )
{
   return( hypre_JacobiGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetZeroGuess
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructJacobiSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_JacobiSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetZeroGuess
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructJacobiGetZeroGuess( HYPRE_StructSolver solver,
                                HYPRE_Int * zeroguess )
{
   return( hypre_JacobiGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructJacobiSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_JacobiSetZeroGuess( (void *) solver, 0 ) );
}





/* NOT YET IMPLEMENTED */

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetNumIterations( HYPRE_StructSolver  solver,
                                    HYPRE_Int          *num_iterations )
{
   return( hypre_JacobiGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                double             *norm   )
{
   return( hypre_JacobiGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
