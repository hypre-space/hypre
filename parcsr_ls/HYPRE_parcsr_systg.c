/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_SysTGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SysTGCreate( HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (HYPRE_Solver) hypre_SysTGCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SysTGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SysTGDestroy( HYPRE_Solver solver )
{
   return( hypre_SysTGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SysTGSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_SysTGSetup( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SysTGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SysTGSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_SysTGSolve( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetBlockData
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SysTGSetBlockData( HYPRE_Solver solver,
                         HYPRE_Int  block_size, 
                         HYPRE_Int num_coarse_points, 
                         HYPRE_Int  *block_coarse_indexes)
{
   return( hypre_SysTGSetBlockData( (void *) solver, block_size, num_coarse_points, block_coarse_indexes ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetCoarseSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SysTGSetCoarseSolver(HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  coarse_grid_solver_solve,
                             HYPRE_PtrToParSolverFcn  coarse_grid_solver_setup,
                             HYPRE_Solver          coarse_grid_solver )
{
   return( hypre_SysTGSetCoarseSolver( (void *) solver,
                                  coarse_grid_solver_solve, coarse_grid_solver_setup,
                                  (void *) coarse_grid_solver ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetMaxLevels
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetMaxCoarseLevels( HYPRE_Solver solver, HYPRE_Int maxlev )
{
   return hypre_SysTGSetMaxCoarseLevels(solver, maxlev);
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetBlockSize
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetBlockSize( HYPRE_Solver solver, HYPRE_Int bsize )
{
   return hypre_SysTGSetBlockSize(solver, bsize );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetNumWells
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetNumWells( HYPRE_Solver solver, HYPRE_Int num_wells )
{
   return hypre_SysTGSetNumWells(solver, num_wells );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetRelaxType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetRelaxType(HYPRE_Solver solver, HYPRE_Int relax_type )
{
   return hypre_SysTGSetRelaxType(solver, relax_type );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetNumRelaxSweeps
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetNumRelaxSweeps( HYPRE_Solver solver, HYPRE_Int nsweeps )
{
   return hypre_SysTGSetNumRelaxSweeps(solver, nsweeps);
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetInterpType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetInterpType( HYPRE_Solver solver, HYPRE_Int interpType )
{
   return hypre_SysTGSetInterpType(solver, interpType);
}

/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetNumInterpSweeps
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetNumInterpSweeps( HYPRE_Solver solver, HYPRE_Int nsweeps )
{
   return hypre_SysTGSetNumInterpSweeps(solver, nsweeps);
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetPrintLevel
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetPrintLevel( HYPRE_Solver solver, HYPRE_Int print_level )
{
   return hypre_SysTGSetPrintLevel( solver, print_level );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetLogging
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetLogging( HYPRE_Solver solver, HYPRE_Int logging )
{
   return hypre_SysTGSetLogging(solver, logging );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetMaxIters
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetMaxIters( HYPRE_Solver solver, HYPRE_Int max_iter )
{
   return hypre_SysTGSetMaxIters( solver, max_iter );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetConvTol
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetConvTol( HYPRE_Solver solver, HYPRE_Real conv_tol )
{
   return hypre_SysTGSetConvTol( solver, conv_tol );
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetMaxGlobalsmoothIters
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SysTGSetMaxGlobalsmoothIters( HYPRE_Solver solver, HYPRE_Int max_iter )
{
	return hypre_SysTGSetMaxGlobalsmoothIters(solver, max_iter);
}
/*--------------------------------------------------------------------------
 * HYPRE_SysTGSetGlobalsmoothType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SysTGSetGlobalsmoothType( HYPRE_Solver solver, HYPRE_Int iter_type )
{
	return hypre_SysTGSetGlobalsmoothType(solver, iter_type);
}
