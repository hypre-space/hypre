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
 * HYPRE_MGRCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRCreate( HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (HYPRE_Solver) hypre_MGRCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_MGRDestroy( HYPRE_Solver solver )
{
   return( hypre_MGRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_MGRSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_MGRSetup( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_MGRSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_MGRSolve( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}


HYPRE_Int
HYPRE_MGRSetBlockDataWrapper( HYPRE_Solver solver, 
                                HYPRE_Int block_size, 
                                HYPRE_Int coarse_grid_index)
{
  return ( hypre_MGRSetBlockDataWrapper( (void *) solver, block_size, coarse_grid_index) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetBlockData
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetBlockData( HYPRE_Solver solver,
                         HYPRE_Int  block_size, 
                         HYPRE_Int num_coarse_points, 
                         HYPRE_Int  *block_coarse_indexes)
{
   return( hypre_MGRSetBlockData( (void *) solver, block_size, num_coarse_points, block_coarse_indexes ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetBlockDataExp
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetBlockDataExp( HYPRE_Solver solver,
                         HYPRE_Int  block_size,
                         HYPRE_Int  *num_coarse_points,
                         HYPRE_Int  **block_coarse_indexes)
{
   return( hypre_MGRSetBlockDataExp( (void *) solver, block_size, num_coarse_points, block_coarse_indexes ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCoarseSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetCoarseSolver(HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  coarse_grid_solver_solve,
                             HYPRE_PtrToParSolverFcn  coarse_grid_solver_setup,
                             HYPRE_Solver          coarse_grid_solver )
{
   return( hypre_MGRSetCoarseSolver( (void *) solver,
									   (HYPRE_Int (*)(void*, void*, void*, void*)) coarse_grid_solver_solve,
									   (HYPRE_Int (*)(void*, void*, void*, void*)) coarse_grid_solver_setup,
									   (void *) coarse_grid_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetAdditionalCoarseIndices
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetAdditionalCoarseIndices( HYPRE_Solver solver, HYPRE_Int num_add_coarse_idx, HYPRE_Int *additional_coarse_indices)
{
   return hypre_MGRSetAdditionalCoarseIndices(solver, num_add_coarse_idx, additional_coarse_indices);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxLevels
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetMaxCoarseLevels( HYPRE_Solver solver, HYPRE_Int maxlev )
{
   return hypre_MGRSetMaxCoarseLevels(solver, maxlev);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetBlockSize
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetBlockSize( HYPRE_Solver solver, HYPRE_Int bsize )
{
   return hypre_MGRSetBlockSize(solver, bsize );
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetReservedCoarseNodes
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetReservedCoarseNodes( HYPRE_Solver solver, HYPRE_Int reserved_coarse_size, HYPRE_Int *reserved_coarse_indexes )
{
   return hypre_MGRSetReservedCoarseNodes(solver, reserved_coarse_size, reserved_coarse_indexes );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetRestrictType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetRestrictType(HYPRE_Solver solver, HYPRE_Int restrict_type )
{
   return hypre_MGRSetRestrictType(solver, restrict_type );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetRelaxMethod
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetRelaxMethod(HYPRE_Solver solver, HYPRE_Int relax_method )
{
   return hypre_MGRSetRelaxMethod(solver, relax_method );
}

HYPRE_Int
HYPRE_MGRSetSplittingStrategy(HYPRE_Solver solver, HYPRE_Int splitting_strategy )
{
   return hypre_MGRSetSplittingStrategy(solver, splitting_strategy );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetRelaxType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetRelaxType(HYPRE_Solver solver, HYPRE_Int relax_type )
{
   return hypre_MGRSetRelaxType(solver, relax_type );
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumRelaxSweeps
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetNumRelaxSweeps( HYPRE_Solver solver, HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumRelaxSweeps(solver, nsweeps);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetInterpType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetInterpType( HYPRE_Solver solver, HYPRE_Int interpType )
{
   return hypre_MGRSetInterpType(solver, interpType);
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumInterpSweeps
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetNumInterpSweeps( HYPRE_Solver solver, HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumInterpSweeps(solver, nsweeps);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetPrintLevel
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetPrintLevel( HYPRE_Solver solver, HYPRE_Int print_level )
{
   return hypre_MGRSetPrintLevel( solver, print_level );
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLogging
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetLogging( HYPRE_Solver solver, HYPRE_Int logging )
{
   return hypre_MGRSetLogging(solver, logging );
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxIters
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetMaxIters( HYPRE_Solver solver, HYPRE_Int max_iter )
{
   return hypre_MGRSetMaxIters( solver, max_iter );
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetConvTol
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetConvTol( HYPRE_Solver solver, HYPRE_Real conv_tol )
{
   return hypre_MGRSetConvTol( solver, conv_tol );
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxGlobalsmoothIters
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRSetMaxGlobalsmoothIters( HYPRE_Solver solver, HYPRE_Int max_iter )
{
	return hypre_MGRSetMaxGlobalsmoothIters(solver, max_iter);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRSetGlobalsmoothType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MGRSetGlobalsmoothType( HYPRE_Solver solver, HYPRE_Int iter_type )
{
	return hypre_MGRSetGlobalsmoothType(solver, iter_type);
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRGetNumIterations
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRGetNumIterations( HYPRE_Solver solver, HYPRE_Int *num_iterations )
{
   return hypre_MGRGetNumIterations( solver, num_iterations );
}
/*--------------------------------------------------------------------------
 * HYPRE_MGRGetResidualNorm
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MGRGetResidualNorm(  HYPRE_Solver solver, HYPRE_Real *res_norm )
{
   return hypre_MGRGetResidualNorm(solver, res_norm);
}
