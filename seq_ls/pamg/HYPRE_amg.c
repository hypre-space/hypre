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




/******************************************************************************
 *
 * HYPRE_AMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMGInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Solver
HYPRE_AMGInitialize( )
{
   return ( (HYPRE_Solver) hypre_AMGInitialize( ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGFinalize( HYPRE_Solver solver )
{
   return( hypre_AMGFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGSetup( HYPRE_Solver solver,
                HYPRE_CSRMatrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_AMGSetup( (void *) solver,
                           (hypre_CSRMatrix *) A,
                           (hypre_Vector *) b,
                           (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGSolve( HYPRE_Solver solver,
                HYPRE_CSRMatrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{


   return( hypre_AMGSolve( (void *) solver,
                           (hypre_CSRMatrix *) A,
                           (hypre_Vector *) b,
                           (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetMaxLevels( HYPRE_Solver solver,
                       int          max_levels  )
{
   return( hypre_AMGSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetStrongThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetStrongThreshold( HYPRE_Solver solver,
                             double       strong_threshold  )
{
   return( hypre_AMGSetStrongThreshold( (void *) solver, strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetMode
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetMode( HYPRE_Solver solver,
                       int          mode  )
{
   return( hypre_AMGSetMode( (void *) solver, mode ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetATruncFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetATruncFactor( HYPRE_Solver solver,
                          double       A_trunc_factor)
{
   return( hypre_AMGSetATruncFactor( (void *) solver, A_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetAMaxElmts
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetAMaxElmts( HYPRE_Solver solver,
                       int          A_max_elmts)
{
   return( hypre_AMGSetAMaxElmts( (void *) solver, A_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetPTruncFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetPTruncFactor( HYPRE_Solver solver,
                          double       P_trunc_factor)
{
   return( hypre_AMGSetPTruncFactor( (void *) solver, P_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetPMaxElmts
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetPMaxElmts( HYPRE_Solver solver,
                       int          P_max_elmts)
{
   return( hypre_AMGSetPMaxElmts( (void *) solver, P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetCoarsenType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetCoarsenType( HYPRE_Solver solver,
                         int          coarsen_type  )
{
   return( hypre_AMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetAggCoarsenType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetAggCoarsenType( HYPRE_Solver solver,
                         int          agg_coarsen_type  )
{
   return( hypre_AMGSetAggCoarsenType( (void *) solver, agg_coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetAggLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetAggLevels( HYPRE_Solver solver,
                         int          agg_levels  )
{
   return( hypre_AMGSetAggLevels( (void *) solver, agg_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetInterpType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetInterpType( HYPRE_Solver solver,
                        int          interp_type  )
{
   return( hypre_AMGSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetAggInterpType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetAggInterpType( HYPRE_Solver solver,
                        int          agg_interp_type  )
{
   return( hypre_AMGSetAggInterpType( (void *) solver, agg_interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetNumJacs
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetNumJacs( HYPRE_Solver solver,
                     int          num_jacs  )
{
   return( hypre_AMGSetNumJacs( (void *) solver, num_jacs ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetMaxIter( HYPRE_Solver solver,
                     int          max_iter  )
{
   return( hypre_AMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetCycleType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetCycleType( HYPRE_Solver solver,
                       int          cycle_type  )
{
   return( hypre_AMGSetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetTol( HYPRE_Solver solver,
                 double       tol    )
{
   return( hypre_AMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetNumRelaxSteps
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetNumRelaxSteps( HYPRE_Solver  solver,
                           int           num_relax_steps  )
{
   return( hypre_AMGSetNumRelaxSteps( (void *) solver, num_relax_steps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetNumGridSweeps
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetNumGridSweeps( HYPRE_Solver  solver,
                           int          *num_grid_sweeps  )
{
   return( hypre_AMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetGridRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetGridRelaxType( HYPRE_Solver  solver,
                           int          *grid_relax_type  )
{
   return( hypre_AMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetGridRelaxPoints( HYPRE_Solver   solver,
                             int          **grid_relax_points  )
{
   return( hypre_AMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetRelaxWeight( HYPRE_Solver   solver,
                         double        *relax_weight  )
{
   return( hypre_AMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetSchwarzOption
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetSchwarzOption( HYPRE_Solver   solver,
                         int        *schwarz_option  )
{
   return( hypre_AMGSetSchwarzOption( (void *) solver, schwarz_option ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetIOutDat
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetIOutDat( HYPRE_Solver solver,
                     int          ioutdat  )
{
   return( hypre_AMGSetIOutDat( (void *) solver, ioutdat ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetLogFileName
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetLogFileName( HYPRE_Solver  solver,
                         char         *log_file_name  )
{
   return( hypre_AMGSetLogFileName( (void *) solver, log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetLogging( HYPRE_Solver  solver,
                     int           ioutdat,
                     char         *log_file_name  )
{
   return( hypre_AMGSetLogging( (void *) solver, ioutdat, log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetNumFunctions
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetNumFunctions( HYPRE_Solver solver,
                       int          num_functions  )
{
   return( hypre_AMGSetNumFunctions( (void *) solver, num_functions ) );
}

int
HYPRE_AMGSetDofFunc( HYPRE_Solver solver,
                     int          *dof_func  )
{
   return( hypre_AMGSetDofFunc( (void *) solver, dof_func ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetUseBlockFlag
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetUseBlockFlag( HYPRE_Solver solver,
                       int          use_block_flag  )
{
   return( hypre_AMGSetUseBlockFlag( (void *) solver, use_block_flag ) );
}
