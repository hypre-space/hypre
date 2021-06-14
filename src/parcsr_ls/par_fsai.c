/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * 
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

#include "_hypre_parcsr_ls.h"

/******************************************************************************
 * HYPRE_FSAICreate
 ******************************************************************************/

HYPRE_Int
HYPRE_FSAICreate( HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (HYPRE_Solver) hypre_FSAICreate( );

   return hypre_error_flag;
}

/******************************************************************************
 * HYPRE_FSAIDestroy
 ******************************************************************************/

HYPRE_Int
HYPRE_FSAIDestroy( HYPRE_Solver solver )
{
   return( hypre_FSAIDestroy( (void*) solver ) );
}

/******************************************************************************
 * HYPRE_FSAISetup
 ******************************************************************************/



/******************************************************************************
 * HYPRE_FSAISetMaxIterations 
 ******************************************************************************/

HYPRE_Int
HYPRE_FSAISetMaxIterations( HYPRE_Solver solver, HYPRE_Int max_iterations )
{
   return( hypre_FSAISetMaxIterations( (void*) solver, max_iterations ) ); 
}

/******************************************************************************
 * HYPRE_FSAISetTolerance 
 ******************************************************************************/

HYPRE_Int
HYPRE_FSAISetTolerance( HYPRE_Solver solver, HYPRE_Real tolerance )
{
   return( hypre_FSAISetTolerance( (void*) solver, tolerance ) ); 
}

/******************************************************************************
 * HYPRE_FSAISetMaxs 
 ******************************************************************************/

HYPRE_Int
HYPRE_FSAISetMaxs( HYPRE_Solver solver, HYPRE_Int max_s )
{
   return( hypre_FSAISetMaxs( (void*) solver, max_s ) ); 
}
