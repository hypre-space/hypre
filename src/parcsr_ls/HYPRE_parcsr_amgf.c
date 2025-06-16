/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMGFCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_AMGFCreate( HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (HYPRE_Solver) hypre_AMGFCreate( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_AMGFDestroy( HYPRE_Solver solver )
{
   return ( hypre_AMGFDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_AMGFSetup( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   return ( hypre_AMGFSetup( (void *) solver,
                                  (hypre_ParCSRMatrix *) A,
                                  (hypre_ParVector *) b,
                                  (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_AMGFSolve( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!b)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!x)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   return ( hypre_AMGFSolve( (void *) solver,
                                  (hypre_ParCSRMatrix *) A,
                                  (hypre_ParVector *) b,
                                  (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFSetRestriction
 *--------------------------------------------------------------------------*/

/*HYPRE_Int
HYPRE_BoomerAMGSetRestriction( HYPRE_Solver solver,
                               HYPRE_Int    restr_par  )
{
   return ( hypre_BoomerAMGSetRestriction( (void *) solver, restr_par ) );
}
*/

