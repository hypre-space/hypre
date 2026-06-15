/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_Schwarz - Public API for unified Schwarz Preconditioner
 *
 * Supports both legacy domain-based Schwarz (variants 0-4) and
 * recent overlapping Schwarz (variants 10+) implementations.
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * Create a Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzCreate( HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (HYPRE_Solver) hypre_SchwarzCreate( ) ;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Destroy a Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzDestroy( HYPRE_Solver solver )
{
   return ( hypre_SchwarzDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * Setup a Schwarz solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetup(HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   return ( hypre_SchwarzSetup( (void *) solver,
                                (hypre_ParCSRMatrix *) A,
                                (hypre_ParVector *) b,
                                (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * Solve a Schwarz problem
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSolve( HYPRE_Solver solver,
                    HYPRE_ParCSRMatrix A,
                    HYPRE_ParVector b,
                    HYPRE_ParVector x      )
{


   return ( hypre_SchwarzSolve( (void *) solver,
                                (hypre_ParCSRMatrix *) A,
                                (hypre_ParVector *) b,
                                (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * Domain-based Schwarz variants (0-4):
 *   0 = Multiplicative Schwarz (MPSchwarz)
 *   1 = Additive Schwarz with scaling (AdSchwarz)
 *   2 = Parallel Additive Schwarz (ParAdSchwarz)
 *   3 = Parallel Multiplicative Schwarz with boundary (ParMPSchwarz)
 *   4 = Forward Multiplicative Schwarz (MPSchwarzFW)
 *
 * Overlapping Schwarz variants (10+):
 *   10 = RAS + ILU(k)
 *   11 = AS + ILU(k)
 *   20 = RAS + ILUT
 *   21 = AS + ILUT
 *   30 = RAS + AMG
 *   31 = AS + AMG
 *   40 = RAS + SuperLU
 *   41 = AS + SuperLU
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetVariant( HYPRE_Solver solver,
                         HYPRE_Int    variant )
{
   return ( hypre_SchwarzSetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * Set the overlap order for the Schwarz solver
 *
 * For Domain-based (old) Schwarz: minimal overlap (default 1)
 * For Overlapping Schwarz: overlap order delta (default 1)
 *   - delta = 0: No overlap (block Jacobi)
 *   - delta = 1: One level of overlap (neighbors)
 *   - delta = k: k levels of overlap
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetOverlap( HYPRE_Solver solver, HYPRE_Int overlap)
{
   return ( hypre_SchwarzSetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * Set the domain type for the Schwarz solver
 * (Old Schwarz only - ignored for new overlapping Schwarz variants)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetDomainType( HYPRE_Solver solver,
                            HYPRE_Int    domain_type  )
{
   return ( hypre_SchwarzSetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * Set the domain structure for the Schwarz solver
 * (Old Schwarz only - ignored for new overlapping Schwarz variants)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetDomainStructure( HYPRE_Solver solver,
                                 HYPRE_CSRMatrix domain_structure  )
{
   return ( hypre_SchwarzSetDomainStructure(
               (void *) solver, (hypre_CSRMatrix *) domain_structure ) );
}

/*--------------------------------------------------------------------------
 * Set the number of functions for the Schwarz solver
 * (Old Schwarz only - ignored for new overlapping Schwarz variants)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetNumFunctions( HYPRE_Solver  solver,
                              HYPRE_Int     num_functions  )
{
   return ( hypre_SchwarzSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * Set the non-symmetry for the Schwarz solver
 * (Old Schwarz only - ignored for new overlapping Schwarz variants)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetNonSymm( HYPRE_Solver  solver,
                         HYPRE_Int     use_nonsymm  )
{
   return ( hypre_SchwarzSetNonSymm( (void *) solver, use_nonsymm ));
}

/*--------------------------------------------------------------------------
 * Set the relaxation weight for the Schwarz solver
 * Relaxation weight for the update (default 1.0)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetRelaxWeight( HYPRE_Solver  solver,
                             HYPRE_Real relax_weight)
{
   return ( hypre_SchwarzSetRelaxWeight((void *) solver, relax_weight));
}

/*--------------------------------------------------------------------------
 * Set the degree of freedom function for the Schwarz solver
 * (Old Schwarz only - ignored for new overlapping Schwarz variants)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetDofFunc( HYPRE_Solver  solver,
                         HYPRE_Int    *dof_func  )
{
   return ( hypre_SchwarzSetDofFunc( (void *) solver, dof_func ) );
}

/*--------------------------------------------------------------------------
 * Set the local solver type for the Schwarz solver
 *
 * Set the local solver type for overlapping Schwarz variants (10+):
 *   0 = ILU(k) (default)
 *   1 = ILUT
 *   2 = AMG
 *   3 = SuperLU_dist
 *
 * Note: The local solver type can also be specified via the variant number
 *       (e.g., variant=10 uses ILU(k), variant=14 uses AMG)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetLocalSolverType( HYPRE_Solver solver,
                                 HYPRE_Int    local_solver_type )
{
   return ( hypre_SchwarzSetLocalSolverType( (void *) solver, local_solver_type ) );
}

/*--------------------------------------------------------------------------
 * Set the level of fill for ILU(k) local solver
 *
 * Set the level of fill for ILU(k) local solver.
 * Default is 0 (ILU(0)).
 * (Overlapping Schwarz variants only)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetILUKLevelOfFill( HYPRE_Solver solver,
                                 HYPRE_Int    level_of_fill )
{
   return ( hypre_SchwarzSetILUKLevelOfFill( (void *) solver, level_of_fill ) );
}

/*--------------------------------------------------------------------------
 * Set the maximum nonzeros per row for ILUT local solver
 *
 * Set the maximum nonzeros per row for ILUT local solver.
 * Default is 1000.
 * (Overlapping Schwarz variants only)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetILUTMaxNnzPerRow( HYPRE_Solver solver,
                                  HYPRE_Int    max_nnz_row )
{
   return ( hypre_SchwarzSetILUTMaxNnzPerRow( (void *) solver, max_nnz_row ) );
}

/*--------------------------------------------------------------------------
 * Set the drop tolerance for ILUT local solver
 *
 * Set the drop tolerance for ILUT local solver.
 * Default is 1e-2.
 * (Overlapping Schwarz variants only)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetILUTDroptol( HYPRE_Solver solver,
                             HYPRE_Real   droptol )
{
   return ( hypre_SchwarzSetILUTDroptol( (void *) solver, droptol ) );
}

/*--------------------------------------------------------------------------
 * Set the maximum number of iterations for the Schwarz solver
 *
 * Set maximum number of iterations.
 * Default is 1 (for use as preconditioner).
 * (Overlapping Schwarz variants only)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetMaxIter( HYPRE_Solver solver,
                         HYPRE_Int    max_iter )
{
   return ( hypre_SchwarzSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * Set the convergence tolerance for the Schwarz solver
 *
 * Set convergence tolerance.
 * Default is 0.0 (no convergence checking).
 * (Overlapping Schwarz variants only)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetTol( HYPRE_Solver solver,
                     HYPRE_Real   tol )
{
   return ( hypre_SchwarzSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * Set the print level for the Schwarz solver
 *
 * Set print level for output:
 *   0 = no output (default)
 *   1 = setup/solve summary
 *   2 = per-iteration info
 * (Overlapping Schwarz variants only)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetPrintLevel( HYPRE_Solver solver,
                            HYPRE_Int    print_level )
{
   return ( hypre_SchwarzSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * Set the logging level for the Schwarz solver
 *
 * Set logging level:
 *   0 = no logging (default)
 *   1 = store residual norms
 * (Overlapping Schwarz variants only)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetLogging( HYPRE_Solver solver,
                         HYPRE_Int    logging )
{
   return ( hypre_SchwarzSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * Get the number of iterations performed
 *
 * Get the number of iterations performed.
 * (For old Schwarz, always returns 1)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzGetNumIterations( HYPRE_Solver solver,
                               HYPRE_Int   *num_iterations )
{
   return ( hypre_SchwarzGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * Get the final residual norm
 *
 * Get the final residual norm.
 * (For old Schwarz, always returns 0.0)
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzGetFinalResidualNorm( HYPRE_Solver solver,
                                   HYPRE_Real  *norm )
{
   return ( hypre_SchwarzGetFinalResidualNorm( (void *) solver, norm ) );
}
