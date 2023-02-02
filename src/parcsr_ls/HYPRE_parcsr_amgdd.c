/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDCreate( HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (HYPRE_Solver) hypre_BoomerAMGDDCreate( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDDestroy( HYPRE_Solver solver )
{
   return ( hypre_BoomerAMGDDDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x )
{
   return ( hypre_BoomerAMGDDSetup( (void *) solver,
                                    (hypre_ParCSRMatrix *) A,
                                    (hypre_ParVector *) b,
                                    (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x )
{
   return ( hypre_BoomerAMGDDSolve( (void *) solver,
                                    (hypre_ParCSRMatrix *) A,
                                    (hypre_ParVector *) b,
                                    (hypre_ParVector *) x ) );
}

/*-------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetStartLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetStartLevel( HYPRE_Solver solver,
                                HYPRE_Int    start_level )
{
   return ( hypre_BoomerAMGDDSetStartLevel( (void *) solver, start_level ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetStartLevel( HYPRE_Solver  solver,
                                HYPRE_Int    *start_level )
{
   return ( hypre_BoomerAMGDDGetStartLevel( (void *) solver, start_level ) );
}

/*-------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetFACNumRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetFACNumRelax( HYPRE_Solver solver,
                                 HYPRE_Int    amgdd_fac_num_relax )
{
   return ( hypre_BoomerAMGDDSetFACNumRelax( (void *) solver, amgdd_fac_num_relax ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetFACNumRelax( HYPRE_Solver  solver,
                                 HYPRE_Int    *amgdd_fac_num_relax )
{
   return ( hypre_BoomerAMGDDGetFACNumRelax( (void *) solver, amgdd_fac_num_relax ) );
}

/*-------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetFACNumCycles
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetFACNumCycles( HYPRE_Solver solver,
                                  HYPRE_Int    amgdd_fac_num_cycles )
{
   return ( hypre_BoomerAMGDDSetFACNumCycles( (void *) solver, amgdd_fac_num_cycles ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetFACNumCycles( HYPRE_Solver  solver,
                                  HYPRE_Int    *amgdd_fac_num_cycles  )
{
   return ( hypre_BoomerAMGDDGetFACNumCycles( (void *) solver, amgdd_fac_num_cycles ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetFACCycleType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetFACCycleType( HYPRE_Solver solver,
                                  HYPRE_Int    amgdd_fac_cycle_type )
{
   return ( hypre_BoomerAMGDDSetFACCycleType( (void *) solver, amgdd_fac_cycle_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetFACCycleType( HYPRE_Solver  solver,
                                  HYPRE_Int    *amgdd_fac_cycle_type )
{
   return ( hypre_BoomerAMGDDGetFACCycleType( (void *) solver, amgdd_fac_cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetFACRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetFACRelaxType( HYPRE_Solver solver,
                                  HYPRE_Int    amgdd_fac_relax_type )
{
   return ( hypre_BoomerAMGDDSetFACRelaxType( (void *) solver, amgdd_fac_relax_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetFACRelaxType( HYPRE_Solver  solver,
                                  HYPRE_Int    *amgdd_fac_relax_type )
{
   return ( hypre_BoomerAMGDDGetFACRelaxType( (void *) solver, amgdd_fac_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetFACRelaxWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetFACRelaxWeight( HYPRE_Solver solver,
                                    HYPRE_Real   amgdd_fac_relax_weight )
{
   return ( hypre_BoomerAMGDDSetFACRelaxWeight( (void *) solver, amgdd_fac_relax_weight ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetFACRelaxWeight( HYPRE_Solver  solver,
                                    HYPRE_Real   *amgdd_fac_relax_weight )
{
   return ( hypre_BoomerAMGDDGetFACRelaxWeight( (void *) solver, amgdd_fac_relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetPadding
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetPadding( HYPRE_Solver solver,
                             HYPRE_Int    padding )
{
   return ( hypre_BoomerAMGDDSetPadding( (void *) solver, padding ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetPadding( HYPRE_Solver  solver,
                             HYPRE_Int    *padding  )
{
   return ( hypre_BoomerAMGDDGetPadding( (void *) solver, padding ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetNumGhostLayers
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetNumGhostLayers( HYPRE_Solver solver,
                                    HYPRE_Int    num_ghost_layers )
{
   return ( hypre_BoomerAMGDDSetNumGhostLayers( (void *) solver, num_ghost_layers ) );
}

HYPRE_Int
HYPRE_BoomerAMGDDGetNumGhostLayers( HYPRE_Solver  solver,
                                    HYPRE_Int    *num_ghost_layers )
{
   return ( hypre_BoomerAMGDDGetNumGhostLayers( (void *) solver, num_ghost_layers ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDSetUserFACRelaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDSetUserFACRelaxation( HYPRE_Solver solver,
                                       HYPRE_Int (*userFACRelaxation)( void      *amgdd_vdata,
                                                                       HYPRE_Int  level,
                                                                       HYPRE_Int  cycle_param ) )
{
   return ( hypre_BoomerAMGDDSetUserFACRelaxation( (void *) solver, userFACRelaxation ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDGetAMG
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDGetAMG( HYPRE_Solver  solver,
                         HYPRE_Solver *amg_solver )
{
   return ( hypre_BoomerAMGDDGetAMG( (void *) solver, (void **) amg_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               HYPRE_Real   *rel_resid_norm )
{
   HYPRE_Solver amg_solver;

   HYPRE_BoomerAMGDDGetAMG(solver, &amg_solver);
   return ( hypre_BoomerAMGGetRelResidualNorm( (void *) amg_solver, rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDDGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGDDGetNumIterations( HYPRE_Solver   solver,
                                   HYPRE_Int     *num_iterations )
{
   HYPRE_Solver amg_solver;

   HYPRE_BoomerAMGDDGetAMG(solver, &amg_solver);
   return ( hypre_BoomerAMGGetNumIterations( (void *) amg_solver, num_iterations ) );
}
