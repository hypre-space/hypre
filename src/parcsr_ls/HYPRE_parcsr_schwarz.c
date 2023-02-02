/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzCreate
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
 * HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzDestroy( HYPRE_Solver solver )
{
   return ( hypre_SchwarzDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetup
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
 * HYPRE_SchwarzSolve
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
 * HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetVariant( HYPRE_Solver solver,
                         HYPRE_Int    variant )
{
   return ( hypre_SchwarzSetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetOverlap( HYPRE_Solver solver, HYPRE_Int overlap)
{
   return ( hypre_SchwarzSetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetDomainType( HYPRE_Solver solver,
                            HYPRE_Int    domain_type  )
{
   return ( hypre_SchwarzSetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetDomainStructure( HYPRE_Solver solver,
                                 HYPRE_CSRMatrix domain_structure  )
{
   return ( hypre_SchwarzSetDomainStructure(
               (void *) solver, (hypre_CSRMatrix *) domain_structure ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetNumFunctions( HYPRE_Solver  solver,
                              HYPRE_Int     num_functions  )
{
   return ( hypre_SchwarzSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetNonSymm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetNonSymm( HYPRE_Solver  solver,
                         HYPRE_Int     use_nonsymm  )
{
   return ( hypre_SchwarzSetNonSymm( (void *) solver, use_nonsymm ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetRelaxWeight( HYPRE_Solver  solver,
                             HYPRE_Real relax_weight)
{
   return ( hypre_SchwarzSetRelaxWeight((void *) solver, relax_weight));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SchwarzSetDofFunc( HYPRE_Solver  solver,
                         HYPRE_Int    *dof_func  )
{
   return ( hypre_SchwarzSetDofFunc( (void *) solver, dof_func ) );
}

