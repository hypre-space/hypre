/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParAMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SchwarzCreate( HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_SchwarzCreate( ) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_SchwarzDestroy( HYPRE_Solver solver )
{
   return( hypre_SchwarzDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_SchwarzSetup(HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   return( hypre_SchwarzSetup( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_SchwarzSolve( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{


   return( hypre_SchwarzSolve( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

int
HYPRE_SchwarzSetVariant( HYPRE_Solver solver,
                         int          variant )
{
   return( hypre_SchwarzSetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

int
HYPRE_SchwarzSetOverlap( HYPRE_Solver solver, int overlap)
{
   return( hypre_SchwarzSetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

int
HYPRE_SchwarzSetDomainType( HYPRE_Solver solver,
                              int          domain_type  )
{
   return( hypre_SchwarzSetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

int
HYPRE_SchwarzSetDomainStructure( HYPRE_Solver solver,
                                 HYPRE_CSRMatrix domain_structure  )
{
   return( hypre_SchwarzSetDomainStructure( (void *) solver, 
			(hypre_CSRMatrix *) domain_structure ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

int
HYPRE_SchwarzSetNumFunctions( HYPRE_Solver  solver,
                              int          num_functions  )
{
   return( hypre_SchwarzSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_SchwarzSetRelaxWeight( HYPRE_Solver  solver,
                                double relax_weight)
{
   return( hypre_SchwarzSetRelaxWeight((void *) solver,relax_weight));
}

/*--------------------------------------------------------------------------
 * HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

int
HYPRE_SchwarzSetDofFunc( HYPRE_Solver  solver,
                              int          *dof_func  )
{
   return( hypre_SchwarzSetDofFunc( (void *) solver, dof_func ) );
}

