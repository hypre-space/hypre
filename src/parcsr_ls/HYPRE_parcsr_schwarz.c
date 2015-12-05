/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/



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
   if (!solver) hypre_error_in_arg(1);
   return hypre_error_flag;
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

