/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorCreate( MPI_Comm comm,
                       int      global_size, 
                       int     *partitioning,
		       HYPRE_ParVector *vector )
{
   *vector = (HYPRE_ParVector) hypre_ParVectorCreate(comm, global_size,
                                                    partitioning) ;
   if (!vector) hypre_error_in_arg(4);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParMultiVectorCreate( MPI_Comm comm,
                            int      global_size, 
                            int     *partitioning,
                            int      number_vectors,
                            HYPRE_ParVector *vector )
{
   *vector = (HYPRE_ParVector) hypre_ParMultiVectorCreate
      (comm, global_size, partitioning, number_vectors );

   if (!vector) hypre_error_in_arg(5);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParVectorDestroy( HYPRE_ParVector vector )
{
   return ( hypre_ParVectorDestroy( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParVectorInitialize( HYPRE_ParVector vector )
{
   return ( hypre_ParVectorInitialize( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorRead( MPI_Comm         comm,
                     const char      *file_name, 
		     HYPRE_ParVector *vector)
{
   *vector = (HYPRE_ParVector) hypre_ParVectorRead( comm, file_name ) ;
   if (!vector) hypre_error_in_arg(3);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorPrint( HYPRE_ParVector  vector,
                      const char      *file_name )
{
   return ( hypre_ParVectorPrint( (hypre_ParVector *) vector,
                                  file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorSetConstantValues( HYPRE_ParVector  vector,
                      		  double	   value )
{
   return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) vector,
                                  value ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorSetRandomValues( HYPRE_ParVector  vector,
                      		int	         seed  )
{
   return ( hypre_ParVectorSetRandomValues( (hypre_ParVector *) vector,
                                  seed ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorCopy( HYPRE_ParVector x, HYPRE_ParVector y)
{
   return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCloneShallow
 *--------------------------------------------------------------------------*/

HYPRE_ParVector
HYPRE_ParVectorCloneShallow( HYPRE_ParVector x )
{
   return ( (HYPRE_ParVector) hypre_ParVectorCloneShallow( (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorScale( double value, HYPRE_ParVector x)
{
   return ( hypre_ParVectorScale( value, (hypre_ParVector *) x) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/
int
HYPRE_ParVectorAxpy( double        alpha,
                     HYPRE_ParVector x,
                     HYPRE_ParVector y     )
{
   return hypre_ParVectorAxpy( alpha, (hypre_ParVector *)x, (hypre_ParVector *)y );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

int
HYPRE_ParVectorInnerProd( HYPRE_ParVector x, HYPRE_ParVector y, double *prod)
{
   if (!x) 
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (!y) 
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   *prod = hypre_ParVectorInnerProd( (hypre_ParVector *) x, 
			(hypre_ParVector *) y) ;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_VectorToParVector
 *--------------------------------------------------------------------------*/

int
HYPRE_VectorToParVector( MPI_Comm comm, HYPRE_Vector b, int *partitioning,
			 HYPRE_ParVector *vector)
{
   *vector = (HYPRE_ParVector) hypre_VectorToParVector (comm, 
		(hypre_Vector *) b, partitioning );
   if (!vector) hypre_error_in_arg(4);
   return hypre_error_flag;
}
