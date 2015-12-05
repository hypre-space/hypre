/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
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

HYPRE_Int
HYPRE_ParVectorCreate( MPI_Comm comm,
                       HYPRE_Int      global_size, 
                       HYPRE_Int     *partitioning,
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

HYPRE_Int
HYPRE_ParMultiVectorCreate( MPI_Comm comm,
                            HYPRE_Int      global_size, 
                            HYPRE_Int     *partitioning,
                            HYPRE_Int      number_vectors,
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

HYPRE_Int 
HYPRE_ParVectorDestroy( HYPRE_ParVector vector )
{
   return ( hypre_ParVectorDestroy( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParVectorInitialize( HYPRE_ParVector vector )
{
   return ( hypre_ParVectorInitialize( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

HYPRE_Int
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

HYPRE_Int
HYPRE_ParVectorPrint( HYPRE_ParVector  vector,
                      const char      *file_name )
{
   return ( hypre_ParVectorPrint( (hypre_ParVector *) vector,
                                  file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParVectorSetConstantValues( HYPRE_ParVector  vector,
                      		  double	   value )
{
   return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) vector,
                                  value ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParVectorSetRandomValues( HYPRE_ParVector  vector,
                      		HYPRE_Int	         seed  )
{
   return ( hypre_ParVectorSetRandomValues( (hypre_ParVector *) vector,
                                  seed ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
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

HYPRE_Int
HYPRE_ParVectorScale( double value, HYPRE_ParVector x)
{
   return ( hypre_ParVectorScale( value, (hypre_ParVector *) x) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParVectorAxpy( double        alpha,
                     HYPRE_ParVector x,
                     HYPRE_ParVector y     )
{
   return hypre_ParVectorAxpy( alpha, (hypre_ParVector *)x, (hypre_ParVector *)y );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Int
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

HYPRE_Int
HYPRE_VectorToParVector( MPI_Comm comm, HYPRE_Vector b, HYPRE_Int *partitioning,
			 HYPRE_ParVector *vector)
{
   *vector = (HYPRE_ParVector) hypre_VectorToParVector (comm, 
		(hypre_Vector *) b, partitioning );
   if (!vector) hypre_error_in_arg(4);
   return hypre_error_flag;
}
