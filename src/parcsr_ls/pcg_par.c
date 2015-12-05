/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Struct matrix-vector implementation of PCG interface routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_ParKrylovCAlloc( HYPRE_Int count,
                 HYPRE_Int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovFree
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovFree( char *ptr )
{
   HYPRE_Int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovCreateVector( void *vvector )
{
   hypre_ParVector *vector = vvector;
   hypre_ParVector *new_vector;

   new_vector = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
				       hypre_ParVectorGlobalSize(vector),	
                                       hypre_ParVectorPartitioning(vector) );
   hypre_ParVectorSetPartitioningOwner(new_vector,0);
   hypre_ParVectorInitialize(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
   hypre_ParVector *vector = vvector;
   hypre_ParVector **new_vector;
   HYPRE_Int i;

   new_vector = hypre_CTAlloc(hypre_ParVector*,n);
   for (i=0; i < n; i++)
   {
      new_vector[i] = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
				       hypre_ParVectorGlobalSize(vector),	
                                       hypre_ParVectorPartitioning(vector) );
      hypre_ParVectorSetPartitioningOwner(new_vector[i],0);
      hypre_ParVectorInitialize(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovDestroyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovDestroyVector( void *vvector )
{
   hypre_ParVector *vector = vvector;

   return( hypre_ParVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovMatvec( void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_ParCSRMatrixMatvec ( alpha,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) x,
                               beta,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovMatvecT(void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_ParCSRMatrixMatvecT( alpha,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) x,
                               beta,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_ParKrylovInnerProd( void *x, 
                    void *y )
{
   return ( hypre_ParVectorInnerProd( (hypre_ParVector *) x,
                                (hypre_ParVector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovCopyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovCopyVector( void *x, 
                     void *y )
{
   return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovClearVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovClearVector( void *x )
{
   return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovScaleVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_ParVectorScale( alpha, (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x,
                              (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCommInfo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovCommInfo( void   *A, HYPRE_Int *my_id, HYPRE_Int *num_procs)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm ( (hypre_ParCSRMatrix *) A);
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentitySetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovIdentitySetup( void *vdata,
                        void *A,
                        void *b,
                        void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentity
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovIdentity( void *vdata,
                   void *A,
                   void *b,
                   void *x     )

{
   return( hypre_ParKrylovCopyVector( b, x ) );
}

