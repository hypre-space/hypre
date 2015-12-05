/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Struct matrix-vector implementation of PCG interface routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_StructKrylovCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_StructKrylovCAlloc( HYPRE_Int count,
                 HYPRE_Int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovFree
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovFree( char *ptr )
{
   HYPRE_Int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_StructKrylovCreateVector( void *vvector )
{
   hypre_StructVector *vector = vvector;
   hypre_StructVector *new_vector;
   HYPRE_Int          *num_ghost= hypre_StructVectorNumGhost(vector);

   new_vector = hypre_StructVectorCreate( hypre_StructVectorComm(vector),
                                          hypre_StructVectorGrid(vector) );
   hypre_StructVectorSetNumGhost(new_vector, num_ghost);
   hypre_StructVectorInitialize(new_vector);
   hypre_StructVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_StructKrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
   hypre_StructVector *vector = vvector;
   hypre_StructVector **new_vector;
   HYPRE_Int          *num_ghost= hypre_StructVectorNumGhost(vector);
   HYPRE_Int i;

   new_vector = hypre_CTAlloc(hypre_StructVector*,n);
   for (i=0; i < n; i++)
   {
      HYPRE_StructVectorCreate(hypre_StructVectorComm(vector),
                                hypre_StructVectorGrid(vector),
                                (HYPRE_StructVector *) &new_vector[i] );
      hypre_StructVectorSetNumGhost(new_vector[i], num_ghost);
      HYPRE_StructVectorInitialize((HYPRE_StructVector) new_vector[i]);
      HYPRE_StructVectorAssemble((HYPRE_StructVector) new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovDestroyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovDestroyVector( void *vvector )
{
   hypre_StructVector *vector = vvector;

   return( hypre_StructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_StructKrylovMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup(matvec_data, A, x);

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovMatvec( void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_StructMatvecCompute( matvec_data,
                                       alpha,
                                       (hypre_StructMatrix *) A,
                                       (hypre_StructVector *) x,
                                       beta,
                                       (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovMatvecDestroy( void *matvec_data )
{
   return ( hypre_StructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_StructKrylovInnerProd( void *x, 
                    void *y )
{
   return ( hypre_StructInnerProd( (hypre_StructVector *) x,
                                   (hypre_StructVector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_StructKrylovCopyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovCopyVector( void *x, 
                     void *y )
{
   return ( hypre_StructCopy( (hypre_StructVector *) x,
                              (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovClearVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovClearVector( void *x )
{
   return ( hypre_StructVectorSetConstantValues( (hypre_StructVector *) x,
                                                 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovScaleVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_StructScale( alpha, (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_StructAxpy( alpha, (hypre_StructVector *) x,
                              (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovIdentitySetup (for a default preconditioner)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovIdentitySetup( void *vdata,
                        void *A,
                        void *b,
                        void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovIdentity (for a default preconditioner)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovIdentity( void *vdata,
                   void *A,
                   void *b,
                   void *x     )

{
   return( hypre_StructKrylovCopyVector( b, x ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructKrylovCommInfo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovCommInfo( void  *A,
                      HYPRE_Int   *my_id,
                      HYPRE_Int   *num_procs )
{
   MPI_Comm comm = hypre_StructMatrixComm((hypre_StructMatrix *) A);
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return 0;
}

