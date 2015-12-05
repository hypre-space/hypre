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
 * csr matrix-vector implementation of PCG interface routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CGCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_CGCAlloc( HYPRE_Int count,
                 HYPRE_Int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGFree
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGFree( char *ptr )
{
   HYPRE_Int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_CGCreateVector( void *vvector )
{
   hypre_Vector *vector = vvector;
   hypre_Vector *new_vector;

   new_vector = hypre_SeqVectorCreate( hypre_VectorSize(vector));
   hypre_SeqVectorInitialize(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_CGCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_CGCreateVectorArray(HYPRE_Int n, void *vvector )
{
   hypre_Vector *vector = vvector;
   hypre_Vector **new_vector;
   HYPRE_Int i;

   new_vector = hypre_CTAlloc(hypre_Vector*,n);
   for (i=0; i < n; i++)
   {
      new_vector[i] = hypre_SeqVectorCreate( hypre_VectorSize(vector));
      hypre_SeqVectorInitialize(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_CGDestroyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGDestroyVector( void *vvector )
{
   hypre_Vector *vector = vvector;

   return( hypre_SeqVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_CGMatvecCreate( void   *A,
                       void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGMatvec( void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_CSRMatrixMatvec ( alpha,
                              (hypre_CSRMatrix *) A,
                              (hypre_Vector *) x,
                               beta,
                              (hypre_Vector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvecT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGMatvecT(void   *matvec_data,
                 double  alpha,
                 void   *A,
                 void   *x,
                 double  beta,
                 void   *y           )
{
   return ( hypre_CSRMatrixMatvecT( alpha,
                              (hypre_CSRMatrix *) A,
                              (hypre_Vector *) x,
                               beta,
                              (hypre_Vector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CGInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_CGInnerProd( void *x, 
                    void *y )
{
   return ( hypre_SeqVectorInnerProd( (hypre_Vector *) x,
                                (hypre_Vector *) y ) );
}


/*--------------------------------------------------------------------------
 * hypre_CGCopyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGCopyVector( void *x, 
                     void *y )
{
   return ( hypre_SeqVectorCopy( (hypre_Vector *) x,
                                 (hypre_Vector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGClearVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGClearVector( void *x )
{
   return ( hypre_SeqVectorSetConstantValues( (hypre_Vector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGScaleVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGScaleVector( double  alpha,
                      void   *x     )
{
   return ( hypre_SeqVectorScale( alpha, (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGAxpy( double alpha,
               void   *x,
               void   *y )
{
   return ( hypre_SeqVectorAxpy( alpha, (hypre_Vector *) x,
                              (hypre_Vector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGCommInfo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGCommInfo( void   *A, HYPRE_Int *my_id, HYPRE_Int *num_procs)
{
   *num_procs = 1;
   *my_id = 0;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CGIdentitySetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGIdentitySetup( void *vdata,
                        void *A,
                        void *b,
                        void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CGIdentity
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGIdentity( void *vdata,
                   void *A,
                   void *b,
                   void *x     )

{
   return( hypre_CGCopyVector( b, x ) );
}


