/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_parcsr_ls.h"

HYPRE_Int
hypre_KrylovFree( void *ptr )
{
   HYPRE_Int ierr = 0;

   hypre_Free( ptr , HYPRE_MEMORY_HOST);

   return ierr;
}


HYPRE_Int
hypre_KrylovCommInfo( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs)
{
   *num_procs = 1;
   *my_id = 0;
   
   return 0;
}


void *
hypre_KrylovCreateVector( void *vvector )
{
   hypre_Vector *vector = (hypre_Vector *) vvector;
   hypre_Vector *new_vector = hypre_SeqVectorCreate( hypre_VectorSize(vector) );
   hypre_SeqVectorInitialize(new_vector);

   return ( (void *) new_vector );
}


void *
hypre_KrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
   hypre_Vector *vector = (hypre_Vector *) vvector;
   hypre_Vector **new_vector;
   HYPRE_Int i;

   new_vector = hypre_CTAlloc(hypre_Vector*, n, HYPRE_MEMORY_HOST);
   for (i=0; i < n; i++)
   {
      new_vector[i] = hypre_SeqVectorCreate( hypre_VectorSize(vector) );
      hypre_SeqVectorInitialize(new_vector[i]);
   }

   return ( (void *) new_vector );
}


HYPRE_Int
hypre_KrylovDestroyVector( void *vvector )
{
   hypre_Vector *vector = (hypre_Vector *) vvector;

   return ( hypre_SeqVectorDestroy(vector) );
}


void *
hypre_KrylovMatvecCreate( void   *A,
                          void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}


HYPRE_Int
hypre_KrylovMatvec( void   *matvec_data,
                    HYPRE_Complex  alpha,
                    void   *A,
                    void   *x,
                    HYPRE_Complex  beta,
                    void   *y           )
{
   return ( hypre_CSRMatrixMatvec ( alpha,
                                    (hypre_CSRMatrix *) A,
                                    (hypre_Vector *) x,
                                    beta,
                                    (hypre_Vector *) y ) );
}


HYPRE_Int
hypre_KrylovMatvecT(void   *matvec_data,
                    HYPRE_Complex  alpha,
                    void   *A,
                    void   *x,
                    HYPRE_Complex  beta,
                    void   *y           )
{
   return ( hypre_CSRMatrixMatvecT( alpha,
                                    (hypre_CSRMatrix *) A,
                                    (hypre_Vector *) x,
                                    beta,
                                    (hypre_Vector *) y ) );
}


HYPRE_Int
hypre_KrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}


HYPRE_Real
hypre_KrylovInnerProd( void *x, 
                       void *y )
{
   return ( hypre_SeqVectorInnerProd( (hypre_Vector *) x,
                                      (hypre_Vector *) y ) );
}


HYPRE_Int
hypre_KrylovCopyVector( void *x, 
                        void *y )
{
   return ( hypre_SeqVectorCopy( (hypre_Vector *) x,
                                 (hypre_Vector *) y ) );
}


HYPRE_Int
hypre_KrylovClearVector( void *x )
{
   return ( hypre_SeqVectorSetConstantValues( (hypre_Vector *) x, 0.0 ) );
}


HYPRE_Int
hypre_KrylovScaleVector( HYPRE_Complex  alpha,
                         void   *x )
{
   return ( hypre_SeqVectorScale( alpha, (hypre_Vector *) x ) );
}


HYPRE_Int
hypre_KrylovAxpy( HYPRE_Complex alpha,
                  void   *x,
                  void   *y )
{
   return ( hypre_SeqVectorAxpy( alpha, (hypre_Vector *) x,
                                 (hypre_Vector *) y ) );
}


HYPRE_Int
hypre_KrylovIdentitySetup( void *vdata,
                              void *A,
                              void *b,
                              void *x     )

{
   return 0;
}


HYPRE_Int
hypre_KrylovIdentity( void *vdata,
                      void *A,
                      void *b,
                      void *x     )

{
   return ( hypre_KrylovCopyVector( b, x ) );
}

