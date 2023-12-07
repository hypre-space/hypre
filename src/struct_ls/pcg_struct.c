/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_StructKrylovCAlloc( size_t               count,
                          size_t               elt_size,
                          HYPRE_MemoryLocation location)
{
   return ( (void*) hypre_CTAlloc(char, count * elt_size, location) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovFree( void *ptr )
{
   hypre_TFree( ptr, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_StructKrylovCreateVector( void *vvector )
{
   hypre_StructVector *vector = (hypre_StructVector *)vvector;
   hypre_StructVector *new_vector;
   HYPRE_Int          *num_ghost = hypre_StructVectorNumGhost(vector);

   new_vector = hypre_StructVectorCreate( hypre_StructVectorComm(vector),
                                          hypre_StructVectorGrid(vector) );
   hypre_StructVectorSetNumGhost(new_vector, num_ghost);
   hypre_StructVectorInitialize(new_vector);
   hypre_StructVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_StructKrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
   hypre_StructVector *vector = (hypre_StructVector *)vvector;
   hypre_StructVector **new_vector;
   HYPRE_Int          *num_ghost = hypre_StructVectorNumGhost(vector);
   HYPRE_Int i;

   new_vector = hypre_CTAlloc(hypre_StructVector*, n, HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovDestroyVector( void *vvector )
{
   hypre_StructVector *vector = (hypre_StructVector *)vvector;

   return ( hypre_StructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_StructKrylovMatvecCreate( void   *A,
                                void   *x )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup(matvec_data, (hypre_StructMatrix *)A, (hypre_StructVector *)x);

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovMatvec( void   *matvec_data,
                          HYPRE_Complex  alpha,
                          void   *A,
                          void   *x,
                          HYPRE_Complex  beta,
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovMatvecDestroy( void *matvec_data )
{
   return ( hypre_StructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_StructKrylovInnerProd( void *x,
                             void *y )
{
   return ( hypre_StructInnerProd( (hypre_StructVector *) x,
                                   (hypre_StructVector *) y ) );
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovCopyVector( void *x,
                              void *y )
{
   return ( hypre_StructCopy( (hypre_StructVector *) x,
                              (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovClearVector( void *x )
{
   return ( hypre_StructVectorSetConstantValues( (hypre_StructVector *) x,
                                                 0.0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovScaleVector( HYPRE_Complex  alpha,
                               void   *x     )
{
   return ( hypre_StructScale( alpha, (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovAxpy( HYPRE_Complex alpha,
                        void   *x,
                        void   *y )
{
   return ( hypre_StructAxpy( alpha, (hypre_StructVector *) x,
                              (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovIdentitySetup( void *vdata,
                                 void *A,
                                 void *b,
                                 void *x     )

{
   HYPRE_UNUSED_VAR(vdata);
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovIdentity( void *vdata,
                            void *A,
                            void *b,
                            void *x     )

{
   HYPRE_UNUSED_VAR(vdata);
   HYPRE_UNUSED_VAR(A);

   return ( hypre_StructKrylovCopyVector( b, x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructKrylovCommInfo( void  *A,
                            HYPRE_Int   *my_id,
                            HYPRE_Int   *num_procs )
{
   MPI_Comm comm = hypre_StructMatrixComm((hypre_StructMatrix *) A);
   hypre_MPI_Comm_size(comm, num_procs);
   hypre_MPI_Comm_rank(comm, my_id);
   return hypre_error_flag;
}
