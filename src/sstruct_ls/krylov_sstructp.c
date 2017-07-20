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

/******************************************************************************
 *
 * SStruct Pmatrix-Pvector implementation of Krylov interface routines.
 *
 * This is for SStruct_PMatrix and SStruct_PVector 
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"



HYPRE_Int
hypre_SStructPKrylovIdentitySetup( void *vdata,
                                   void *A,
                                   void *b,
                                   void *x )

{
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovIdentity( void *vdata,
                              void *A,
                              void *b,
                              void *x )

{
   return( hypre_SStructPKrylovCopyVector(b, x) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

char *
hypre_SStructPKrylovCAlloc( HYPRE_Int count,
                            HYPRE_Int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovFree( char *ptr )
{
   hypre_Free( ptr );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SStructPKrylovCreateVector( void *vvector )
{
	hypre_SStructPVector  *vector = (hypre_SStructPVector  *)vvector;	
   hypre_SStructPVector  *new_vector;

   hypre_StructVector   *svector;
   hypre_StructVector   *new_svector;
   HYPRE_Int            *num_ghost;
   
   HYPRE_Int    nvars, var;

   hypre_SStructPVectorCreate(hypre_SStructPVectorComm(vector),
                              hypre_SStructPVectorPGrid(vector),
                              &new_vector);

   nvars = hypre_SStructPVectorNVars(vector);
   for (var= 0; var< nvars; var++)
   {
      svector= hypre_SStructPVectorSVector(vector, var);
      num_ghost= hypre_StructVectorNumGhost(svector);

      new_svector= hypre_SStructPVectorSVector(new_vector, var);
      hypre_StructVectorSetNumGhost(new_svector, num_ghost);
   }

   hypre_SStructPVectorInitialize(new_vector);
   hypre_SStructPVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SStructPKrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
	hypre_SStructPVector  *vector = (hypre_SStructPVector  *)vvector;
   hypre_SStructPVector  **new_vector;

   hypre_StructVector   *svector;
   hypre_StructVector   *new_svector;
   HYPRE_Int            *num_ghost;
   
   HYPRE_Int nvars = hypre_SStructPVectorNVars(vector);
   HYPRE_Int var, i;

   new_vector = hypre_CTAlloc(hypre_SStructPVector*, n);
   
   for (i=0; i < n; i++)
   {
      hypre_SStructPVectorCreate(hypre_SStructPVectorComm(vector),
                                 hypre_SStructPVectorPGrid(vector),
                                &new_vector[i]);

      for (var= 0; var< nvars; var++)
      {
         svector= hypre_SStructPVectorSVector(vector, var);
         num_ghost= hypre_StructVectorNumGhost(svector);

         new_svector= hypre_SStructPVectorSVector(new_vector[i], var);
         hypre_StructVectorSetNumGhost(new_svector, num_ghost);
      }

      hypre_SStructPVectorInitialize(new_vector[i]);
      hypre_SStructPVectorAssemble(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovDestroyVector( void *vvector )
{
	hypre_SStructPVector *vector = (hypre_SStructPVector  *)vvector;

   return( hypre_SStructPVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SStructPKrylovMatvecCreate( void   *A,
                                  void   *x )
{
   void *matvec_data;

   hypre_SStructPMatvecCreate( &matvec_data );
   hypre_SStructPMatvecSetup( matvec_data,
                             (hypre_SStructPMatrix *) A,
                             (hypre_SStructPVector *) x );

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovMatvec( void   *matvec_data,
                            HYPRE_Complex  alpha,
                            void   *A,
                            void   *x,
                            HYPRE_Complex  beta,
                            void   *y )
{
   return ( hypre_SStructPMatvec( alpha,
                                 (hypre_SStructPMatrix *) A,
                                 (hypre_SStructPVector *) x,
                                 beta,
                                 (hypre_SStructPVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovMatvecDestroy( void *matvec_data )
{
   return ( hypre_SStructPMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Complex
hypre_SStructPKrylovInnerProd( void *x, 
                               void *y )
{
   HYPRE_Complex result;

   hypre_SStructPInnerProd( (hypre_SStructPVector *) x,
                            (hypre_SStructPVector *) y, &result );

   return result;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovCopyVector( void *x, 
                                void *y )
{
   return ( hypre_SStructPCopy( (hypre_SStructPVector *) x,
                                (hypre_SStructPVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovClearVector( void *x )
{
   return ( hypre_SStructPVectorSetConstantValues( (hypre_SStructPVector *) x,
                                                  0.0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovScaleVector( HYPRE_Complex  alpha,
                                 void   *x )
{
   return ( hypre_SStructPScale( alpha, (hypre_SStructPVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovAxpy( HYPRE_Complex alpha,
                         void   *x,
                         void   *y )
{
   return ( hypre_SStructPAxpy( alpha, (hypre_SStructPVector *) x,
                               (hypre_SStructPVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPKrylovCommInfo( void  *A,
                             HYPRE_Int   *my_id,
                             HYPRE_Int   *num_procs )
{
   MPI_Comm comm = hypre_SStructPMatrixComm((hypre_SStructPMatrix *) A);
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return hypre_error_flag;
}

