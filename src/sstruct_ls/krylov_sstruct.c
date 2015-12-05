/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * SStruct matrix-vector implementation of Krylov interface routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_SStructKrylovCAlloc( HYPRE_Int count,
                    HYPRE_Int elt_size )
{
   return( hypre_CAlloc( count, elt_size ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovFree
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovFree( char *ptr )
{
   HYPRE_Int ierr = 0;

   hypre_Free( ptr );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovCreateVector( void *vvector )
{
   hypre_SStructVector  *vector = vvector;
   hypre_SStructVector  *new_vector;
   HYPRE_Int             object_type;

   HYPRE_Int             nparts= hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   hypre_StructVector   *svector;
   hypre_SStructPVector *new_pvector;
   hypre_StructVector   *new_svector;
   HYPRE_Int            *num_ghost;
   
   HYPRE_Int    part;
   HYPRE_Int    nvars, var;

   object_type= hypre_SStructVectorObjectType(vector);

   HYPRE_SStructVectorCreate(hypre_SStructVectorComm(vector),
                             hypre_SStructVectorGrid(vector),
                             &new_vector);
   HYPRE_SStructVectorSetObjectType(new_vector, object_type);

   if (object_type == HYPRE_SSTRUCT || object_type == HYPRE_STRUCT)
   {
      for (part= 0; part< nparts; part++)
      {
         pvector    = hypre_SStructVectorPVector(vector, part);
         new_pvector= hypre_SStructVectorPVector(new_vector, part);
         nvars      = hypre_SStructPVectorNVars(pvector);

         for (var= 0; var< nvars; var++)
         {
            svector= hypre_SStructPVectorSVector(pvector, var);
            num_ghost= hypre_StructVectorNumGhost(svector);

            new_svector= hypre_SStructPVectorSVector(new_pvector, var);
            hypre_StructVectorSetNumGhost(new_svector, num_ghost);
         }
      }
   }

   HYPRE_SStructVectorInitialize(new_vector);
   HYPRE_SStructVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
   hypre_SStructVector  *vector = vvector;
   hypre_SStructVector  **new_vector;
   HYPRE_Int             object_type;

   HYPRE_Int             nparts= hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   hypre_StructVector   *svector;
   hypre_SStructPVector *new_pvector;
   hypre_StructVector   *new_svector;
   HYPRE_Int            *num_ghost;
   
   HYPRE_Int    part;
   HYPRE_Int    nvars, var;

   HYPRE_Int i;

   object_type= hypre_SStructVectorObjectType(vector);

   new_vector = hypre_CTAlloc(hypre_SStructVector*,n);
   for (i=0; i < n; i++)
   {
      HYPRE_SStructVectorCreate(hypre_SStructVectorComm(vector),
                                hypre_SStructVectorGrid(vector),
                                &new_vector[i]);
      HYPRE_SStructVectorSetObjectType(new_vector[i], object_type);

      if (object_type == HYPRE_SSTRUCT || object_type == HYPRE_STRUCT)
      {
         for (part= 0; part< nparts; part++)
         {
            pvector    = hypre_SStructVectorPVector(vector, part);
            new_pvector= hypre_SStructVectorPVector(new_vector[i], part);
            nvars      = hypre_SStructPVectorNVars(pvector);

            for (var= 0; var< nvars; var++)
            {
               svector= hypre_SStructPVectorSVector(pvector, var);
               num_ghost= hypre_StructVectorNumGhost(svector);

               new_svector= hypre_SStructPVectorSVector(new_pvector, var);
               hypre_StructVectorSetNumGhost(new_svector, num_ghost);
            }
         }
      }

      HYPRE_SStructVectorInitialize(new_vector[i]);
      HYPRE_SStructVectorAssemble(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovDestroyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovDestroyVector( void *vvector )
{
   hypre_SStructVector *vector = vvector;

   return( HYPRE_SStructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovMatvecCreate( void   *A,
                          void   *x )
{
   void *matvec_data;

   hypre_SStructMatvecCreate( &matvec_data );
   hypre_SStructMatvecSetup( matvec_data,
                             (hypre_SStructMatrix *) A,
                             (hypre_SStructVector *) x );

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovMatvec( void   *matvec_data,
                    double  alpha,
                    void   *A,
                    void   *x,
                    double  beta,
                    void   *y )
{
   return ( hypre_SStructMatvec( alpha,
                                 (hypre_SStructMatrix *) A,
                                 (hypre_SStructVector *) x,
                                 beta,
                                 (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovMatvecDestroy( void *matvec_data )
{
   return ( hypre_SStructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovInnerProd
 *--------------------------------------------------------------------------*/

double
hypre_SStructKrylovInnerProd( void *x, 
                       void *y )
{
   double result;

   hypre_SStructInnerProd( (hypre_SStructVector *) x,
                           (hypre_SStructVector *) y, &result );

   return result;
}


/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCopyVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovCopyVector( void *x, 
                        void *y )
{
   return ( hypre_SStructCopy( (hypre_SStructVector *) x,
                               (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovClearVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovClearVector( void *x )
{
   return ( hypre_SStructVectorSetConstantValues( (hypre_SStructVector *) x,
                                                  0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovScaleVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovScaleVector( double  alpha,
                         void   *x )
{
   return ( hypre_SStructScale( alpha, (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovAxpy( double alpha,
                  void   *x,
                  void   *y )
{
   return ( hypre_SStructAxpy( alpha, (hypre_SStructVector *) x,
                                     (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructKrylovCommInfo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovCommInfo( void  *A,
                      HYPRE_Int   *my_id,
                      HYPRE_Int   *num_procs )
{
   MPI_Comm comm = hypre_SStructMatrixComm((hypre_SStructMatrix *) A);
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return 0;
}

