/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "headers.h"
#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCreate( HYPRE_Int size )
{
   hypre_Vector  *vector;

   vector = hypre_CTAlloc(hypre_Vector, 1);

   hypre_VectorData(vector) = NULL;
   hypre_VectorSize(vector) = size;

   hypre_VectorNumVectors(vector) = 1;
   hypre_VectorMultiVecStorageMethod(vector) = 0;

   /* set defaults */
   hypre_VectorOwnsData(vector) = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultiVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqMultiVectorCreate( HYPRE_Int size, HYPRE_Int num_vectors )
{
   hypre_Vector *vector = hypre_SeqVectorCreate(size);
   hypre_VectorNumVectors(vector) = num_vectors;
   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SeqVectorDestroy( hypre_Vector *vector )
{
   HYPRE_Int  ierr=0;

   if (vector)
   {
      if ( hypre_VectorOwnsData(vector) )
      {
         hypre_TFree(hypre_VectorData(vector));
      }
      hypre_TFree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SeqVectorInitialize( hypre_Vector *vector )
{
   HYPRE_Int  size = hypre_VectorSize(vector);
   HYPRE_Int  ierr = 0;
   HYPRE_Int  num_vectors = hypre_VectorNumVectors(vector);
   HYPRE_Int  multivec_storage_method = hypre_VectorMultiVecStorageMethod(vector);

   if ( ! hypre_VectorData(vector) )
      hypre_VectorData(vector) = hypre_CTAlloc(double, num_vectors*size);

   if ( multivec_storage_method == 0 )
   {
      hypre_VectorVectorStride(vector) = size;
      hypre_VectorIndexStride(vector) = 1;
   }
   else if ( multivec_storage_method == 1 )
   {
      hypre_VectorVectorStride(vector) = 1;
      hypre_VectorIndexStride(vector) = num_vectors;
   }
   else
      ++ierr;


   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_SeqVectorSetDataOwner( hypre_Vector *vector,
                          HYPRE_Int           owns_data   )
{
   HYPRE_Int    ierr=0;

   hypre_VectorOwnsData(vector) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * ReadVector
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorRead( char *file_name )
{
   hypre_Vector  *vector;

   FILE    *fp;

   double  *data;
   HYPRE_Int      size;
   
   HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   hypre_fscanf(fp, "%d", &size);

   vector = hypre_SeqVectorCreate(size);
   hypre_SeqVectorInitialize(vector);

   data = hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      hypre_fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   /* multivector code not written yet >>> */
   hypre_assert( hypre_VectorNumVectors(vector) == 1 );

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPrint( hypre_Vector *vector,
                   char         *file_name )
{
   FILE    *fp;

   double  *data;
   HYPRE_Int      size, num_vectors, vecstride, idxstride;
   
   HYPRE_Int      i, j;

   HYPRE_Int      ierr = 0;

   num_vectors = hypre_VectorNumVectors(vector);
   vecstride = hypre_VectorVectorStride(vector);
   idxstride = hypre_VectorIndexStride(vector);

   /*----------------------------------------------------------
    * Print in the data
    *----------------------------------------------------------*/

   data = hypre_VectorData(vector);
   size = hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   if ( hypre_VectorNumVectors(vector) == 1 )
   {
      hypre_fprintf(fp, "%d\n", size);
   }
   else
   {
      hypre_fprintf(fp, "%d vectors of size %d\n", num_vectors, size );
   }

   if ( num_vectors>1 )
   {
      for ( j=0; j<num_vectors; ++j )
      {
         hypre_fprintf(fp, "vector %d\n", j );
         for (i = 0; i < size; i++)
         {
            hypre_fprintf(fp, "%.14e\n",  data[ j*vecstride + i*idxstride ] );
         }
      }
   }
   else
   {
      for (i = 0; i < size; i++)
      {
         hypre_fprintf(fp, "%.14e\n", data[i]);
      }
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetConstantValues( hypre_Vector *v,
                               double        value )
{
   double  *vector_data = hypre_VectorData(v);
   HYPRE_Int      size        = hypre_VectorSize(v);
           
   HYPRE_Int      i;
           
   HYPRE_Int      ierr  = 0;

   size *=hypre_VectorNumVectors(v);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      vector_data[i] = value;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetRandomValues
 *
 *     returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetRandomValues( hypre_Vector *v,
                             HYPRE_Int           seed )
{
   double  *vector_data = hypre_VectorData(v);
   HYPRE_Int      size        = hypre_VectorSize(v);
           
   HYPRE_Int      i;
           
   HYPRE_Int      ierr  = 0;
   hypre_SeedRand(seed);

   size *=hypre_VectorNumVectors(v);

/* RDF: threading this loop may cause problems because of hypre_Rand() */
   for (i = 0; i < size; i++)
      vector_data[i] = 2.0 * hypre_Rand() - 1.0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopy
 * copies data from x to y
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorCopy( hypre_Vector *x,
                  hypre_Vector *y )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(x);
           
   HYPRE_Int      i;
           
   HYPRE_Int      ierr = 0;

   size *=hypre_VectorNumVectors(x);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      y_data[i] = x_data[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCloneDeep( hypre_Vector *x )
{
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Int      num_vectors   = hypre_VectorNumVectors(x);
   hypre_Vector * y = hypre_SeqMultiVectorCreate( size, num_vectors );

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

   hypre_SeqVectorInitialize(y);
   hypre_SeqVectorCopy( x, y );

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneShallow
 * Returns a complete copy of x - a shallow copy, pointing the data of x
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCloneShallow( hypre_Vector *x )
{
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Int      num_vectors   = hypre_VectorNumVectors(x);
   hypre_Vector * y = hypre_SeqMultiVectorCreate( size, num_vectors );

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

   hypre_VectorData(y) = hypre_VectorData(x);
   hypre_SeqVectorSetDataOwner( y, 0 );
   hypre_SeqVectorInitialize(y);

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorScale( double        alpha,
                   hypre_Vector *y     )
{
   double  *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(y);
           
   HYPRE_Int      i;
           
   HYPRE_Int      ierr = 0;

   size *=hypre_VectorNumVectors(y);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      y_data[i] *= alpha;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpy( double        alpha,
            hypre_Vector *x,
            hypre_Vector *y     )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(x);
           
   HYPRE_Int      i;
           
   HYPRE_Int      ierr = 0;

   size *=hypre_VectorNumVectors(x);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      y_data[i] += alpha * x_data[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

double   hypre_SeqVectorInnerProd( hypre_Vector *x,
                          hypre_Vector *y )
{
   double  *x_data = hypre_VectorData(x);
   double  *y_data = hypre_VectorData(y);
   HYPRE_Int      size   = hypre_VectorSize(x);
           
   HYPRE_Int      i;

   double      result = 0.0;

   size *=hypre_VectorNumVectors(x);

#define HYPRE_SMP_PRIVATE i
#define HYPRE_SMP_REDUCTION_OP +
#define HYPRE_SMP_REDUCTION_VARS result
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < size; i++)
      result += y_data[i] * x_data[i];

   return result;
}

/*--------------------------------------------------------------------------
 * hypre_VectorSumElts:
 * Returns the sum of all vector elements.
 *--------------------------------------------------------------------------*/

double hypre_VectorSumElts( hypre_Vector *vector )
{
   double sum = 0;
   double * data = hypre_VectorData( vector );
   HYPRE_Int size = hypre_VectorSize( vector );
   HYPRE_Int i;

#define HYPRE_SMP_PRIVATE i
#define HYPRE_SMP_REDUCTION_OP +
#define HYPRE_SMP_REDUCTION_VARS sum
#include "../utilities/hypre_smp_forloop.h"
   for ( i=0; i<size; ++i ) sum += data[i];

   return sum;
}
