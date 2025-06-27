/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCreate( HYPRE_Int size )
{
   hypre_Vector  *vector;

   vector = hypre_CTAlloc(hypre_Vector, 1, HYPRE_MEMORY_HOST);

   hypre_VectorData(vector)                  = NULL;
   hypre_VectorSize(vector)                  = size;
   hypre_VectorNumTags(vector)               = 1;
   hypre_VectorOwnsTags(vector)              = 1;
   hypre_VectorTags(vector)                  = NULL;
   hypre_VectorNumVectors(vector)            = 1;
   hypre_VectorMultiVecStorageMethod(vector) = 0;
   hypre_VectorOwnsData(vector)              = 1;
   hypre_VectorMemoryLocation(vector)        = hypre_HandleMemoryLocation(hypre_handle());

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
   if (vector)
   {
      HYPRE_MemoryLocation memory_location = hypre_VectorMemoryLocation(vector);

      if (hypre_VectorOwnsTags(vector))
      {
         hypre_TFree(hypre_VectorTags(vector), memory_location);
      }

      if (hypre_VectorOwnsData(vector))
      {
         hypre_TFree(hypre_VectorData(vector), memory_location);
      }

      hypre_TFree(vector, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitializeShell
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorInitializeShell( hypre_Vector *vector )
{
   HYPRE_Int  size = hypre_VectorSize(vector);
   HYPRE_Int  num_vectors = hypre_VectorNumVectors(vector);
   HYPRE_Int  multivec_storage_method = hypre_VectorMultiVecStorageMethod(vector);

   if (multivec_storage_method == 0)
   {
      hypre_VectorVectorStride(vector) = size;
      hypre_VectorIndexStride(vector)  = 1;
   }
   else if (multivec_storage_method == 1)
   {
      hypre_VectorVectorStride(vector) = 1;
      hypre_VectorIndexStride(vector)  = num_vectors;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid multivec storage method!\n");
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetData( hypre_Vector  *vector,
                        HYPRE_Complex *data )
{
   /* Free data array if already present */
   if (hypre_VectorData(vector) && hypre_VectorOwnsData(vector))
   {
      hypre_TFree(hypre_VectorData(vector), hypre_VectorMemoryLocation(vector));
   }

   /* Set data pointer passed via input  */
   hypre_VectorData(vector) = data;

   /* Remove data pointer ownership */
   hypre_VectorOwnsData(vector) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set an ownership flag about the tags array.
 *
 *  0: vector tags point to the user-input tags, vector does not own tags
 *  1: vector tags is created and user-input tags copied into it, vector owns the tags
 *  2: vector tags point to the user-input tags, vector owns the tags
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetOwnsTags( hypre_Vector *vector,
                            HYPRE_Int     owns_tags )
{
   /* Set owns tags info passed via input */
   hypre_VectorOwnsTags(vector) = owns_tags;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetNumTags( hypre_Vector *vector,
                           HYPRE_Int     num_tags )
{
   /* Set number of tags info passed via input */
   hypre_VectorNumTags(vector) = num_tags;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set tags array to a SeqVector. See hypre_SeqVectorSetOwnsTags for
 * ownership logic.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetTags( hypre_Vector          *vector,
                        HYPRE_MemoryLocation   memory_location,
                        HYPRE_Int             *tags )
{
   HYPRE_Int  size = hypre_VectorSize(vector);
   HYPRE_Int  owns_tags = hypre_VectorOwnsTags(vector);
   HYPRE_Int *new_tags;

   /* Return if tags array does not exist */
   if (!tags)
   {
      hypre_TFree(hypre_VectorTags(vector), hypre_VectorMemoryLocation(vector));
      hypre_VectorTags(vector) = NULL;

      return hypre_error_flag;
   }

   if (owns_tags == 1)
   {
      /* Deallocate existing tags if present */
      if (hypre_VectorTags(vector))
      {
         hypre_TFree(hypre_VectorTags(vector), hypre_VectorMemoryLocation(vector));
      }

      /* Allocate new tags array */
      new_tags = hypre_TAlloc(HYPRE_Int, size, hypre_VectorMemoryLocation(vector));

      /* Copy tags */
      hypre_TMemcpy(new_tags, tags, HYPRE_Int, size,
                    hypre_VectorMemoryLocation(vector),
                    memory_location);

      /* Attach new tags */
      hypre_VectorTags(vector) = new_tags;
   }
   else
   {
      /* Deallocate existing tags if present */
      if (hypre_VectorTags(vector) && owns_tags == 2)
      {
         hypre_TFree(hypre_VectorTags(vector), hypre_VectorMemoryLocation(vector));
      }

      /* Just point to the input tags array */
      hypre_VectorTags(vector) = tags;

      /* Check whether the memory location for the tags array
         match the memory location used by the vector */
      if (hypre_GetActualMemLocation(hypre_VectorMemoryLocation(vector)) !=
          hypre_GetActualMemLocation(memory_location))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Memory location mismatch!");
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetValuesTaggedHost( hypre_Vector  *vector,
                                    HYPRE_Complex *values )
{
   HYPRE_Int   size = hypre_VectorSize(vector);
   HYPRE_Int  *tags = hypre_VectorTags(vector);
   HYPRE_Int   i;

   /* Setup scaling vector */
#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      hypre_VectorEntryI(vector, i) = values[tags[i]];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetValuesTagged( hypre_Vector  *vector,
                                HYPRE_Complex *values )
{
   /* Sanity checks */
   if (hypre_VectorNumVectors(vector) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_vectors > 1 not implemented!");
      return hypre_error_flag;
   }

   if ((!hypre_VectorTags(vector) && hypre_VectorSize(vector) > 0) || hypre_VectorNumTags(vector) < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "This function is valid only for tagged vectors");
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_VectorMemoryLocation(vector));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SeqVectorSetValuesTaggedDevice(vector, values);
   }
   else
#endif
   {
      hypre_SeqVectorSetValuesTaggedHost(vector, values);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize_v2
 *
 * Initialize a vector at a given memory location
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorInitialize_v2( hypre_Vector         *vector,
                              HYPRE_MemoryLocation  memory_location )
{
   HYPRE_Int  size        = hypre_VectorSize(vector);
   HYPRE_Int  num_vectors = hypre_VectorNumVectors(vector);

   /* Set up the basic structure and metadata for the local vector */
   hypre_SeqVectorInitializeShell(vector);

   /* Set memory location */
   hypre_VectorMemoryLocation(vector) = memory_location;

   /* Caveat: for pre-existing data, the memory location must be guaranteed
    * to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered
    * when being used, and freed */
   if (!hypre_VectorData(vector))
   {
      hypre_VectorData(vector) = hypre_CTAlloc(HYPRE_Complex, num_vectors * size, memory_location);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorInitialize( hypre_Vector *vector )
{
   return hypre_SeqVectorInitialize_v2(vector, hypre_VectorMemoryLocation(vector));
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetDataOwner( hypre_Vector *vector,
                             HYPRE_Int     owns_data )
{
   hypre_VectorOwnsData(vector) = owns_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetSize( hypre_Vector *vector,
                        HYPRE_Int     size   )
{
   HYPRE_Int  multivec_storage_method = hypre_VectorMultiVecStorageMethod(vector);

   hypre_VectorSize(vector) = size;
   if (multivec_storage_method == 0)
   {
      hypre_VectorVectorStride(vector) = size;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorResize
 *
 * Resize a sequential vector when changing its number of components or
 * local size.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorResize( hypre_Vector *vector,
                       HYPRE_Int     size_in,
                       HYPRE_Int     num_vectors_in )
{
   HYPRE_Int  method        = hypre_VectorMultiVecStorageMethod(vector);
   HYPRE_Int  size          = hypre_VectorSize(vector);
   HYPRE_Int  num_vectors   = hypre_VectorNumVectors(vector);
   HYPRE_Int  total_size    = num_vectors * size;
   HYPRE_Int  total_size_in = num_vectors_in * size_in;

   /* Reallocate data array */
   if (total_size_in > total_size)
   {
      hypre_VectorData(vector) = hypre_TReAlloc_v2(hypre_VectorData(vector),
                                                   HYPRE_Complex,
                                                   total_size,
                                                   HYPRE_Complex,
                                                   total_size_in,
                                                   hypre_VectorMemoryLocation(vector));
   }

   /* Update vector info */
   hypre_VectorSize(vector) = size_in;
   hypre_VectorNumVectors(vector) = num_vectors_in;
   if (method == 0)
   {
      hypre_VectorVectorStride(vector) = size_in;
      hypre_VectorIndexStride(vector)  = 1;
   }
   else if (method == 1)
   {
      hypre_VectorVectorStride(vector) = 1;
      hypre_VectorIndexStride(vector)  = num_vectors;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorRead
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorRead( char *file_name )
{
   hypre_Vector  *vector;

   FILE    *fp;

   HYPRE_Complex *data;
   HYPRE_Int      size;

   HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   hypre_fscanf(fp, "%d", &size);

   vector = hypre_SeqVectorCreate(size);

   hypre_VectorMemoryLocation(vector) = HYPRE_MEMORY_HOST;

   hypre_SeqVectorInitialize(vector);

   data = hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      hypre_fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   /* multivector code not written yet */
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
   HYPRE_Int             num_vectors     = hypre_VectorNumVectors(vector);
   HYPRE_Int             vecstride       = hypre_VectorVectorStride(vector);
   HYPRE_Int             idxstride       = hypre_VectorIndexStride(vector);
   HYPRE_Int             size            = hypre_VectorSize(vector);
   HYPRE_MemoryLocation  memory_location = hypre_VectorMemoryLocation(vector);

   FILE                 *fp;

   hypre_Vector         *h_vector;
   HYPRE_Complex        *data;
   HYPRE_Int             i, j;
   HYPRE_Complex         value;

   /*----------------------------------------------------------
    * Print in the data
    *----------------------------------------------------------*/

   /* Create temporary vector on host memory if needed */
   h_vector = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
              hypre_SeqVectorCloneDeep_v2(vector, HYPRE_MEMORY_HOST) : vector;

   data = hypre_VectorData(h_vector);

   fp = fopen(file_name, "w");

   if (num_vectors == 1)
   {
      hypre_fprintf(fp, "%d\n", size);
   }
   else
   {
      hypre_fprintf(fp, "%d vectors of size %d\n", num_vectors, size );
   }

   if (num_vectors > 1)
   {
      for ( j = 0; j < num_vectors; ++j )
      {
         hypre_fprintf(fp, "vector %d\n", j );
         for (i = 0; i < size; i++)
         {
            value = data[j * vecstride + i * idxstride];
#ifdef HYPRE_COMPLEX
            hypre_fprintf(fp, "%.14e , %.14e\n",
                          hypre_creal(value), hypre_cimag(value));
#else
            hypre_fprintf(fp, "%.14e\n", value);
#endif
         }
      }
   }
   else
   {
      for (i = 0; i < size; i++)
      {
#ifdef HYPRE_COMPLEX
         hypre_fprintf(fp, "%.14e , %.14e\n",
                       hypre_creal(data[i]), hypre_cimag(data[i]));
#else
         hypre_fprintf(fp, "%.14e\n", data[i]);
#endif
      }
   }

   fclose(fp);

   /* Free temporary vector */
   if (h_vector != vector)
   {
      hypre_SeqVectorDestroy(h_vector);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValuesHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetConstantValuesHost( hypre_Vector *v,
                                      HYPRE_Complex value )
{
   HYPRE_Complex *vector_data = hypre_VectorData(v);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(v);
   HYPRE_Int      size        = hypre_VectorSize(v);
   HYPRE_Int      total_size  = size * num_vectors;
   HYPRE_Int      i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      vector_data[i] = value;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetConstantValues( hypre_Vector *v,
                                  HYPRE_Complex value )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int   num_vectors = hypre_VectorNumVectors(v);
   HYPRE_Int   size        = hypre_VectorSize(v);
   HYPRE_Int   total_size  = size * num_vectors;

   /* Trivial case */
   if (total_size <= 0)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_VectorMemoryLocation(v));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SeqVectorSetConstantValuesDevice(v, value);
   }
   else
#endif
   {
      hypre_SeqVectorSetConstantValuesHost(v, value);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetRandomValues
 *
 * returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorSetRandomValues( hypre_Vector *v,
                                HYPRE_Int     seed )
{
   HYPRE_Complex *vector_data = hypre_VectorData(v);
   HYPRE_Int      size        = hypre_VectorSize(v);
   HYPRE_Int      i;

   hypre_SeedRand(seed);
   size *= hypre_VectorNumVectors(v);

   if (hypre_GetActualMemLocation(hypre_VectorMemoryLocation(v)) == hypre_MEMORY_HOST)
   {
      /* RDF: threading this loop may cause problems because of hypre_Rand() */
      for (i = 0; i < size; i++)
      {
         vector_data[i] = 2.0 * hypre_Rand() - 1.0;
      }
   }
   else
   {
      HYPRE_Complex *h_data = hypre_TAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
      for (i = 0; i < size; i++)
      {
         h_data[i] = 2.0 * hypre_Rand() - 1.0;
      }
      hypre_TMemcpy(vector_data, h_data, HYPRE_Complex, size, hypre_VectorMemoryLocation(v),
                    HYPRE_MEMORY_HOST);
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopy
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorCopy( hypre_Vector *x,
                     hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   hypre_GpuProfilingPushRange("SeqVectorCopy");

   size_t size = hypre_min(hypre_VectorSize(x), hypre_VectorSize(y)) * hypre_VectorNumVectors(x);

   hypre_TMemcpy( hypre_VectorData(y),
                  hypre_VectorData(x),
                  HYPRE_Complex,
                  size,
                  hypre_VectorMemoryLocation(y),
                  hypre_VectorMemoryLocation(x) );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopyTags
 *
 * Copy tags array from x to y.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorCopyTags( hypre_Vector *x,
                         hypre_Vector *y )
{
   HYPRE_Int  size = hypre_min(hypre_VectorSize(x), hypre_VectorSize(y));

   /* Return if x does not have tags */
   if (!hypre_VectorTags(x))
   {
      return hypre_error_flag;
   }

   /* Deallocate existing tags if present */
   if (hypre_VectorTags(y))
   {
      hypre_TFree(hypre_VectorTags(y), hypre_VectorMemoryLocation(y));
   }

   /* Allocate new tags array */
   hypre_VectorTags(y) = hypre_TAlloc(HYPRE_Int, size, hypre_VectorMemoryLocation(y));

   /* Copy tags */
   hypre_TMemcpy(hypre_VectorTags(y),
                 hypre_VectorTags(x),
                 HYPRE_Int, size,
                 hypre_VectorMemoryLocation(y),
                 hypre_VectorMemoryLocation(x));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorStridedCopy
 *
 * Perform strided copy from a data array to x->data.
 *
 * We assume that the data array lives in the same memory location as x->data
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorStridedCopy( hypre_Vector  *x,
                            HYPRE_Int      istride,
                            HYPRE_Int      ostride,
                            HYPRE_Int      size,
                            HYPRE_Complex *data)
{
   HYPRE_Int        x_size = hypre_VectorSize(x);
   HYPRE_Complex   *x_data = hypre_VectorData(x);

   HYPRE_Int        i;

   /* Sanity checks */
   if (istride < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Input stride needs to be greater than zero!");
      return hypre_error_flag;
   }

   if (ostride < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Output stride needs to be greater than zero!");
      return hypre_error_flag;
   }

   if (x_size < (size / istride) * ostride)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not enough space in x!");
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_VectorMemoryLocation(x));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SeqVectorStridedCopyDevice(x, istride, ostride, size, data);
   }
   else
#endif
   {
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i += istride)
      {
         x_data[(i / istride) * ostride] = data[i];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneDeep_v2
 *--------------------------------------------------------------------------*/

hypre_Vector*
hypre_SeqVectorCloneDeep_v2( hypre_Vector *x, HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int      size          = hypre_VectorSize(x);
   HYPRE_Int      num_vectors   = hypre_VectorNumVectors(x);

   hypre_Vector  *y = hypre_SeqMultiVectorCreate(size, num_vectors);

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y)  = hypre_VectorIndexStride(x);

   hypre_SeqVectorSetNumTags(y, hypre_VectorNumTags(x));
   hypre_SeqVectorSetOwnsTags(y, 1);
   hypre_SeqVectorInitialize_v2(y, memory_location);
   hypre_SeqVectorCopy(x, y);
   hypre_SeqVectorCopyTags(x, y);

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneDeep
 *
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_Vector*
hypre_SeqVectorCloneDeep( hypre_Vector *x )
{
   return hypre_SeqVectorCloneDeep_v2(x, hypre_VectorMemoryLocation(x));
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneShallow
 *
 * Returns a complete copy of x - a shallow copy, pointing the data of x
 *--------------------------------------------------------------------------*/

hypre_Vector *
hypre_SeqVectorCloneShallow( hypre_Vector *x )
{
   HYPRE_Int     size         = hypre_VectorSize(x);
   HYPRE_Int     num_vectors  = hypre_VectorNumVectors(x);
   hypre_Vector *y            = hypre_SeqMultiVectorCreate(size, num_vectors);

   hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
   hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
   hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

   hypre_VectorMemoryLocation(y) = hypre_VectorMemoryLocation(x);

   hypre_VectorData(y) = hypre_VectorData(x);
   hypre_SeqVectorSetDataOwner(y, 0);
   hypre_SeqVectorInitialize(y);

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMigrate
 *
 * Migrates the vector data to memory_location.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorMigrate(hypre_Vector         *x,
                       HYPRE_MemoryLocation  memory_location )
{
   HYPRE_Complex       *data = hypre_VectorData(x);
   HYPRE_Int            size = hypre_VectorSize(x);
   HYPRE_Int            num_vectors = hypre_VectorNumVectors(x);
   HYPRE_MemoryLocation old_memory_location = hypre_VectorMemoryLocation(x);
   HYPRE_Int            total_size = size * num_vectors;

   /* Update x's memory location */
   hypre_VectorMemoryLocation(x) = memory_location;

   if ( hypre_GetActualMemLocation(memory_location) !=
        hypre_GetActualMemLocation(old_memory_location) )
   {
      if (data)
      {
         HYPRE_Complex *new_data;

         new_data = hypre_TAlloc(HYPRE_Complex, total_size, memory_location);
         hypre_TMemcpy(new_data, data, HYPRE_Complex, total_size,
                       memory_location, old_memory_location);
         hypre_VectorData(x) = new_data;
         hypre_VectorOwnsData(x) = 1;

         /* Free old data */
         hypre_TFree(data, old_memory_location);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScaleHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorScaleHost( HYPRE_Complex alpha,
                          hypre_Vector *y )
{
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(y);
   HYPRE_Int      size        = hypre_VectorSize(y);
   HYPRE_Int      total_size  = size * num_vectors;
   HYPRE_Int      i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      y_data[i] *= alpha;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorScale( HYPRE_Complex alpha,
                      hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   /* special cases */
   if (alpha == 1.0)
   {
      return hypre_error_flag;
   }

   if (alpha == 0.0)
   {
      return hypre_SeqVectorSetConstantValues(y, 0.0);
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_VectorMemoryLocation(y));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SeqVectorScaleDevice(alpha, y);
   }
   else
#endif
   {
      hypre_SeqVectorScaleHost(alpha, y);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpyHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpyHost( HYPRE_Complex alpha,
                         hypre_Vector *x,
                         hypre_Vector *y )
{
   HYPRE_Complex *x_data      = hypre_VectorData(x);
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int      size        = hypre_VectorSize(x);
   HYPRE_Int      total_size  = size * num_vectors;
   HYPRE_Int      i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      y_data[i] += alpha * x_data[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpy( HYPRE_Complex alpha,
                     hypre_Vector *x,
                     hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(y) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SeqVectorAxpyDevice(alpha, x, y);
   }
   else
#endif
   {
      hypre_SeqVectorAxpyHost(alpha, x, y);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpyzHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpyzHost( HYPRE_Complex alpha,
                          hypre_Vector *x,
                          HYPRE_Complex beta,
                          hypre_Vector *y,
                          hypre_Vector *z )
{
   HYPRE_Complex *x_data      = hypre_VectorData(x);
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Complex *z_data      = hypre_VectorData(z);

   HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int      size        = hypre_VectorSize(x);
   HYPRE_Int      total_size  = size * num_vectors;
   HYPRE_Int      i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      z_data[i] = alpha * x_data[i] + beta * y_data[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpyz
 *
 * Computes z = a*x + b*y
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpyz( HYPRE_Complex alpha,
                      hypre_Vector *x,
                      HYPRE_Complex beta,
                      hypre_Vector *y,
                      hypre_Vector *z )
{
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(y));
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SeqVectorAxpyzDevice(alpha, x, beta, y, z);
   }
   else
#endif
   {
      hypre_SeqVectorAxpyzHost(alpha, x, beta, y, z);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPointwiseDivpyHost
 *
 * if marker != NULL: only for marker[i] == marker_val
 *
 * TODO:
 *        1) Change to hypre_SeqVectorPointwiseDivpyMarkedHost?
 *        2) Add vecstride/idxstride variables
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseDivpyHost( hypre_Vector *x,
                                   hypre_Vector *b,
                                   hypre_Vector *y,
                                   HYPRE_Int    *marker,
                                   HYPRE_Int     marker_val )
{
   HYPRE_Complex   *x_data        = hypre_VectorData(x);
   HYPRE_Complex   *b_data        = hypre_VectorData(b);
   HYPRE_Complex   *y_data        = hypre_VectorData(y);
   HYPRE_Int        num_vectors_x = hypre_VectorNumVectors(x);
   HYPRE_Int        num_vectors_y = hypre_VectorNumVectors(y);
   HYPRE_Int        num_vectors_b = hypre_VectorNumVectors(b);
   HYPRE_Int        size          = hypre_VectorSize(b);
   HYPRE_Int        i, j;
   HYPRE_Complex    val;

   if (num_vectors_b == 1)
   {
      if (num_vectors_x == 1 &&
          num_vectors_y == 1)
      {
         if (marker)
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               if (marker[i] == marker_val)
               {
                  y_data[i] += x_data[i] / b_data[i];
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               y_data[i] += x_data[i] / b_data[i];
            }
         } /* if (marker) */
      }
      else if (num_vectors_x == 2 &&
               num_vectors_y == 2)
      {
         if (marker)
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];
               if (marker[i] == marker_val)
               {
                  y_data[i]        += x_data[i]        * val;
                  y_data[i + size] += x_data[i + size] * val;
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];

               y_data[i]        += x_data[i]        * val;
               y_data[i + size] += x_data[i + size] * val;
            }
         } /* if (marker) */
      }
      else if (num_vectors_x == num_vectors_y)
      {
         if (marker)
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i, j) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];
               if (marker[i] == marker_val)
               {
                  for (j = 0; j < num_vectors_x; j++)
                  {
                     y_data[i + size * j] += x_data[i + size * j] * val;
                  }
               }
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i, j) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];
               for (j = 0; j < num_vectors_x; j++)
               {
                  y_data[i + size * j] += x_data[i + size * j] * val;
               }
            }
         } /* if (marker) */
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
      }
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_vectors_b != 1 not supported!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPointwiseDivpyMarked
 *
 * Computes: y[i] = y[i] + x[i] / b[i] for marker[i] = marker_val
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseDivpyMarked( hypre_Vector *x,
                                     hypre_Vector *b,
                                     hypre_Vector *y,
                                     HYPRE_Int    *marker,
                                     HYPRE_Int     marker_val)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   /* Sanity checks */
   if (hypre_VectorSize(x) < hypre_VectorSize(b))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "sizes of x and b do not match!\n");
      return hypre_error_flag;
   }

   if (!hypre_VectorSize(x))
   {
      /* VPM: Do not throw an error message here since this can happen for idle processors */
      return hypre_error_flag;
   }

   if (!hypre_VectorData(x))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "x_data is not present!\n");
      return hypre_error_flag;
   }

   if (!hypre_VectorData(b))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "b_data is not present!\n");
      return hypre_error_flag;
   }

   if (!hypre_VectorData(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "y_data is not present!\n");
      return hypre_error_flag;
   }

   /* row-wise multivec is not supported */
   hypre_assert(hypre_VectorMultiVecStorageMethod(x) == 0);
   hypre_assert(hypre_VectorMultiVecStorageMethod(b) == 0);
   hypre_assert(hypre_VectorMultiVecStorageMethod(y) == 0);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(b) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SeqVectorPointwiseDivpyDevice(x, b, y, marker, marker_val);
   }
   else
#endif
   {
      hypre_SeqVectorPointwiseDivpyHost(x, b, y, marker, marker_val);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorPointwiseDivpy
 *
 * Computes: y = y + x ./ b
 *
 * Notes:
 *    1) x and b must have the same sizes
 *    2) x and y can have different sizes
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseDivpy( hypre_Vector *x,
                               hypre_Vector *b,
                               hypre_Vector *y )
{
   return hypre_SeqVectorPointwiseDivpyMarked(x, b, y, NULL, -1);
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProdHost
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_SeqVectorInnerProdHost( hypre_Vector *x,
                              hypre_Vector *y )
{
   HYPRE_Complex *x_data      = hypre_VectorData(x);
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int      size        = hypre_VectorSize(x);
   HYPRE_Int      total_size  = size * num_vectors;

   HYPRE_Real     result      = 0.0;
   HYPRE_Int      i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) reduction(+:result) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      result += hypre_conj(y_data[i]) * x_data[i];
   }

   return result;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_SeqVectorInnerProd( hypre_Vector *x,
                          hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Real result;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(y) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      result = hypre_SeqVectorInnerProdDevice(x, y);
   }
   else
#endif
   {
      result = hypre_SeqVectorInnerProdHost(x, y);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return result;
}

/*--------------------------------------------------------------------------
 * Computes the "marked" inner product of two vectors x and y.
 *
 *  - iprod[0]: inner product of full vector
 *  - iprod[i + 1]: inner product computed from entries marked with "i" tag
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorInnerProdTaggedHost( hypre_Vector  *x,
                                    hypre_Vector  *y,
                                    HYPRE_Complex *iprod )
{
   HYPRE_Complex *x_data      = hypre_VectorData(x);
   HYPRE_Complex *y_data      = hypre_VectorData(y);
   HYPRE_Int     *tags        = hypre_VectorTags(x);
   HYPRE_Int      num_tags    = hypre_VectorNumTags(x);
   HYPRE_Int      num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int      size        = hypre_VectorSize(x);
   HYPRE_Int      total_size  = size * num_vectors;
   HYPRE_Int      i;

#if defined(HYPRE_USING_OPENMP)
   HYPRE_Int      j, num_threads = hypre_NumThreads();
   HYPRE_Complex *thread_sums    = hypre_CTAlloc(HYPRE_Complex,
                                                 num_threads * num_tags,
                                                 HYPRE_MEMORY_HOST);
#endif

   /* Initialize result */
   for (i = 0; i < num_tags + 1; i++)
   {
      iprod[i] = 0.0;
   }

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel private(i, j)
   {
      HYPRE_Int      tid = hypre_GetThreadNum();
      HYPRE_Complex *sum = &thread_sums[tid * num_tags];

      #pragma omp for HYPRE_SMP_SCHEDULE
      for (i = 0; i < total_size; i++)
      {
         sum[tags[i] + 1] += hypre_conj(y_data[i]) * x_data[i];
      }

      #pragma omp critical
      {
         for (j = 0; j < num_tags; j++)
         {
            iprod[j + 1] += sum[j + 1];
         }
      }
   }
#else
   for (i = 0; i < total_size; i++)
   {
      iprod[tags[i] + 1] += hypre_conj(y_data[i]) * x_data[i];
   }
#endif

   /* Compute inner product of the full vectors */
   for (i = 0; i < num_tags; i++)
   {
      iprod[0] += iprod[i + 1];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProdTagged
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorInnerProdTagged( hypre_Vector  *x,
                                hypre_Vector  *y,
                                HYPRE_Complex *iprod )
{
   HYPRE_Int      num_tags_x = hypre_VectorNumTags(x);
   HYPRE_Int      num_tags_y = hypre_VectorNumTags(y);

   /* Sanity checks */
   if (!iprod)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "iprod not allocated!");
      return hypre_error_flag;
   }

   if (num_tags_x != num_tags_y)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Number of tags in x and y don't match!");
      return hypre_error_flag;
   }

   if (num_tags_x == 1)
   {
      iprod[0] = hypre_SeqVectorInnerProd(x, y);
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(y) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      /* TODO (VPM): add hypre_SeqVectorInnerProdTaggedDevice */
      hypre_Vector *x_h = hypre_SeqVectorCloneDeep_v2(x, HYPRE_MEMORY_HOST);
      hypre_Vector *y_h = hypre_SeqVectorCloneDeep_v2(y, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInnerProdTaggedHost(x_h, y_h, iprod);
      hypre_SeqVectorCopy(x_h, x);
      hypre_SeqVectorCopy(y_h, y);
      hypre_SeqVectorDestroy(x_h);
      hypre_SeqVectorDestroy(y_h);
   }
   else
#endif
   {
      hypre_SeqVectorInnerProdTaggedHost(x, y, iprod);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSumEltsHost
 *--------------------------------------------------------------------------*/

HYPRE_Complex
hypre_SeqVectorSumEltsHost( hypre_Vector *vector )
{
   HYPRE_Complex  *data        = hypre_VectorData( vector );
   HYPRE_Int       num_vectors = hypre_VectorNumVectors(vector);
   HYPRE_Int       size        = hypre_VectorSize(vector);
   HYPRE_Int       total_size  = size * num_vectors;

   HYPRE_Complex   sum  = 0;
   HYPRE_Int       i;

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:sum) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      sum += data[i];
   }

   return sum;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSumElts:
 *
 * Returns the sum of all vector elements.
 *--------------------------------------------------------------------------*/

HYPRE_Complex
hypre_SeqVectorSumElts( hypre_Vector *v )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

   HYPRE_Complex sum;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_VectorMemoryLocation(v));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      sum = hypre_SeqVectorSumEltsDevice(v);
   }
   else
#endif
   {
      sum = hypre_SeqVectorSumEltsHost(v);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

   return sum;
}

/*--------------------------------------------------------------------------
 * See hypre_SeqVectorPointwiseProduct
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseProductHost( hypre_Vector *x,
                                     hypre_Vector *y,
                                     hypre_Vector *z )
{
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Complex *z_data = hypre_VectorData(z);
   HYPRE_Int      i;

   /* Element-wise multiplication z[i] = y[i] * x[i] */
#if defined (HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i)
#endif
   for (i = 0; i < size; i++)
   {
      z_data[i] = y_data[i] * x_data[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * See hypre_SeqVectorPointwiseDivision
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseDivisionHost( hypre_Vector *x,
                                      hypre_Vector *y,
                                      hypre_Vector *z )
{
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Complex *z_data = hypre_VectorData(z);
   HYPRE_Int      i;

   /* Element-wise division z[i] = y[i] / x[i] */
#if defined (HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i)
#endif
   for (i = 0; i < size; i++)
   {
      hypre_assert(x_data[i] != 0.0);

      z_data[i] = y_data[i] / x_data[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * See hypre_SeqVectorPointwiseInverse
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseInverseHost( hypre_Vector *x,
                                     hypre_Vector *y )
{
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Int      i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i)
#endif
   for (i = 0; i < size; i++)
   {
      hypre_assert(x_data[i] != 0.0);
      y_data[i] = 1.0 / x_data[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the element-wise product of two vectors: z[i] = x[i] * y[i]
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseProduct( hypre_Vector  *x,
                                 hypre_Vector  *y,
                                 hypre_Vector **z_ptr )
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Complex  x_size = hypre_VectorSize(x);
   HYPRE_Complex  y_size = hypre_VectorSize(y);
   HYPRE_Complex  z_size = hypre_VectorSize(*z_ptr);

   /* Check if vectors are initialized */
   if ((!x_data && !x_size) || (!y_data && !y_size))
   {
      return hypre_error_flag;
   }

   /* Check if vectors are initialized */
   if ((!x_data && x_size) || (!y_data && y_size))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Vectors have different sizes!");
      return hypre_error_flag;
   }

   /* Check if vectors have same size */
   if (y_size != x_size || (*z_ptr && y_size != z_size))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Vectors have different sizes!");
      return hypre_error_flag;
   }

   /* Create and initialize z if it doesn't exist, or resize it */
   if (!*z_ptr)
   {
      *z_ptr = hypre_SeqVectorCreate(hypre_VectorSize(x));
      hypre_SeqVectorInitialize_v2(*z_ptr, hypre_VectorMemoryLocation(x));
   }
   else if (*z_ptr != y)
   {
      /* No-op if z has the same size as x */
      hypre_SeqVectorResize(*z_ptr, hypre_VectorSize(x), hypre_VectorNumVectors(x));
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(y) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      return hypre_SeqVectorPointwiseProductDevice(x, y, *z_ptr);
   }
   else
#endif
   {
      return hypre_SeqVectorPointwiseProductHost(x, y, *z_ptr);
   }
}

/*--------------------------------------------------------------------------
 * Computes the element-wise division of two vectors: z[i] = y[i] / x[i]
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseDivision( hypre_Vector  *x,
                                  hypre_Vector  *y,
                                  hypre_Vector **z_ptr )
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);

   /* Check if vectors are initialized */
   if (!x_data || !y_data)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Vectors are not initialized!");
      return hypre_error_flag;
   }

   /* Check if vectors have same size */
   if (hypre_VectorSize(y) != hypre_VectorSize(x) ||
       (*z_ptr && hypre_VectorSize(y) != hypre_VectorSize(*z_ptr)))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Vectors have different sizes!");
      return hypre_error_flag;
   }

   /* Create and initialize z if it doesn't exist, or resize it */
   if (!*z_ptr)
   {
      *z_ptr = hypre_SeqVectorCreate(hypre_VectorSize(x));
      hypre_SeqVectorInitialize_v2(*z_ptr, hypre_VectorMemoryLocation(x));
   }
   else if (*z_ptr != y)
   {
      /* No-op if z has the same size as x */
      hypre_SeqVectorResize(*z_ptr, hypre_VectorSize(x), hypre_VectorNumVectors(x));
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_VectorMemoryLocation(x),
                                                      hypre_VectorMemoryLocation(y) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      return hypre_SeqVectorPointwiseDivisionDevice(x, y, *z_ptr);
   }
   else
#endif
   {
      return hypre_SeqVectorPointwiseDivisionHost(x, y, *z_ptr);
   }
}

/*--------------------------------------------------------------------------
 * Computes the element-wise inverse of a vector: y[i] = 1.0 / x[i]
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorPointwiseInverse( hypre_Vector  *x,
                                 hypre_Vector **y_ptr )
{
   HYPRE_Complex *x_data = hypre_VectorData(x);

   /* Check if vector is initialized */
   if (!x_data)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Vector is not initialized!");
      return hypre_error_flag;
   }

   /* Create and initialize z if it doesn't exist, or resize it */
   if (!*y_ptr)
   {
      *y_ptr = hypre_SeqVectorCreate(hypre_VectorSize(x));
      hypre_SeqVectorInitialize_v2(*y_ptr, hypre_VectorMemoryLocation(x));
   }
   else if (*y_ptr != x)
   {
      /* No-op if y has the same size as x */
      hypre_SeqVectorResize(*y_ptr, hypre_VectorSize(x), hypre_VectorNumVectors(x));
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_VectorMemoryLocation(x));
   if (exec == HYPRE_EXEC_DEVICE)
   {
      return hypre_SeqVectorPointwiseInverseDevice(x, *y_ptr);
   }
   else
#endif
   {
      return hypre_SeqVectorPointwiseInverseHost(x, *y_ptr);
   }
}
