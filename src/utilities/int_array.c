/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/******************************************************************************
 *
 * Routines for hypre_IntArray struct for holding an array of integers
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_IntArrayCreate
 *--------------------------------------------------------------------------*/

hypre_IntArray *
hypre_IntArrayCreate( HYPRE_Int size )
{
   hypre_IntArray  *array;

   array = hypre_CTAlloc(hypre_IntArray, 1, HYPRE_MEMORY_HOST);

   hypre_IntArrayData(array) = NULL;
   hypre_IntArraySize(array) = size;

   hypre_IntArrayMemoryLocation(array) = hypre_HandleMemoryLocation(hypre_handle());

   return array;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayDestroy( hypre_IntArray *array )
{
   if (array)
   {
      HYPRE_MemoryLocation memory_location = hypre_IntArrayMemoryLocation(array);

      hypre_TFree(hypre_IntArrayData(array), memory_location);

      hypre_TFree(array, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayInitialize_v2( hypre_IntArray *array, HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int  size = hypre_IntArraySize(array);

   hypre_IntArrayMemoryLocation(array) = memory_location;

   /* Caveat: for pre-existing data, the memory location must be guaranteed
    * to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered
    * when being used, and freed */
   if (!hypre_IntArrayData(array))
   {
      hypre_IntArrayData(array) = hypre_CTAlloc(HYPRE_Int, size, memory_location);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayInitialize( hypre_IntArray *array )
{
   hypre_IntArrayInitialize_v2( array, hypre_IntArrayMemoryLocation(array) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayCopy
 *
 * Copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayCopy( hypre_IntArray *x,
                    hypre_IntArray *y )
{
   size_t size = hypre_min( hypre_IntArraySize(x), hypre_IntArraySize(y) );

   hypre_TMemcpy( hypre_IntArrayData(y),
                  hypre_IntArrayData(x),
                  HYPRE_Int,
                  size,
                  hypre_IntArrayMemoryLocation(y),
                  hypre_IntArrayMemoryLocation(x) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayCloneDeep_v2
 *--------------------------------------------------------------------------*/

hypre_IntArray *
hypre_IntArrayCloneDeep_v2( hypre_IntArray *x, HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int    size = hypre_IntArraySize(x);

   hypre_IntArray *y = hypre_IntArrayCreate( size );

   hypre_IntArrayInitialize_v2(y, memory_location);
   hypre_IntArrayCopy( x, y );

   return y;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_IntArray *
hypre_IntArrayCloneDeep( hypre_IntArray *x )
{
   return hypre_IntArrayCloneDeep_v2(x, hypre_IntArrayMemoryLocation(x));
}

/*--------------------------------------------------------------------------
 * hypre_IntArraySetConstantValuesHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArraySetConstantValuesHost( hypre_IntArray *v,
                                     HYPRE_Int       value )
{
   HYPRE_Int *array_data = hypre_IntArrayData(v);
   HYPRE_Int  size       = hypre_IntArraySize(v);
   HYPRE_Int  i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      array_data[i] = value;
   }

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_IntArraySetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArraySetConstantValues( hypre_IntArray *v,
                                 HYPRE_Int       value )
{
   if (hypre_IntArraySize(v) <= 0)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_IntArrayMemoryLocation(v));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IntArraySetConstantValuesDevice(v, value);
   }
   else
#endif
   {
      hypre_IntArraySetConstantValuesHost(v, value);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayCountHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayCountHost( hypre_IntArray *v,
                         HYPRE_Int       value,
                         HYPRE_Int      *num_values_ptr )
{
   HYPRE_Int  *array_data  = hypre_IntArrayData(v);
   HYPRE_Int   size        = hypre_IntArraySize(v);
   HYPRE_Int   num_values  = 0;
   HYPRE_Int   i;

#if !defined(_MSC_VER) && defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) reduction(+:num_values) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      num_values += (array_data[i] == value) ? 1 : 0;
   }

   *num_values_ptr = num_values;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayCount
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayCount( hypre_IntArray *v,
                     HYPRE_Int       value,
                     HYPRE_Int      *num_values_ptr )
{
   if (hypre_IntArraySize(v) <= 0)
   {
      *num_values_ptr = 0;
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_IntArrayMemoryLocation(v));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IntArrayCountDevice(v, value, num_values_ptr);
   }
   else
#endif
   {
      hypre_IntArrayCountHost(v, value, num_values_ptr);
   }

   return hypre_error_flag;
}
