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
 * hypre_IntArrayMigrate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayMigrate( hypre_IntArray      *v,
                       HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int            size                = hypre_IntArraySize(v);
   HYPRE_Int           *v_data              = hypre_IntArrayData(v);
   HYPRE_MemoryLocation old_memory_location = hypre_IntArrayMemoryLocation(v);

   HYPRE_Int           *w_data;

   /* Update v's memory location */
   hypre_IntArrayMemoryLocation(v) = memory_location;

   if ( hypre_GetActualMemLocation(memory_location) !=
        hypre_GetActualMemLocation(old_memory_location) )
   {
      w_data = hypre_TAlloc(HYPRE_Int, size, memory_location);
      hypre_TMemcpy(w_data, v_data, HYPRE_Int, size,
                    memory_location, old_memory_location);
      hypre_TFree(v_data, old_memory_location);
      hypre_IntArrayData(v) = w_data;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayPrint( MPI_Comm        comm,
                     hypre_IntArray *array,
                     const char     *filename )
{
   HYPRE_Int             size            = hypre_IntArraySize(array);
   HYPRE_MemoryLocation  memory_location = hypre_IntArrayMemoryLocation(array);

   hypre_IntArray       *h_array;
   HYPRE_Int            *data;

   FILE                 *file;
   HYPRE_Int             i, myid;
   char                  new_filename[1024];

   hypre_MPI_Comm_rank(comm, &myid);

   /* Move data to host if needed*/
   h_array = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
             hypre_IntArrayCloneDeep_v2(array, HYPRE_MEMORY_HOST) : array;
   data = hypre_IntArrayData(h_array);

   /* Open file */
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open output file\n");
      return hypre_error_flag;
   }

   /* Print to file */
   hypre_fprintf(file, "%d\n", size);
   for (i = 0; i < size; i++)
   {
      hypre_fprintf(file, "%d\n", data[i]);
   }
   fclose(file);

   /* Free memory */
   if (h_array != array)
   {
      hypre_IntArrayDestroy(h_array);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayRead
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayRead( MPI_Comm         comm,
                    const char      *filename,
                    hypre_IntArray **array_ptr )
{
   hypre_IntArray       *array;
   HYPRE_Int             size;
   FILE                 *file;
   HYPRE_Int             i, myid;
   char                  new_filename[1024];

   hypre_MPI_Comm_rank(comm, &myid);

   /* Open file */
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open input file\n");
      return hypre_error_flag;
   }

   /* Read array size from file */
   hypre_fscanf(file, "%d\n", &size);

   /* Create IntArray on the host */
   array = hypre_IntArrayCreate(size);
   hypre_IntArrayInitialize_v2(array, HYPRE_MEMORY_HOST);

   /* Read array values from file */
   for (i = 0; i < size; i++)
   {
      hypre_fscanf(file, "%d\n", &hypre_IntArrayData(array)[i]);
   }
   fclose(file);

   /* Migrate to final memory location */
   hypre_IntArrayMigrate(array, hypre_HandleMemoryLocation(hypre_handle()));

   /* Set output pointer */
   *array_ptr = array;

   return hypre_error_flag;
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

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
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
 * hypre_IntArraySetInterleavedValuesHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArraySetInterleavedValuesHost( hypre_IntArray *v,
                                        HYPRE_Int       cycle )
{
   HYPRE_Int *array_data = hypre_IntArrayData(v);
   HYPRE_Int  size       = hypre_IntArraySize(v);
   HYPRE_Int  i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      array_data[i] = i % cycle;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArraySetInterleavedValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArraySetInterleavedValues( hypre_IntArray *v,
                                    HYPRE_Int       cycle )
{
   if (cycle < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid cycle value!");
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_IntArrayMemoryLocation(v));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IntArraySetInterleavedValuesDevice(v, cycle);
   }
   else
#endif
   {
      hypre_IntArraySetInterleavedValuesHost(v, cycle);
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

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
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

/*--------------------------------------------------------------------------
 * hypre_IntArrayInverseMappingHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayInverseMappingHost( hypre_IntArray  *v,
                                  hypre_IntArray  *w )
{
   HYPRE_Int   size    = hypre_IntArraySize(v);
   HYPRE_Int  *v_data  = hypre_IntArrayData(v);
   HYPRE_Int  *w_data  = hypre_IntArrayData(w);

   HYPRE_Int   i;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      w_data[v_data[i]] = i;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayInverseMapping
 *
 * Compute the reverse mapping (w) given an input array (v)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayInverseMapping( hypre_IntArray  *v,
                              hypre_IntArray **w_ptr )
{
   HYPRE_Int             size = hypre_IntArraySize(v);
   HYPRE_MemoryLocation  memory_location = hypre_IntArrayMemoryLocation(v);
   hypre_IntArray       *w;

   /* Create and initialize output array */
   w = hypre_IntArrayCreate(size);
   hypre_IntArrayInitialize_v2(w, memory_location);

   /* Exit if array has no elements */
   if (hypre_IntArraySize(w) <= 0)
   {
      *w_ptr = w;

      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IntArrayInverseMappingDevice(v, w);
   }
   else
#endif
   {
      hypre_IntArrayInverseMappingHost(v, w);
   }

   /* Set output pointer */
   *w_ptr = w;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayNegate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayNegate( hypre_IntArray *v )
{
   HYPRE_Int  *array_data  = hypre_IntArrayData(v);
   HYPRE_Int   size        = hypre_IntArraySize(v);
   HYPRE_Int   i;

   if (size <= 0)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_IntArrayMemoryLocation(v));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IntArrayNegateDevice(v);
   }
   else
#endif
   {
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         array_data[i] = - array_data[i];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArraySeparateByValue
 *
 * This function separates the indices of an array "v" based on
 * specified values.
 *
 * Input parameters:
 *   - num_values: number of unique values found in "v"
 *   - values: array of size "num_values" containing a list of unique values
 *   - sizes: array of size "num_values" containing the number of occurrences
 *            in "v" of each value from "values".
 *   - v: object of type hypre_IntArray containing values for categorization.
 *
 * Output parameter:
 *   - w: object of type hypre_IntArrayArray containing pointers to "num_values"
 *        arrays. Each array contains the set of indices in "v" that map to a
 *        particular value of the array "values".
 *
 * Example:
 *   Consider the following:
 *   - v = {-1, -1, 1, -1, -1, 1, 1, -1}
 *   - values = {-1, 1}
 *   - num_values = 2
 *
 *   This function computes:
 *   - w[0] = {0, 1, 3, 4, 7}
 *   - w[1] = {2, 5, 6}
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArraySeparateByValue( HYPRE_Int             num_values,
                               HYPRE_Int            *values,
                               HYPRE_Int            *sizes,
                               hypre_IntArray       *v,
                               hypre_IntArrayArray **w_ptr )
{
   hypre_IntArrayArray *w;

   HYPRE_Int           *v_data = hypre_IntArrayData(v);
   HYPRE_Int            v_size = hypre_IntArraySize(v);
   HYPRE_Int            i, k, val;
   HYPRE_Int           *count;

   /* Create output array */
   w = hypre_IntArrayArrayCreate(num_values, sizes);
   hypre_IntArrayArrayInitializeIn(w, hypre_IntArrayMemoryLocation(v));

   /* Fill arrays */
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_IntArrayMemoryLocation(v));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_IntArraySeparateByValueDevice(num_values, values, sizes, v, w);
   }
   else
#endif
   {
      count = hypre_CTAlloc(HYPRE_Int, num_values, HYPRE_MEMORY_HOST);
      for (k = 0; k < v_size; k++)
      {
         val = v_data[k];

         /* Find which entry "val" belongs to */
         for (i = 0; i < num_values; i++)
         {
            if (values[i] == val)
            {
               hypre_IntArrayArrayEntryIDataJ(w, i, count[i]++) = k;
               break;
            }
         }
      }
      hypre_TFree(count, HYPRE_MEMORY_HOST);
   }

   /* Set output pointer */
   *w_ptr = w;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * Routines for hypre_IntArrayArray struct
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_IntArrayArrayCreate
 *--------------------------------------------------------------------------*/

hypre_IntArrayArray *
hypre_IntArrayArrayCreate( HYPRE_Int  num_entries,
                           HYPRE_Int *sizes )
{
   hypre_IntArrayArray  *w;
   HYPRE_Int             i;

   w = hypre_CTAlloc(hypre_IntArrayArray, 1, HYPRE_MEMORY_HOST);

   hypre_IntArrayArraySize(w)    = num_entries;
   hypre_IntArrayArrayEntries(w) = hypre_TAlloc(hypre_IntArray*, num_entries, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_entries; i++)
   {
      hypre_IntArrayArrayEntryI(w, i) = hypre_IntArrayCreate(sizes[i]);
   }

   return w;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayArrayDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayArrayDestroy( hypre_IntArrayArray *w )
{
   HYPRE_Int  i;

   if (w)
   {
      for (i = 0; i < hypre_IntArrayArraySize(w); i++)
      {
         hypre_IntArrayDestroy(hypre_IntArrayArrayEntryI(w, i));
      }
      hypre_TFree(hypre_IntArrayArrayEntries(w), HYPRE_MEMORY_HOST);
      hypre_TFree(w, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayArrayInitializeIn
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayArrayInitializeIn( hypre_IntArrayArray  *w,
                                 HYPRE_MemoryLocation  memory_location )
{
   HYPRE_Int  i;

   for (i = 0; i < hypre_IntArrayArraySize(w); i++)
   {
      hypre_IntArrayInitialize_v2(hypre_IntArrayArrayEntryI(w, i), memory_location);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayArrayInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayArrayInitialize( hypre_IntArrayArray *w )
{
   hypre_IntArray  *v = hypre_IntArrayArrayEntryI(w, 0);
   HYPRE_Int        i;

   for (i = 0; i < hypre_IntArrayArraySize(w); i++)
   {
      hypre_IntArrayInitialize_v2(v, hypre_IntArrayMemoryLocation(v));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayArrayMigrate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayArrayMigrate( hypre_IntArrayArray  *w,
                            HYPRE_MemoryLocation  memory_location )
{
   HYPRE_Int i;

   for (i = 0; i < hypre_IntArrayArraySize(w); i++)
   {
      hypre_IntArrayMigrate(hypre_IntArrayArrayEntryI(w, i), memory_location);
   }

   return hypre_error_flag;
}
