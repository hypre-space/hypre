/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

HYPRE_Int
hypre_ParVectorGetValuesDevice(hypre_ParVector *vector,
                               HYPRE_Int        num_values,
                               HYPRE_BigInt    *indices,
                               HYPRE_BigInt     base,
                               HYPRE_Complex   *values)
{
   HYPRE_Int     ierr = 0;
   HYPRE_BigInt  first_index = hypre_ParVectorFirstIndex(vector);
   HYPRE_BigInt  last_index = hypre_ParVectorLastIndex(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(vector);
   HYPRE_Complex *data = hypre_VectorData(local_vector);

   /* If indices == NULL, assume that num_values components
      are to be retrieved from block starting at vec_start */
   if (indices)
   {
      ierr = HYPRE_THRUST_CALL( count_if,
                                indices,
                                indices + num_values,
                                out_of_range<HYPRE_BigInt>(first_index + base, last_index + base) );
      if (ierr)
      {
         hypre_error_in_arg(3);
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Index out of range! -- hypre_ParVectorGetValues.");
         hypre_printf(" error: %d indices out of range! -- hypre_ParVectorGetValues\n", ierr);

         HYPRE_THRUST_CALL( gather_if,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values,
                            indices,
                            data,
                            values,
                            in_range<HYPRE_BigInt>(first_index + base, last_index + base) );
      }
      else
      {
         HYPRE_THRUST_CALL( gather,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values,
                            data,
                            values);
      }
   }
   else
   {
      if (num_values > hypre_VectorSize(local_vector))
      {
         hypre_error_in_arg(3);
         return hypre_error_flag;
      }

      hypre_TMemcpy(values, data, HYPRE_Complex, num_values, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
