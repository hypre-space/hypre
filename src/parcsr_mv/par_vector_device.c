/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

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
#ifdef HYPRE_USING_SYCL
      ierr = HYPRE_ONEDPL_CALL( std::count_if,
                                indices,
                                indices + num_values,
                                [low = first_index + base, high = last_index + base] (const auto & x) -> bool {return (x < low || x > high);} );
#else
      ierr = HYPRE_THRUST_CALL( count_if,
                                indices,
                                indices + num_values,
                                out_of_range<HYPRE_BigInt>(first_index + base, last_index + base) );
#endif

      if (ierr)
      {
         hypre_error_in_arg(3);
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Index out of range! -- hypre_ParVectorGetValues.");
         hypre_printf(" error: %d indices out of range! -- hypre_ParVectorGetValues\n", ierr);

#ifdef HYPRE_USING_SYCL
         auto pred = in_range<HYPRE_BigInt>(first_index + base, last_index + base);
         auto map_first  = oneapi::dpl::make_transform_iterator(indices, [ = ](auto _1) {return _1 - base - first_index;});
         auto perm_begin = oneapi::dpl::make_permutation_iterator(data, map_first);
         hypreSycl_transform_if( perm_begin, perm_begin + num_values, indices, values,
         [ = ](auto &&v) { return v; },
         [ = ](auto &&m) { return pred(m); } );
#else
         HYPRE_THRUST_CALL( gather_if,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),              /* map_first   */
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values, /* map_last    */
                            indices,                                                                        /* mask        */
                            data,                                                                           /* input_first */
                            values,                                                                         /* result      */
                            in_range<HYPRE_BigInt>(first_index + base, last_index + base) );                /* pred        */
#endif
      }
      else
      {
#ifdef HYPRE_USING_SYCL
         auto map_first = oneapi::dpl::make_transform_iterator(indices, [ = ](auto _1) {return _1 - base - first_index;});
         hypreSycl_gather(map_first,
                          map_first + num_values,
                          data,
                          values);
#else
         HYPRE_THRUST_CALL( gather,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values,
                            data,
                            values);
#endif
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

#endif // #if defined(HYPRE_USING_GPU)
