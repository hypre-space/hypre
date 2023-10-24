/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_hypre_onedpl.hpp"
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
   HYPRE_BigInt    first_index  = hypre_ParVectorFirstIndex(vector);
   HYPRE_BigInt    last_index   = hypre_ParVectorLastIndex(vector);
   hypre_Vector   *local_vector = hypre_ParVectorLocalVector(vector);

   HYPRE_Int       component    = hypre_VectorComponent(local_vector);
   HYPRE_Int       vecstride    = hypre_VectorVectorStride(local_vector);
   HYPRE_Int       idxstride    = hypre_VectorIndexStride(local_vector);
   HYPRE_Complex  *data         = hypre_VectorData(local_vector);
   HYPRE_Int       vecoffset    = component * vecstride;

   HYPRE_Int       ierr = 0;

   if (idxstride != 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "hypre_ParVectorGetValuesDevice not implemented for non-columnwise vector storage\n");
      return hypre_error_flag;
   }

   /* If indices == NULL, assume that num_values components
      are to be retrieved from block starting at vec_start */
   if (indices)
   {
#if defined(HYPRE_USING_SYCL)
      ierr = HYPRE_ONEDPL_CALL( std::count_if,
                                indices,
                                indices + num_values,
                                out_of_range<HYPRE_BigInt>(first_index + base, last_index + base) );
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

#if defined(HYPRE_USING_SYCL)
         /* /1* WM: todo - why can't I combine transform iterator and gather? *1/ */
         /* HYPRE_ONEDPL_CALL( std::transform, */
         /*                    indices, */
         /*                    indices + num_values, */
         /*                    indices, */
         /*                    [base, first_index] (const auto & x) {return x - base - first_index;} ); */
         /* hypreSycl_gather_if( indices, */
         /*                      indices+ num_values, */
         /*                      indices, */
         /*                      data + vecoffset, */
         /*                      values, */
         /*                      in_range<HYPRE_BigInt>(first_index + base, last_index + base) ); */
         /* } */
         /* else */
         /* { */
         /* /1* WM: todo - why can't I combine transform iterator and gather? *1/ */
         /* HYPRE_ONEDPL_CALL( std::transform, */
         /*                    indices, */
         /*                    indices + num_values, */
         /*                    indices, */
         /*                    [base, first_index] (const auto & x) {return x - base - first_index;} ); */
         /* hypreSycl_gather( indices, */
         /*                   indices+ num_values, */
         /*                   data + vecoffset, */
         /*                   values); */
         auto trans_it = oneapi::dpl::make_transform_iterator(indices, [base,
         first_index] (const auto & x) {return x - base - first_index;} );
         hypreSycl_gather_if( trans_it,
                              trans_it + num_values,
                              indices,
                              data + vecoffset,
                              values,
                              in_range<HYPRE_BigInt>(first_index + base, last_index + base) );
      }
      else
      {
         auto trans_it = oneapi::dpl::make_transform_iterator(indices, [base,
         first_index] (const auto & x) {return x - base - first_index;} );
         hypreSycl_gather( trans_it,
                           trans_it + num_values,
                           data + vecoffset,
                           values);
#else
         HYPRE_THRUST_CALL( gather_if,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values,
                            indices,
                            data + vecoffset,
                            values,
                            in_range<HYPRE_BigInt>(first_index + base, last_index + base) );
      }
      else
      {
         HYPRE_THRUST_CALL( gather,
                            thrust::make_transform_iterator(indices, _1 - base - first_index),
                            thrust::make_transform_iterator(indices, _1 - base - first_index) + num_values,
                            data + vecoffset,
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

      hypre_TMemcpy(values, data + vecoffset, HYPRE_Complex, num_values,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)
