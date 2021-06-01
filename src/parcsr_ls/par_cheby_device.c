/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Chebyshev setup and solve Device
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "float.h"

#if defined(HYPRE_USING_CUDA)
#include "_hypre_utilities.hpp"


#if defined(HYPRE_USING_CUDA)


/**
 * @brief Grabs the diagonal, takes the square root of the reciporacl and save
 *
 * Performs
 * o = u
 * u = r * a
 * For vectors o and u, with scalar a
 */
template<typename T1, typename T2>
struct grab_diagonal
{
   const T1* rowptrs;
   const T2* vals;

   grab_diagonal(T1* _rowptrs, T2* _vals): rowptrs(_rowptrs), vals(_vals) {}

   __host__ __device__ T2 operator()(T1 v) 
   {
      return 1/sqrt(vals[rowptrs[v]]);
   }
};
#endif


HYPRE_Int hypre_ParCSRRelax_Cheby_SolveDevice(hypre_ParCSRMatrix *A, /* matrix to relax with */
                            hypre_ParVector *f,    /* right-hand side */
                            HYPRE_Real *ds_data,
                            HYPRE_Real *coefs,
                            HYPRE_Int order,            /* polynomial order */
                            HYPRE_Int scale,            /* scale by diagonal?*/
                            HYPRE_Int variant,           
                            hypre_ParVector *u,   /* initial/updated approximation */
                            hypre_ParVector *v    /* temporary vector */,
                            hypre_ParVector *r    /*another temp vector */  )
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real *v_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   HYPRE_Real  *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   HYPRE_Int i, j;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
 
   HYPRE_Real mult;
   HYPRE_Real *orig_u;
   
   HYPRE_Int cheby_order;

   HYPRE_Real  *tmp_data;

   hypre_ParVector    *tmp_vec;

   /* u = u + p(A)r */

   if (order > 4)
      order = 4;
   if (order < 1)
      order = 1;

   /* we are using the order of p(A) */
   cheby_order = order -1;
   
   orig_u = hypre_CTAlloc(HYPRE_Real,  num_rows, HYPRE_MEMORY_DEVICE);

   if (!scale)
   {
      /* get residual: r = f - A*u */
      hypre_ParVectorCopy(f, r); 
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      /* o = u; u = r .* coef */
      HYPRE_THRUST_CALL(for_each,
      thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
      thrust::make_zip_iterator(thrust::make_tuple(orig_u + num_rows, u_data + num_rows, r_data + num_rows)),
      save_and_scale<HYPRE_Real>(coefs[cheby_order]));

      for (i = cheby_order - 1; i >= 0; i-- ) 
      {
         hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, v);
         mult = coefs[i];
         /* u = mult * r + v */

         HYPRE_THRUST_CALL(for_each,
         thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, v_data)),
         thrust::make_zip_iterator(thrust::make_tuple(u_data + num_rows, r_data + num_rows, v_data + num_rows)),
         oop_axpy<HYPRE_Real>(mult));
      }

      /* u = o + u */
      HYPRE_THRUST_CALL(transform,
      orig_u, u_data,
      orig_u + num_rows, u_data + num_rows,
      xpy<HYPRE_Real>(mult));
   }
   else /* scaling! */
   {
      
      /*grab 1/sqrt(diagonal) */
      
      tmp_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                      hypre_ParCSRMatrixGlobalNumRows(A),
                                      hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize_v2(tmp_vec, HYPRE_MEMORY_DEVICE);
      hypre_ParVectorSetPartitioningOwner(tmp_vec,0);
      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

    /* get ds_data and get scaled residual: r = D^(-1/2)f -
       * D^(-1/2)A*u */

      hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);
      /* r = ds .* (f + tmp) */

      HYPRE_THRUST_CALL(for_each,
      thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data)),
      thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data )) + num_rows,
      oop_xypz<HYPRE_Real>());

      /* save original u, then start 
         the iteration by multiplying r by the cheby coef.*/

      /* o = u;  u = r * coef */
      HYPRE_THRUST_CALL(for_each,
      thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
      thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data))+num_rows,
      save_and_scale<HYPRE_Real>(coefs[cheby_order]));

      /* now do the other coefficients */   
      for (i = cheby_order - 1; i >= 0; i-- ) 
      {
         /* v = D^(-1/2)AD^(-1/2)u */
         /* tmp = ds .* u */
         HYPRE_THRUST_CALL(for_each,
         thrust::make_zip_iterator(thrust::make_tuple(tmp_data, ds_data, u_data)),
         thrust::make_zip_iterator(thrust::make_tuple(tmp_data , ds_data, u_data ))+num_rows,
         oop_xy<HYPRE_Real>());

         hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

         /* u_new = coef*r + v*/
         mult = coefs[i];

         /* u = coef * r + ds .* v */
         HYPRE_THRUST_CALL(for_each,
         thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, ds_data, v_data)),
         thrust::make_zip_iterator(thrust::make_tuple(u_data , r_data , ds_data , v_data))+num_rows,
         oop_axpyz<HYPRE_Real>(mult));
      } /* end of cheby_order loop */

      /* now we have to scale u_data before adding it to u_orig*/

      /* u = orig_u + ds .* u */
      HYPRE_THRUST_CALL(for_each,
      thrust::make_zip_iterator(thrust::make_tuple(u_data, orig_u, ds_data)),
      thrust::make_zip_iterator(thrust::make_tuple(u_data + num_rows, orig_u + num_rows, ds_data + num_rows)),
      oop_xpyz<HYPRE_Real>());
   
      hypre_ParVectorDestroy(tmp_vec);  

   }/* end of scaling code */

   hypre_TFree(orig_u, HYPRE_MEMORY_DEVICE);
  
   return hypre_error_flag;
}
#endif
