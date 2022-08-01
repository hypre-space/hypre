/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#include "_hypre_utilities.hpp"
#include <algorithm>

/**
 * @brief waxpyz
 *
 * Performs
 * w = a*x+y.*z
 * For scalars w,x,y,z and constant a (indices 0, 1, 2, 3 respectively)
 */
template <typename T>
struct waxpyz
{
   typedef thrust::tuple<T &, T, T, T> Tuple;

   const T scale;
   waxpyz(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = scale * thrust::get<1>(t) + thrust::get<2>(t) * thrust::get<3>(t);
   }
};

/**
 * @brief wxypz
 *
 * Performs
 * o = x * (y .+ z)
 * For scalars o,x,y,z (indices 0, 1, 2, 3 respectively)
 */
template <typename T>
struct wxypz
{
   typedef thrust::tuple<T &, T, T, T> Tuple;
   __host__ __device__ void            operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t) * (thrust::get<2>(t) + thrust::get<3>(t));
   }
};
/**
 * @brief Saves u into o, then scales r placing the result in u
 *
 * Performs
 * o = u
 * u = r * a
 * For scalars o and u, with constant a
 */
template <typename T>
struct save_and_scale
{
   typedef thrust::tuple<T &, T &, T> Tuple;

   const T scale;

   save_and_scale(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t);
      thrust::get<1>(t) = thrust::get<2>(t) * scale;
   }
};

/**
 * @brief xpyz
 *
 * Performs
 * y = x + y .* z
 * For scalars x,y,z (indices 1,0,2 respectively)
 */
template <typename T>
struct xpyz
{
   typedef thrust::tuple<T &, T, T> Tuple;

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t) + thrust::get<2>(t) * thrust::get<0>(t);
   }
};

/**
 * @brief scale
 *
 * Performs
 * x = d .* x
 * For scalars x, d
 */
template <typename T>
struct scaleInPlace
{
   typedef thrust::tuple<T, T&> Tuple;

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<1>(t);
   }
};

/**
 * @brief add
 *
 * Performs
 * x = x + coef * y
 * For scalars x, d
 */
template <typename T>
struct add
{
   typedef thrust::tuple<T, T&> Tuple;

   const T coef;
   add(T _coef = 1.0) : coef(_coef) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = coef * thrust::get<0>(t) + thrust::get<1>(t);
   }
};

/**
 * @brief add
 *
 * Performs
 * x = x + coef * d.*y
 * For scalars x, d
 */
template <typename T>
struct scaledAdd
{
   typedef thrust::tuple<T, T, T&> Tuple;
   const T scale;
   scaledAdd(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = thrust::get<2>(t) + scale * thrust::get<0>(t) * thrust::get<1>(t);
   }
};

/**
 * @brief add
 *
 * Performs
 * r = r - D .* tmp
 * v = coef0 * r + coef1 * v
 * 
 */
template <typename T>
struct updateRAndV
{
   typedef thrust::tuple<T, T, T&, T&> Tuple;
   const T coef0;
   const T coef1;
   updateRAndV(T _coef0, T _coef1) : coef0(_coef0), coef1(_coef1) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = thrust::get<2>(t) - thrust::get<0>(t) * thrust::get<1>(t);
      thrust::get<3>(t) = coef0 * thrust::get<2>(t) + coef1 * thrust::get<3>(t);
   }
};

/**
 * @brief scale
 *
 * Performs
 * y = coef * d .* x
 * For scalars x, d, y
 */
template <typename T>
struct applySmoother
{
   typedef thrust::tuple<T, T, T&> Tuple;

   const T coef;
   applySmoother(T _coef = 1.0) : coef(_coef) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = coef * thrust::get<0>(t) * thrust::get<1>(t);
   }
};

/**
 * @brief waxpyz
 *
 * Performs
 * y = a * x
 * constant a
 */
template <typename T>
struct scaleConstant
{
   typedef thrust::tuple<T, T&> Tuple;

   const T scale;
   scaleConstant(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = scale * thrust::get<0>(t);
   }
};

/**
 * @brief update
 *
 * Performs
 * d = scale0 * r + scale1 * d
 */
template <typename T>
struct update
{
   typedef thrust::tuple<T, T&> Tuple;

   const T scale0;
   const T scale1;
   update(T _scale0, T _scale1) : scale0(_scale0), scale1(_scale1) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = scale1 * thrust::get<1>(t) + scale0 * thrust::get<0>(t);
   }
};

/**
 * @brief updateCol
 *
 * Performs
 * d = scale0 * x.*r + scale1 * d
 */
template <typename T>
struct updateCol
{
   typedef thrust::tuple<T, T, T&> Tuple;

   const T scale0;
   const T scale1;
   updateCol(T _scale0, T _scale1) : scale0(_scale0), scale1(_scale1) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = scale1 * thrust::get<2>(t) + scale0 * thrust::get<1>(t) * thrust::get<0>(t);
   }
};
/**
 * @brief Solve using a chebyshev polynomial on the device
 *
 * @param[in] A Matrix to relax with
 * @param[in] f right-hand side
 * @param[in] ds_data Diagonal information
 * @param[in] coefs Polynomial coefficients
 * @param[in] order Order of the polynomial
 * @param[in] scale Whether or not to scale by diagonal
 * @param[in] scale Whether or not to use a variant
 * @param[in,out] u Initial/updated approximation
 * @param[out] v Temp vector
 * @param[out] v Temp Vector
 */
HYPRE_Int
hypre_ParCSRRelax_Cheby_SolveDevice(hypre_ParCSRMatrix *A, /* matrix to relax with */
                                    hypre_ParVector    *f, /* right-hand side */
                                    HYPRE_Real         *ds_data,
                                    HYPRE_Real         *coefs,
                                    HYPRE_Int           order, /* polynomial order */
                                    HYPRE_Int           scale, /* scale by diagonal?*/
                                    HYPRE_Int           variant,
                                    hypre_ParVector    *u,          /* initial/updated approximation */
                                    hypre_ParVector    *v,          /* temporary vector */
                                    hypre_ParVector    *r,          /*another temp vector */
                                    hypre_ParVector    *orig_u_vec, /*another temp vector */
                                    hypre_ParVector    *tmp_vec)       /*a potential temp vector */
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real      *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real      *v_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   HYPRE_Real *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   HYPRE_Int i;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Real  mult;

   HYPRE_Int cheby_order;

   HYPRE_Real *tmp_data;

   /* u = u + p(A)r */
   if (order < 1)
   {
      order = 1;
   }

   HYPRE_Int maxOrder = 4;
   if(variant == 2 || variant == 3){
      maxOrder = 9999;
   }
   if(variant == 4) maxOrder = 16; // due to writing out coefficients

   if (order > maxOrder)
   {
      order = maxOrder;
   }

   /* we are using the order of p(A) */
   cheby_order = order - 1;

   hypre_assert(hypre_VectorSize(hypre_ParVectorLocalVector(orig_u_vec)) >= num_rows);
   HYPRE_Real *orig_u = hypre_VectorData(hypre_ParVectorLocalVector(orig_u_vec));

   if(variant == 0 || variant == 1){

     if (!scale)
     {
        /* get residual: r = f - A*u */
        hypre_ParVectorCopy(f, r);
        hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

        /* o = u; u = r .* coef */
        HYPRE_THRUST_CALL(
           for_each,
           thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
           thrust::make_zip_iterator(thrust::make_tuple(orig_u + num_rows, u_data + num_rows,
                                                        r_data + num_rows)),
           save_and_scale<HYPRE_Real>(coefs[cheby_order]));

        for (i = cheby_order - 1; i >= 0; i--)
        {
           hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, v);
           mult = coefs[i];

           /* u = mult * r + v */
           hypreDevice_ComplexAxpyn( r_data, num_rows, v_data, u_data, mult );
        }

        /* u = o + u */
        hypreDevice_ComplexAxpyn( orig_u, num_rows, u_data, u_data, 1.0);
     }
     else /* scaling! */
     {

        /*grab 1/sqrt(diagonal) */

        tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

        /* get ds_data and get scaled residual: r = D^(-1/2)f -
         * D^(-1/2)A*u */

        hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);
        /* r = ds .* (f + tmp) */

        /* TODO: It might be possible to merge this and the next call to:
         * r[j] = ds_data[j] * (f_data[j] + tmp_data[j]); o[j] = u[j]; u[j] = r[j] * coef */
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data)) + num_rows,
                          wxypz<HYPRE_Real>());

        /* save original u, then start
           the iteration by multiplying r by the cheby coef.*/

        /* o = u;  u = r * coef */
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)) + num_rows,
                          save_and_scale<HYPRE_Real>(coefs[cheby_order]));

        /* now do the other coefficients */
        for (i = cheby_order - 1; i >= 0; i--)
        {
           /* v = D^(-1/2)AD^(-1/2)u */
           /* tmp = ds .* u */
           HYPRE_THRUST_CALL( transform, ds_data, ds_data + num_rows, u_data, tmp_data, _1 * _2 );

           hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

           /* u_new = coef*r + v*/
           mult = coefs[i];

           /* u = coef * r + ds .* v */
           HYPRE_THRUST_CALL(for_each,
                             thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, ds_data, v_data)),
                             thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, ds_data, v_data)) + num_rows,
                             waxpyz<HYPRE_Real>(mult));
        } /* end of cheby_order loop */

        /* now we have to scale u_data before adding it to u_orig*/

        /* u = orig_u + ds .* u */
        HYPRE_THRUST_CALL(
           for_each,
           thrust::make_zip_iterator(thrust::make_tuple(u_data, orig_u, ds_data)),
           thrust::make_zip_iterator(thrust::make_tuple(u_data + num_rows, orig_u + num_rows,
                                                        ds_data + num_rows)),
           xpyz<HYPRE_Real>());


     } /* end of scaling code */
   }
   else if(variant == 2)
   {
      const auto lambda_max = coefs[0];
      const auto lambda_min = coefs[1];

      const auto theta = 0.5 * (lambda_max + lambda_min);
      const auto delta = 0.5 * (lambda_max - lambda_min);
      const auto sigma = theta / delta;
      auto rho = 1.0 / sigma;

      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

      // r := f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      // TODO: consolidate two calls below

      // r = D^{-1} r
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(ds_data, r_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(ds_data, r_data)) + num_rows,
                        scaleInPlace<HYPRE_Real>());
      
      // v := 1/theta r
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, v_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, v_data)) + num_rows,
                        scaleConstant<HYPRE_Real>(1.0 / theta));
      
      for(int i = 0; i < cheby_order; ++i){
        // u += v
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                          add<HYPRE_Real>());
        // tmp = Av
        hypre_ParCSRMatrixMatvec(1.0, A, v, 0.0, tmp_vec);

        const auto rhoSave = rho;
        rho = 1.0 / (2 * sigma - rho);

        const auto vcoef = rho * rhoSave;
        const auto rcoef = 2.0 * rho / delta;

        // r = r - D^{-1} Av
        // v = rho_{k+1} rho_k * v + 2 rho_{k+1} / delta r
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(ds_data, tmp_data, r_data, v_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(ds_data, tmp_data, r_data, v_data)) + num_rows,
                          updateRAndV<HYPRE_Real>(rcoef, vcoef));
      }

      // u += v;
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                        add<HYPRE_Real>());

   }
   else if(variant == 3 || variant == 4)
   {
      const auto lambda_max = coefs[0];
      const auto coeffOffset = 2;

      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

      // r := f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      // v := \dfrac{4}{3} \dfrac{1}{\rho(D^{-1}A)} D^{-1} r
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)) + num_rows,
                        applySmoother<HYPRE_Real>(4.0 / 3.0 / lambda_max));
      
      for(int i = 0; i < cheby_order; ++i){
        // u += \beta_k v
        // since this is _not_ the optimized variant, \beta := 1.0
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                          add<HYPRE_Real>(coefs[coeffOffset + i]));
        // r = r - Av
        hypre_ParCSRMatrixMatvec(-1.0, A, v, 1.0, r);

        // + 2 offset is due to two issues:
        // + 1 is from https://arxiv.org/pdf/2202.08830.pdf being written in 1-based indexing
        // + 1 is from pre-computing z_1 _outside_ of the loop
        // v = \dfrac{(2i-3)}{(2i+1)} v + \dfrac{(8i-4)}{(2i+1)} \dfrac{1}{\rho(SA)} S r
        const auto id = i + 2;
        const auto vScale = (2.0 * id - 3.0) / (2.0 * id + 1.0);
        const auto rScale = (8.0 * id - 4.0) / (2.0 * id + 1.0) / lambda_max;

        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)) + num_rows,
                          updateCol<HYPRE_Real>(rScale, vScale));

      }

      // u += \beta v;
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                        add<HYPRE_Real>(coefs[coeffOffset + cheby_order]));

   }

   return hypre_error_flag;
}
#endif
