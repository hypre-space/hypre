/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * Setup Chebyshev solver/relaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetup( hypre_ParChebyData *cheby_data,
                     hypre_ParCSRMatrix *A,
                     hypre_ParVector    *f,
                     hypre_ParVector    *u )
{
   MPI_Comm              comm             = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt          global_num_rows  = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt         *row_starts       = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_MemoryLocation  memory_location  = hypre_ParCSRMatrixMemoryLocation(A);

   HYPRE_Int             logging          = hypre_ParChebyDataLogging(cheby_data);
   HYPRE_Int             owns_temp        = hypre_ParChebyDataOwnsTemp(cheby_data);
   HYPRE_Int             scale            = hypre_ParChebyDataScale(cheby_data);
   HYPRE_Int             variant          = hypre_ParChebyDataVariant(cheby_data);
   HYPRE_Int             order            = hypre_ParChebyDataOrder(cheby_data);
   HYPRE_Int             eig_est          = hypre_ParChebyDataEigEst(cheby_data);
   HYPRE_Real            eig_ratio        = hypre_ParChebyDataEigRatio(cheby_data);
   HYPRE_Real            eig_provided     = hypre_ParChebyDataEigProvided(cheby_data);
   HYPRE_Real            min_eig          = hypre_ParChebyDataMinEigEst(cheby_data);
   HYPRE_Real            max_eig          = hypre_ParChebyDataMaxEigEst(cheby_data);

   hypre_ParVector      *scaling          = hypre_ParChebyDataScaling(cheby_data);
   hypre_ParVector      *residual         = hypre_ParChebyDataResidual(cheby_data);
   hypre_ParVector      *Ptemp            = hypre_ParChebyDataPtemp(cheby_data);
   hypre_ParVector      *Rtemp            = hypre_ParChebyDataRtemp(cheby_data);
   hypre_ParVector      *Vtemp            = hypre_ParChebyDataVtemp(cheby_data);
   hypre_ParVector      *Ztemp            = hypre_ParChebyDataZtemp(cheby_data);

   HYPRE_ANNOTATE_FUNC_BEGIN;
   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   /* Allocate work vectors when either of the conditions hold:
      - Chebyshev solver owns them.
      - They haven't been allocated yet (check on Ptemp). */
   if (owns_temp || (!owns_temp && !Ptemp))
   {
      Ptemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
      Rtemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
      Vtemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
      Ztemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);

      hypre_ParVectorInitialize_v2(Ptemp, memory_location);
      hypre_ParVectorInitialize_v2(Rtemp, memory_location);
      hypre_ParVectorInitialize_v2(Vtemp, memory_location);
      hypre_ParVectorInitialize_v2(Ztemp, memory_location);

      hypre_ParChebyDataPtemp(cheby_data) = Ptemp;
      hypre_ParChebyDataRtemp(cheby_data) = Rtemp;
      hypre_ParChebyDataVtemp(cheby_data) = Vtemp;
      hypre_ParChebyDataZtemp(cheby_data) = Ztemp;

      hypre_ParChebyDataOwnsTemp(cheby_data) = 1;
   }

   /* Allocate scaling vector */
   if (scale)
   {
      scaling = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
      hypre_ParVectorInitialize_v2(scaling, hypre_ParCSRMatrixMemoryLocation(A));
      hypre_ParChebyDataScaling(cheby_data) = scaling;
   }

   /* Allocate residual vector for logging purposes */
   if (logging > 1)
   {
      residual = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
      hypre_ParVectorInitialize_v2(residual, hypre_ParCSRMatrixMemoryLocation(A));
      hypre_ParChebyDataResidual(cheby_data) = residual;
   }

   /* Estimate eigenvalues if not provided by user */
   if (!eig_provided)
   {
      if (eig_est > 0)
      {
         hypre_ParCSRMaxEigEstimateCG(A, scale, eig_est, &max_eig, &min_eig);
      }
      else
      {
         hypre_ParCSRMaxEigEstimate(A, scale, &max_eig, &min_eig);
      }
      hypre_ParChebyDataMinEigEst(cheby_data) = min_eig;
      hypre_ParChebyDataMaxEigEst(cheby_data) = max_eig;
   }

   /* Setup coefficients and scaling vector */
   hypre_ParCSRRelax_Cheby_Setup(A, max_eig, min_eig, eig_ratio,
                                 order, scale, variant,
                                 &hypre_ParChebyDataCoefs(cheby_data),
                                 (scale) ? &hypre_ParVectorLocalData(scaling) : NULL);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Setup coefficients and optional diagonal scaling elements for Chebyshev
 *
 * Will calculate ds_ptr on device/host depending on where A is located
 *
 * - A: Matrix for which to seteup
 * - max_eig: Maximum eigenvalue
 * - min_eig: Minimum eigenvalue
 * - eig_ratio: Fraction used to calculate lower bound
 * - order: Polynomial order to use [1,4]
 * - scale: Whether or not to scale by the diagonal
 * - variant: Chebyshev variant (0 standard, 1 variant)
 * - coefs_ptr: output polynomial coefficients
 * - ds_ptr: output diagonal scaling
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRRelax_Cheby_Setup(hypre_ParCSRMatrix *A,
                              HYPRE_Real          max_eig,
                              HYPRE_Real          min_eig,
                              HYPRE_Real          eig_ratio,
                              HYPRE_Int           order,
                              HYPRE_Int           scale,
                              HYPRE_Int           variant,
                              HYPRE_Real        **coefs_ptr,
                              HYPRE_Real        **ds_ptr)
{
   HYPRE_MemoryLocation   memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_CSRMatrix       *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              num_rows        = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int              cheby_order;
   HYPRE_Real             theta, delta;
   HYPRE_Real             den, upper_bound, lower_bound;
   HYPRE_Real            *coefs   = NULL;
   HYPRE_Real            *ds_data = NULL;

   /* u = u + p(A)r */
   if (order > 4)
   {
      order = 4;
   }

   if (order < 1)
   {
      order = 1;
   }

   /* Allocate coefficients */
   coefs = hypre_CTAlloc(HYPRE_Real, order + 1, HYPRE_MEMORY_HOST);

   /* we are using the order of p(A) */
   cheby_order = order - 1;

   if (max_eig <= 0.0)
   {
      upper_bound = min_eig * 1.1;
      lower_bound = max_eig - (max_eig - upper_bound) * eig_ratio;
   }
   else
   {
      /* make sure we are large enough - Adams et al. 2003 */
      upper_bound = max_eig * 1.1;

      /* lower_bound = max_eig/fraction; */
      lower_bound = (upper_bound - min_eig) * eig_ratio + min_eig;
   }

   /* theta and delta */
   theta = (upper_bound + lower_bound) / 2;
   delta = (upper_bound - lower_bound) / 2;

   if (variant == 1)
   {
      /* These are the corresponding cheby polynomials: u = u_o + s(A) r_0
         So order is one less than  resid poly: r(t) = 1 - t*s(t) */
      switch (cheby_order)
      {
         case 0:
            if (hypre_cabs(theta) > 0.0)
            {
               coefs[0] = 1.0 / theta;
            }
            else
            {
               coefs[0] = 0.0;
            }
            break;

         case 1:  /* (del - t + 2*th)/(th^2 + del*th) */
            den = (theta * theta + delta * theta);

            if (hypre_cabs(den) > 0.0)
            {
               coefs[0] = (delta + 2 * theta) / den;
               coefs[1] = -1.0 / den;
            }
            else
            {
               coefs[0] = coefs[1] = 0.0;
            }

            break;

         case 2:  /* (4*del*th - del^2 - t*(2*del + 6*th) + 2*t^2 + 6*th^2)/
                     (2*del*th^2 - del^2*th - del^3 + 2*th^3) */
            den = 2 * delta * theta * theta - delta * delta * theta -
                  hypre_pow(delta, 3) + 2 * hypre_pow(theta, 3);

            if (hypre_cabs(den) > 0.0)
            {
               coefs[0] = (4 * delta * theta - hypre_pow(delta, 2) +
                           6 * hypre_pow(theta, 2)) / den;
               coefs[1] = -(2 * delta + 6 * theta) / den;
               coefs[2] =  2 / den;
            }
            else
            {
               coefs[0] = coefs[1] = coefs[2] = 0.0;
            }
            break;

         case 3: /* -(6*del^2*th - 12*del*th^2 - t^2*(4*del + 16*th) +
                      t*(12*del*th - 3*del^2 + 24*th^2) + 3*del^3 + 4*t^3 - 16*th^3)/
                      (4*del*th^3 - 3*del^2*th^2 - 3*del^3*th + 4*th^4) */
            den = - 4 * delta * hypre_pow(theta, 3) +
                  3 * hypre_pow(delta, 2) * hypre_pow(theta, 2) +
                  3 * hypre_pow(delta, 3) * theta -
                  4 * hypre_pow(theta, 4);

            if (hypre_cabs(den) > 0.0)
            {
               coefs[0] = (6 * hypre_pow(delta, 2) * theta -
                           12 * delta * hypre_pow(theta, 2) +
                           3 * hypre_pow(delta, 3) -
                           16 * hypre_pow(theta, 3) ) / den;
               coefs[1] = (12 * delta * theta -
                           3 * hypre_pow(delta, 2) +
                           24 * hypre_pow(theta, 2)) / den;
               coefs[2] = -(4 * delta + 16 * theta) / den;
               coefs[3] = 4 / den;
            }
            else
            {
               coefs[0] = coefs[1] = coefs[2] = coefs[3] = 0.0;
            }
            break;
      }
   }
   else /* standard chebyshev */
   {
      /* these are the corresponding cheby polynomials: u = u_o + s(A)r_0
         so order is one less than resid poly: r(t) = 1 - t*s(t) */
      switch (cheby_order)
      {
         case 0:
            if (hypre_cabs(theta) > 0.0)
            {
               coefs[0] = 1.0 / theta;
            }
            else
            {
               coefs[0] = 0.0;
            }
            break;

         case 1:  /* (2*t - 4*th)/(del^2 - 2*th^2) */
            den = delta * delta - 2 * theta * theta;

            if (hypre_cabs(den) > 0.0)
            {
               coefs[0] = -4 * theta / den;
               coefs[1] = 2 / den;
            }
            else
            {
               coefs[0] = coefs[1] = 0.0;
            }

            break;

         case 2: /* (3*del^2 - 4*t^2 + 12*t*th - 12*th^2)/(3*del^2*th - 4*th^3) */
            den = 3 * (delta * delta) * theta - 4 * (theta * theta * theta);

            if (hypre_cabs(den) > 0.0)
            {
               coefs[0] = (3 * delta * delta - 12 * theta * theta) / den;
               coefs[1] = 12 * theta / den;
               coefs[2] = -4 / den;
            }
            else
            {
               coefs[0] = coefs[1] = coefs[2] = 0.0;
            }

            break;

         case 3: /*(t*(8*del^2 - 48*th^2) - 16*del^2*th + 32*t^2*th - 8*t^3 + 32*th^3)/
                   (del^4 - 8*del^2*th^2 + 8*th^4)*/
            den = hypre_pow(delta, 4) - 8 * hypre_squared(delta * theta) +
                  hypre_pow(theta, 4) * 8;

            if (hypre_cabs(den) > 0.0)
            {
               coefs[0] = (32 * hypre_pow(theta, 3) - 16 * delta * delta * theta) / den;
               coefs[1] = (8 * delta * delta - 48 * theta * theta) / den;
               coefs[2] = 32 * theta / den;
               coefs[3] = -8 / den;
            }
            else
            {
               coefs[0] = coefs[1] = coefs[2] = coefs[3] = 0.0;
            }

            break;
      }
   }

   /* Compute scaling vector: 1/hypre_sqrt(abs(diagonal)) */
   if (scale)
   {
      if (ds_ptr && !*ds_ptr)
      {
         ds_data = hypre_CTAlloc(HYPRE_Real, num_rows, memory_location);
      }
      else if (ds_ptr)
      {
         ds_data = *ds_ptr;
      }
      hypre_CSRMatrixExtractDiagonal(A_diag, ds_data, 4);
   }

   /* Set output pointers */
   if (ds_ptr)
   {
      *ds_ptr = ds_data;
   }
   *coefs_ptr = coefs;

   return hypre_error_flag;
}
