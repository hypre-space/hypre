/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySolve( hypre_ParChebyData *cheby_data,
                     hypre_ParCSRMatrix *A,
                     hypre_ParVector    *f,
                     hypre_ParVector    *u )
{
   MPI_Comm         comm          = hypre_ParCSRMatrixComm(A);

   HYPRE_Int        print_level   = hypre_ParChebyDataPrintLevel(cheby_data);
   HYPRE_Int        logging       = hypre_ParChebyDataLogging(cheby_data);
   HYPRE_Int        max_iter      = hypre_ParChebyDataMaxIterations(cheby_data);
   HYPRE_Int        tol           = hypre_ParChebyDataTol(cheby_data);
   HYPRE_Int        scale         = hypre_ParChebyDataScale(cheby_data);
   HYPRE_Int        variant       = hypre_ParChebyDataVariant(cheby_data);
   HYPRE_Int        order         = hypre_ParChebyDataOrder(cheby_data);
   HYPRE_Int        converge_type = 0;

   HYPRE_Real      *coefs         = hypre_ParChebyDataCoefs(cheby_data);
   hypre_ParVector *scaling       = hypre_ParChebyDataScaling(cheby_data);
   hypre_ParVector *residual      = hypre_ParChebyDataResidual(cheby_data);
   hypre_ParVector *Ptemp         = hypre_ParChebyDataPtemp(cheby_data);
   hypre_ParVector *Rtemp         = hypre_ParChebyDataRtemp(cheby_data);
   hypre_ParVector *Vtemp         = hypre_ParChebyDataVtemp(cheby_data);
   hypre_ParVector *Ztemp         = hypre_ParChebyDataZtemp(cheby_data);
   HYPRE_Complex   *scaling_data  = (scaling) ? hypre_ParVectorLocalData(scaling) : NULL;

   hypre_ParVector *Ltemp;
   HYPRE_Int        myid, iter;
   HYPRE_Complex    alpha = 1.0;
   HYPRE_Complex    beta  = -1.0;
   HYPRE_Real       conv_factor;
   HYPRE_Complex    relative_resid, old_resid, resid_nrm, resid_nrm_init;
   HYPRE_Complex    rhs_norm, ieee_check = 0.0;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------------------
    *   Write some initial info
    *-----------------------------------------------------------------------*/

   if (!myid && print_level > 1 && tol > 0.)
   {
      hypre_printf("\n\nChebyshev SOLUTION INFO:\n");
   }

   /*-----------------------------------------------------------------------
    *   Compute initial residual and print
    *-----------------------------------------------------------------------*/

   if (print_level > 1 || logging > 1 || tol > 0.)
   {
      /* Set pointer to work vector for logging purposes */
      Ltemp = (logging > 1) ? residual : Vtemp;

      /* Compute residual */
      hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, Ltemp);

      /* Compute residual L2-norm */
      resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Ltemp, Ltemp));

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resid_nrm != 0.)
      {
         ieee_check = resid_nrm / resid_nrm; /* INF -> NaN conversion */
      }

      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            hypre_printf("ERROR -- hypre_ParChebySolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }

      /* r0 */
      resid_nrm_init = resid_nrm;

      if (0 == converge_type)
      {
         rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
         relative_resid = (rhs_norm) ? resid_nrm_init / rhs_norm : resid_nrm_init;
      }
      else
      {
         /* converge_type != 0, test convergence with ||r|| / ||r0|| */
         relative_resid = 1.0;
      }
   }
   else
   {
      relative_resid = 1.0;
   }

   if (!myid && print_level > 1)
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n",
                   resid_nrm_init, relative_resid);
   }

   /*-----------------------------------------------------------------------
    *   Main loop
    *-----------------------------------------------------------------------*/

   iter = 0;
   while (relative_resid >= tol && iter < max_iter)
   {
      /* Apply Chebyshev polynomial */
      hypre_ParCSRRelax_Cheby_Solve(A, f, scaling_data,
                                    coefs, order, scale, variant, u,
                                    Vtemp, Ztemp, Ptemp, Rtemp);

      /*---------------------------------------------------------------
       *  Compute residual and its L2-norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.0)
      {
         old_resid = resid_nrm;

         /* Set pointer to work vector for logging purposes */
         Ltemp = (logging > 1) ? residual : Vtemp;

         /* Compute residual */
         hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, Ltemp);

         conv_factor = (old_resid) ? resid_nrm / old_resid : resid_nrm;
         if (0 == converge_type)
         {
            relative_resid = (rhs_norm) ? resid_nrm / rhs_norm : resid_nrm;
         }
         else
         {
            relative_resid = resid_nrm / resid_nrm_init;
         }

         hypre_ParChebyDataRelResidNorm(cheby_data) = relative_resid;
      }

      /* Increase iterations counter */
      iter++;

      if (!myid && print_level > 1)
      {
         hypre_printf("     Iter %2d   %e    %f     %e \n",
                      iter, resid_nrm, conv_factor, relative_resid);
      }
   }
   hypre_ParChebyDataNumIterations(cheby_data) = iter;

   if (iter == max_iter && tol > 0.)
   {
      hypre_error(HYPRE_ERROR_CONV);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Legacy function for applying the chebyshev polynomial on the host
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRRelax_Cheby_SolveHost(hypre_ParCSRMatrix *A,
                                  hypre_ParVector    *f,
                                  HYPRE_Real         *ds_data,
                                  HYPRE_Real         *coefs,
                                  HYPRE_Int           order,
                                  HYPRE_Int           scale,
                                  HYPRE_Int           variant,
                                  hypre_ParVector    *u,
                                  hypre_ParVector    *v,
                                  hypre_ParVector    *r,
                                  hypre_ParVector    *orig_u_vec,
                                  hypre_ParVector    *tmp_vec)
{
   HYPRE_UNUSED_VAR(variant);

   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *u_data   = hypre_ParVectorLocalData(u);
   HYPRE_Real      *f_data   = hypre_ParVectorLocalData(f);
   HYPRE_Real      *v_data   = hypre_ParVectorLocalData(v);
   HYPRE_Real      *r_data   = hypre_ParVectorLocalData(r);
   HYPRE_Real      *tmp_data = hypre_ParVectorLocalData(tmp_vec);
   HYPRE_Int        num_rows = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int        i, j;
   HYPRE_Int        cheby_order;
   HYPRE_Real       mult;
   HYPRE_Real      *orig_u;

   /* u = u + p(A) r */
   if (order > 4)
   {
      order = 4;
   }
   if (order < 1)
   {
      order = 1;
   }

   /* we are using the order of p(A) */
   cheby_order = order - 1;

   hypre_assert(hypre_VectorSize(hypre_ParVectorLocalVector(orig_u_vec)) >= num_rows);
   orig_u = hypre_VectorData(hypre_ParVectorLocalVector(orig_u_vec));

   if (!scale)
   {
      /* get residual: r = f - A*u */
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, r);

      /* o = u; u = r .* coef */
      for (i = 0; i < num_rows; i++)
      {
         orig_u[i] = u_data[i];
         u_data[i] = r_data[i] * coefs[cheby_order];
      }

      for (i = cheby_order - 1; i >= 0; i--)
      {
         /* v = A*u */
         hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, v);
         mult = coefs[i];

         /* u = mult * r + v */
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            u_data[j] = mult * r_data[j] + v_data[j];
         }
      }

      /* u = o + u */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for ( i = 0; i < num_rows; i++ )
      {
         u_data[i] = orig_u[i] + u_data[i];
      }
   }
   else /* with scaling */
   {
      /* Compute scaled residual: r = D^(-1/2) f - D^(-1/2) A*u */
      /* tmp = -A*u */
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);

      /* r = ds .* (f + tmp) */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < num_rows; j++)
      {
         r_data[j] = ds_data[j] * (f_data[j] + tmp_data[j]);
      }

      /* save original u, then start
         the iteration by multiplying r by the cheby coef.*/

      /* o = u;  u = r * coef */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for ( j = 0; j < num_rows; j++ )
      {
         orig_u[j] = u_data[j]; /* orig, unscaled u */
         u_data[j] = r_data[j] * coefs[cheby_order];
      }

      /* now do the other coefficients */
      for (i = cheby_order - 1; i >= 0; i--)
      {
         /* v = D^(-1/2)AD^(-1/2)u */
         /* tmp = ds .* u */
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            tmp_data[j] = ds_data[j] * u_data[j];
         }
         hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

         /* u_new = coef*r + v*/
         mult = coefs[i];

         /* u = coef * r + ds .* v */
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            u_data[j] = mult * r_data[j] + ds_data[j] * v_data[j];
         }
      } /* end of cheby_order loop */

      /* now we have to scale u_data before adding it to u_orig */

      /* u = orig_u + ds .* u */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for ( j = 0; j < num_rows; j++ )
      {
         u_data[j] = orig_u[j] + ds_data[j] * u_data[j];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Legacy function for applying the chebyshev polynomial
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRRelax_Cheby_Solve(hypre_ParCSRMatrix *A, /* matrix to relax with */
                              hypre_ParVector    *f, /* right-hand side */
                              HYPRE_Real         *ds_data,
                              HYPRE_Real         *coefs,
                              HYPRE_Int           order, /* polynomial order */
                              HYPRE_Int           scale, /* scale by diagonal?*/
                              HYPRE_Int           variant,
                              hypre_ParVector    *u, /* initial/updated approximation */
                              hypre_ParVector    *v, /* temporary vector */
                              hypre_ParVector    *r, /*another temp vector */
                              hypre_ParVector    *orig_u_vec, /*another temp vector */
                              hypre_ParVector    *tmp_vec) /*another temp vector */
{
   /* Sanity check */
   if (hypre_ParVectorNumVectors(f) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Chebyshev doesn't support multicomponent vectors");
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A));
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRRelax_Cheby_SolveDevice(A, f, ds_data, coefs, order,
                                          scale, variant, u, v, r,
                                          orig_u_vec, tmp_vec);
   }
   else
#endif
   {
      hypre_ParCSRRelax_Cheby_SolveHost(A, f, ds_data, coefs, order,
                                        scale, variant, u, v, r,
                                        orig_u_vec, tmp_vec);
   }

   return hypre_error_flag;
}
