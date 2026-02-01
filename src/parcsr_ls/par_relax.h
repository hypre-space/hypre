/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_PAR_RELAX_HEADER
#define HYPRE_PAR_RELAX_HEADER

/* Non-Scale version: inner loop kernel with diagonal skip */
static inline void
hypre_HybridGaussSeidelNSskip_core( const HYPRE_Int     *A_diag_i,
                                    const HYPRE_Int     *A_diag_j,
                                    const HYPRE_Complex *A_diag_data,
                                    const HYPRE_Int     *A_offd_i,
                                    const HYPRE_Int     *A_offd_j,
                                    const HYPRE_Complex *A_offd_data,
                                    const HYPRE_Complex *f_data,
                                    const HYPRE_Complex *norms,
                                    HYPRE_Complex       *u_data,
                                    const HYPRE_Complex *v_ext_data,
                                    HYPRE_Int            i )
{
   const HYPRE_Int diag_start = A_diag_i[i] + 1;
   const HYPRE_Int diag_end   = A_diag_i[i + 1];
   const HYPRE_Int offd_start = A_offd_i[i];
   const HYPRE_Int offd_end   = A_offd_i[i + 1];

   HYPRE_Complex   res = f_data[i];
   HYPRE_Int       jj;

   /* Diagonal block contribution */
   for (jj = diag_start; jj < diag_end; jj++)
   {
      res -= A_diag_data[jj] * u_data[A_diag_j[jj]];
   }

   /* Off-diagonal block contribution */
   for (jj = offd_start; jj < offd_end; jj++)
   {
      res -= A_offd_data[jj] * v_ext_data[A_offd_j[jj]];
   }

   /* Update solution */
   u_data[i] = res / norms[i];
}

/* Non-Scale version: inner loop kernel without diagonal skip */
static inline void
hypre_HybridGaussSeidelNS_core( const HYPRE_Int     *A_diag_i,
                                const HYPRE_Int     *A_diag_j,
                                const HYPRE_Complex *A_diag_data,
                                const HYPRE_Int     *A_offd_i,
                                const HYPRE_Int     *A_offd_j,
                                const HYPRE_Complex *A_offd_data,
                                const HYPRE_Complex *f_data,
                                const HYPRE_Complex *norms,
                                HYPRE_Complex       *u_data,
                                const HYPRE_Complex *v_ext_data,
                                HYPRE_Int            i )
{
   const HYPRE_Int diag_start = A_diag_i[i];
   const HYPRE_Int diag_end   = A_diag_i[i + 1];
   const HYPRE_Int offd_start = A_offd_i[i];
   const HYPRE_Int offd_end   = A_offd_i[i + 1];

   HYPRE_Complex   res = f_data[i];
   HYPRE_Int       jj;

   /* Diagonal block contribution */
   for (jj = diag_start; jj < diag_end; jj++)
   {
      res -= A_diag_data[jj] * u_data[A_diag_j[jj]];
   }

   /* Off-diagonal block contribution */
   for (jj = offd_start; jj < offd_end; jj++)
   {
      res -= A_offd_data[jj] * v_ext_data[A_offd_j[jj]];
   }

   /* Update solution */
   u_data[i] += res / norms[i];
}

/* Non-Scale version */
static inline void
hypre_HybridGaussSeidelNS( HYPRE_Int     *A_diag_i,
                           HYPRE_Int     *A_diag_j,
                           HYPRE_Complex *A_diag_data,
                           HYPRE_Int     *A_offd_i,
                           HYPRE_Int     *A_offd_j,
                           HYPRE_Complex *A_offd_data,
                           HYPRE_Complex *f_data,
                           HYPRE_Int     *cf_marker,
                           HYPRE_Int      relax_points,
                           HYPRE_Complex *l1_norms,
                           HYPRE_Complex *u_data,
                           HYPRE_Complex *v_tmp_data,
                           HYPRE_Complex *v_ext_data,
                           HYPRE_Int      ibegin,
                           HYPRE_Int      iend,
                           HYPRE_Int      iorder,
                           HYPRE_Int      Skip_diag )
{
   HYPRE_UNUSED_VAR(v_tmp_data);

   HYPRE_Int i;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *
    * Split into 4 code paths to eliminate inner-loop branches:
    * (l1_norms vs diagonal) x (Skip_diag vs not)
    *-----------------------------------------------------------*/

   if (l1_norms)
   {
      if (Skip_diag)
      {
         /* l1_norms provided, Skip_diag = 1 (Jacobi-style update) */
         if (relax_points == 0)
         {
            for (i = ibegin; i != iend; i += iorder)
            {
               /* TODO: remove this check */
#if 0               
               if (l1_norms[i] != 0.0)
#endif
               {
                  hypre_HybridGaussSeidelNSskip_core(A_diag_i, A_diag_j, A_diag_data,
                                                     A_offd_i, A_offd_j, A_offd_data,
                                                     f_data, l1_norms, u_data, v_ext_data, i);
               }
            }
         }
         else
         {
            for (i = ibegin; i != iend; i += iorder)
            {
               if (cf_marker[i] == relax_points && l1_norms[i] != 0.0)
               {
                  hypre_HybridGaussSeidelNSskip_core(A_diag_i, A_diag_j, A_diag_data,
                                                     A_offd_i, A_offd_j, A_offd_data,
                                                     f_data, l1_norms, u_data, v_ext_data, i);
               }
            }
         }
      }
      else
      {
         /* l1_norms provided, Skip_diag = 0 (GS-style update) */
         if (relax_points == 0)
         {
            for (i = ibegin; i != iend; i += iorder)
            {
#if 0                
               if (l1_norms[i] != 0.0)
#endif
               {
                  hypre_HybridGaussSeidelNS_core(A_diag_i, A_diag_j, A_diag_data,
                                                 A_offd_i, A_offd_j, A_offd_data,
                                                 f_data, l1_norms, u_data, v_ext_data, i);
               }
            }
         }
         else
         {
            for (i = ibegin; i != iend; i += iorder)
            {
               if (cf_marker[i] == relax_points && l1_norms[i] != 0.0)
               {
                  hypre_HybridGaussSeidelNS_core(A_diag_i, A_diag_j, A_diag_data,
                                                 A_offd_i, A_offd_j, A_offd_data,
                                                 f_data, l1_norms, u_data, v_ext_data, i);
               }
            }
         }
      }
   }
   else
   {
      /* Use diagonal entry as norm */
      if (Skip_diag)
      {
         /* No l1_norms, Skip_diag = 1 (Jacobi-style update) */
         if (relax_points == 0)
         {
            for (i = ibegin; i != iend; i += iorder)
            {
               /* TODO: remove this check maybe? */
#if 0               
               const HYPRE_Complex diag_val = A_diag_data[A_diag_i[i]];
               if (diag_val != 0.0)
#endif
               {
                  hypre_HybridGaussSeidelNSskip_core(A_diag_i, A_diag_j, A_diag_data,
                                                     A_offd_i, A_offd_j, A_offd_data,
                                                     f_data, A_diag_data, u_data, v_ext_data, i);
               }
            }
         }
         else
         {
            for (i = ibegin; i != iend; i += iorder)
            {
               /* TODO: remove this check maybe? */
               const HYPRE_Complex diag_val = A_diag_data[A_diag_i[i]];
               if (cf_marker[i] == relax_points && diag_val != 0.0)
               {
                  hypre_HybridGaussSeidelNSskip_core(A_diag_i, A_diag_j, A_diag_data,
                                                     A_offd_i, A_offd_j, A_offd_data,
                                                     f_data, A_diag_data, u_data, v_ext_data, i);
               }
            }
         }
      }
      else
      {
         /* No l1_norms, Skip_diag = 0 (GS-style update) */
         if (relax_points == 0)
         {
            for (i = ibegin; i != iend; i += iorder)
            {
               /* TODO: remove this check maybe? */
#if 0               
               const HYPRE_Complex diag_val = A_diag_data[A_diag_i[i]];
               if (diag_val != 0.0)
#endif
               {
                  hypre_HybridGaussSeidelNS_core(A_diag_i, A_diag_j, A_diag_data,
                                                 A_offd_i, A_offd_j, A_offd_data,
                                                 f_data, A_diag_data, u_data, v_ext_data, i);
               }
            }
         }
         else
         {
            for (i = ibegin; i != iend; i += iorder)
            {
               /* TODO: remove this check maybe? */
               const HYPRE_Complex diag_val = A_diag_data[A_diag_i[i]];
               if (cf_marker[i] == relax_points && diag_val != 0.0)
               {
                  hypre_HybridGaussSeidelNS_core(A_diag_i, A_diag_j, A_diag_data,
                                                 A_offd_i, A_offd_j, A_offd_data,
                                                 f_data, A_diag_data, u_data, v_ext_data, i);
               }
            }
         }
      }
   }
}

/* Non-Scale Threaded version */
static inline void
hypre_HybridGaussSeidelNSThreads( HYPRE_Int     *A_diag_i,
                                  HYPRE_Int     *A_diag_j,
                                  HYPRE_Complex *A_diag_data,
                                  HYPRE_Int     *A_offd_i,
                                  HYPRE_Int     *A_offd_j,
                                  HYPRE_Complex *A_offd_data,
                                  HYPRE_Complex *f_data,
                                  HYPRE_Int     *cf_marker,
                                  HYPRE_Int      relax_points,
                                  HYPRE_Complex *l1_norms,
                                  HYPRE_Complex *u_data,
                                  HYPRE_Complex *v_tmp_data,
                                  HYPRE_Complex *v_ext_data,
                                  HYPRE_Int      ns,
                                  HYPRE_Int      ne,
                                  HYPRE_Int      ibegin,
                                  HYPRE_Int      iend,
                                  HYPRE_Int      iorder,
                                  HYPRE_Int      Skip_diag )
{
   HYPRE_Int i;
   const HYPRE_Complex zero = 0.0;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *-----------------------------------------------------------*/

   if (l1_norms)
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         /*-----------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          * If i is of the right type ( C or F or All) and diagonal is
          * nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && l1_norms[i] != zero )
         {
            HYPRE_Int jj;
            HYPRE_Complex res = f_data[i];

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res -= A_diag_data[jj] * u_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] = res / l1_norms[i];
            }
            else
            {
               u_data[i] += res / l1_norms[i];
            }
         }
      } /* for ( i = ...) */
   }
   else
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
         {
            HYPRE_Int jj;
            HYPRE_Complex res = f_data[i];

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res -= A_diag_data[jj] * u_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] = res / A_diag_data[A_diag_i[i]];
            }
            else
            {
               u_data[i] += res / A_diag_data[A_diag_i[i]];
            }
         }
      } /* for ( i = ...) */
   }
}

/* Scaled version */
static inline void
hypre_HybridGaussSeidel( HYPRE_Int     *A_diag_i,
                         HYPRE_Int     *A_diag_j,
                         HYPRE_Complex *A_diag_data,
                         HYPRE_Int     *A_offd_i,
                         HYPRE_Int     *A_offd_j,
                         HYPRE_Complex *A_offd_data,
                         HYPRE_Complex *f_data,
                         HYPRE_Int     *cf_marker,
                         HYPRE_Int      relax_points,
                         HYPRE_Real     relax_weight,
                         HYPRE_Real     omega,
                         HYPRE_Real     one_minus_omega,
                         HYPRE_Real     prod,
                         HYPRE_Complex *l1_norms,
                         HYPRE_Complex *u_data,
                         HYPRE_Complex *v_tmp_data,
                         HYPRE_Complex *v_ext_data,
                         HYPRE_Int      ibegin,
                         HYPRE_Int      iend,
                         HYPRE_Int      iorder,
                         HYPRE_Int      Skip_diag )
{
   HYPRE_Int i;
   const HYPRE_Complex zero = 0.0;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *-----------------------------------------------------------*/

   if (l1_norms)
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && l1_norms[i] != zero )
         {
            HYPRE_Int jj;
            HYPRE_Complex res = f_data[i];
            HYPRE_Complex res0 = 0.0;
            HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_diag_j[jj];
               res0 -= A_diag_data[jj] * u_data[ii];
               res2 += A_diag_data[jj] * v_tmp_data[ii];
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
         }
      } /* for ( i = ...) */
   }
   else
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
         {
            HYPRE_Int jj;
            HYPRE_Complex res = f_data[i];
            HYPRE_Complex res0 = 0.0;
            HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_diag_j[jj];
               res0 -= A_diag_data[jj] * u_data[ii];
               res2 += A_diag_data[jj] * v_tmp_data[ii];
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
         }
      } /* for ( i = ...) */
   }
}

/* Scaled Threaded version */
static inline void
hypre_HybridGaussSeidelThreads( HYPRE_Int     *A_diag_i,
                                HYPRE_Int     *A_diag_j,
                                HYPRE_Complex *A_diag_data,
                                HYPRE_Int     *A_offd_i,
                                HYPRE_Int     *A_offd_j,
                                HYPRE_Complex *A_offd_data,
                                HYPRE_Complex *f_data,
                                HYPRE_Int     *cf_marker,
                                HYPRE_Int      relax_points,
                                HYPRE_Real     relax_weight,
                                HYPRE_Real     omega,
                                HYPRE_Real     one_minus_omega,
                                HYPRE_Real     prod,
                                HYPRE_Complex *l1_norms,
                                HYPRE_Complex *u_data,
                                HYPRE_Complex *v_tmp_data,
                                HYPRE_Complex *v_ext_data,
                                HYPRE_Int      ns,
                                HYPRE_Int      ne,
                                HYPRE_Int      ibegin,
                                HYPRE_Int      iend,
                                HYPRE_Int      iorder,
                                HYPRE_Int      Skip_diag )
{
   HYPRE_Int i;
   const HYPRE_Complex zero = 0.0;

   /*-----------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    * If i is of the right type ( C or F or All) and diagonal is
    * nonzero, relax point i; otherwise, skip it.
    *-----------------------------------------------------------*/
   if (l1_norms)
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && l1_norms[i] != zero )
         {
            HYPRE_Int jj;
            HYPRE_Complex res = f_data[i];
            HYPRE_Complex res0 = 0.0;
            HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res0 -= A_diag_data[jj] * u_data[ii];
                  res2 += A_diag_data[jj] * v_tmp_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) / l1_norms[i];
            }
         }
      } /* for ( i = ...) */
   }
   else
   {
      for (i = ibegin; i != iend; i += iorder)
      {
         if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
         {
            HYPRE_Int jj;
            HYPRE_Complex res = f_data[i];
            HYPRE_Complex res0 = 0.0;
            HYPRE_Complex res2 = 0.0;

            for (jj = A_diag_i[i] + Skip_diag; jj < A_diag_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_diag_j[jj];
               if (ii >= ns && ii < ne)
               {
                  res0 -= A_diag_data[jj] * u_data[ii];
                  res2 += A_diag_data[jj] * v_tmp_data[ii];
               }
               else
               {
                  res -= A_diag_data[jj] * v_tmp_data[ii];
               }
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               const HYPRE_Int ii = A_offd_j[jj];
               res -= A_offd_data[jj] * v_ext_data[ii];
            }

            if (Skip_diag)
            {
               u_data[i] *= prod;
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
            else
            {
               u_data[i] += relax_weight * (omega * res + res0 + one_minus_omega * res2) /
                            A_diag_data[A_diag_i[i]];
            }
         }
      } /* for ( i = ...) */
   }
}


/* Ordered Version */
static inline void
hypre_HybridGaussSeidelOrderedNS( HYPRE_Int     *A_diag_i,
                                  HYPRE_Int     *A_diag_j,
                                  HYPRE_Complex *A_diag_data,
                                  HYPRE_Int     *A_offd_i,
                                  HYPRE_Int     *A_offd_j,
                                  HYPRE_Complex *A_offd_data,
                                  HYPRE_Complex *f_data,
                                  HYPRE_Int     *cf_marker,
                                  HYPRE_Int      relax_points,
                                  HYPRE_Complex *u_data,
                                  HYPRE_Complex *v_tmp_data,
                                  HYPRE_Complex *v_ext_data,
                                  HYPRE_Int      ibegin,
                                  HYPRE_Int      iend,
                                  HYPRE_Int      iorder,
                                  HYPRE_Int     *proc_ordering )
{
   HYPRE_UNUSED_VAR(v_tmp_data);

   HYPRE_Int j;
   const HYPRE_Complex zero = 0.0;

   for (j = ibegin; j != iend; j += iorder)
   {
      const HYPRE_Int i = proc_ordering[j];
      /*-----------------------------------------------------------
       * Relax only C or F points as determined by relax_points.
       * If i is of the right type ( C or F or All) and diagonal is
       * nonzero, relax point i; otherwise, skip it.
       *-----------------------------------------------------------*/
      if ( (relax_points == 0 || cf_marker[i] == relax_points) && A_diag_data[A_diag_i[i]] != zero )
      {
         HYPRE_Int jj;
         HYPRE_Complex res = f_data[i];

         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            const HYPRE_Int ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
         }

         for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
         {
            const HYPRE_Int ii = A_offd_j[jj];
            res -= A_offd_data[jj] * v_ext_data[ii];
         }

         u_data[i] = res / A_diag_data[A_diag_i[i]];
      }
   } /* for ( i = ...) */
}

#endif /* #ifndef HYPRE_PAR_RELAX_HEADER */
