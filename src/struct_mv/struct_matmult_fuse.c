/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured matrix-matrix multiply kernel functions
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"
#include "struct_matmult_fuse.h"

#if defined (HYPRE_FUSE_FCC_FC_F)

/*--------------------------------------------------------------------------
 * Compute the fused product for FCC, FC and F terms combined
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_fcc_fc_f( HYPRE_Int            nprod_fcc,
                                          HYPRE_Complex       *cprod_fcc,
                                          hypre_3Cptrs        *tptrs_fcc,
                                          hypre_1Cptr         *mptrs_fcc,
                                          HYPRE_Int           *mentries_fcc,
                                          HYPRE_Int            nprod_fc,
                                          HYPRE_Complex       *cprod_fc,
                                          hypre_3Cptrs        *tptrs_fc,
                                          hypre_1Cptr         *mptrs_fc,
                                          HYPRE_Int           *mentries_fc,
                                          HYPRE_Int            nprod_f,
                                          HYPRE_Complex       *cprod_f,
                                          hypre_3Cptrs        *tptrs_f,
                                          hypre_1Cptr         *mptrs_f,
                                          HYPRE_Int           *mentries_f,
                                          HYPRE_Int            ndim,
                                          hypre_Index          loop_size,
                                          hypre_Box           *fdbox,
                                          hypre_Index          fdstart,
                                          hypre_Index          fdstride,
                                          hypre_Box           *cdbox,
                                          hypre_Index          cdstart,
                                          hypre_Index          cdstride,
                                          hypre_Box           *Mdbox,
                                          hypre_Index          Mdstart,
                                          hypre_Index          Mdstride,
                                          HYPRE_Int            Mnum_values,
                                          hypre_1Cptr         *mptrs,
                                          HYPRE_Int           *combined_ptr)
{
   /* Declare Mptrs, FFC, FC, and F data pointers */
   HYPRE_SMMFUSE_DECLARE_MPTRS_UP_TO_27;
   HYPRE_SMMFUSE_DECLARE_FCC_UP_TO_36;
   HYPRE_SMMFUSE_DECLARE_FC_UP_TO_72;
   HYPRE_SMMFUSE_DECLARE_F_UP_TO_9;

   /* Set flag for combined execution of FCC, FC, and F products */
   *combined_ptr = 0;
   if ((nprod_fcc == 11 && nprod_fc == 6  && nprod_f == 3 && Mnum_values == 8)  ||
       (nprod_fcc == 19 && nprod_fc == 14 && nprod_f == 5 && Mnum_values == 14) ||
       (nprod_fcc == 19 && nprod_fc == 38 && nprod_f == 5 && Mnum_values == 14) ||
       (nprod_fcc == 20 && nprod_fc == 8  && nprod_f == 5 && Mnum_values == 15) ||
       (nprod_fcc == 36 && nprod_fc == 24 && nprod_f == 9 && Mnum_values == 27) ||
       (nprod_fcc == 36 && nprod_fc == 72 && nprod_f == 9 && Mnum_values == 27))
   {
      *combined_ptr = 1;
   }
   else
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_FCC_FC_F POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of (FCC, FC, F) products: (%d, %d, %d)\n",
                   nprod_fcc, nprod_fc, nprod_f);
   for (k = 0; k < nprod_fcc; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "FCC Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod_fcc[k], cprod_fcc[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs_fcc[k], mentries_fcc[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs_fcc[k][0]);
   }
   for (k = 0; k < nprod_fc; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "FC Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod_fc[k], cprod_fc[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs_fc[k], mentries_fc[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs_fc[k][0]);
   }
   for (k = 0; k < nprod_f; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "F Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod_f[k], cprod_f[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs_f[k], mentries_f[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs_f[k][0]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=====================\n");

#else
   HYPRE_UNUSED_VAR(mentries_f);
   HYPRE_UNUSED_VAR(mentries_fc);
   HYPRE_UNUSED_VAR(mentries_fcc);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("fcc_fc_f");

   if (nprod_fcc == 11 && nprod_fc == 6 && nprod_f == 3 && Mnum_values == 8)
   {
      /* Load Mptrs, FCC, FC, and F data pointers */
      HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_8;
      HYPRE_SMMFUSE_LOAD_FCC_UP_TO_11;
      HYPRE_SMMFUSE_LOAD_FC_UP_TO_6;
      HYPRE_SMMFUSE_LOAD_F_UP_TO_3;

      hypre_BoxLoop3Begin(ndim, loop_size,
                          Mdbox, Mdstart, Mdstride, Mi,
                          fdbox, fdstart, fdstride, fi,
                          cdbox, cdstart, cdstride, ci);
      {
         HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_8;

         HYPRE_SMMFUSE_CALC_FCC_UP_TO_11;
         HYPRE_SMMFUSE_CALC_FC_UP_TO_6;
         HYPRE_SMMFUSE_CALC_F_UP_TO_3;
      }
      hypre_BoxLoop3End(Mi, fi, ci);
   }
   else if (nprod_fcc == 19 && nprod_fc == 14 && nprod_f == 5 && Mnum_values == 14)
   {
      /* Load Mptrs, FCC, FC, and F data pointers */
      HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_14;
      HYPRE_SMMFUSE_LOAD_FCC_UP_TO_19;
      HYPRE_SMMFUSE_LOAD_FC_UP_TO_14;
      HYPRE_SMMFUSE_LOAD_F_UP_TO_5;

      hypre_BoxLoop3Begin(ndim, loop_size,
                          Mdbox, Mdstart, Mdstride, Mi,
                          fdbox, fdstart, fdstride, fi,
                          cdbox, cdstart, cdstride, ci);
      {
         HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_14;

         HYPRE_SMMFUSE_CALC_FCC_UP_TO_19;
         HYPRE_SMMFUSE_CALC_FC_UP_TO_14;
         HYPRE_SMMFUSE_CALC_F_UP_TO_5;
      }
      hypre_BoxLoop3End(Mi, fi, ci);
   }
   else if (nprod_fcc == 19 && nprod_fc == 38 && nprod_f == 5 && Mnum_values == 14)
   {
      /* Load Mptrs, FCC, FC, and F data pointers */
      HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_14;
      HYPRE_SMMFUSE_LOAD_FCC_UP_TO_19;
      HYPRE_SMMFUSE_LOAD_FC_UP_TO_38;
      HYPRE_SMMFUSE_LOAD_F_UP_TO_5;

      hypre_BoxLoop3Begin(ndim, loop_size,
                          Mdbox, Mdstart, Mdstride, Mi,
                          fdbox, fdstart, fdstride, fi,
                          cdbox, cdstart, cdstride, ci);
      {
         HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_14;

         HYPRE_SMMFUSE_CALC_FCC_UP_TO_19;
         HYPRE_SMMFUSE_CALC_FC_UP_TO_38;
         HYPRE_SMMFUSE_CALC_F_UP_TO_5;
      }
      hypre_BoxLoop3End(Mi, fi, ci);
   }
   else if (nprod_fcc == 20 && nprod_fc == 8 && nprod_f == 5 && Mnum_values == 15)
   {
      /* Load Mptrs, FCC, FC, and F data pointers */
      HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_15;
      HYPRE_SMMFUSE_LOAD_FCC_UP_TO_20;
      HYPRE_SMMFUSE_LOAD_FC_UP_TO_8;
      HYPRE_SMMFUSE_LOAD_F_UP_TO_5;

      hypre_BoxLoop3Begin(ndim, loop_size,
                          Mdbox, Mdstart, Mdstride, Mi,
                          fdbox, fdstart, fdstride, fi,
                          cdbox, cdstart, cdstride, ci);
      {
         HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_15;

         HYPRE_SMMFUSE_CALC_FCC_UP_TO_20;
         HYPRE_SMMFUSE_CALC_FC_UP_TO_8;
         HYPRE_SMMFUSE_CALC_F_UP_TO_5;
      }
      hypre_BoxLoop3End(Mi, fi, ci);
   }
   else if (nprod_fcc == 36 && nprod_fc == 24 && nprod_f == 9 && Mnum_values == 27)
   {
      /* Load Mptrs, FCC, FC, and F data pointers */
      HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_27;
      HYPRE_SMMFUSE_LOAD_FCC_UP_TO_36;
      HYPRE_SMMFUSE_LOAD_FC_UP_TO_24;
      HYPRE_SMMFUSE_LOAD_F_UP_TO_9;

      hypre_BoxLoop3Begin(ndim, loop_size,
                          Mdbox, Mdstart, Mdstride, Mi,
                          fdbox, fdstart, fdstride, fi,
                          cdbox, cdstart, cdstride, ci);
      {
         HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_27;

         HYPRE_SMMFUSE_CALC_FCC_UP_TO_36;
         HYPRE_SMMFUSE_CALC_FC_UP_TO_24;
         HYPRE_SMMFUSE_CALC_F_UP_TO_9;
      }
      hypre_BoxLoop3End(Mi, fi, ci);
   }
   else if (nprod_fcc == 36 && nprod_fc == 72 && nprod_f == 9 && Mnum_values == 27)
   {
      /* Load Mptrs, FCC, FC, and F data pointers */
      HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_27;
      HYPRE_SMMFUSE_LOAD_FCC_UP_TO_36;
      HYPRE_SMMFUSE_LOAD_FC_UP_TO_72;
      HYPRE_SMMFUSE_LOAD_F_UP_TO_9;

      hypre_BoxLoop3Begin(ndim, loop_size,
                          Mdbox, Mdstart, Mdstride, Mi,
                          fdbox, fdstart, fdstride, fi,
                          cdbox, cdstart, cdstride, ci);
      {
         HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_27;

         HYPRE_SMMFUSE_CALC_FCC_UP_TO_36;
         HYPRE_SMMFUSE_CALC_FC_UP_TO_72;
         HYPRE_SMMFUSE_CALC_F_UP_TO_9;
      }
      hypre_BoxLoop3End(Mi, fi, ci);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!");
   }

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream();
#endif

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
#endif

#if defined(USE_FUSE_SORT)
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_fuse_order_bigints( HYPRE_Int      nprod,
                          HYPRE_Int     *order,
                          HYPRE_BigInt  *bigints )
{
   HYPRE_Int     k;
   HYPRE_BigInt  tmp_bigints[HYPRE_MAX_MMTERMS];

   if (nprod >= HYPRE_MAX_MMTERMS)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Reached maximum allowed product index! Increase HYPRE_MAX_MMTERMS!");
      return hypre_error_flag;
   }

   for (k = 0; k < nprod; k++)
   {
      tmp_bigints[k] = bigints[order[k]];
   }
   for (k = 0; k < nprod; k++)
   {
      bigints[k] = tmp_bigints[k];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_fuse_order_ptrs( HYPRE_Int       nprod,
                       HYPRE_Int      *order,
                       hypre_3Cptrs   *tptrs,
                       hypre_1Cptr    *mptrs )
{
   HYPRE_Int     k, i;
   hypre_3Cptrs  tmp_tptrs[HYPRE_MAX_MMTERMS];
   hypre_1Cptr   tmp_mptrs[HYPRE_MAX_MMTERMS];

   if (nprod >= HYPRE_MAX_MMTERMS)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Reached maximum allowed product index! Increase HYPRE_MAX_MMTERMS!");
      return hypre_error_flag;
   }

   for (k = 0; k < nprod; k++)
   {
      for (i = 0; i < 3; i++)
      {
         tmp_tptrs[k][i] = tptrs[order[k]][i];
      }
   }
   for (k = 0; k < nprod; k++)
   {
      for (i = 0; i < 3; i++)
      {
         tptrs[k][i] = tmp_tptrs[k][i];
      }
   }
   for (k = 0; k < nprod; k++)
   {
      tmp_mptrs[k] = mptrs[order[k]];
   }
   for (k = 0; k < nprod; k++)
   {
      mptrs[k] = tmp_mptrs[k];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_fuse_sort( HYPRE_Int        nprod,
                 hypre_3Cptrs    *tptrs,
                 hypre_1Cptr     *mptrs )
{
   HYPRE_Int       approach = 1;

   HYPRE_Int       k;
   HYPRE_Complex  *minptrs[4];
   HYPRE_BigInt    distances[4][HYPRE_MAX_MMTERMS];
   HYPRE_Int       order[HYPRE_MAX_MMTERMS];

   if ((nprod < 1) || (approach == 0))
   {
      return hypre_error_flag;
   }

   if (nprod >= HYPRE_MAX_MMTERMS)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Reached maximum allowed product index! Increase HYPRE_MAX_MMTERMS!");
      return hypre_error_flag
   }

   /* Get minimum pointer addresses */
   minptrs[0] = tptrs[0][0];
   minptrs[1] = tptrs[0][1];
   minptrs[2] = tptrs[0][2];
   minptrs[3] = mptrs[0];
   for (k = 1; k < nprod; k++)
   {
      minptrs[0] = hypre_min( minptrs[0], tptrs[k][0] );
      minptrs[1] = hypre_min( minptrs[1], tptrs[k][1] );
      minptrs[2] = hypre_min( minptrs[2], tptrs[k][2] );
      minptrs[3] = hypre_min( minptrs[3], mptrs[k] );
   }

   /* Compute pointer distances and order array */
   for (k = 0; k < nprod; k++)
   {
      distances[0][k] = (HYPRE_Int) (tptrs[k][0] - minptrs[0]);
      distances[1][k] = (HYPRE_Int) (tptrs[k][1] - minptrs[1]);
      distances[2][k] = (HYPRE_Int) (tptrs[k][2] - minptrs[2]);
      distances[3][k] = (HYPRE_Int) (mptrs[k]    - minptrs[3]);
      order[k] = k;
   }

#if defined(DEBUG_MATMULT)
   /* Print distances */
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "distances[%2d]   %16d %16d %16d   %16d\n",
                      k, distances[0][k], distances[1][k], distances[2][k], distances[3][k]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "\n");
#endif

   /* Sort according to middle column (index 1) */
   hypre_BigQsortbi(distances[1], order, 0, nprod - 1);

#if defined(DEBUG_MATMULT)
   hypre_fuse_order_bigints(nprod, order, distances[0]);
   hypre_fuse_order_bigints(nprod, order, distances[2]);
   hypre_fuse_order_bigints(nprod, order, distances[3]);

   /* Print order array */
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, " %d", order[k]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "\n\n");

   /* Print distances */
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "distances[%2d]   %16b %16b %16b   %16b\n",
                      k, distances[0][k], distances[1][k], distances[2][k], distances[3][k]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "\n");
#endif

   /* Reorder data pointers */
   hypre_fuse_order_ptrs(nprod, order, tptrs, mptrs);

   return hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 * Compute the fused product for all terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse( HYPRE_Int nterms,
                                 hypre_StructMatmultDataMH *a,
                                 HYPRE_Int    na,
                                 HYPRE_Int    ndim,
                                 hypre_Index  loop_size,
                                 HYPRE_Int    stencil_size,
                                 hypre_Box   *fdbox,
                                 hypre_Index  fdstart,
                                 hypre_Index  fdstride,
                                 hypre_Box   *cdbox,
                                 hypre_Index  cdstart,
                                 hypre_Index  cdstride,
                                 hypre_Box   *Mdbox,
                                 hypre_Index  Mdstart,
                                 hypre_Index  Mdstride,
                                 hypre_StructMatrix *M )
{
#if defined(HYPRE_FUSE_FCC_FC_F)
   HYPRE_Int       Mnum_values   = hypre_StructMatrixNumValues(M);
   hypre_1Cptr     mmptrs[HYPRE_MAX_MMTERMS];
#else
   HYPRE_UNUSED_VAR(M);
   HYPRE_UNUSED_VAR(stencil_size);
#endif

   HYPRE_Int       nprod[8] = {0};
   HYPRE_Complex   cprod[8][HYPRE_MAX_MMTERMS];
   hypre_3Cptrs    tptrs[8][HYPRE_MAX_MMTERMS];
   hypre_1Cptr     mptrs[8][HYPRE_MAX_MMTERMS];
   HYPRE_Int       mentries[8][HYPRE_MAX_MMTERMS];

   HYPRE_Int       ptype = 0, nf, nc, nt;
   HYPRE_Int       i, k, t;
   HYPRE_Int       combined_fcc_fc_f = 0;

   /* Sanity check */
   if (nterms > 3)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Can't have more than 3 terms in StructMatmultCompute_fuse!");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("fuse");

   /* Build product arrays */
   for (i = 0; i < na; i++)
   {
      /* Determine number of fine and coarse terms */
      nf = nc = 0;
      for (t = 0; t < nterms; t++)
      {
         if (a[i].types[t] == 1)
         {
            /* Type 1 -> coarse data space */
            nc++;
         }
         else if (a[i].types[t] != 3)
         {
            /* Type 0 or 2 -> fine data space */
            nf++;
         }
      }
      nt = nf + nc;

      /* Determine product type */
      switch (nt)
      {
         case 3:
            switch (nc)
            {
               case 0: /* fff term (call fuse_fff) */
                  ptype = 0;
                  break;

               case 1: /* ffc term (call fuse_ffc) */
                  ptype = 1;
                  break;

               case 2: /* fcc term (call fuse_fcc) */
                  ptype = 2;
                  break;
            }
            break;

         case 2:
            switch (nc)
            {
               case 0: /* ff term (call core_ff) */
                  ptype = 3;
                  break;

               case 1: /* cf term (call core_fc) */
                  ptype = 4;
                  break;

               case 2: /* cc term (call core_cc) */
                  ptype = 5;
                  break;
            }
            break;

         case 1:
            switch (nc)
            {
               case 0: /* f term (call core_f) */
                  ptype = 6;
                  break;

               case 1: /* c term (call core_c) */
                  ptype = 7;
                  break;
            }
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Can't have zero terms in StructMatmult!");
            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_FUNC_END;

            return hypre_error_flag;
      }

      /* Retrieve product index */
      k = nprod[ptype];
      if (k >= HYPRE_MAX_MMTERMS)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Reached maximum allowed product index! Increase HYPRE_MAX_MMTERMS!");
         hypre_GpuProfilingPopRange();
         HYPRE_ANNOTATE_FUNC_END;
         return hypre_error_flag;
      }

      /* Set array values for k-th product of type "ptype" */
      cprod[ptype][k] = a[i].cprod;
      for (t = 0; t < nterms; t++)
      {
         tptrs[ptype][k][t] = NULL;
      }
      for (nf = 0, nc = 0, t = 0; t < nterms; t++)
      {
         if (a[i].types[t] == 1)
         {
            /* Type 1 -> coarse data space */
            tptrs[ptype][k][nt - 1 - nc] = a[i].tptrs[t];  /* put last */
            nc++;
         }
         else if (a[i].types[t] != 3)
         {
            /* Type 0 or 2 -> fine data space */
            tptrs[ptype][k][nf] = a[i].tptrs[t];  /* put first */
            nf++;
         }
      }
      mentries[ptype][k] = a[i].mentry;
      mptrs[ptype][k] = a[i].mptr;
      nprod[ptype]++;
   } /* loop i < na*/

#if defined(USE_FUSE_SORT)
   for (i = 0; i < 8; i++)
   {
      hypre_fuse_sort(nprod[i], tptrs[i], mptrs[i]);
   }
#endif

#if defined(DEBUG_MATMULT)
   const char *cases[8] = {"FFF", "FFC", "FCC", "FF", "FC", "CC", "F", "C"};

   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products - ");
   for (t = 0, k = 0, i = 0; i < 8; i++)
   {
      if (nprod[i] > 0)
      {
         hypre_ParPrintf(hypre_MPI_COMM_WORLD, "%s: %d | ", cases[i], nprod[i]);
      }
      t += nprod[i];
      k += hypre_ceildiv(nprod[i], HYPRE_FUSE_MAXDEPTH);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Sum: %d (%d BoxLoops)\n", t, k);
#endif

#if defined(HYPRE_FUSE_FCC_FC_F)
   /* Build pointers to M data arrays */
   for (k = 0, t = 0; k < stencil_size; k++)
   {
      for (i = 0; i < na; i++)
      {
         if (a[i].mentry != k)
         {
            continue;
         }

         mmptrs[t++] = a[i].mptr;
         i = na;
      }
   }

   /* Call fully fused (combined) functions */
   hypre_StructMatmultCompute_fuse_fcc_fc_f(nprod[2], cprod[2], tptrs[2], mptrs[2], mentries[2],
                                            nprod[4], cprod[4], tptrs[4], mptrs[4], mentries[4],
                                            nprod[6], cprod[6], tptrs[6], mptrs[6], mentries[6],
                                            ndim, loop_size,
                                            fdbox, fdstart, fdstride,
                                            cdbox, cdstart, cdstride,
                                            Mdbox, Mdstart, Mdstride,
                                            Mnum_values, mmptrs,
                                            &combined_fcc_fc_f);
#endif

   /* Call individual fuse functions */
   if (!combined_fcc_fc_f)
   {
      hypre_StructMatmultCompute_fuse_fcc(nprod[2], cprod[2], tptrs[2],
                                          mptrs[2], mentries[2],
                                          ndim, loop_size,
                                          fdbox, fdstart, fdstride,
                                          cdbox, cdstart, cdstride,
                                          Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_fuse_fc(nprod[4], cprod[4], tptrs[4],
                                         mptrs[4], mentries[4],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         cdbox, cdstart, cdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_fuse_f(nprod[6], cprod[6], tptrs[6],
                                        mptrs[6], mentries[6],
                                        ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        Mdbox, Mdstart, Mdstride);
   }

   hypre_StructMatmultCompute_fuse_fff(nprod[0], cprod[0], tptrs[0],
                                       mptrs[0], mentries[0],
                                       ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_ffc(nprod[1], cprod[1], tptrs[1],
                                       mptrs[1], mentries[1],
                                       ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       cdbox, cdstart, cdstride,
                                       Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_ff(nprod[3], cprod[3], tptrs[3],
                                      mptrs[3], mentries[3],
                                      ndim, loop_size,
                                      fdbox, fdstart, fdstride,
                                      Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_cc(nprod[5], cprod[5], tptrs[5],
                                      mptrs[5], mentries[5],
                                      ndim, loop_size,
                                      cdbox, cdstart, cdstride,
                                      Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_c(nprod[7], cprod[7], tptrs[7],
                                     mptrs[7], mentries[7],
                                     ndim, loop_size,
                                     cdbox, cdstart, cdstride,
                                     Mdbox, Mdstart, Mdstride);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
