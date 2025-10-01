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

/*--------------------------------------------------------------------------
 * Compute the fused product for F terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_f( HYPRE_Int            nprod,
                                   HYPRE_Complex       *cprod,
                                   hypre_3Cptrs        *tptrs,
                                   hypre_1Cptr         *mptrs,
                                   HYPRE_Int           *mentries,
                                   HYPRE_Int            ndim,
                                   hypre_Index          loop_size,
                                   hypre_Box           *fdbox,
                                   hypre_Index          fdstart,
                                   hypre_Index          fdstride,
                                   hypre_Box           *Mdbox,
                                   hypre_Index          Mdstart,
                                   hypre_Index          Mdstride )
{
   HYPRE_Int     k, depth;

   /* Variable declarations for up to HYPRE_FUSE_MAXDEPTH variables */
   HYPRE_SMMFUSE_DECLARE_20_3VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_F POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=====================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("f");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 20:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 20);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_20;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 19:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 19);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_19;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 18);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_18;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 17);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_17;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 16);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_16;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 15);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_15;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 14);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_14;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 13);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_13;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 12);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_12;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 11);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_11;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 10);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_10;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 9);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_9;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 8);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_8;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_7;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_6;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_5;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_4;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_3;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_2;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_F_UP_TO_1;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            hypre_GpuProfilingPopRange();
            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }

#if defined(HYPRE_USING_GPU)
      hypre_SyncComputeStream();
#endif
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
