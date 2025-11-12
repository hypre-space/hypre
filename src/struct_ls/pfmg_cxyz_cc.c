/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"
#include "pfmg_cxyz.h"

/*--------------------------------------------------------------------------
 * Core function for computing stencil collapsing for constant coefficients.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeCxyz_core_CC(hypre_StructMatrix *A,
                              HYPRE_Int           Ab,
                              HYPRE_Int           diag_is_constant,
                              HYPRE_Int           diag_entry,
                              HYPRE_Int          *nentries,
                              HYPRE_Int         **entries,
                              hypre_IndexRef      start,
                              hypre_Index         loop_size,
                              hypre_Box          *A_dbox,
                              hypre_Box          *w_dbox,
                              HYPRE_Real        **w_data)
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_Index           ustride;
   HYPRE_Int             k = 0, d, depth, all_zero;

   HYPRE_Real           *w_data_d, *w_data_0, *w_data_1, *w_data_2;
   HYPRE_Real           *A_diag = NULL;
   HYPRE_AP_DECLARE_UP_TO_9;

   /* Exit if there are no constant coefficients */
   all_zero = 1;
   for (d = 0; d < ndim; d++)
   {
      if (nentries[d] > 0)
      {
         all_zero = 0;
         break;
      }
   }
   if (all_zero)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("CC");

   hypre_SetIndex(ustride, 1);
   A_diag = (diag_is_constant) ?
            hypre_StructMatrixConstData(A, diag_entry) :
            hypre_StructMatrixBoxData(A, Ab, diag_entry);

   /* Set w_data pointers */
   switch (ndim)
   {
      case 3:
         w_data_2 = w_data[2];
         HYPRE_FALLTHROUGH;

      case 2:
         w_data_1 = w_data[1];
         HYPRE_FALLTHROUGH;

      case 1:
         w_data_0 = w_data[0];
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!");
         return hypre_error_flag;
   }

#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(a0, a1, a2)                        \
   /* Load Ap pointers */                                  \
   HYPRE_CAP_LOAD_UP_TO_##a0(0);                           \
   HYPRE_CAP_LOAD_UP_TO_##a1(1);                           \
   HYPRE_CAP_LOAD_UP_TO_##a2(2);                           \
                                                           \
   /* Compute w_data */                                    \
   hypre_BoxLoop2Begin(ndim, loop_size,                    \
                       A_dbox, start, ustride, Ai,         \
                       w_dbox, start, ustride, wi);        \
   {                                                       \
      HYPRE_CXYZ_DEFINE_SIGN;                              \
                                                           \
      w_data_0[wi] = sign * (HYPRE_CAP_SUM_UP_TO_##a0(0)); \
      w_data_1[wi] = sign * (HYPRE_CAP_SUM_UP_TO_##a1(1)); \
      w_data_2[wi] = sign * (HYPRE_CAP_SUM_UP_TO_##a2(2)); \
   }                                                       \
   hypre_BoxLoop2End(Ai, wi)

   /* Compute w_data pointers by summing contributions from A.
      Specialization is used for common stencil sizes to optimize for performance.
      A generic fallback algorithm is used otherwise */
   if (ndim == 3 && nentries[0] == 9 && nentries[1] == 9 && nentries[2] == 9)
   {
      HYPRE_CORE_CASE(9, 9, 9);
   }
   else if (ndim == 3 && nentries[0] == 2 && nentries[1] == 2 && nentries[2] == 2)
   {
      HYPRE_CORE_CASE(2, 2, 2);
   }
   else
   {
      /* Fallback to generic algorithm */
      for (d = 0; d < ndim; d++)
      {
         w_data_d = w_data[d];

         /* Compute row sums */
         for (k = 0; k < nentries[d]; k += HYPRE_UNROLL_MAXDEPTH)
         {
            depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nentries[d] - k));

#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                               \
           case n:                                                       \
              HYPRE_CAP_LOAD_UP_TO_##n(d);                               \
              hypre_BoxLoop2Begin(ndim, loop_size,                       \
                                  A_dbox, start, ustride, Ai,            \
                                  w_dbox, start, ustride, wi);           \
              {                                                          \
                 HYPRE_CXYZ_DEFINE_SIGN;                                 \
                 if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }  \
                                                                         \
                 w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_##n(d));    \
              }                                                          \
              hypre_BoxLoop2End(Ai, wi);                                 \
              break

            switch (depth)
            {
                  HYPRE_CORE_CASE(9);
                  HYPRE_CORE_CASE(8);
                  HYPRE_CORE_CASE(7);
                  HYPRE_CORE_CASE(6);
                  HYPRE_CORE_CASE(5);
                  HYPRE_CORE_CASE(4);
                  HYPRE_CORE_CASE(3);
                  HYPRE_CORE_CASE(2);
                  HYPRE_CORE_CASE(1);

               default:
                  break;
            } /* switch (nentries) */
         }

#if defined(HYPRE_USING_GPU)
         hypre_SyncComputeStream();
#endif
      } /* for (d = 0; d < ndim; d++) */
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
