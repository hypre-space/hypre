/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

#if defined(HYPRE_UNROLL_MAXDEPTH)
#undef HYPRE_UNROLL_MAXDEPTH
#endif
#define HYPRE_UNROLL_MAXDEPTH 9

#define HYPRE_CXYZ_DEFINE_SIGN         \
  HYPRE_Real sign = diag_is_constant ? \
    (A_diag[0]  < 0.0 ? 1.0 : -1.0) :  \
    (A_diag[Ai] < 0.0 ? 1.0 : -1.0)

#define HYPRE_AP_DECLARE(n)     \
  HYPRE_Complex *Ap##n##_d, *Ap##n##_0, *Ap##n##_1, *Ap##n##_2

#define HYPRE_AP_LOAD(n, d)     \
  Ap##n##_##d = hypre_StructMatrixBoxData(A, Ab, entries[d][n])

#define HYPRE_CAP_LOAD(n, d)    \
  Ap##n##_##d = hypre_StructMatrixConstData(A, entries[d][n])

#define HYPRE_AP_EVAL(n, d)     \
  Ap##n##_##d[Ai]

#define HYPRE_CAP_EVAL(n, d)    \
  Ap##n##_##d[0]

#define HYPRE_AP_DECLARE_UP_TO_9 \
  HYPRE_AP_DECLARE(0);           \
  HYPRE_AP_DECLARE(1);           \
  HYPRE_AP_DECLARE(2);           \
  HYPRE_AP_DECLARE(3);           \
  HYPRE_AP_DECLARE(4);           \
  HYPRE_AP_DECLARE(5);           \
  HYPRE_AP_DECLARE(6);           \
  HYPRE_AP_DECLARE(7);           \
  HYPRE_AP_DECLARE(8)

#define HYPRE_AP_LOAD_UP_TO_1(d) \
  HYPRE_AP_LOAD(0, d)

#define HYPRE_AP_LOAD_UP_TO_2(d) \
  HYPRE_AP_LOAD_UP_TO_1(d);      \
  HYPRE_AP_LOAD(1, d)

#define HYPRE_AP_LOAD_UP_TO_3(d) \
  HYPRE_AP_LOAD_UP_TO_2(d);      \
  HYPRE_AP_LOAD(2, d)

#define HYPRE_AP_LOAD_UP_TO_4(d) \
  HYPRE_AP_LOAD_UP_TO_3(d);      \
  HYPRE_AP_LOAD(3, d)

#define HYPRE_AP_LOAD_UP_TO_5(d) \
  HYPRE_AP_LOAD_UP_TO_4(d);      \
  HYPRE_AP_LOAD(4, d)

#define HYPRE_AP_LOAD_UP_TO_6(d) \
  HYPRE_AP_LOAD_UP_TO_5(d);      \
  HYPRE_AP_LOAD(5, d)

#define HYPRE_AP_LOAD_UP_TO_7(d) \
  HYPRE_AP_LOAD_UP_TO_6(d);      \
  HYPRE_AP_LOAD(6, d)

#define HYPRE_AP_LOAD_UP_TO_8(d) \
  HYPRE_AP_LOAD_UP_TO_7(d);      \
  HYPRE_AP_LOAD(7, d)

#define HYPRE_AP_LOAD_UP_TO_9(d) \
  HYPRE_AP_LOAD_UP_TO_8(d);      \
  HYPRE_AP_LOAD(8, d)

#define HYPRE_CAP_LOAD_UP_TO_1(d) \
  HYPRE_CAP_LOAD(0, d)

#define HYPRE_CAP_LOAD_UP_TO_2(d) \
  HYPRE_CAP_LOAD_UP_TO_1(d);      \
  HYPRE_CAP_LOAD(1, d)

#define HYPRE_CAP_LOAD_UP_TO_3(d) \
  HYPRE_CAP_LOAD_UP_TO_2(d);      \
  HYPRE_CAP_LOAD(2, d)

#define HYPRE_CAP_LOAD_UP_TO_4(d) \
  HYPRE_CAP_LOAD_UP_TO_3(d);      \
  HYPRE_CAP_LOAD(3, d)

#define HYPRE_CAP_LOAD_UP_TO_5(d) \
  HYPRE_CAP_LOAD_UP_TO_4(d);      \
  HYPRE_CAP_LOAD(4, d)

#define HYPRE_CAP_LOAD_UP_TO_6(d) \
  HYPRE_CAP_LOAD_UP_TO_5(d);      \
  HYPRE_CAP_LOAD(5, d)

#define HYPRE_CAP_LOAD_UP_TO_7(d) \
  HYPRE_CAP_LOAD_UP_TO_6(d);      \
  HYPRE_CAP_LOAD(6, d)

#define HYPRE_CAP_LOAD_UP_TO_8(d) \
  HYPRE_CAP_LOAD_UP_TO_7(d);      \
  HYPRE_CAP_LOAD(7, d)

#define HYPRE_CAP_LOAD_UP_TO_9(d) \
  HYPRE_CAP_LOAD_UP_TO_8(d);      \
  HYPRE_CAP_LOAD(8, d)

#define HYPRE_AP_SUM_UP_TO_1(d) \
  HYPRE_AP_EVAL(0, d)

#define HYPRE_AP_SUM_UP_TO_2(d) \
  HYPRE_AP_SUM_UP_TO_1(d) +     \
  HYPRE_AP_EVAL(1, d)

#define HYPRE_AP_SUM_UP_TO_3(d) \
  HYPRE_AP_SUM_UP_TO_2(d) +     \
  HYPRE_AP_EVAL(2, d)

#define HYPRE_AP_SUM_UP_TO_4(d) \
  HYPRE_AP_SUM_UP_TO_3(d) +     \
  HYPRE_AP_EVAL(3, d)

#define HYPRE_AP_SUM_UP_TO_5(d) \
  HYPRE_AP_SUM_UP_TO_4(d) +     \
  HYPRE_AP_EVAL(4, d)

#define HYPRE_AP_SUM_UP_TO_6(d) \
  HYPRE_AP_SUM_UP_TO_5(d) +     \
  HYPRE_AP_EVAL(5, d)

#define HYPRE_AP_SUM_UP_TO_7(d) \
  HYPRE_AP_SUM_UP_TO_6(d) +     \
  HYPRE_AP_EVAL(6, d)

#define HYPRE_AP_SUM_UP_TO_8(d) \
  HYPRE_AP_SUM_UP_TO_7(d) +     \
  HYPRE_AP_EVAL(7, d)

#define HYPRE_AP_SUM_UP_TO_9(d) \
  HYPRE_AP_SUM_UP_TO_8(d) +     \
  HYPRE_AP_EVAL(8, d)

#define HYPRE_CAP_SUM_UP_TO_1(d) \
  HYPRE_CAP_EVAL(0, d)

#define HYPRE_CAP_SUM_UP_TO_2(d) \
  HYPRE_CAP_SUM_UP_TO_1(d) +     \
  HYPRE_CAP_EVAL(1, d)

#define HYPRE_CAP_SUM_UP_TO_3(d) \
  HYPRE_CAP_SUM_UP_TO_2(d) +     \
  HYPRE_CAP_EVAL(2, d)

#define HYPRE_CAP_SUM_UP_TO_4(d) \
  HYPRE_CAP_SUM_UP_TO_3(d) +     \
  HYPRE_CAP_EVAL(3, d)

#define HYPRE_CAP_SUM_UP_TO_5(d) \
  HYPRE_CAP_SUM_UP_TO_4(d) +     \
  HYPRE_CAP_EVAL(4, d)

#define HYPRE_CAP_SUM_UP_TO_6(d) \
  HYPRE_CAP_SUM_UP_TO_5(d) +     \
  HYPRE_CAP_EVAL(5, d)

#define HYPRE_CAP_SUM_UP_TO_7(d) \
  HYPRE_CAP_SUM_UP_TO_6(d) +     \
  HYPRE_CAP_EVAL(6, d)

#define HYPRE_CAP_SUM_UP_TO_8(d) \
  HYPRE_CAP_SUM_UP_TO_7(d) +     \
  HYPRE_CAP_EVAL(7, d)

#define HYPRE_CAP_SUM_UP_TO_9(d) \
  HYPRE_CAP_SUM_UP_TO_8(d) +     \
  HYPRE_CAP_EVAL(8, d)

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeMaxLevels( hypre_StructGrid   *grid,
                            HYPRE_Int          *max_levels_ptr )
{
   HYPRE_Int        ndim = hypre_StructGridNDim(grid);
   hypre_Box       *bbox = hypre_StructGridBoundingBox(grid);
   HYPRE_Int        max_levels, d;

   max_levels = 1;
   for (d = 0; d < ndim; d++)
   {
      max_levels += hypre_Log2(hypre_BoxSizeD(bbox, d)) + 2;
   }

   *max_levels_ptr = max_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeCxyz_core_VC
 *
 * Core function for computing stencil collapsing for variable coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeCxyz_core_VC(hypre_StructMatrix *A,
                              HYPRE_Int           Ab,
                              HYPRE_Int           diag_is_constant,
                              HYPRE_Int           diag_entry,
                              HYPRE_Int          *nentries,
                              HYPRE_Int         **entries,
                              hypre_IndexRef      start,
                              hypre_Index         loop_size,
                              hypre_Box          *A_dbox,
                              hypre_Box          *w_dbox,
                              HYPRE_Complex     **w_data)
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_Index           ustride;
   HYPRE_Int             k, d, depth;

   HYPRE_Complex        *w_data_d, *w_data_0, *w_data_1, *w_data_2;
   HYPRE_Complex        *A_diag = NULL;
   HYPRE_AP_DECLARE_UP_TO_9;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("VC");

   hypre_SetIndex(ustride, 1);

   /* Set A_diag pointer */
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

   /* Compute w_data pointers by summing contributions from A.
      Specialization is used for common stencil sizes to optimize for performance.
      A generic fallback algorithm is used otherwise */
   if (ndim == 3 && nentries[0] == 9 && nentries[1] == 9 && nentries[2] == 9)
   {
      /* Load Ap pointers (9, 9, 9) */
      HYPRE_AP_LOAD_UP_TO_9(0);
      HYPRE_AP_LOAD_UP_TO_9(1);
      HYPRE_AP_LOAD_UP_TO_9(2);

      /* Compute w_data */
      hypre_BoxLoop2Begin(ndim, loop_size,
                          A_dbox, start, ustride, Ai,
                          w_dbox, start, ustride, wi);
      {
         HYPRE_CXYZ_DEFINE_SIGN;

         w_data_0[wi] = sign * (HYPRE_AP_SUM_UP_TO_9(0));
         w_data_1[wi] = sign * (HYPRE_AP_SUM_UP_TO_9(1));
         w_data_2[wi] = sign * (HYPRE_AP_SUM_UP_TO_9(2));
      }
      hypre_BoxLoop2End(Ai, wi);
   }
   else if (ndim == 3 && nentries[0] == 2 && nentries[1] == 2 && nentries[2] == 2)
   {
      /* Load Ap pointers (2, 2, 2) */
      HYPRE_AP_LOAD_UP_TO_2(0);
      HYPRE_AP_LOAD_UP_TO_2(1);
      HYPRE_AP_LOAD_UP_TO_2(2);

      /* Compute w_data */
      hypre_BoxLoop2Begin(ndim, loop_size,
                          A_dbox, start, ustride, Ai,
                          w_dbox, start, ustride, wi);
      {
         HYPRE_CXYZ_DEFINE_SIGN;

         w_data_0[wi] = sign * (HYPRE_AP_SUM_UP_TO_2(0));
         w_data_1[wi] = sign * (HYPRE_AP_SUM_UP_TO_2(1));
         w_data_2[wi] = sign * (HYPRE_AP_SUM_UP_TO_2(2));
      }
      hypre_BoxLoop2End(Ai, wi);
   }
   else
   {
      /* Fallback to general algorithm */
      for (d = 0; d < ndim; d++)
      {
         w_data_d = w_data[d];
         for (k = 0; k < nentries[d]; k += HYPRE_UNROLL_MAXDEPTH)
         {
            depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nentries[d] - k));

            switch (depth)
            {
               case 9:
                  HYPRE_AP_LOAD_UP_TO_9(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_9(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 8:
                  HYPRE_AP_LOAD_UP_TO_8(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_8(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 7:
                  HYPRE_AP_LOAD_UP_TO_7(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_7(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 6:
                  HYPRE_AP_LOAD_UP_TO_6(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_6(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 5:
                  HYPRE_AP_LOAD_UP_TO_5(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_5(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 4:
                  HYPRE_AP_LOAD_UP_TO_4(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_4(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 3:
                  HYPRE_AP_LOAD_UP_TO_3(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_3(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 2:
                  HYPRE_AP_LOAD_UP_TO_2(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_2(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 1:
                  HYPRE_AP_LOAD_UP_TO_1(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_AP_SUM_UP_TO_1(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;
            }
         }
      }

#if defined(DEBUG_CXYZ)
      hypre_printf("ndim = %d | nentries[0] = %d | nentries[1] = %d | nentries[2] = %d\n",
                   ndim, nentries[0], nentries[1], nentries[2]);
#endif
   }

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream();
#endif
   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeCxyz_core_CC
 *
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
                              HYPRE_Complex     **w_data)
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_Index           ustride;
   HYPRE_Int             k, d, depth, all_zero;

   HYPRE_Complex        *w_data_d, *w_data_0, *w_data_1, *w_data_2;
   HYPRE_Complex        *A_diag = NULL;
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

   /* Compute w_data pointers by summing contributions from A.
      Specialization is used for common stencil sizes to optimize for performance.
      A generic fallback algorithm is used otherwise */
   if (ndim == 3 && nentries[0] == 9 && nentries[1] == 9 && nentries[2] == 9)
   {
      /* Load Ap pointers (9, 9, 9) */
      HYPRE_CAP_LOAD_UP_TO_9(0);
      HYPRE_CAP_LOAD_UP_TO_9(1);
      HYPRE_CAP_LOAD_UP_TO_9(2);

      /* Compute w_data */
      hypre_BoxLoop2Begin(ndim, loop_size,
                          A_dbox, start, ustride, Ai,
                          w_dbox, start, ustride, wi);
      {
         HYPRE_CXYZ_DEFINE_SIGN;

         w_data_0[wi] = sign * (HYPRE_CAP_SUM_UP_TO_9(0));
         w_data_1[wi] = sign * (HYPRE_CAP_SUM_UP_TO_9(1));
         w_data_2[wi] = sign * (HYPRE_CAP_SUM_UP_TO_9(2));
      }
      hypre_BoxLoop2End(Ai, wi);
   }
   else if (ndim == 3 && nentries[0] == 2 && nentries[1] == 2 && nentries[2] == 2)
   {
      /* Load Ap pointers (2, 2, 2) */
      HYPRE_CAP_LOAD_UP_TO_2(0);
      HYPRE_CAP_LOAD_UP_TO_2(1);
      HYPRE_CAP_LOAD_UP_TO_2(2);

      /* Compute w_data */
      hypre_BoxLoop2Begin(ndim, loop_size,
                          A_dbox, start, ustride, Ai,
                          w_dbox, start, ustride, wi);
      {
         HYPRE_CXYZ_DEFINE_SIGN;

         w_data_0[wi] = sign * (HYPRE_CAP_SUM_UP_TO_2(0));
         w_data_1[wi] = sign * (HYPRE_CAP_SUM_UP_TO_2(1));
         w_data_2[wi] = sign * (HYPRE_CAP_SUM_UP_TO_2(2));
      }
      hypre_BoxLoop2End(Ai, wi);
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

            switch (depth)
            {
               case 9:
                  HYPRE_CAP_LOAD_UP_TO_9(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_9(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 8:
                  HYPRE_CAP_LOAD_UP_TO_8(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_8(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 7:
                  HYPRE_CAP_LOAD_UP_TO_7(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_7(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 6:
                  HYPRE_CAP_LOAD_UP_TO_6(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_6(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 5:
                  HYPRE_CAP_LOAD_UP_TO_5(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_5(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 4:
                  HYPRE_CAP_LOAD_UP_TO_4(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_4(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 3:
                  HYPRE_CAP_LOAD_UP_TO_3(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_3(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 2:
                  HYPRE_CAP_LOAD_UP_TO_2(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_2(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 1:
                  HYPRE_CAP_LOAD_UP_TO_1(d);
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, start, ustride, Ai,
                                      w_dbox, start, ustride, wi);
                  {
                     HYPRE_CXYZ_DEFINE_SIGN;
                     if (k < HYPRE_UNROLL_MAXDEPTH) { w_data_d[wi] = 0.0; }

                     w_data_d[wi] += sign * (HYPRE_CAP_SUM_UP_TO_1(d));
                  }
                  hypre_BoxLoop2End(Ai, wi);
                  break;

               case 0:
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

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeCxyz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeCxyz( hypre_StructMatrix *A,
                       HYPRE_Real         *cxyz,
                       HYPRE_Real         *sqcxyz)
{
   MPI_Comm               comm          = hypre_StructMatrixComm(A);
   HYPRE_Int              ndim          = hypre_StructMatrixNDim(A);
   hypre_StructGrid      *grid          = hypre_StructMatrixGrid(A);
   hypre_StructStencil   *stencil       = hypre_StructMatrixStencil(A);
   HYPRE_Int             *constant      = hypre_StructMatrixConstant(A);
   HYPRE_Int             *const_indices = hypre_StructMatrixConstIndices(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size  = hypre_StructStencilSize(stencil);
   HYPRE_Int              diag_entry    = hypre_StructStencilDiagEntry(stencil);
   hypre_BoxArray        *compute_boxes = hypre_StructGridBoxes(grid);

   hypre_Box             *A_dbox;
   hypre_Box             *compute_box;
   hypre_Index            loop_size, ustride;
   hypre_IndexRef         start;

   hypre_StructVector    *work[HYPRE_MAXDIM];
   HYPRE_Complex         *w_data[HYPRE_MAXDIM];
   hypre_Box             *w_dbox;

   HYPRE_Int              d, i, k, si;
   HYPRE_Int              cdepth[HYPRE_MAXDIM];
   HYPRE_Int              vdepth[HYPRE_MAXDIM];
   HYPRE_Int             *entries[HYPRE_MAXDIM];
   HYPRE_Int              csi[HYPRE_MAXDIM][stencil_size];
   HYPRE_Int              vsi[HYPRE_MAXDIM][stencil_size];
   HYPRE_Int              diag_is_constant;

#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(A);
   HYPRE_ExecutionPolicy  exec_policy     = hypre_GetExecPolicy1(memory_location);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("PFMGComputeCxyz");

   /*----------------------------------------------------------
    * Initialize data
    *----------------------------------------------------------*/

   hypre_SetIndex(ustride, 1);
   for (d = 0; d < ndim; d++)
   {
      cxyz[d] = 0.0;
      sqcxyz[d] = 0.0;
   }

   /* Check if diagonal entry is constant (1) or variable (0) */
   diag_is_constant = constant[diag_entry] ? 1 : 0;

   /* Create work arrays */
   for (d = 0; d < ndim; d++)
   {
      work[d] = hypre_StructVectorCreate(comm, grid);
      for (i = 0; i < 2 * ndim; i++)
      {
         hypre_StructVectorNumGhost(work[d])[i] = hypre_StructMatrixNumGhost(A)[i];
      }
      hypre_StructVectorInitialize(work[d], 0);
   }

   /* Initialize csi/vsi stencil pointers */
   for (d = 0; d < ndim; d++)
   {
      cdepth[d] = vdepth[d] = 0;
      for (k = 0; k < stencil_size; k++)
      {
         csi[d][k] = vsi[d][k] = 0;
      }
   }

   /* Setup csi/vsi stencil pointers */
   for (si = 0; si < stencil_size; si++)
   {
      if (hypre_StructMatrixConstEntry(A, si))
      {
         for (d = 0; d < ndim; d++)
         {
            if (hypre_IndexD(stencil_shape[si], d) != 0)
            {
               csi[d][cdepth[d]++] = const_indices[si];
            }
         }
      }
      else
      {
         for (d = 0; d < ndim; d++)
         {
            if (hypre_IndexD(stencil_shape[si], d) != 0)
            {
               vsi[d][vdepth[d]++] = si;
            }
         }
      }
   }

   /*----------------------------------------------------------
    * Compute cxyz (use arithmetic mean)
    *----------------------------------------------------------*/

   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);
      start = hypre_BoxIMin(compute_box);
      hypre_BoxGetSize(compute_box, loop_size);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      w_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(work[0]), i);
      for (d = 0; d < ndim; d++)
      {
         w_data[d] = hypre_StructVectorBoxData(work[d], i);
      }

      /* Collect pointers to variable stencil entries */
      for (d = 0; d < ndim; d++)
      {
         entries[d] = vsi[d];
      }

      /* Compute variable coefficient contributions */
      hypre_PFMGComputeCxyz_core_VC(A, i, diag_is_constant, diag_entry,
                                    vdepth, entries, start, loop_size,
                                    A_dbox, w_dbox, w_data);

      /* Collect pointers to constant stencil entries */
      for (d = 0; d < ndim; d++)
      {
         entries[d] = csi[d];
      }

      /* Compute constant coefficient contributions */
      hypre_PFMGComputeCxyz_core_CC(A, i, diag_is_constant, diag_entry,
                                    cdepth, entries, start, loop_size,
                                    A_dbox, w_dbox, w_data);


      /* Compute cxyz/sqcxyz */
      hypre_GpuProfilingPushRange("Reduction");
      if (ndim == 3)
      {
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)
         /* TODO: Use a single BoxLoopReduction */
         HYPRE_Real cdb_0 = cxyz[0], cdb_1 = cxyz[1], cdb_2 = cxyz[2];
         HYPRE_Real sqcdb_0 = sqcxyz[0], sqcdb_1 = sqcxyz[1], sqcdb_2 = sqcxyz[2];

         hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                      start, ustride, wi, cdb_0)
         {
            cdb_0 += w_data[0][wi];
         }
         hypre_BoxLoop1ReductionEnd(wi, cdb_0)

         hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                      start, ustride, wi, cdb_1)
         {
            cdb_1 += w_data[1][wi];
         }
         hypre_BoxLoop1ReductionEnd(wi, cdb_1)

         hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                      start, ustride, wi, cdb_2)
         {
            cdb_2 += w_data[2][wi];
         }
         hypre_BoxLoop1ReductionEnd(wi, cdb_2)

         hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                      start, ustride, wi, sqcdb_0)
         {
            sqcdb_0 += hypre_squared(w_data[0][wi]);
         }
         hypre_BoxLoop1ReductionEnd(wi, sqcdb_0)

         hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                      start, ustride, wi, sqcdb_1)
         {
            sqcdb_1 += hypre_squared(w_data[1][wi]);
         }
         hypre_BoxLoop1ReductionEnd(wi, sqcdb_1)

         hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                      start, ustride, wi, sqcdb_2)
         {
            sqcdb_2 += hypre_squared(w_data[2][wi]);
         }
         hypre_BoxLoop1ReductionEnd(wi, sqcdb_2)
#else
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         HYPRE_double6 d6(cxyz[0], cxyz[1], cxyz[2], sqcxyz[0], sqcxyz[1], sqcxyz[2]);
         ReduceSum<HYPRE_double6> sum6(d6);
#else
         HYPRE_Real cdb_0 = cxyz[0], cdb_1 = cxyz[1], cdb_2 = cxyz[2];
         HYPRE_Real sqcdb_0 = sqcxyz[0], sqcdb_1 = sqcxyz[1], sqcdb_2 = sqcxyz[2];

#if defined(HYPRE_BOX_REDUCTION)
#undef HYPRE_BOX_REDUCTION
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_BOX_REDUCTION map(tofrom:cdb_0,cdb_1,cdb_2,sqcdb_0,sqcdb_1,sqcdb_2) reduction(+:cdb_0,cdb_1,cdb_2,sqcdb_0,sqcdb_1,sqcdb_2)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cdb_0,cdb_1,cdb_2,sqcdb_0,sqcdb_1,sqcdb_2)
#endif

#endif
         hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                      start, ustride, wi, sum6)
         {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
            HYPRE_double6 temp6(w_data[0][wi], w_data[1][wi], w_data[2][wi],
                                hypre_squared(w_data[0][wi]),
                                hypre_squared(w_data[1][wi]),
                                hypre_squared(w_data[2][wi]));
            sum6 += temp6;
#else
            cdb_0 += w_data[0][wi];
            cdb_1 += w_data[1][wi];
            cdb_2 += w_data[2][wi];

            sqcdb_0 += hypre_squared(w_data[0][wi]);
            sqcdb_1 += hypre_squared(w_data[1][wi]);
            sqcdb_2 += hypre_squared(w_data[2][wi]);
#endif
         }
         hypre_BoxLoop1ReductionEnd(wi, sum6)
#endif

#if !defined(HYPRE_USING_KOKKOS) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
         HYPRE_double6 temp6 = (HYPRE_double6) sum6;
         cxyz[0]   = temp6.x;
         cxyz[1]   = temp6.y;
         cxyz[2]   = temp6.z;
         sqcxyz[0] = temp6.u;
         sqcxyz[1] = temp6.v;
         sqcxyz[2] = temp6.w;
#else
         cxyz[0]   = (HYPRE_Real) cdb_0;
         cxyz[1]   = (HYPRE_Real) cdb_1;
         cxyz[2]   = (HYPRE_Real) cdb_2;
         sqcxyz[0] = (HYPRE_Real) sqcdb_0;
         sqcxyz[1] = (HYPRE_Real) sqcdb_1;
         sqcxyz[2] = (HYPRE_Real) sqcdb_2;
#endif
      }
      else
      {
         for (d = 0; d < ndim; d++)
         {
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)
            HYPRE_Real cdb   = cxyz[d];
            HYPRE_Real sqcdb = sqcxyz[d];

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                         start, ustride, wi, cdb)
            {
               cdb += w_data[d][wi];
            }
            hypre_BoxLoop1ReductionEnd(wi, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                         start, ustride, wi, sqcdb)
            {
               sqcdb += hypre_squared(w_data[d][wi]);
            }
            hypre_BoxLoop1ReductionEnd(wi, sqcdb)
#else
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
            HYPRE_Real2 d2(cxyz[d], sqcxyz[d]);
            ReduceSum<HYPRE_Real2> sum2(d2);
#else
            HYPRE_Real cdb   = cxyz[d];
            HYPRE_Real sqcdb = sqcxyz[d];

#if defined(HYPRE_BOX_REDUCTION)
#undef HYPRE_BOX_REDUCTION
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_BOX_REDUCTION map(tofrom:cdb,sqcdb) reduction(+:cdb,sqcdb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cdb,sqcdb)
#endif

#endif
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, w_dbox,
                                         start, ustride, wi, sum2)
            {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(w_data[d][wi], hypre_squared(w_data[d][wi]));
               sum2  += temp2;
#else
               cdb   += w_data[d][wi];
               sqcdb += hypre_squared(w_data[d][wi]);
#endif
            }
            hypre_BoxLoop1ReductionEnd(wi, sum2)
#endif

#if !defined(HYPRE_USING_KOKKOS) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
            HYPRE_Real2 temp2 = (HYPRE_Real2) sum2;
            cxyz[d]   = temp2.x;
            sqcxyz[d] = temp2.y;
#else
            cxyz[d]   = (HYPRE_Real) cdb;
            sqcxyz[d] = (HYPRE_Real) sqcdb;
#endif
         } /* for (d = 0; d < ndim; d++) */
      }
      hypre_GpuProfilingPopRange(); // "Reduction"
   } /* hypre_ForBoxI(i, compute_boxes) */

   /* Free work arrays */
   for (d = 0; d < ndim; d++)
   {
      hypre_StructVectorDestroy(work[d]);
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeDxyz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz( hypre_StructMatrix *A,
                       HYPRE_Real         *dxyz,
                       HYPRE_Int          *dxyz_flag )
{
   MPI_Comm           comm = hypre_StructMatrixComm(A);
   hypre_StructGrid  *grid = hypre_StructMatrixGrid(A);

   HYPRE_Int          cte_coeff;
   HYPRE_Real         cxyz_max;
   HYPRE_Real         cxyz[HYPRE_MAXDIM];
   HYPRE_Real         sqcxyz[HYPRE_MAXDIM];
   HYPRE_Real         tcxyz[HYPRE_MAXDIM];
   HYPRE_Real         mean[HYPRE_MAXDIM];
   HYPRE_Real         deviation[HYPRE_MAXDIM];

   HYPRE_Int          d, ndim;
   HYPRE_BigInt       global_size;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*----------------------------------------------------------
    * Exit if user gives dxyz different than zero
    *----------------------------------------------------------*/

   if ((dxyz[0] != 0) && (dxyz[1] != 0) && (dxyz[2] != 0))
   {
      *dxyz_flag = 0;

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   ndim        = hypre_StructMatrixNDim(A);
   cte_coeff   = hypre_StructMatrixConstantCoefficient(A);
   global_size = hypre_StructGridGlobalSize(grid);

   /* Compute cxyz and sqcxyz arrays */
   hypre_PFMGComputeCxyz(A, cxyz, sqcxyz);

   /*----------------------------------------------------------
    * Compute dxyz
    *----------------------------------------------------------*/

   if (cte_coeff)
   {
      /* all coefficients constant or variable diagonal */
      global_size = 1;
   }
   else
   {
      /* all coefficients vary with space */
      for (d = 0; d < ndim; d++)
      {
         tcxyz[d] = cxyz[d];
      }
      hypre_MPI_Allreduce(tcxyz, cxyz, ndim, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

      for (d = 0; d < ndim; d++)
      {
         tcxyz[d] = sqcxyz[d];
      }
      hypre_MPI_Allreduce(tcxyz, sqcxyz, ndim, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   }

   for (d = 0; d < ndim; d++)
   {
      mean[d] = cxyz[d] / (HYPRE_Real) global_size;
      deviation[d] = sqcxyz[d] / (HYPRE_Real) global_size;
   }

   cxyz_max = 0.0;
   for (d = 0; d < ndim; d++)
   {
      cxyz_max = hypre_max(cxyz_max, cxyz[d]);
   }

   if (cxyz_max == 0.0)
   {
      /* Do isotropic coarsening */
      for (d = 0; d < ndim; d++)
      {
         cxyz[d] = 1.0;
      }
      cxyz_max = 1.0;
   }

   /* Set dxyz values that are scaled appropriately for the coarsening routine */
   for (d = 0; d < ndim; d++)
   {
      HYPRE_Real max_anisotropy = HYPRE_REAL_MAX / 1000;
      if (cxyz[d] > (cxyz_max / max_anisotropy))
      {
         cxyz[d] /= cxyz_max;
         dxyz[d] = hypre_sqrt(1.0 / cxyz[d]);
      }
      else
      {
         dxyz[d] = hypre_sqrt(max_anisotropy);
      }
   }

   /* Set 'dxyz_flag' if the matrix-coefficient variation is "too large".
    * This is used later to set relaxation weights for Jacobi.
    *
    * Use the "square of the coefficient of variation" = (sigma/mu)^2,
    * where sigma is the standard deviation and mu is the mean.  This is
    * equivalent to computing (d - mu^2)/mu^2 where d is the average of
    * the squares of the coefficients stored in 'deviation'.  Care is
    * taken to avoid dividing by zero when the mean is zero. */

   *dxyz_flag = 0;
   for (d = 0; d < ndim; d++)
   {
      deviation[d] -= mean[d] * mean[d];
      if ( deviation[d] > 0.1 * (mean[d]*mean[d]) )
      {
         *dxyz_flag = 1;
         break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGCoarsen( hypre_Box     *cbox,
                   hypre_Index    periodic,
                   HYPRE_Int      max_levels,
                   HYPRE_Int      dxyz_flag,
                   HYPRE_Real    *dxyz,
                   HYPRE_Int    **cdir_l_ptr,
                   HYPRE_Int    **active_l_ptr,
                   HYPRE_Real   **relax_weights_ptr,
                   HYPRE_Int     *num_levels )
{
   HYPRE_Int      ndim = hypre_BoxNDim(cbox);
   HYPRE_Int     *cdir_l;
   HYPRE_Int     *active_l;
   HYPRE_Real    *relax_weights;

   hypre_Index    coarsen;
   hypre_Index    cindex;
   hypre_Index    stride;

   HYPRE_Real     alpha, beta, min_dxyz;
   HYPRE_Int      d, l, cdir;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Allocate data */
   cdir_l        = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   active_l      = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   relax_weights = hypre_CTAlloc(HYPRE_Real, max_levels, HYPRE_MEMORY_HOST);

   /* Force relaxation on finest grid */
   hypre_SetIndex(coarsen, 1);
   for (l = 0; l < max_levels; l++)
   {
      /* Initialize min_dxyz */
      min_dxyz = 1;
      for (d = 0; d < ndim; d++)
      {
         min_dxyz += dxyz[d];
      }

      /* Determine cdir */
      cdir = -1;
      alpha = 0.0;
      for (d = 0; d < ndim; d++)
      {
         if ((hypre_BoxIMaxD(cbox, d) > hypre_BoxIMinD(cbox, d)) &&
             (dxyz[d] < min_dxyz))
         {
            min_dxyz = dxyz[d];
            cdir = d;
         }
         alpha += 1.0 / (dxyz[d] * dxyz[d]);
      }
      relax_weights[l] = 1.0;

      /* If it's possible to coarsen, change relax_weights */
      beta = 0.0;
      if (cdir != -1)
      {
         if (dxyz_flag || (ndim == 1))
         {
            relax_weights[l] = 2.0 / 3.0;
         }
         else
         {
            for (d = 0; d < ndim; d++)
            {
               if (d != cdir)
               {
                  beta += 1.0 / (dxyz[d] * dxyz[d]);
               }
            }

            /* determine level Jacobi weights */
            relax_weights[l] = 2.0 / (3.0 - beta / alpha);
         }

         /*    don't coarsen if a periodic direction and not divisible by 2
            or don't coarsen if we've reached max_levels*/
         if (((periodic[cdir]) && (periodic[cdir] % 2)) || l == (max_levels - 1))
         {
            cdir = -1;
         }
      }

      /* stop coarsening */
      if (cdir == -1)
      {
         active_l[l] = 1; /* forces relaxation on coarsest grid */
         break;
      }

      cdir_l[l] = cdir;

      if (hypre_IndexD(coarsen, cdir) != 0)
      {
         /* coarsened previously in this direction, relax level l */
         active_l[l] = 1;
         hypre_SetIndex(coarsen, 0);
      }
      else
      {
         active_l[l] = 0;
      }
      hypre_IndexD(coarsen, cdir) = 1;

      /* set cindex and stride */
      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetStride(cdir, stride);

      /* update dxyz and coarsen cbox*/
      dxyz[cdir] *= 2;
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride, hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride, hypre_BoxIMax(cbox));

      /* update periodic */
      periodic[cdir] /= 2;
   }
   *num_levels = l + 1;

   *cdir_l_ptr        = cdir_l;
   *active_l_ptr      = active_l;
   *relax_weights_ptr = relax_weights;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
