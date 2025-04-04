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

/*--------------------------------------------------------------------------
 * Defines used below
 *--------------------------------------------------------------------------*/

//#include "struct_matmult_core.h"

#ifdef HYPRE_UNROLL_MAXDEPTH
#undef HYPRE_UNROLL_MAXDEPTH
#endif
#define HYPRE_UNROLL_MAXDEPTH 8

typedef HYPRE_Complex *hypre_3Cptrs[3];
typedef HYPRE_Complex *hypre_1Cptr;

/*--------------------------------------------------------------------------
 * Macros used in the kernel loops below
 *
 * F/C means fine/coarse data space.  For example, CFF means the data spaces for
 * the three terms are respectively coarse, fine, and fine.
 *--------------------------------------------------------------------------*/

#define HYPRE_SMMCORE_FF(k) \
   cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi]

#define HYPRE_SMMCORE_CF(k) \
   cprod[k] * tptrs[k][0][ci] * tptrs[k][1][fi]

#define HYPRE_SMMCORE_FFF(k) \
   cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi] * tptrs[k][2][fi]

#define HYPRE_SMMCORE_CFF(k) \
   cprod[k] * tptrs[k][0][ci] * tptrs[k][1][fi] * tptrs[k][2][fi]

#define HYPRE_SMMCORE_CCF(k) \
   cprod[k] * tptrs[k][0][ci] * tptrs[k][1][ci] * tptrs[k][2][fi]

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_ff( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    const hypre_3Cptrs  *tptrs,
                                    hypre_1Cptr         *mptrs,
                                    HYPRE_Int            ndim,
                                    hypre_Index          loop_size,
                                    hypre_Box           *fdbox,
                                    hypre_Index          fdstart,
                                    hypre_Index          fdstride,
                                    hypre_Box           *Mdbox,
                                    hypre_Index          Mdstart,
                                    hypre_Index          Mdstride )
{
   HYPRE_Complex  *mptr = mptrs[0];  /* Requires all pointers to be the same */
   HYPRE_Int       k;
   HYPRE_Int       depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0) +
                                   HYPRE_SMMCORE_FF(k + 1) +
                                   HYPRE_SMMCORE_FF(k + 2) +
                                   HYPRE_SMMCORE_FF(k + 3) +
                                   HYPRE_SMMCORE_FF(k + 4) +
                                   HYPRE_SMMCORE_FF(k + 5) +
                                   HYPRE_SMMCORE_FF(k + 6) +
                                   HYPRE_SMMCORE_FF(k + 7);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0) +
                                   HYPRE_SMMCORE_FF(k + 1) +
                                   HYPRE_SMMCORE_FF(k + 2) +
                                   HYPRE_SMMCORE_FF(k + 3) +
                                   HYPRE_SMMCORE_FF(k + 4) +
                                   HYPRE_SMMCORE_FF(k + 5) +
                                   HYPRE_SMMCORE_FF(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0) +
                                   HYPRE_SMMCORE_FF(k + 1) +
                                   HYPRE_SMMCORE_FF(k + 2) +
                                   HYPRE_SMMCORE_FF(k + 3) +
                                   HYPRE_SMMCORE_FF(k + 4) +
                                   HYPRE_SMMCORE_FF(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0) +
                                   HYPRE_SMMCORE_FF(k + 1) +
                                   HYPRE_SMMCORE_FF(k + 2) +
                                   HYPRE_SMMCORE_FF(k + 3) +
                                   HYPRE_SMMCORE_FF(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0) +
                                   HYPRE_SMMCORE_FF(k + 1) +
                                   HYPRE_SMMCORE_FF(k + 2) +
                                   HYPRE_SMMCORE_FF(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0) +
                                   HYPRE_SMMCORE_FF(k + 1) +
                                   HYPRE_SMMCORE_FF(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0) +
                                   HYPRE_SMMCORE_FF(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_cf( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    const hypre_3Cptrs  *tptrs,
                                    hypre_1Cptr         *mptrs,
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
                                    hypre_Index          Mdstride )
{
   HYPRE_Complex  *mptr = mptrs[0];  /* Requires all pointers to be the same */
   HYPRE_Int       k;
   HYPRE_Int       depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF(k + 0) +
                                   HYPRE_SMMCORE_CF(k + 1) +
                                   HYPRE_SMMCORE_CF(k + 2) +
                                   HYPRE_SMMCORE_CF(k + 3) +
                                   HYPRE_SMMCORE_CF(k + 4) +
                                   HYPRE_SMMCORE_CF(k + 5) +
                                   HYPRE_SMMCORE_CF(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF(k + 0) +
                                   HYPRE_SMMCORE_CF(k + 1) +
                                   HYPRE_SMMCORE_CF(k + 2) +
                                   HYPRE_SMMCORE_CF(k + 3) +
                                   HYPRE_SMMCORE_CF(k + 4) +
                                   HYPRE_SMMCORE_CF(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF(k + 0) +
                                   HYPRE_SMMCORE_CF(k + 1) +
                                   HYPRE_SMMCORE_CF(k + 2) +
                                   HYPRE_SMMCORE_CF(k + 3) +
                                   HYPRE_SMMCORE_CF(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF(k + 0) +
                                   HYPRE_SMMCORE_CF(k + 1) +
                                   HYPRE_SMMCORE_CF(k + 2) +
                                   HYPRE_SMMCORE_CF(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF(k + 0) +
                                   HYPRE_SMMCORE_CF(k + 1) +
                                   HYPRE_SMMCORE_CF(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF(k + 0) +
                                   HYPRE_SMMCORE_CF(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_fff( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     const hypre_3Cptrs  *tptrs,
                                     hypre_1Cptr         *mptrs,
                                     HYPRE_Int            ndim,
                                     hypre_Index          loop_size,
                                     hypre_Box           *fdbox,
                                     hypre_Index          fdstart,
                                     hypre_Index          fdstride,
                                     hypre_Box           *Mdbox,
                                     hypre_Index          Mdstart,
                                     hypre_Index          Mdstride )
{
   HYPRE_Complex  *mptr = mptrs[0];  /* Requires all pointers to be the same */
   HYPRE_Int       k;
   HYPRE_Int       depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0) +
                                   HYPRE_SMMCORE_FFF(k + 1) +
                                   HYPRE_SMMCORE_FFF(k + 2) +
                                   HYPRE_SMMCORE_FFF(k + 3) +
                                   HYPRE_SMMCORE_FFF(k + 4) +
                                   HYPRE_SMMCORE_FFF(k + 5) +
                                   HYPRE_SMMCORE_FFF(k + 6) +
                                   HYPRE_SMMCORE_FFF(k + 7);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0) +
                                   HYPRE_SMMCORE_FFF(k + 1) +
                                   HYPRE_SMMCORE_FFF(k + 2) +
                                   HYPRE_SMMCORE_FFF(k + 3) +
                                   HYPRE_SMMCORE_FFF(k + 4) +
                                   HYPRE_SMMCORE_FFF(k + 5) +
                                   HYPRE_SMMCORE_FFF(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0) +
                                   HYPRE_SMMCORE_FFF(k + 1) +
                                   HYPRE_SMMCORE_FFF(k + 2) +
                                   HYPRE_SMMCORE_FFF(k + 3) +
                                   HYPRE_SMMCORE_FFF(k + 4) +
                                   HYPRE_SMMCORE_FFF(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0) +
                                   HYPRE_SMMCORE_FFF(k + 1) +
                                   HYPRE_SMMCORE_FFF(k + 2) +
                                   HYPRE_SMMCORE_FFF(k + 3) +
                                   HYPRE_SMMCORE_FFF(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0) +
                                   HYPRE_SMMCORE_FFF(k + 1) +
                                   HYPRE_SMMCORE_FFF(k + 2) +
                                   HYPRE_SMMCORE_FFF(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0) +
                                   HYPRE_SMMCORE_FFF(k + 1) +
                                   HYPRE_SMMCORE_FFF(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0) +
                                   HYPRE_SMMCORE_FFF(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_cff( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     const hypre_3Cptrs  *tptrs,
                                     hypre_1Cptr         *mptrs,
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
                                     hypre_Index          Mdstride )
{
   HYPRE_Complex  *mptr = mptrs[0];  /* Requires all pointers to be the same */
   HYPRE_Int       k;
   HYPRE_Int       depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF(k + 0) +
                                   HYPRE_SMMCORE_CFF(k + 1) +
                                   HYPRE_SMMCORE_CFF(k + 2) +
                                   HYPRE_SMMCORE_CFF(k + 3) +
                                   HYPRE_SMMCORE_CFF(k + 4) +
                                   HYPRE_SMMCORE_CFF(k + 5) +
                                   HYPRE_SMMCORE_CFF(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF(k + 0) +
                                   HYPRE_SMMCORE_CFF(k + 1) +
                                   HYPRE_SMMCORE_CFF(k + 2) +
                                   HYPRE_SMMCORE_CFF(k + 3) +
                                   HYPRE_SMMCORE_CFF(k + 4) +
                                   HYPRE_SMMCORE_CFF(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF(k + 0) +
                                   HYPRE_SMMCORE_CFF(k + 1) +
                                   HYPRE_SMMCORE_CFF(k + 2) +
                                   HYPRE_SMMCORE_CFF(k + 3) +
                                   HYPRE_SMMCORE_CFF(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF(k + 0) +
                                   HYPRE_SMMCORE_CFF(k + 1) +
                                   HYPRE_SMMCORE_CFF(k + 2) +
                                   HYPRE_SMMCORE_CFF(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF(k + 0) +
                                   HYPRE_SMMCORE_CFF(k + 1) +
                                   HYPRE_SMMCORE_CFF(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF(k + 0) +
                                   HYPRE_SMMCORE_CFF(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_ccf( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     const hypre_3Cptrs  *tptrs,
                                     hypre_1Cptr         *mptrs,
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
                                     hypre_Index          Mdstride )
{
   HYPRE_Complex  *mptr = mptrs[0];  /* Requires all pointers to be the same */
   HYPRE_Int       k;
   HYPRE_Int       depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF(k + 0) +
                                   HYPRE_SMMCORE_CCF(k + 1) +
                                   HYPRE_SMMCORE_CCF(k + 2) +
                                   HYPRE_SMMCORE_CCF(k + 3) +
                                   HYPRE_SMMCORE_CCF(k + 4) +
                                   HYPRE_SMMCORE_CCF(k + 5) +
                                   HYPRE_SMMCORE_CCF(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF(k + 0) +
                                   HYPRE_SMMCORE_CCF(k + 1) +
                                   HYPRE_SMMCORE_CCF(k + 2) +
                                   HYPRE_SMMCORE_CCF(k + 3) +
                                   HYPRE_SMMCORE_CCF(k + 4) +
                                   HYPRE_SMMCORE_CCF(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF(k + 0) +
                                   HYPRE_SMMCORE_CCF(k + 1) +
                                   HYPRE_SMMCORE_CCF(k + 2) +
                                   HYPRE_SMMCORE_CCF(k + 3) +
                                   HYPRE_SMMCORE_CCF(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF(k + 0) +
                                   HYPRE_SMMCORE_CCF(k + 1) +
                                   HYPRE_SMMCORE_CCF(k + 2) +
                                   HYPRE_SMMCORE_CCF(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF(k + 0) +
                                   HYPRE_SMMCORE_CCF(k + 1) +
                                   HYPRE_SMMCORE_CCF(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF(k + 0) +
                                   HYPRE_SMMCORE_CCF(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Core function for computing the double-product of coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_double( hypre_StructMatmultDataMH *a,
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
                                        hypre_Index  Mdstride )
{
   HYPRE_Int       nprod[2];
   HYPRE_Complex   cprod[2][na];
   hypre_3Cptrs    tptrs[2][na];
   hypre_1Cptr     mptrs[2][na];

   HYPRE_Int       mentry, ptype, nf, nc;
   HYPRE_Int       e, p, i, k, t;

   for (e = 0; e < stencil_size; e++)
   {
      /* Reset product counters */
      for (p = 0; p < 2; p++)
      {
         nprod[p] = 0;
      }

      /* Build products arrays */
      for (i = 0; i < na; i++)
      {
         mentry = a[i].mentry;
         if (mentry != e)
         {
            continue;
         }

         /* Determine number of fine and coarse terms */
         nf = nc = 0;
         for (t = 0; t < 2; t++)
         {
            if (a[i].types[t] == 1)
            {
               /* Type 1 -> coarse data space */
               nc++;
            }
            else
            {
               /* Type 0 or 2 -> fine data space */
               nf++;
            }
         }

         /* Determine product type */
         switch (nc)
         {
            case 0: /* ff term (call core_ff) */
               ptype = 0;
               break;
            case 1: /* cf term (call core_cf) */
               ptype = 1;
               break;
         }

         /* Set array values for product k of product type ptype */
         k = nprod[ptype];
         cprod[ptype][k] = a[i].cprod;
         nf = nc = 0;
         for (t = 0; t < 2; t++)
         {
            if (a[i].types[t] == 1)
            {
               /* Type 1 -> coarse data space */
               tptrs[ptype][k][nc] = a[i].tptrs[t];  /* put first */
               nc++;
            }
            else
            {
               /* Type 0 or 2 -> fine data space */
               tptrs[ptype][k][1 - nf] = a[i].tptrs[t];  /* put last */
               nf++;
            }
         }
         mptrs[ptype][k] = a[i].mptr;
         nprod[ptype]++;
      }

      /* Call core functions */
      hypre_StructMatmultCompute_core_ff(nprod[0], cprod[0], tptrs[0], mptrs[0],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_cf(nprod[1], cprod[1], tptrs[1], mptrs[1],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         cdbox, cdstart, cdstride,
                                         Mdbox, Mdstart, Mdstride);
   } /* loop on M stencil entries */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Core function for computing the triple-product of coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_triple( hypre_StructMatmultDataMH *a,
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
                                        hypre_Index  Mdstride )
{
   HYPRE_Int       nprod[3];
   HYPRE_Complex   cprod[3][na];
   hypre_3Cptrs    tptrs[3][na];
   hypre_1Cptr     mptrs[3][na];

   HYPRE_Int       mentry, ptype, nf, nc;
   HYPRE_Int       e, p, i, k, t;

   for (e = 0; e < stencil_size; e++)
   {
      /* Reset product counters */
      for (p = 0; p < 3; p++)
      {
         nprod[p] = 0;
      }

      /* Build products arrays */
      for (i = 0; i < na; i++)
      {
         mentry = a[i].mentry;
         if (mentry != e)
         {
            continue;
         }

         /* Determine number of fine and coarse terms */
         nf = nc = 0;
         for (t = 0; t < 3; t++)
         {
            if (a[i].types[t] == 1)
            {
               /* Type 1 -> coarse data space */
               nc++;
            }
            else
            {
               /* Type 0 or 2 -> fine data space */
               nf++;
            }
         }

         /* Determine product type */
         switch (nc)
         {
            case 0: /* fff term (call core_fff) */
               ptype = 0;
               break;
            case 1: /* cff term (call core_cff) */
               ptype = 1;
               break;
            case 2: /* ccf term (call core_ccf) */
               ptype = 2;
               break;
         }

         /* Set array values for product k of product type ptype */
         k = nprod[ptype];
         cprod[ptype][k] = a[i].cprod;
         nf = nc = 0;
         for (t = 0; t < 3; t++)
         {
            if (a[i].types[t] == 1)
            {
               /* Type 1 -> coarse data space */
               tptrs[ptype][k][nc] = a[i].tptrs[t];  /* put first */
               nc++;
            }
            else
            {
               /* Type 0 or 2 -> fine data space */
               tptrs[ptype][k][2 - nf] = a[i].tptrs[t];  /* put last */
               nf++;
            }
         }
         mptrs[ptype][k] = a[i].mptr;
         nprod[ptype]++;
      }

      /* Call core functions */
      hypre_StructMatmultCompute_core_fff(nprod[0], cprod[0], tptrs[0], mptrs[0],
                                          ndim, loop_size,
                                          fdbox, fdstart, fdstride,
                                          Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_cff(nprod[1], cprod[1], tptrs[1], mptrs[1],
                                          ndim, loop_size,
                                          fdbox, fdstart, fdstride,
                                          cdbox, cdstart, cdstride,
                                          Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_ccf(nprod[2], cprod[2], tptrs[2], mptrs[2],
                                          ndim, loop_size,
                                          fdbox, fdstart, fdstride,
                                          cdbox, cdstart, cdstride,
                                          Mdbox, Mdstart, Mdstride);
   } /* loop on M stencil entries */

   return hypre_error_flag;
}

