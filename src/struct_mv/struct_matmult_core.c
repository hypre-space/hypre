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

#define HYPRE_SMMCORE_FFF(k) \
   cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi] * tptrs[k][2][fi]

#define HYPRE_SMMCORE_CFF(k) \
   cprod[k] * tptrs[k][0][ci] * tptrs[k][1][fi] * tptrs[k][2][fi]

#define HYPRE_SMMCORE_CCF(k) \
   cprod[k] * tptrs[k][0][ci] * tptrs[k][1][ci] * tptrs[k][2][fi]

/* Don't need CCC */

#define HYPRE_SMMCORE_FF(k)                     \
   cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi]

#define HYPRE_SMMCORE_CF(k) \
   cprod[k] * tptrs[k][0][ci] * tptrs[k][1][fi]

#define HYPRE_SMMCORE_CC(k) \
   cprod[k] * tptrs[k][0][ci] * tptrs[k][1][ci]

#define HYPRE_SMMCORE_F(k) \
   cprod[k] * tptrs[k][0][fi]

#define HYPRE_SMMCORE_C(k) \
   cprod[k] * tptrs[k][0][ci]

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_fff( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     hypre_3Cptrs  *tptrs,
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
                                     hypre_3Cptrs  *tptrs,
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
                                     hypre_3Cptrs  *tptrs,
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_ff( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    hypre_3Cptrs  *tptrs,
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
                                    hypre_3Cptrs  *tptrs,
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
hypre_StructMatmultCompute_core_cc( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    hypre_3Cptrs  *tptrs,
                                    hypre_1Cptr         *mptrs,
                                    HYPRE_Int            ndim,
                                    hypre_Index          loop_size,
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
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC(k + 0) +
                                   HYPRE_SMMCORE_CC(k + 1) +
                                   HYPRE_SMMCORE_CC(k + 2) +
                                   HYPRE_SMMCORE_CC(k + 3) +
                                   HYPRE_SMMCORE_CC(k + 4) +
                                   HYPRE_SMMCORE_CC(k + 5) +
                                   HYPRE_SMMCORE_CC(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC(k + 0) +
                                   HYPRE_SMMCORE_CC(k + 1) +
                                   HYPRE_SMMCORE_CC(k + 2) +
                                   HYPRE_SMMCORE_CC(k + 3) +
                                   HYPRE_SMMCORE_CC(k + 4) +
                                   HYPRE_SMMCORE_CC(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC(k + 0) +
                                   HYPRE_SMMCORE_CC(k + 1) +
                                   HYPRE_SMMCORE_CC(k + 2) +
                                   HYPRE_SMMCORE_CC(k + 3) +
                                   HYPRE_SMMCORE_CC(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC(k + 0) +
                                   HYPRE_SMMCORE_CC(k + 1) +
                                   HYPRE_SMMCORE_CC(k + 2) +
                                   HYPRE_SMMCORE_CC(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC(k + 0) +
                                   HYPRE_SMMCORE_CC(k + 1) +
                                   HYPRE_SMMCORE_CC(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC(k + 0) +
                                   HYPRE_SMMCORE_CC(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
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
hypre_StructMatmultCompute_core_f( HYPRE_Int            nprod,
                                   HYPRE_Complex       *cprod,
                                   hypre_3Cptrs  *tptrs,
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
         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F(k + 0) +
                                   HYPRE_SMMCORE_F(k + 1) +
                                   HYPRE_SMMCORE_F(k + 2) +
                                   HYPRE_SMMCORE_F(k + 3) +
                                   HYPRE_SMMCORE_F(k + 4) +
                                   HYPRE_SMMCORE_F(k + 5) +
                                   HYPRE_SMMCORE_F(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F(k + 0) +
                                   HYPRE_SMMCORE_F(k + 1) +
                                   HYPRE_SMMCORE_F(k + 2) +
                                   HYPRE_SMMCORE_F(k + 3) +
                                   HYPRE_SMMCORE_F(k + 4) +
                                   HYPRE_SMMCORE_F(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F(k + 0) +
                                   HYPRE_SMMCORE_F(k + 1) +
                                   HYPRE_SMMCORE_F(k + 2) +
                                   HYPRE_SMMCORE_F(k + 3) +
                                   HYPRE_SMMCORE_F(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F(k + 0) +
                                   HYPRE_SMMCORE_F(k + 1) +
                                   HYPRE_SMMCORE_F(k + 2) +
                                   HYPRE_SMMCORE_F(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F(k + 0) +
                                   HYPRE_SMMCORE_F(k + 1) +
                                   HYPRE_SMMCORE_F(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F(k + 0) +
                                   HYPRE_SMMCORE_F(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F(k + 0);

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
hypre_StructMatmultCompute_core_c( HYPRE_Int            nprod,
                                   HYPRE_Complex       *cprod,
                                   hypre_3Cptrs  *tptrs,
                                   hypre_1Cptr         *mptrs,
                                   HYPRE_Int            ndim,
                                   hypre_Index          loop_size,
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
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C(k + 0) +
                                   HYPRE_SMMCORE_C(k + 1) +
                                   HYPRE_SMMCORE_C(k + 2) +
                                   HYPRE_SMMCORE_C(k + 3) +
                                   HYPRE_SMMCORE_C(k + 4) +
                                   HYPRE_SMMCORE_C(k + 5) +
                                   HYPRE_SMMCORE_C(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C(k + 0) +
                                   HYPRE_SMMCORE_C(k + 1) +
                                   HYPRE_SMMCORE_C(k + 2) +
                                   HYPRE_SMMCORE_C(k + 3) +
                                   HYPRE_SMMCORE_C(k + 4) +
                                   HYPRE_SMMCORE_C(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C(k + 0) +
                                   HYPRE_SMMCORE_C(k + 1) +
                                   HYPRE_SMMCORE_C(k + 2) +
                                   HYPRE_SMMCORE_C(k + 3) +
                                   HYPRE_SMMCORE_C(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C(k + 0) +
                                   HYPRE_SMMCORE_C(k + 1) +
                                   HYPRE_SMMCORE_C(k + 2) +
                                   HYPRE_SMMCORE_C(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C(k + 0) +
                                   HYPRE_SMMCORE_C(k + 1) +
                                   HYPRE_SMMCORE_C(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C(k + 0) +
                                   HYPRE_SMMCORE_C(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
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
 * Core function for computing the nterms-product of coefficients.
 * Here, nterms can only be 2 or 3.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core( HYPRE_Int                  nterms,
                                 hypre_StructMatmultDataMH *a,
                                 HYPRE_Int                  na,
                                 HYPRE_Int                  ndim,
                                 hypre_Index                loop_size,
                                 HYPRE_Int                  stencil_size,
                                 hypre_Box                 *fdbox,
                                 hypre_Index                fdstart,
                                 hypre_Index                fdstride,
                                 hypre_Box                 *cdbox,
                                 hypre_Index                cdstart,
                                 hypre_Index                cdstride,
                                 hypre_Box                 *Mdbox,
                                 hypre_Index                Mdstart,
                                 hypre_Index                Mdstride )
{
   HYPRE_Int       nprod[8];
   HYPRE_Complex   cprod[8][na];
   hypre_3Cptrs    tptrs[8][na];
   hypre_1Cptr     mptrs[8][na];

   HYPRE_Int       mentry, ptype, nf, nc, nt;
   HYPRE_Int       e, p, i, k, t;

   for (e = 0; e < stencil_size; e++)
   {
      /* Reset product counters */
      for (p = 0; p < 8; p++)
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
               break;

            case 2:
               switch (nc)
               {
                  case 0: /* ff term (call core_ff) */
                     ptype = 3;
                     break;
                  case 1: /* cf term (call core_cf) */
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
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Can't have zero terms in StructMatmult!");
               return hypre_error_flag;
         }

         /* Set array values for product k of product type ptype */
         k = nprod[ptype];
         cprod[ptype][k] = a[i].cprod;
         nf = nc = 0;
         for (t = 0; t < nterms; t++)
         {
            if (a[i].types[t] == 1)
            {
               /* Type 1 -> coarse data space */
               tptrs[ptype][k][nc] = a[i].tptrs[t];  /* put first */
               nc++;
            }
            else if (a[i].types[t] != 3)
            {
               /* Type 0 or 2 -> fine data space */
               tptrs[ptype][k][nt - 1 - nf] = a[i].tptrs[t];  /* put last */
               nf++;
            }
         }
         mptrs[ptype][k] = a[i].mptr;
         nprod[ptype]++;
      }

      //hypre_ParPrintf(MPI_COMM_WORLD, "%d %d %d %d %d %d %d %d\n",
      //                nprod[0], nprod[1], nprod[2], nprod[3], nprod[4],
      //                nprod[5], nprod[6], nprod[7]);

      /* Call core functions */
      if (nterms > 2)
      {
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
      }

      hypre_StructMatmultCompute_core_ff(nprod[3], cprod[3], tptrs[3], mptrs[3],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_cf(nprod[4], cprod[4], tptrs[4], mptrs[4],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         cdbox, cdstart, cdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_cc(nprod[5], cprod[5], tptrs[5], mptrs[5],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_f(nprod[6], cprod[6], tptrs[6], mptrs[6],
                                        ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_c(nprod[7], cprod[7], tptrs[7], mptrs[7],
                                        ndim, loop_size,
                                        cdbox, cdstart, cdstride,
                                        Mdbox, Mdstart, Mdstride);
   } /* loop on M stencil entries */

   return hypre_error_flag;
}

