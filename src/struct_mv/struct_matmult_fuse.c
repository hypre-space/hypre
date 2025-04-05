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

#define HYPRE_FUSE_MAXDEPTH 9

typedef HYPRE_Complex *hypre_3Cptrs[3];
typedef HYPRE_Complex *hypre_1Cptr;

/*--------------------------------------------------------------------------
 * Macros used in the kernel loops below
 *
 * F/C means fine/coarse data space.  For example, CFF means the data spaces for
 * the three terms are respectively coarse, fine, and fine.
 *--------------------------------------------------------------------------*/

#define HYPRE_SMMFUSE_FFF(k) \
   mptrs[k][Mi] += cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi]* tptrs[k][2][fi]

#define HYPRE_SMMFUSE_FFC(k) \
   mptrs[k][Mi] += cprod[k] * tptrs[k][0][fi] * tptrs[k][1][fi]* tptrs[k][2][ci]
//   locmp[k][Mi] += loccp[k] * loctp[k][0][fi] * loctp[k][1][fi]* loctp[k][2][ci]

#define HYPRE_SMMFUSE_FCC(k) \
   mptrs[k][Mi] += cprod[k] * tptrs[k][0][fi] * tptrs[k][1][ci]* tptrs[k][2][ci]

#define HYPRE_SMMFUSE_FCC_M1M(k) \
   mptrs[k][Mi] += cprod[k] * tptrs[k][0][fi] * a1ptr[ci]* tptrs[k][2][ci]

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_fff( HYPRE_Int            nprod,
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
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   //for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   for (k = 0; k < nprod; /* increment k at the end of the body below */ )
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 15:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
               HYPRE_SMMFUSE_FFF(k + 8);
               HYPRE_SMMFUSE_FFF(k + 9);
               HYPRE_SMMFUSE_FFF(k + 10);
               HYPRE_SMMFUSE_FFF(k + 11);
               HYPRE_SMMFUSE_FFF(k + 12);
               HYPRE_SMMFUSE_FFF(k + 13);
               HYPRE_SMMFUSE_FFF(k + 14);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 14:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
               HYPRE_SMMFUSE_FFF(k + 8);
               HYPRE_SMMFUSE_FFF(k + 9);
               HYPRE_SMMFUSE_FFF(k + 10);
               HYPRE_SMMFUSE_FFF(k + 11);
               HYPRE_SMMFUSE_FFF(k + 12);
               HYPRE_SMMFUSE_FFF(k + 13);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 13:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
               HYPRE_SMMFUSE_FFF(k + 8);
               HYPRE_SMMFUSE_FFF(k + 9);
               HYPRE_SMMFUSE_FFF(k + 10);
               HYPRE_SMMFUSE_FFF(k + 11);
               HYPRE_SMMFUSE_FFF(k + 12);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 12:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
               HYPRE_SMMFUSE_FFF(k + 8);
               HYPRE_SMMFUSE_FFF(k + 9);
               HYPRE_SMMFUSE_FFF(k + 10);
               HYPRE_SMMFUSE_FFF(k + 11);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 11:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
               HYPRE_SMMFUSE_FFF(k + 8);
               HYPRE_SMMFUSE_FFF(k + 9);
               HYPRE_SMMFUSE_FFF(k + 10);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 10:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
               HYPRE_SMMFUSE_FFF(k + 8);
               HYPRE_SMMFUSE_FFF(k + 9);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 9:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
               HYPRE_SMMFUSE_FFF(k + 8);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
               HYPRE_SMMFUSE_FFF(k + 7);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
               HYPRE_SMMFUSE_FFF(k + 6);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
               HYPRE_SMMFUSE_FFF(k + 5);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
               HYPRE_SMMFUSE_FFF(k + 4);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
               HYPRE_SMMFUSE_FFF(k + 3);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
               HYPRE_SMMFUSE_FFF(k + 2);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
               HYPRE_SMMFUSE_FFF(k + 1);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF(k + 0);
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }

      k += depth;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_ffc( HYPRE_Int            nprod,
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
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   //for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   for (k = 0; k < nprod; /* increment k at the end of the body below */ )
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 15:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
               HYPRE_SMMFUSE_FFC(k + 8);
               HYPRE_SMMFUSE_FFC(k + 9);
               HYPRE_SMMFUSE_FFC(k + 10);
               HYPRE_SMMFUSE_FFC(k + 11);
               HYPRE_SMMFUSE_FFC(k + 12);
               HYPRE_SMMFUSE_FFC(k + 13);
               HYPRE_SMMFUSE_FFC(k + 14);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
               HYPRE_SMMFUSE_FFC(k + 8);
               HYPRE_SMMFUSE_FFC(k + 9);
               HYPRE_SMMFUSE_FFC(k + 10);
               HYPRE_SMMFUSE_FFC(k + 11);
               HYPRE_SMMFUSE_FFC(k + 12);
               HYPRE_SMMFUSE_FFC(k + 13);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
               HYPRE_SMMFUSE_FFC(k + 8);
               HYPRE_SMMFUSE_FFC(k + 9);
               HYPRE_SMMFUSE_FFC(k + 10);
               HYPRE_SMMFUSE_FFC(k + 11);
               HYPRE_SMMFUSE_FFC(k + 12);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
               HYPRE_SMMFUSE_FFC(k + 8);
               HYPRE_SMMFUSE_FFC(k + 9);
               HYPRE_SMMFUSE_FFC(k + 10);
               HYPRE_SMMFUSE_FFC(k + 11);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
               HYPRE_SMMFUSE_FFC(k + 8);
               HYPRE_SMMFUSE_FFC(k + 9);
               HYPRE_SMMFUSE_FFC(k + 10);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
               HYPRE_SMMFUSE_FFC(k + 8);
               HYPRE_SMMFUSE_FFC(k + 9);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
               HYPRE_SMMFUSE_FFC(k + 8);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
               HYPRE_SMMFUSE_FFC(k + 7);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
               HYPRE_SMMFUSE_FFC(k + 6);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
               HYPRE_SMMFUSE_FFC(k + 5);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
               HYPRE_SMMFUSE_FFC(k + 4);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
               HYPRE_SMMFUSE_FFC(k + 3);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
               HYPRE_SMMFUSE_FFC(k + 2);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
               HYPRE_SMMFUSE_FFC(k + 1);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC(k + 0);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }

      k += depth;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_fcc( HYPRE_Int            nprod,
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
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   //for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   for (k = 0; k < nprod; /* increment k at the end of the body below */ )
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 31:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
               HYPRE_SMMFUSE_FCC(k + 9);
               HYPRE_SMMFUSE_FCC(k + 10);
               HYPRE_SMMFUSE_FCC(k + 11);
               HYPRE_SMMFUSE_FCC(k + 12);
               HYPRE_SMMFUSE_FCC(k + 13);
               HYPRE_SMMFUSE_FCC(k + 14);
               HYPRE_SMMFUSE_FCC(k + 15);
               HYPRE_SMMFUSE_FCC(k + 16);
               HYPRE_SMMFUSE_FCC(k + 17);
               HYPRE_SMMFUSE_FCC(k + 18);
               HYPRE_SMMFUSE_FCC(k + 19);
               HYPRE_SMMFUSE_FCC(k + 20);
               HYPRE_SMMFUSE_FCC(k + 21);
               HYPRE_SMMFUSE_FCC(k + 22);
               HYPRE_SMMFUSE_FCC(k + 23);
               HYPRE_SMMFUSE_FCC(k + 24);
               HYPRE_SMMFUSE_FCC(k + 25);
               HYPRE_SMMFUSE_FCC(k + 26);
               HYPRE_SMMFUSE_FCC(k + 27);
               HYPRE_SMMFUSE_FCC(k + 28);
               HYPRE_SMMFUSE_FCC(k + 29);
               HYPRE_SMMFUSE_FCC(k + 30);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 15:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
               HYPRE_SMMFUSE_FCC(k + 9);
               HYPRE_SMMFUSE_FCC(k + 10);
               HYPRE_SMMFUSE_FCC(k + 11);
               HYPRE_SMMFUSE_FCC(k + 12);
               HYPRE_SMMFUSE_FCC(k + 13);
               HYPRE_SMMFUSE_FCC(k + 14);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
               HYPRE_SMMFUSE_FCC(k + 9);
               HYPRE_SMMFUSE_FCC(k + 10);
               HYPRE_SMMFUSE_FCC(k + 11);
               HYPRE_SMMFUSE_FCC(k + 12);
               HYPRE_SMMFUSE_FCC(k + 13);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
               HYPRE_SMMFUSE_FCC(k + 9);
               HYPRE_SMMFUSE_FCC(k + 10);
               HYPRE_SMMFUSE_FCC(k + 11);
               HYPRE_SMMFUSE_FCC(k + 12);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
               HYPRE_SMMFUSE_FCC(k + 9);
               HYPRE_SMMFUSE_FCC(k + 10);
               HYPRE_SMMFUSE_FCC(k + 11);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
               HYPRE_SMMFUSE_FCC(k + 9);
               HYPRE_SMMFUSE_FCC(k + 10);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
               HYPRE_SMMFUSE_FCC(k + 9);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
               HYPRE_SMMFUSE_FCC(k + 8);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
               HYPRE_SMMFUSE_FCC(k + 7);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
               HYPRE_SMMFUSE_FCC(k + 6);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
               HYPRE_SMMFUSE_FCC(k + 5);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
               HYPRE_SMMFUSE_FCC(k + 4);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
               HYPRE_SMMFUSE_FCC(k + 3);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
               HYPRE_SMMFUSE_FCC(k + 2);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
               HYPRE_SMMFUSE_FCC(k + 1);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC(k + 0);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }

      k += depth;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Notation: f/c is fine/coarse data space; m/1 is multi/1 ptr
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_fcc_m1m( HYPRE_Int        nprod,
                                         HYPRE_Complex   *cprod,
                                         hypre_3Cptrs    *tptrs,
                                         hypre_1Cptr     *mptrs,
                                         HYPRE_Int        ndim,
                                         hypre_Index      loop_size,
                                         hypre_Box       *fdbox,
                                         hypre_Index      fdstart,
                                         hypre_Index      fdstride,
                                         hypre_Box       *cdbox,
                                         hypre_Index      cdstart,
                                         hypre_Index      cdstride,
                                         hypre_Box       *Mdbox,
                                         hypre_Index      Mdstart,
                                         hypre_Index      Mdstride )
{
   HYPRE_Complex  *a1ptr;
   HYPRE_Int       k;
   HYPRE_Int       depth, repeat_depth;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < nprod; /* increment k at the end of the body below */ )
   {
      a1ptr = tptrs[k][1];

      repeat_depth = 1;
      while ( ((k + repeat_depth) < nprod) && (tptrs[k + repeat_depth][1] == a1ptr) )
      {
         repeat_depth++;
      }
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));
      depth = hypre_min(depth, repeat_depth);

      //hypre_ParPrintf(MPI_COMM_WORLD, "depth = %d\n", depth);

      switch (depth)
      {
         case 15:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
               HYPRE_SMMFUSE_FCC_M1M(k + 8);
               HYPRE_SMMFUSE_FCC_M1M(k + 9);
               HYPRE_SMMFUSE_FCC_M1M(k + 10);
               HYPRE_SMMFUSE_FCC_M1M(k + 11);
               HYPRE_SMMFUSE_FCC_M1M(k + 12);
               HYPRE_SMMFUSE_FCC_M1M(k + 13);
               HYPRE_SMMFUSE_FCC_M1M(k + 14);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
               HYPRE_SMMFUSE_FCC_M1M(k + 8);
               HYPRE_SMMFUSE_FCC_M1M(k + 9);
               HYPRE_SMMFUSE_FCC_M1M(k + 10);
               HYPRE_SMMFUSE_FCC_M1M(k + 11);
               HYPRE_SMMFUSE_FCC_M1M(k + 12);
               HYPRE_SMMFUSE_FCC_M1M(k + 13);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
               HYPRE_SMMFUSE_FCC_M1M(k + 8);
               HYPRE_SMMFUSE_FCC_M1M(k + 9);
               HYPRE_SMMFUSE_FCC_M1M(k + 10);
               HYPRE_SMMFUSE_FCC_M1M(k + 11);
               HYPRE_SMMFUSE_FCC_M1M(k + 12);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
               HYPRE_SMMFUSE_FCC_M1M(k + 8);
               HYPRE_SMMFUSE_FCC_M1M(k + 9);
               HYPRE_SMMFUSE_FCC_M1M(k + 10);
               HYPRE_SMMFUSE_FCC_M1M(k + 11);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
               HYPRE_SMMFUSE_FCC_M1M(k + 8);
               HYPRE_SMMFUSE_FCC_M1M(k + 9);
               HYPRE_SMMFUSE_FCC_M1M(k + 10);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
               HYPRE_SMMFUSE_FCC_M1M(k + 8);
               HYPRE_SMMFUSE_FCC_M1M(k + 9);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
               HYPRE_SMMFUSE_FCC_M1M(k + 8);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
               HYPRE_SMMFUSE_FCC_M1M(k + 7);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
               HYPRE_SMMFUSE_FCC_M1M(k + 6);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
               HYPRE_SMMFUSE_FCC_M1M(k + 5);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
               HYPRE_SMMFUSE_FCC_M1M(k + 4);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
               HYPRE_SMMFUSE_FCC_M1M(k + 3);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
               HYPRE_SMMFUSE_FCC_M1M(k + 2);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
               HYPRE_SMMFUSE_FCC_M1M(k + 1);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_M1M(k + 0);
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported loop fusion depth!");

            HYPRE_ANNOTATE_FUNC_END;
            return hypre_error_flag;
      }

      k += depth;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_fuse_order_bigints( HYPRE_Int      nprod,
                          HYPRE_Int     *order,
                          HYPRE_BigInt  *bigints )
{
   HYPRE_Int     k;
   HYPRE_BigInt  tmp_bigints[nprod];

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
   hypre_3Cptrs  tmp_tptrs[nprod];
   hypre_1Cptr   tmp_mptrs[nprod];

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
   HYPRE_Int       approach = 0;
   HYPRE_Int       debug    = 0;

   HYPRE_Int       k;
   HYPRE_Complex  *minptrs[4];
   HYPRE_BigInt    distances[4][nprod];
   HYPRE_Int       order[nprod];

   if ((nprod < 1) || (approach == 0))
   {
      return hypre_error_flag;
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

   /* Print distances */
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(MPI_COMM_WORLD, "distances[%2d]   %16d %16d %16d   %16d\n",
                      k, distances[0][k], distances[1][k], distances[2][k], distances[3][k]);
   }
   hypre_ParPrintf(MPI_COMM_WORLD, "\n");

   /* Sort according to middle column (index 1) */
   hypre_BigQsortbi(distances[1], order, 0, nprod - 1);

   if (debug)
   {
      hypre_fuse_order_bigints(nprod, order, distances[0]);
      hypre_fuse_order_bigints(nprod, order, distances[2]);
      hypre_fuse_order_bigints(nprod, order, distances[3]);

      /* Print order array */
      for (k = 0; k < nprod; k++)
      {
         hypre_ParPrintf(MPI_COMM_WORLD, " %d", order[k]);
      }
      hypre_ParPrintf(MPI_COMM_WORLD, "\n\n");

      /* Print distances */
      for (k = 0; k < nprod; k++)
      {
         hypre_ParPrintf(MPI_COMM_WORLD, "distances[%2d]   %16b %16b %16b   %16b\n",
                         k, distances[0][k], distances[1][k], distances[2][k], distances[3][k]);
      }
      hypre_ParPrintf(MPI_COMM_WORLD, "\n");
   }

   /* Reorder data pointers */
   hypre_fuse_order_ptrs(nprod, order, tptrs, mptrs);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_triple( hypre_StructMatmultDataMH *a,
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

   HYPRE_Int       ptype, nf, nc;
   HYPRE_Int       p, i, k, t;

   /* Initialize product counters */
   for (p = 0; p < 3; p++)
   {
      nprod[p] = 0;
   }

   /* Build product arrays */
   for (i = 0; i < na; i++)
   {
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

      /* Set array values for product k of product type ptype */
      k = nprod[ptype];
      cprod[ptype][k] = a[i].cprod;
      nf = nc = 0;
      for (t = 0; t < 3; t++)
      {
         if (a[i].types[t] == 1)
         {
            /* Type 1 -> coarse data space */
            tptrs[ptype][k][2 - nc] = a[i].tptrs[t];  /* put last */
            nc++;
         }
         else
         {
            /* Type 0 or 2 -> fine data space */
            tptrs[ptype][k][nf] = a[i].tptrs[t];  /* put first */
            nf++;
         }
      }
      mptrs[ptype][k] = a[i].mptr;
      nprod[ptype]++;

   } /* loop i < na*/

   //hypre_ParPrintf(MPI_COMM_WORLD, "nprod = %d, %d, %d\n", nprod[0], nprod[1], nprod[2]);

   hypre_fuse_sort(nprod[0], tptrs[0], mptrs[0]);
   hypre_fuse_sort(nprod[1], tptrs[1], mptrs[1]);
   hypre_fuse_sort(nprod[2], tptrs[2], mptrs[2]);

   /* Call fuse functions */
   hypre_StructMatmultCompute_fuse_fff(nprod[0], cprod[0], tptrs[0], mptrs[0],
                                       ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_ffc(nprod[1], cprod[1], tptrs[1], mptrs[1],
                                       ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       cdbox, cdstart, cdstride,
                                       Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_fcc(nprod[2], cprod[2], tptrs[2], mptrs[2],
                                       ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       cdbox, cdstart, cdstride,
                                       Mdbox, Mdstart, Mdstride);
   //hypre_StructMatmultCompute_fuse_fcc_m1m(nprod[2], cprod[2], tptrs[2], mptrs[2],
   //                                        ndim, loop_size,
   //                                        fdbox, fdstart, fdstride,
   //                                        cdbox, cdstart, cdstride,
   //                                        Mdbox, Mdstart, Mdstride);

   return hypre_error_flag;
}

