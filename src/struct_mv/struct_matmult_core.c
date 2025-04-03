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

/*--------------------------------------------------------------------------
 * Macros used in the kernel loops below
 *
 * F/C means fine/coarse data space.  For example, CFF means the data spaces for
 * the three terms are respectively coarse, fine, and fine.
 *--------------------------------------------------------------------------*/

#define HYPRE_SMMCORE_FF(k) \
   cprod[k]*                \
   tptrs[k][0][fi]*         \
   tptrs[k][1][fi]

#define HYPRE_SMMCORE_CF(k) \
    cprod[k]*               \
    tptrs[k][0][ci]*        \
    tptrs[k][1][fi]

#define HYPRE_SMMCORE_FFF(k) \
   cprod[k]*                 \
   tptrs[k][0][fi]*          \
   tptrs[k][1][fi]*          \
   tptrs[k][2][fi]

#define HYPRE_SMMCORE_CFF(k) \
    cprod[k]*                \
    tptrs[k][0][ci]*         \
    tptrs[k][1][fi]*         \
    tptrs[k][2][fi]

#define HYPRE_SMMCORE_CCF(k) \
    cprod[k]*                \
    tptrs[k][0][ci]*         \
    tptrs[k][1][ci]*         \
    tptrs[k][2][fi]

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_ff( HYPRE_Int                  ncomponents,
                                    HYPRE_Complex             *cprod,
                                    const HYPRE_Complex     ***tptrs,
                                    HYPRE_Complex             *mptr,
                                    HYPRE_Int                  ndim,
                                    hypre_Index                loop_size,
                                    hypre_Box                 *fdbox,
                                    hypre_Index                fdstart,
                                    hypre_Index                fdstride,
                                    hypre_Box                 *Mdbox,
                                    hypre_Index                Mdstart,
                                    hypre_Index                Mdstride )
{
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

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
hypre_StructMatmultCompute_core_cf( HYPRE_Int                  ncomponents,
                                    HYPRE_Int                **order,
                                    HYPRE_Complex             *cprod,
                                    const HYPRE_Complex     ***tptrs,
                                    HYPRE_Complex             *mptr,
                                    HYPRE_Int                  ndim,
                                    hypre_Index                loop_size,
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
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

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
hypre_StructMatmultCompute_core_fff( HYPRE_Int                  ncomponents,
                                     HYPRE_Complex             *cprod,
                                     const HYPRE_Complex     ***tptrs,
                                     HYPRE_Complex             *mptr,
                                     HYPRE_Int                  ndim,
                                     hypre_Index                loop_size,
                                     hypre_Box                 *fdbox,
                                     hypre_Index                fdstart,
                                     hypre_Index                fdstride,
                                     hypre_Box                 *Mdbox,
                                     hypre_Index                Mdstart,
                                     hypre_Index                Mdstride )
{
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

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
hypre_StructMatmultCompute_core_cff( HYPRE_Int                  ncomponents,
                                     HYPRE_Int                **order,
                                     HYPRE_Complex             *cprod,
                                     const HYPRE_Complex     ***tptrs,
                                     HYPRE_Complex             *mptr,
                                     HYPRE_Int                  ndim,
                                     hypre_Index                loop_size,
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
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

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
hypre_StructMatmultCompute_core_ccf( HYPRE_Int                  ncomponents,
                                     HYPRE_Int                **order,
                                     HYPRE_Complex             *cprod,
                                     const HYPRE_Complex     ***tptrs,
                                     HYPRE_Complex             *mptr,
                                     HYPRE_Int                  ndim,
                                     hypre_Index                loop_size,
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
   HYPRE_Int     k;
   HYPRE_Int     depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

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
   HYPRE_Int               *ncomp;
   HYPRE_Complex          **cprod;
   HYPRE_Int             ***order;
   const HYPRE_Complex  ****tptrs;
   HYPRE_Complex          **mptrs;

   HYPRE_Int                max_components = 2;
   HYPRE_Int                mentry, comptype, nf, nc;
   HYPRE_Int                e, c, i, k, t;

   /* Allocate memory */
   ncomp = hypre_CTAlloc(HYPRE_Int, max_components, HYPRE_MEMORY_HOST);
   cprod = hypre_TAlloc(HYPRE_Complex *, max_components, HYPRE_MEMORY_HOST);
   order = hypre_TAlloc(HYPRE_Int **, max_components, HYPRE_MEMORY_HOST);
   tptrs = hypre_TAlloc(const HYPRE_Complex***, max_components, HYPRE_MEMORY_HOST);
   mptrs = hypre_TAlloc(HYPRE_Complex*, max_components, HYPRE_MEMORY_HOST);
   for (c = 0; c < max_components; c++)
   {
      cprod[c] = hypre_CTAlloc(HYPRE_Complex, na, HYPRE_MEMORY_HOST);
      order[c] = hypre_TAlloc(HYPRE_Int *, na, HYPRE_MEMORY_HOST);
      tptrs[c] = hypre_TAlloc(const HYPRE_Complex **, na, HYPRE_MEMORY_HOST);
      for (t = 0; t < na; t++)
      {
         order[c][t] = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
         tptrs[c][t] = hypre_TAlloc(const HYPRE_Complex *, 2, HYPRE_MEMORY_HOST);
      }
   }

   for (e = 0; e < stencil_size; e++)
   {
      /* Reset component counters */
      for (c = 0; c < max_components; c++)
      {
         ncomp[c] = 0;
      }

      /* Build components arrays */
      for (i = 0; i < na; i++)
      {
         mentry = a[i].mentry;
         if (mentry != e)
         {
            continue;
         }

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
         switch (nc)
         {
            case 0: /* ff term (call core_ff) */
               comptype = 0;
               break;
            case 1: /* cf term (call core_cf) */
               comptype = 1;
               break;
         }
         k = ncomp[comptype];
         cprod[comptype][k] = a[i].cprod;
         nf = nc = 0;
         for (t = 0; t < 2; t++)
         {
            if (a[i].types[t] == 1)
            {
               /* Type 1 -> coarse data space */
               tptrs[comptype][k][nc] = a[i].tptrs[t];  /* put first */
               nc++;
            }
            else
            {
               /* Type 0 or 2 -> fine data space */
               tptrs[comptype][k][1 - nf] = a[i].tptrs[t];  /* put last */
               nf++;
            }
         }
         mptrs[comptype] = a[i].mptr;
         ncomp[comptype]++;
      }

      /* Call core functions */
      hypre_StructMatmultCompute_core_ff(ncomp[0], cprod[0],
                                         tptrs[0], mptrs[0],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_cf(ncomp[1], order[1],
                                         cprod[1], tptrs[1],
                                         mptrs[1], ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         cdbox, cdstart, cdstride,
                                         Mdbox, Mdstart, Mdstride);
   } /* loop on M stencil entries */

   /* Free memory */
   for (c = 0; c < max_components; c++)
   {
      for (t = 0; t < na; t++)
      {
         hypre_TFree(order[c][t], HYPRE_MEMORY_HOST);
         hypre_TFree(tptrs[c][t], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(cprod[c], HYPRE_MEMORY_HOST);
      hypre_TFree(order[c], HYPRE_MEMORY_HOST);
      hypre_TFree(tptrs[c], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(ncomp, HYPRE_MEMORY_HOST);
   hypre_TFree(cprod, HYPRE_MEMORY_HOST);
   hypre_TFree(order, HYPRE_MEMORY_HOST);
   hypre_TFree(tptrs, HYPRE_MEMORY_HOST);
   hypre_TFree(mptrs, HYPRE_MEMORY_HOST);

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
   HYPRE_Int               *ncomp;
   HYPRE_Complex          **cprod;
   HYPRE_Int             ***order;
   const HYPRE_Complex  ****tptrs;
   HYPRE_Complex          **mptrs;

   HYPRE_Int                max_components = 3;
   HYPRE_Int                mentry, comptype, nf, nc;
   HYPRE_Int                e, c, i, k, t;

   /* Allocate memory */
   ncomp = hypre_CTAlloc(HYPRE_Int, max_components, HYPRE_MEMORY_HOST);
   cprod = hypre_TAlloc(HYPRE_Complex *, max_components, HYPRE_MEMORY_HOST);
   order = hypre_TAlloc(HYPRE_Int **, max_components, HYPRE_MEMORY_HOST);
   tptrs = hypre_TAlloc(const HYPRE_Complex***, max_components, HYPRE_MEMORY_HOST);
   mptrs = hypre_TAlloc(HYPRE_Complex*, max_components, HYPRE_MEMORY_HOST);
   for (c = 0; c < max_components; c++)
   {
      cprod[c] = hypre_CTAlloc(HYPRE_Complex, na, HYPRE_MEMORY_HOST);
      order[c] = hypre_TAlloc(HYPRE_Int *, na, HYPRE_MEMORY_HOST);
      tptrs[c] = hypre_TAlloc(const HYPRE_Complex **, na, HYPRE_MEMORY_HOST);
      for (t = 0; t < na; t++)
      {
         order[c][t] = hypre_CTAlloc(HYPRE_Int, 3, HYPRE_MEMORY_HOST);
         tptrs[c][t] = hypre_TAlloc(const HYPRE_Complex *, 3, HYPRE_MEMORY_HOST);
      }
   }

   for (e = 0; e < stencil_size; e++)
   {
      /* Reset component counters */
      for (c = 0; c < max_components; c++)
      {
         ncomp[c] = 0;
      }

      /* Build components arrays */
      for (i = 0; i < na; i++)
      {
         mentry = a[i].mentry;
         if (mentry != e)
         {
            continue;
         }

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
         switch (nc)
         {
            case 0: /* fff term (call core_fff) */
               comptype = 0;
               break;
            case 1: /* cff term (call core_cff) */
               comptype = 1;
               break;
            case 2: /* ccf term (call core_ccf) */
               comptype = 2;
               break;
         }
         k = ncomp[comptype];
         cprod[comptype][k] = a[i].cprod;
         nf = nc = 0;
         for (t = 0; t < 3; t++)
         {
            if (a[i].types[t] == 1)
            {
               /* Type 1 -> coarse data space */
               tptrs[comptype][k][nc] = a[i].tptrs[t];  /* put first */
               nc++;
            }
            else
            {
               /* Type 0 or 2 -> fine data space */
               tptrs[comptype][k][2 - nf] = a[i].tptrs[t];  /* put last */
               nf++;
            }
         }
         mptrs[comptype] = a[i].mptr;
         ncomp[comptype]++;
      }

      /* Call core functions */
      hypre_StructMatmultCompute_core_fff(ncomp[0], cprod[0],
                                          tptrs[0], mptrs[0],
                                          ndim, loop_size,
                                          fdbox, fdstart, fdstride,
                                          Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_cff(ncomp[1], order[1],
                                          cprod[1], tptrs[1], mptrs[1],
                                          ndim, loop_size,
                                          fdbox, fdstart, fdstride,
                                          cdbox, cdstart, cdstride,
                                          Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_ccf(ncomp[2], order[2],
                                          cprod[2], tptrs[2], mptrs[2],
                                          ndim, loop_size,
                                          fdbox, fdstart, fdstride,
                                          cdbox, cdstart, cdstride,
                                          Mdbox, Mdstart, Mdstride);
   } /* loop on M stencil entries */

   /* Free memory */
   for (c = 0; c < max_components; c++)
   {
      for (t = 0; t < na; t++)
      {
         hypre_TFree(order[c][t], HYPRE_MEMORY_HOST);
         hypre_TFree(tptrs[c][t], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(cprod[c], HYPRE_MEMORY_HOST);
      hypre_TFree(order[c], HYPRE_MEMORY_HOST);
      hypre_TFree(tptrs[c], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(ncomp, HYPRE_MEMORY_HOST);
   hypre_TFree(cprod, HYPRE_MEMORY_HOST);
   hypre_TFree(order, HYPRE_MEMORY_HOST);
   hypre_TFree(tptrs, HYPRE_MEMORY_HOST);
   hypre_TFree(mptrs, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

