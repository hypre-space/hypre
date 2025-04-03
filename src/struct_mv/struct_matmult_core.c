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
 * Nomenclature used in the kernel functions:
 *   1) VCC stands for "Variable Coefficient on Coarse data space".
 *   2) VCF stands for "Variable Coefficient on Fine data space".
 *   3) CCF stands for "Constant Coefficient on Fine data space".
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
 *--------------------------------------------------------------------------*/

#define HYPRE_SMMCORE_1D(k) \
   cprod[k]*                \
   tptrs[k][0][gi]*         \
   tptrs[k][1][gi]

#define HYPRE_SMMCORE_2DB(k) \
    cprod[k]*                \
    tptrs[k][0][hi]*         \
    tptrs[k][1][gi]

#define HYPRE_SMMCORE_1T(k) \
   cprod[k]*                \
   tptrs[k][0][gi]*         \
   tptrs[k][1][gi]*         \
   tptrs[k][2][gi]

#define HYPRE_SMMCORE_2ETB(k) \
    cprod[k]*                 \
    tptrs[k][0][hi]*          \
    tptrs[k][1][hi]*          \
    tptrs[k][2][gi]

#define HYPRE_SMMCORE_2TBB(k) \
    cprod[k]*                 \
    tptrs[k][0][hi]*          \
    tptrs[k][1][gi]*          \
    tptrs[k][2][gi]

/*--------------------------------------------------------------------------
 * Core function for computing the double-product of variable coefficients
 * living on the same data space.
 *
 * "1d" means:
 *   "1": single data space.
 *   "d": double-product.
 *
 * This can be used for the scenarios:
 *   1) VCF * VCF.
 *   2) VCC * VCC.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_1d( hypre_StructMatmultDataMH *a,
                                    HYPRE_Int                  ncomponents,
                                    HYPRE_Complex             *cprod,
                                    const HYPRE_Complex     ***tptrs,
                                    HYPRE_Complex             *mptr,
                                    HYPRE_Int                  ndim,
                                    hypre_Index                loop_size,
                                    hypre_Box                 *gdbox,
                                    hypre_Index                gdstart,
                                    hypre_Index                gdstride,
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
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0) +
                                   HYPRE_SMMCORE_1D(k + 1) +
                                   HYPRE_SMMCORE_1D(k + 2) +
                                   HYPRE_SMMCORE_1D(k + 3) +
                                   HYPRE_SMMCORE_1D(k + 4) +
                                   HYPRE_SMMCORE_1D(k + 5) +
                                   HYPRE_SMMCORE_1D(k + 6) +
                                   HYPRE_SMMCORE_1D(k + 7);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0) +
                                   HYPRE_SMMCORE_1D(k + 1) +
                                   HYPRE_SMMCORE_1D(k + 2) +
                                   HYPRE_SMMCORE_1D(k + 3) +
                                   HYPRE_SMMCORE_1D(k + 4) +
                                   HYPRE_SMMCORE_1D(k + 5) +
                                   HYPRE_SMMCORE_1D(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0) +
                                   HYPRE_SMMCORE_1D(k + 1) +
                                   HYPRE_SMMCORE_1D(k + 2) +
                                   HYPRE_SMMCORE_1D(k + 3) +
                                   HYPRE_SMMCORE_1D(k + 4) +
                                   HYPRE_SMMCORE_1D(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0) +
                                   HYPRE_SMMCORE_1D(k + 1) +
                                   HYPRE_SMMCORE_1D(k + 2) +
                                   HYPRE_SMMCORE_1D(k + 3) +
                                   HYPRE_SMMCORE_1D(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0) +
                                   HYPRE_SMMCORE_1D(k + 1) +
                                   HYPRE_SMMCORE_1D(k + 2) +
                                   HYPRE_SMMCORE_1D(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0) +
                                   HYPRE_SMMCORE_1D(k + 1) +
                                   HYPRE_SMMCORE_1D(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0) +
                                   HYPRE_SMMCORE_1D(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1D(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
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
 * Core function for computing the product of two coefficients living on
 * different data spaces. The second coefficients requires usage of a bitmask
 *
 * "2db" means:
 *   "2": two data spaces.
 *   "d": double-product.
 *   "b": single bitmask.
 *
 * This can be used for the scenarios:
 *   1) VCC * CCF.
 *   2) CCF * VCC.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2db( hypre_StructMatmultDataMH *a,
                                     HYPRE_Int                  ncomponents,
                                     HYPRE_Int                **order,
                                     HYPRE_Complex             *cprod,
                                     const HYPRE_Complex     ***tptrs,
                                     HYPRE_Complex             *mptr,
                                     HYPRE_Int                  ndim,
                                     hypre_Index                loop_size,
                                     hypre_Box                 *gdbox,
                                     hypre_Index                gdstart,
                                     hypre_Index                gdstride,
                                     hypre_Box                 *hdbox,
                                     hypre_Index                hdstart,
                                     hypre_Index                hdstride,
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
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2DB(k + 0) +
                                   HYPRE_SMMCORE_2DB(k + 1) +
                                   HYPRE_SMMCORE_2DB(k + 2) +
                                   HYPRE_SMMCORE_2DB(k + 3) +
                                   HYPRE_SMMCORE_2DB(k + 4) +
                                   HYPRE_SMMCORE_2DB(k + 5) +
                                   HYPRE_SMMCORE_2DB(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2DB(k + 0) +
                                   HYPRE_SMMCORE_2DB(k + 1) +
                                   HYPRE_SMMCORE_2DB(k + 2) +
                                   HYPRE_SMMCORE_2DB(k + 3) +
                                   HYPRE_SMMCORE_2DB(k + 4) +
                                   HYPRE_SMMCORE_2DB(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2DB(k + 0) +
                                   HYPRE_SMMCORE_2DB(k + 1) +
                                   HYPRE_SMMCORE_2DB(k + 2) +
                                   HYPRE_SMMCORE_2DB(k + 3) +
                                   HYPRE_SMMCORE_2DB(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2DB(k + 0) +
                                   HYPRE_SMMCORE_2DB(k + 1) +
                                   HYPRE_SMMCORE_2DB(k + 2) +
                                   HYPRE_SMMCORE_2DB(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2DB(k + 0) +
                                   HYPRE_SMMCORE_2DB(k + 1) +
                                   HYPRE_SMMCORE_2DB(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2DB(k + 0) +
                                   HYPRE_SMMCORE_2DB(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2DB(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
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
 * Core function for computing the triple-product of variable coefficients
 * living on the same data space.
 *
 * "1t" means:
 *   "1": single data space.
 *   "t": triple-product.
 *
 * This can be used for the scenarios:
 *   1) VCF * VCF * CCF.
 *   2) VCC * VCC * CCF.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_1t( hypre_StructMatmultDataMH *a,
                                    HYPRE_Int                  ncomponents,
                                    HYPRE_Complex             *cprod,
                                    const HYPRE_Complex     ***tptrs,
                                    HYPRE_Complex             *mptr,
                                    HYPRE_Int                  ndim,
                                    hypre_Index                loop_size,
                                    hypre_Box                 *gdbox,
                                    hypre_Index                gdstart,
                                    hypre_Index                gdstride,
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
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0) +
                                   HYPRE_SMMCORE_1T(k + 1) +
                                   HYPRE_SMMCORE_1T(k + 2) +
                                   HYPRE_SMMCORE_1T(k + 3) +
                                   HYPRE_SMMCORE_1T(k + 4) +
                                   HYPRE_SMMCORE_1T(k + 5) +
                                   HYPRE_SMMCORE_1T(k + 6) +
                                   HYPRE_SMMCORE_1T(k + 7);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0) +
                                   HYPRE_SMMCORE_1T(k + 1) +
                                   HYPRE_SMMCORE_1T(k + 2) +
                                   HYPRE_SMMCORE_1T(k + 3) +
                                   HYPRE_SMMCORE_1T(k + 4) +
                                   HYPRE_SMMCORE_1T(k + 5) +
                                   HYPRE_SMMCORE_1T(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0) +
                                   HYPRE_SMMCORE_1T(k + 1) +
                                   HYPRE_SMMCORE_1T(k + 2) +
                                   HYPRE_SMMCORE_1T(k + 3) +
                                   HYPRE_SMMCORE_1T(k + 4) +
                                   HYPRE_SMMCORE_1T(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0) +
                                   HYPRE_SMMCORE_1T(k + 1) +
                                   HYPRE_SMMCORE_1T(k + 2) +
                                   HYPRE_SMMCORE_1T(k + 3) +
                                   HYPRE_SMMCORE_1T(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0) +
                                   HYPRE_SMMCORE_1T(k + 1) +
                                   HYPRE_SMMCORE_1T(k + 2) +
                                   HYPRE_SMMCORE_1T(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0) +
                                   HYPRE_SMMCORE_1T(k + 1) +
                                   HYPRE_SMMCORE_1T(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0) +
                                   HYPRE_SMMCORE_1T(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_1T(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, gi);
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
 * Core function for computing the product of three coefficients.
 * Two coefficients are variable and live on data space "h".
 * The third coefficient is constant, it lives on data space "g", and it
 * requires the usage of a bitmask
 *
 * "2etb" means:
 *   "2": two data spaces.
 *   "e": data spaces for variable coefficients are the same.
 *   "t": triple-product.
 *   "b": single bitmask.
 *
 * This can be used for the scenarios:
 *   1) VCC * VCC * CCF.
 *   2) VCC * CCF * VCC.
 *   3) CCF * VCC * VCC.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2etb( hypre_StructMatmultDataMH *a,
                                      HYPRE_Int                  ncomponents,
                                      HYPRE_Int                **order,
                                      HYPRE_Complex             *cprod,
                                      const HYPRE_Complex     ***tptrs,
                                      HYPRE_Complex             *mptr,
                                      HYPRE_Int                  ndim,
                                      hypre_Index                loop_size,
                                      hypre_Box                 *gdbox,
                                      hypre_Index                gdstart,
                                      hypre_Index                gdstride,
                                      hypre_Box                 *hdbox,
                                      hypre_Index                hdstart,
                                      hypre_Index                hdstride,
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
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2ETB(k + 0) +
                                   HYPRE_SMMCORE_2ETB(k + 1) +
                                   HYPRE_SMMCORE_2ETB(k + 2) +
                                   HYPRE_SMMCORE_2ETB(k + 3) +
                                   HYPRE_SMMCORE_2ETB(k + 4) +
                                   HYPRE_SMMCORE_2ETB(k + 5) +
                                   HYPRE_SMMCORE_2ETB(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2ETB(k + 0) +
                                   HYPRE_SMMCORE_2ETB(k + 1) +
                                   HYPRE_SMMCORE_2ETB(k + 2) +
                                   HYPRE_SMMCORE_2ETB(k + 3) +
                                   HYPRE_SMMCORE_2ETB(k + 4) +
                                   HYPRE_SMMCORE_2ETB(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2ETB(k + 0) +
                                   HYPRE_SMMCORE_2ETB(k + 1) +
                                   HYPRE_SMMCORE_2ETB(k + 2) +
                                   HYPRE_SMMCORE_2ETB(k + 3) +
                                   HYPRE_SMMCORE_2ETB(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2ETB(k + 0) +
                                   HYPRE_SMMCORE_2ETB(k + 1) +
                                   HYPRE_SMMCORE_2ETB(k + 2) +
                                   HYPRE_SMMCORE_2ETB(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2ETB(k + 0) +
                                   HYPRE_SMMCORE_2ETB(k + 1) +
                                   HYPRE_SMMCORE_2ETB(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2ETB(k + 0) +
                                   HYPRE_SMMCORE_2ETB(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2ETB(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
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
 * Core function for computing the product of three coefficients.
 * One coefficient is variable and live on data space "g".
 * Two coefficients are constant, live on data space "h", and require
 * the usage of a bitmask.
 *
 * "2etb" means:
 *   "2" : two data spaces.
 *   "t" : triple-product.
 *   "bb": two bitmasks.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2tbb( hypre_StructMatmultDataMH *a,
                                      HYPRE_Int                  ncomponents,
                                      HYPRE_Int                **order,
                                      HYPRE_Complex             *cprod,
                                      const HYPRE_Complex     ***tptrs,
                                      HYPRE_Complex             *mptr,
                                      HYPRE_Int                  ndim,
                                      hypre_Index                loop_size,
                                      hypre_Box                 *gdbox,
                                      hypre_Index                gdstart,
                                      hypre_Index                gdstride,
                                      hypre_Box                 *hdbox,
                                      hypre_Index                hdstart,
                                      hypre_Index                hdstride,
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
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2TBB(k + 0) +
                                   HYPRE_SMMCORE_2TBB(k + 1) +
                                   HYPRE_SMMCORE_2TBB(k + 2) +
                                   HYPRE_SMMCORE_2TBB(k + 3) +
                                   HYPRE_SMMCORE_2TBB(k + 4) +
                                   HYPRE_SMMCORE_2TBB(k + 5) +
                                   HYPRE_SMMCORE_2TBB(k + 6);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2TBB(k + 0) +
                                   HYPRE_SMMCORE_2TBB(k + 1) +
                                   HYPRE_SMMCORE_2TBB(k + 2) +
                                   HYPRE_SMMCORE_2TBB(k + 3) +
                                   HYPRE_SMMCORE_2TBB(k + 4) +
                                   HYPRE_SMMCORE_2TBB(k + 5);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2TBB(k + 0) +
                                   HYPRE_SMMCORE_2TBB(k + 1) +
                                   HYPRE_SMMCORE_2TBB(k + 2) +
                                   HYPRE_SMMCORE_2TBB(k + 3) +
                                   HYPRE_SMMCORE_2TBB(k + 4);
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2TBB(k + 0) +
                                   HYPRE_SMMCORE_2TBB(k + 1) +
                                   HYPRE_SMMCORE_2TBB(k + 2) +
                                   HYPRE_SMMCORE_2TBB(k + 3);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2TBB(k + 0) +
                                   HYPRE_SMMCORE_2TBB(k + 1) +
                                   HYPRE_SMMCORE_2TBB(k + 2);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2TBB(k + 0) +
                                   HYPRE_SMMCORE_2TBB(k + 1);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_2TBB(k + 0);

               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, gi, hi);
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
 * Core function for computing the double-product of coefficients.
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

   HYPRE_Int                max_components = 10;
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
            case 0: /* ff term (call core_1d) */
               comptype = 0;
               break;
            case 1: /* cf term (call core_2db) */
               comptype = 4;
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
      hypre_StructMatmultCompute_core_1d(a, ncomp[0], cprod[0],
                                         tptrs[0], mptrs[0],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_2db(a, ncomp[4], order[4],
                                          cprod[4], tptrs[4],
                                          mptrs[4], ndim, loop_size,
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

   HYPRE_Int                max_components = 10;
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
            case 0: /* fff term (call core_1t) */
               comptype = 0;
               break;
            case 1: /* cff term (call core_2tbb) */
               comptype = 9;
               break;
            case 2: /* ccf term (call core_2etb) */
               comptype = 8;
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
      hypre_StructMatmultCompute_core_1t(a, ncomp[0], cprod[0],
                                         tptrs[0], mptrs[0],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_2etb(a, ncomp[8], order[8],
                                           cprod[8], tptrs[8], mptrs[8],
                                           ndim, loop_size,
                                           fdbox, fdstart, fdstride,
                                           cdbox, cdstart, cdstride,
                                           Mdbox, Mdstart, Mdstride);

      hypre_StructMatmultCompute_core_2tbb(a, ncomp[9], order[9],
                                           cprod[9], tptrs[9], mptrs[9],
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

