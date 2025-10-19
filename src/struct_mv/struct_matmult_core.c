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

#if defined(HYPRE_UNROLL_MAXDEPTH)
#undef HYPRE_UNROLL_MAXDEPTH
#endif
#define HYPRE_UNROLL_MAXDEPTH 7
//#define DEBUG_MATMULT 2

typedef HYPRE_Complex *hypre_3Cptrs[3];
typedef HYPRE_Complex *hypre_1Cptr;

/*--------------------------------------------------------------------------
 * Matrix Multiplication Kernel Optimization Macros
 *
 * These macros implement an optimized framework for structured matrix-matrix
 * multiplication kernels that handle different data access patterns.
 *
 * Overview:
 * ---------
 * F/C notation refers to fine/coarse data spaces. For example:
 *   - FFF: All three operands use fine grid data
 *   - CFF: First operand uses coarse data, others use fine data
 *   - CF: First operand uses coarse data, second uses fine data
 *   etc.
 *
 * Macro types and purpose:
 * ------------------------
 * 1. Variable declaration macros:
 *    HYPRE_DECLARE_*VARS - Define local variables for efficient data access
 *    - 2VARS: Declares a coefficient (cprod) and one data pointer (tptrs)
 *    - 3VARS: Declares a coefficient (cprod) and two data pointers (tptrs)
 *    - 4VARS: Declares a coefficient (cprod) and three data pointers (tptrs)
 *    These are combined with HYPRE_DECLARE_ALL_*VARS to declare variables for
 *    up to 7 terms (defined by HYPRE_UNROLL_MAXDEPTH)
 *
 * 2. Data loading macros:
 *    HYPRE_SMMLOAD_*VARS - Load coefficient and pointer values from arrays
 *    - Basic versions (e.g., HYPRE_SMMLOAD_2VARS) serve as building blocks
 *    - _UP_TO macros load multiple terms up to a given depth
 *    - _IMPL_* macros provide the implementation details for each depth
 *
 *    This design uses C99 variadic macros (..., __VA_ARGS__) to create a
 *    recursive implementation that minimizes code duplication. The variadic
 *    arguments allow macros to be chained together cleanly, with each depth
 *    building on the previous ones.
 *
 *    The macro concatenation (##) is used to dispatch to the appropriate
 *    implementation based on the requested depth. This technique allows us
 *    to write generic code that selects the correct number of unrolled loops
 *    at compile time.
 *
 * 3. Computation core macros:
 *    HYPRE_SMMCORE_* - Define the actual multiplication operations
 *    - Different variants for different data space combinations (FFF, CFF, etc.)
 *    - Each computes a product of coefficient and data values
 *
 *    HYPRE_SMMCORE_*_SUM_UP_TO_n - Create unrolled sums of multiple products at compile time
 *    - Direct entry points for generating compile-time sums of core computations
 *    - The n suffix determines how many terms to include (from 0 to n-1)
 *    - Used inside computation loops after loading variables with HYPRE_SMMLOAD_*_UP_TO
 *
 * Usage pattern in computation:
 * -----------------------------
 * 1. Declare variables using HYPRE_DECLARE_ALL_*VARS
 * 2. For each stencil entry:
 *    a. Load data using HYPRE_SMMLOAD_*VARS_UP_TO(k, n) where n is depth
 *    b. Compute products using HYPRE_SMMCORE_* macros via BoxLoops
 *    c. Accumulate results into the output pointer (mptr)
 *--------------------------------------------------------------------------*/

#define HYPRE_DECLARE_2VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, cprod_##k

#define HYPRE_DECLARE_3VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, *tptrs_##k##_1, cprod_##k

#define HYPRE_DECLARE_4VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, *tptrs_##k##_1, *tptrs_##k##_2, cprod_##k

#define HYPRE_DECLARE_ALL_2VARS \
   HYPRE_DECLARE_2VARS(0);      \
   HYPRE_DECLARE_2VARS(1);      \
   HYPRE_DECLARE_2VARS(2);      \
   HYPRE_DECLARE_2VARS(3);      \
   HYPRE_DECLARE_2VARS(4);      \
   HYPRE_DECLARE_2VARS(5);      \
   HYPRE_DECLARE_2VARS(6)

#define HYPRE_DECLARE_ALL_3VARS \
   HYPRE_DECLARE_3VARS(0);      \
   HYPRE_DECLARE_3VARS(1);      \
   HYPRE_DECLARE_3VARS(2);      \
   HYPRE_DECLARE_3VARS(3);      \
   HYPRE_DECLARE_3VARS(4);      \
   HYPRE_DECLARE_3VARS(5);      \
   HYPRE_DECLARE_3VARS(6)

#define HYPRE_DECLARE_ALL_4VARS \
   HYPRE_DECLARE_4VARS(0);      \
   HYPRE_DECLARE_4VARS(1);      \
   HYPRE_DECLARE_4VARS(2);      \
   HYPRE_DECLARE_4VARS(3);      \
   HYPRE_DECLARE_4VARS(4);      \
   HYPRE_DECLARE_4VARS(5);      \
   HYPRE_DECLARE_4VARS(6)

#define HYPRE_SMMLOAD_2VARS(k, m)     \
   cprod_##m     = cprod[(k + m)];    \
   tptrs_##m##_0 = tptrs[(k + m)][0]

#define HYPRE_SMMLOAD_3VARS(k, m)     \
   cprod_##m     = cprod[(k + m)];    \
   tptrs_##m##_0 = tptrs[(k + m)][0]; \
   tptrs_##m##_1 = tptrs[(k + m)][1]

#define HYPRE_SMMLOAD_4VARS(k, m)     \
   cprod_##m     = cprod[(k + m)];    \
   tptrs_##m##_0 = tptrs[(k + m)][0]; \
   tptrs_##m##_1 = tptrs[(k + m)][1]; \
   tptrs_##m##_2 = tptrs[(k + m)][2]

/* For 4 variable types (cprod and tptrs[k][0], tptrs[k][1], tptrs[k][2]) */
#define HYPRE_SMMLOAD_4VARS_UP_TO(k, n, ...) \
   HYPRE_SMMLOAD_4VARS_IMPL_##n(k, __VA_ARGS__)

#define HYPRE_SMMLOAD_4VARS_IMPL_1(k, ...) \
   HYPRE_SMMLOAD_4VARS(k, 0)

#define HYPRE_SMMLOAD_4VARS_IMPL_2(k, ...) \
   HYPRE_SMMLOAD_4VARS(k, 0); \
   HYPRE_SMMLOAD_4VARS(k, 1)

#define HYPRE_SMMLOAD_4VARS_IMPL_3(k, ...) \
   HYPRE_SMMLOAD_4VARS_IMPL_2(k); \
   HYPRE_SMMLOAD_4VARS(k, 2)

#define HYPRE_SMMLOAD_4VARS_IMPL_4(k, ...) \
   HYPRE_SMMLOAD_4VARS_IMPL_3(k); \
   HYPRE_SMMLOAD_4VARS(k, 3)

#define HYPRE_SMMLOAD_4VARS_IMPL_5(k, ...) \
   HYPRE_SMMLOAD_4VARS_IMPL_4(k); \
   HYPRE_SMMLOAD_4VARS(k, 4)

#define HYPRE_SMMLOAD_4VARS_IMPL_6(k, ...) \
   HYPRE_SMMLOAD_4VARS_IMPL_5(k); \
   HYPRE_SMMLOAD_4VARS(k, 5)

#define HYPRE_SMMLOAD_4VARS_IMPL_7(k, ...) \
   HYPRE_SMMLOAD_4VARS_IMPL_6(k); \
   HYPRE_SMMLOAD_4VARS(k, 6)

/* For 3 variable types (cprod and tptrs[k][0], tptrs[k][1]) */
#define HYPRE_SMMLOAD_3VARS_UP_TO(k, n, ...) \
   HYPRE_SMMLOAD_3VARS_IMPL_##n(k, __VA_ARGS__)

#define HYPRE_SMMLOAD_3VARS_IMPL_1(k, ...) \
   HYPRE_SMMLOAD_3VARS(k, 0)

#define HYPRE_SMMLOAD_3VARS_IMPL_2(k, ...) \
   HYPRE_SMMLOAD_3VARS(k, 0); \
   HYPRE_SMMLOAD_3VARS(k, 1)

#define HYPRE_SMMLOAD_3VARS_IMPL_3(k, ...) \
   HYPRE_SMMLOAD_3VARS_IMPL_2(k); \
   HYPRE_SMMLOAD_3VARS(k, 2)

#define HYPRE_SMMLOAD_3VARS_IMPL_4(k, ...) \
   HYPRE_SMMLOAD_3VARS_IMPL_3(k); \
   HYPRE_SMMLOAD_3VARS(k, 3)

#define HYPRE_SMMLOAD_3VARS_IMPL_5(k, ...) \
   HYPRE_SMMLOAD_3VARS_IMPL_4(k); \
   HYPRE_SMMLOAD_3VARS(k, 4)

#define HYPRE_SMMLOAD_3VARS_IMPL_6(k, ...) \
   HYPRE_SMMLOAD_3VARS_IMPL_5(k); \
   HYPRE_SMMLOAD_3VARS(k, 5)

#define HYPRE_SMMLOAD_3VARS_IMPL_7(k, ...) \
   HYPRE_SMMLOAD_3VARS_IMPL_6(k); \
   HYPRE_SMMLOAD_3VARS(k, 6)

/* For 2 variable types (cprod and tptrs[k][0]) */
#define HYPRE_SMMLOAD_2VARS_UP_TO(k, n, ...) \
   HYPRE_SMMLOAD_2VARS_IMPL_##n(k, __VA_ARGS__)

#define HYPRE_SMMLOAD_2VARS_IMPL_1(k, ...) \
   HYPRE_SMMLOAD_2VARS(k, 0)

#define HYPRE_SMMLOAD_2VARS_IMPL_2(k, ...) \
   HYPRE_SMMLOAD_2VARS(k, 0); \
   HYPRE_SMMLOAD_2VARS(k, 1)

#define HYPRE_SMMLOAD_2VARS_IMPL_3(k, ...) \
   HYPRE_SMMLOAD_2VARS_IMPL_2(k); \
   HYPRE_SMMLOAD_2VARS(k, 2)

#define HYPRE_SMMLOAD_2VARS_IMPL_4(k, ...) \
   HYPRE_SMMLOAD_2VARS_IMPL_3(k); \
   HYPRE_SMMLOAD_2VARS(k, 3)

#define HYPRE_SMMLOAD_2VARS_IMPL_5(k, ...) \
   HYPRE_SMMLOAD_2VARS_IMPL_4(k); \
   HYPRE_SMMLOAD_2VARS(k, 4)

#define HYPRE_SMMLOAD_2VARS_IMPL_6(k, ...) \
   HYPRE_SMMLOAD_2VARS_IMPL_5(k); \
   HYPRE_SMMLOAD_2VARS(k, 5)

#define HYPRE_SMMLOAD_2VARS_IMPL_7(k, ...) \
   HYPRE_SMMLOAD_2VARS_IMPL_6(k); \
   HYPRE_SMMLOAD_2VARS(k, 6)

#define HYPRE_SMMCORE_FFF(k) \
   cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi] * tptrs_##k##_2[fi]

#define HYPRE_SMMCORE_CFF(k) \
   cprod_##k * tptrs_##k##_0[ci] * tptrs_##k##_1[fi] * tptrs_##k##_2[fi]

#define HYPRE_SMMCORE_CCF(k) \
   cprod_##k * tptrs_##k##_0[ci] * tptrs_##k##_1[ci] * tptrs_##k##_2[fi]

/* Don't need CCC */

#define HYPRE_SMMCORE_FF(k) \
   cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi]

#define HYPRE_SMMCORE_CF(k) \
   cprod_##k * tptrs_##k##_0[ci] * tptrs_##k##_1[fi]

#define HYPRE_SMMCORE_CC(k) \
   cprod_##k * tptrs_##k##_0[ci] * tptrs_##k##_1[ci]

#define HYPRE_SMMCORE_F(k) \
   cprod_##k * tptrs_##k##_0[fi]

#define HYPRE_SMMCORE_C(k) \
   cprod_##k * tptrs_##k##_0[ci]

#define HYPRE_SMMCORE_FFF_SUM_UP_TO_1 \
   HYPRE_SMMCORE_FFF(0)

#define HYPRE_SMMCORE_FFF_SUM_UP_TO_2 \
   HYPRE_SMMCORE_FFF(0) + \
   HYPRE_SMMCORE_FFF(1)

#define HYPRE_SMMCORE_FFF_SUM_UP_TO_3 \
   HYPRE_SMMCORE_FFF_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_FFF(2)

#define HYPRE_SMMCORE_FFF_SUM_UP_TO_4 \
   HYPRE_SMMCORE_FFF_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_FFF(3)

#define HYPRE_SMMCORE_FFF_SUM_UP_TO_5 \
   HYPRE_SMMCORE_FFF_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_FFF(4)

#define HYPRE_SMMCORE_FFF_SUM_UP_TO_6 \
   HYPRE_SMMCORE_FFF_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_FFF(5)

#define HYPRE_SMMCORE_FFF_SUM_UP_TO_7 \
   HYPRE_SMMCORE_FFF_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_FFF(6)

#define HYPRE_SMMCORE_CFF_SUM_UP_TO_1 \
   HYPRE_SMMCORE_CFF(0)

#define HYPRE_SMMCORE_CFF_SUM_UP_TO_2 \
   HYPRE_SMMCORE_CFF(0) + \
   HYPRE_SMMCORE_CFF(1)

#define HYPRE_SMMCORE_CFF_SUM_UP_TO_3 \
   HYPRE_SMMCORE_CFF_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_CFF(2)

#define HYPRE_SMMCORE_CFF_SUM_UP_TO_4 \
   HYPRE_SMMCORE_CFF_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_CFF(3)

#define HYPRE_SMMCORE_CFF_SUM_UP_TO_5 \
   HYPRE_SMMCORE_CFF_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_CFF(4)

#define HYPRE_SMMCORE_CFF_SUM_UP_TO_6 \
   HYPRE_SMMCORE_CFF_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_CFF(5)

#define HYPRE_SMMCORE_CFF_SUM_UP_TO_7 \
   HYPRE_SMMCORE_CFF_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_CFF(6)

#define HYPRE_SMMCORE_CCF_SUM_UP_TO_1 \
   HYPRE_SMMCORE_CCF(0)

#define HYPRE_SMMCORE_CCF_SUM_UP_TO_2 \
   HYPRE_SMMCORE_CCF(0) + \
   HYPRE_SMMCORE_CCF(1)

#define HYPRE_SMMCORE_CCF_SUM_UP_TO_3 \
   HYPRE_SMMCORE_CCF_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_CCF(2)

#define HYPRE_SMMCORE_CCF_SUM_UP_TO_4 \
   HYPRE_SMMCORE_CCF_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_CCF(3)

#define HYPRE_SMMCORE_CCF_SUM_UP_TO_5 \
   HYPRE_SMMCORE_CCF_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_CCF(4)

#define HYPRE_SMMCORE_CCF_SUM_UP_TO_6 \
   HYPRE_SMMCORE_CCF_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_CCF(5)

#define HYPRE_SMMCORE_CCF_SUM_UP_TO_7 \
   HYPRE_SMMCORE_CCF_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_CCF(6)

#define HYPRE_SMMCORE_FF_SUM_UP_TO_1 \
   HYPRE_SMMCORE_FF(0)

#define HYPRE_SMMCORE_FF_SUM_UP_TO_2 \
   HYPRE_SMMCORE_FF(0) + \
   HYPRE_SMMCORE_FF(1)

#define HYPRE_SMMCORE_FF_SUM_UP_TO_3 \
   HYPRE_SMMCORE_FF_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_FF(2)

#define HYPRE_SMMCORE_FF_SUM_UP_TO_4 \
   HYPRE_SMMCORE_FF_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_FF(3)

#define HYPRE_SMMCORE_FF_SUM_UP_TO_5 \
   HYPRE_SMMCORE_FF_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_FF(4)

#define HYPRE_SMMCORE_FF_SUM_UP_TO_6 \
   HYPRE_SMMCORE_FF_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_FF(5)

#define HYPRE_SMMCORE_FF_SUM_UP_TO_7 \
   HYPRE_SMMCORE_FF_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_FF(6)

#define HYPRE_SMMCORE_CF_SUM_UP_TO_1 \
   HYPRE_SMMCORE_CF(0)

#define HYPRE_SMMCORE_CF_SUM_UP_TO_2 \
   HYPRE_SMMCORE_CF(0) + \
   HYPRE_SMMCORE_CF(1)

#define HYPRE_SMMCORE_CF_SUM_UP_TO_3 \
   HYPRE_SMMCORE_CF_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_CF(2)

#define HYPRE_SMMCORE_CF_SUM_UP_TO_4 \
   HYPRE_SMMCORE_CF_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_CF(3)

#define HYPRE_SMMCORE_CF_SUM_UP_TO_5 \
   HYPRE_SMMCORE_CF_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_CF(4)

#define HYPRE_SMMCORE_CF_SUM_UP_TO_6 \
   HYPRE_SMMCORE_CF_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_CF(5)

#define HYPRE_SMMCORE_CF_SUM_UP_TO_7 \
   HYPRE_SMMCORE_CF_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_CF(6)

#define HYPRE_SMMCORE_CC_SUM_UP_TO_1 \
   HYPRE_SMMCORE_CC(0)

#define HYPRE_SMMCORE_CC_SUM_UP_TO_2 \
   HYPRE_SMMCORE_CC(0) + \
   HYPRE_SMMCORE_CC(1)

#define HYPRE_SMMCORE_CC_SUM_UP_TO_3 \
   HYPRE_SMMCORE_CC_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_CC(2)

#define HYPRE_SMMCORE_CC_SUM_UP_TO_4 \
   HYPRE_SMMCORE_CC_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_CC(3)

#define HYPRE_SMMCORE_CC_SUM_UP_TO_5 \
   HYPRE_SMMCORE_CC_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_CC(4)

#define HYPRE_SMMCORE_CC_SUM_UP_TO_6 \
   HYPRE_SMMCORE_CC_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_CC(5)

#define HYPRE_SMMCORE_CC_SUM_UP_TO_7 \
   HYPRE_SMMCORE_CC_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_CC(6)

#define HYPRE_SMMCORE_F_SUM_UP_TO_1 \
   HYPRE_SMMCORE_F(0)

#define HYPRE_SMMCORE_F_SUM_UP_TO_2 \
   HYPRE_SMMCORE_F(0) + \
   HYPRE_SMMCORE_F(1)

#define HYPRE_SMMCORE_F_SUM_UP_TO_3 \
   HYPRE_SMMCORE_F_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_F(2)

#define HYPRE_SMMCORE_F_SUM_UP_TO_4 \
   HYPRE_SMMCORE_F_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_F(3)

#define HYPRE_SMMCORE_F_SUM_UP_TO_5 \
   HYPRE_SMMCORE_F_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_F(4)

#define HYPRE_SMMCORE_F_SUM_UP_TO_6 \
   HYPRE_SMMCORE_F_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_F(5)

#define HYPRE_SMMCORE_F_SUM_UP_TO_7 \
   HYPRE_SMMCORE_F_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_F(6)

#define HYPRE_SMMCORE_C_SUM_UP_TO_1 \
   HYPRE_SMMCORE_C(0)

#define HYPRE_SMMCORE_C_SUM_UP_TO_2 \
   HYPRE_SMMCORE_C(0) + \
   HYPRE_SMMCORE_C(1)

#define HYPRE_SMMCORE_C_SUM_UP_TO_3 \
   HYPRE_SMMCORE_C_SUM_UP_TO_2 + \
   HYPRE_SMMCORE_C(2)

#define HYPRE_SMMCORE_C_SUM_UP_TO_4 \
   HYPRE_SMMCORE_C_SUM_UP_TO_3 + \
   HYPRE_SMMCORE_C(3)

#define HYPRE_SMMCORE_C_SUM_UP_TO_5 \
   HYPRE_SMMCORE_C_SUM_UP_TO_4 + \
   HYPRE_SMMCORE_C(4)

#define HYPRE_SMMCORE_C_SUM_UP_TO_6 \
   HYPRE_SMMCORE_C_SUM_UP_TO_5 + \
   HYPRE_SMMCORE_C(5)

#define HYPRE_SMMCORE_C_SUM_UP_TO_7 \
   HYPRE_SMMCORE_C_SUM_UP_TO_6 + \
   HYPRE_SMMCORE_C(6)

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_fff( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_4VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("fff");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FFF_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_cff( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_4VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("cff");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 7);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 6);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 5);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 4);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 3);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 2);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 1);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CFF_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_ccf( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_4VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ccf");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 7);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 6);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 5);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 4);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 3);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 2);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            HYPRE_SMMLOAD_4VARS_UP_TO(k, 1);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CCF_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_ff( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_3VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ff");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_FF_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_cf( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_3VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("cf");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 7);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 6);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 5);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 4);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 3);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 2);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 1);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CF_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_cc( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_3VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("cc");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 6:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 5:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 4:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 3:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 2:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 1:
            HYPRE_SMMLOAD_3VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_CC_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_f( HYPRE_Int            nprod,
                                   HYPRE_Complex       *cprod,
                                   hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_2VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("f");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_F_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_c( HYPRE_Int            nprod,
                                   HYPRE_Complex       *cprod,
                                   hypre_3Cptrs        *tptrs,
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
   HYPRE_Int       k, depth;

   HYPRE_DECLARE_ALL_2VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("c");

   for (k = 0; k < nprod; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 7:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C_SUM_UP_TO_7;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 6:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C_SUM_UP_TO_6;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 5:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C_SUM_UP_TO_5;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 4:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C_SUM_UP_TO_4;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 3:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C_SUM_UP_TO_3;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 2:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C_SUM_UP_TO_2;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 1:
            HYPRE_SMMLOAD_2VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_Complex val = HYPRE_SMMCORE_C_SUM_UP_TO_1;
               mptr[Mi] += val;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         default:
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported depth of loop unrolling!");
      }
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Core function for computing the nterms-product of coefficients.
 * Here, nterms can only be 2 or 3.
 *
 * 8 refers to the max. number of core kernels that can be unrolled:
 * FFF, CFF, CCF, FF, CF, CC, F, C.
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
   HYPRE_Complex   cprod[8][HYPRE_MAX_MMTERMS];
   hypre_3Cptrs    tptrs[8][HYPRE_MAX_MMTERMS];
   hypre_1Cptr     mptrs[8][HYPRE_MAX_MMTERMS];

   HYPRE_Int       mentry, ptype = 0, nf, nc, nt;
   HYPRE_Int       e, p, i, k, t;

#if defined(DEBUG_MATMULT)
   HYPRE_Int       c, nboxloops = 0;
   HYPRE_Int       nprodsum[8] = {0};
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("core");

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

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "\n=== CORE STENCIL ENTRY %d POINTERS ===\n", e);

      /* Show all products for this stencil entry across all types */
      const char* type_names[8] = {"FFF", "CFF", "CCF", "FF", "CF", "CC", "F", "C"};
      for (p = 0; p < 8; p++)
      {
         if (nprod[p] > 0)
         {
            hypre_ParPrintf(hypre_MPI_COMM_WORLD, "%s Products (count: %d):\n", type_names[p], nprod[p]);
            for (k = 0; k < nprod[p]; k++)
            {
               hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  Product %d:\n", k);
               hypre_ParPrintf(hypre_MPI_COMM_WORLD, "    cprod = %p (value: %e)\n",
                               (void*)&cprod[p][k], cprod[p][k]);
               hypre_ParPrintf(hypre_MPI_COMM_WORLD, "    mptr = %p\n",
                               (void*)mptrs[p][k]);

               /* Show tptrs based on product type */
               HYPRE_Int num_tptrs;
               if (p <= 2)
               {
                  num_tptrs = 3; /* FFF, CFF, CCF have 3 tptrs */
               }
               else if (p <= 5)
               {
                  num_tptrs = 2; /* FF, CF, CC have 2 tptrs */
               }
               else
               {
                  num_tptrs = 1; /* F, C have 1 tptr */
               }

               for (t = 0; t < num_tptrs; t++)
               {
                  hypre_ParPrintf(hypre_MPI_COMM_WORLD, "    tptr[%d] = %p\n",
                                  t, (void*)tptrs[p][k][t]);
               }
            }
         }
      }

      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "===============================\n");
#endif

#if defined(DEBUG_MATMULT)
      for (c = 0; c < 8; c++)
      {
         nboxloops += hypre_ceildiv(nprod[c], HYPRE_UNROLL_MAXDEPTH);
         nprodsum[c] += hypre_ceildiv(nprod[c], HYPRE_UNROLL_MAXDEPTH);
      }

      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "stEntry: %02d | ", e);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD,
                      "FFF: %d | CFF: %d | CCF: %d | ",
                      nprod[0], nprod[1], nprod[2]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD,
                      "FF: %d | CF: %d | CC: %d | F: %d | C: %d | ",
                      nprod[3], nprod[4], nprod[5],
                      nprod[6], nprod[7]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Tot: %d\n",
                      nprod[0] + nprod[1] + nprod[2] + \
                      nprod[3] + nprod[4] + nprod[5] + \
                      nprod[6] + nprod[7]);
#endif

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

#if defined(DEBUG_MATMULT)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of box loops: ");
   if (nprodsum[0] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "FFF: %d | ", nprodsum[0]); }
   if (nprodsum[1] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "CFF: %d | ", nprodsum[1]); }
   if (nprodsum[2] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "CCF: %d | ", nprodsum[2]); }
   if (nprodsum[3] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "FF: %d | ", nprodsum[3]); }
   if (nprodsum[4] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "CF: %d | ", nprodsum[4]); }
   if (nprodsum[5] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "CC: %d | ", nprodsum[5]); }
   if (nprodsum[6] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "F: %d | ", nprodsum[6]); }
   if (nprodsum[7] > 0) { hypre_ParPrintf(hypre_MPI_COMM_WORLD, "C: %d | ", nprodsum[7]); }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Total: %d\n\n", nboxloops);
#endif

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
