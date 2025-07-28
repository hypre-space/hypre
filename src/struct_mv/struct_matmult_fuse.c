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

#define HYPRE_FUSE_MAXDEPTH 19
//#define DEBUG_MATMULT 1
//#define USE_FUSE_SORT 1

typedef HYPRE_Complex *hypre_3Cptrs[3];
typedef HYPRE_Complex *hypre_1Cptr;

/*--------------------------------------------------------------------------
 * Matrix Multiplication Kernel Fusion Macros
 *
 * This file implements a macro-based framework for fusing multiple structured
 * matrix-matrix multiplication operations. The goal is to maximize throughput
 * by unrolling and fusing multiple stencil product terms into a single BoxLoop.
 *
 * Macro types and purpose:
 * ------------------------
 * 1. Variable declaration macros:
 *    HYPRE_SMMFUSE_DECLARE_*VARS - Declare local variables for each fused product.
 *    - 3VARS, 4VARS, 5VARS: For different product types (e.g., F, FF, FFF, etc.)
 *    - HYPRE_SMMFUSE_DECLARE_19_*VARS: Declare up to 19 sets of variables for unrolling.
 *
 * 2. Data loading macros:
 *    HYPRE_SMMFUSE_LOAD_*VARS - Load coefficients and data pointers for each product.
 *    - _UP_TO_N macros: Recursively load variables for up to N fused products.
 *
 * 3. Fused multiply-add (FMA) macros:
 *    HYPRE_SMMFUSE_* - Perform the actual fused multiply-add for each product type.
 *    - _UP_TO_N macros: Recursively expand to perform N fused operations in sequence.
 *
 * Usage pattern in computation:
 * -----------------------------
 * 1. Declare variables for all fused products using the appropriate DECLARE macro.
 * 2. For each block of products (up to HYPRE_FUSE_MAXDEPTH):
 *    a. Load variables using the LOAD_*VARS_UP_TO macro.
 *    b. Use a BoxLoop to iterate over the grid and apply the FMA sequence macro.
 *--------------------------------------------------------------------------*/

/* Variable declaration macros */
#define HYPRE_SMMFUSE_DECLARE_3VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, cprod_##k, *mptr_##k

#define HYPRE_SMMFUSE_DECLARE_4VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, *tptrs_##k##_1, cprod_##k, *mptr_##k

#define HYPRE_SMMFUSE_DECLARE_5VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, *tptrs_##k##_1, *tptrs_##k##_2, cprod_##k, *mptr_##k

#define HYPRE_SMMFUSE_DECLARE_19_3VARS  \
   HYPRE_SMMFUSE_DECLARE_3VARS(0);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(1);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(2);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(3);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(4);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(5);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(6);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(7);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(8);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(9);      \
   HYPRE_SMMFUSE_DECLARE_3VARS(10);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(11);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(12);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(13);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(14);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(15);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(16);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(17);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(18)

#define HYPRE_SMMFUSE_DECLARE_19_4VARS  \
   HYPRE_SMMFUSE_DECLARE_4VARS(0);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(1);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(2);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(3);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(4);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(5);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(6);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(7);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(8);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(9);      \
   HYPRE_SMMFUSE_DECLARE_4VARS(10);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(11);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(12);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(13);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(14);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(15);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(16);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(17);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(18)

#define HYPRE_SMMFUSE_DECLARE_19_5VARS  \
   HYPRE_SMMFUSE_DECLARE_5VARS(0);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(1);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(2);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(3);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(4);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(5);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(6);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(7);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(8);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(9);      \
   HYPRE_SMMFUSE_DECLARE_5VARS(10);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(11);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(12);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(13);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(14);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(15);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(16);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(17);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(18)

/* Variable loading macros for different patterns */
#define HYPRE_SMMFUSE_LOAD_5VARS(k, m) \
   cprod_##m     = cprod[(k + m)];     \
   mptr_##m      = mptrs[(k + m)];     \
   tptrs_##m##_0 = tptrs[(k + m)][0];  \
   tptrs_##m##_1 = tptrs[(k + m)][1];  \
   tptrs_##m##_2 = tptrs[(k + m)][2]

#define HYPRE_SMMFUSE_LOAD_4VARS(k, m) \
   cprod_##m     = cprod[(k + m)];     \
   mptr_##m      = mptrs[(k + m)];     \
   tptrs_##m##_0 = tptrs[(k + m)][0];  \
   tptrs_##m##_1 = tptrs[(k + m)][1]

#define HYPRE_SMMFUSE_LOAD_3VARS(k, m) \
   cprod_##m     = cprod[(k + m)];     \
   mptr_##m      = mptrs[(k + m)];     \
   tptrs_##m##_0 = tptrs[(k + m)][0]

/* Load variables up to a certain depth (recursive implementation) */
#define HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, n, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_##n(k, __VA_ARGS__)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_1(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 0)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_2(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 0); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 1)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_3(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_2(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 2)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_4(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_3(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 3)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_5(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_4(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 4)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_6(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_5(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 5)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_7(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_6(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 6)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_8(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_7(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 7)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_9(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_8(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 8)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_10(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_9(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 9)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_11(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_10(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 10)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_12(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_11(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 11)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_13(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_12(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 12)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_14(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_13(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 13)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_15(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_14(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 14)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_16(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_15(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 15)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_17(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_16(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 16)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_18(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_17(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 17)

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_19(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_18(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 18)

#define HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, n, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_##n(k, __VA_ARGS__)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_1(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 0)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_2(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 0); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 1)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_3(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_2(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 2)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_4(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_3(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 3)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_5(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_4(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 4)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_6(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_5(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 5)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_7(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_6(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 6)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_8(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_7(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 7)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_9(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_8(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 8)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_10(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_9(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 9)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_11(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_10(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 10)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_12(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_11(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 11)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_13(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_12(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 12)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_14(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_13(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 13)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_15(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_14(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 14)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_16(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_15(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 15)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_17(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_16(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 16)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_18(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_17(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 17)

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_19(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_18(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 18)

#define HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, n, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_##n(k, __VA_ARGS__)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_1(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 0)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_2(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 0); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 1)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_3(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_2(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 2)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_4(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_3(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 3)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_5(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_4(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 4)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_6(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_5(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 5)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_7(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_6(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 6)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_8(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_7(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 7)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_9(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_8(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 8)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_10(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_9(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 9)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_11(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_10(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 10)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_12(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_11(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 11)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_13(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_12(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 12)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_14(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_13(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 13)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_15(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_14(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 14)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_16(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_15(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 15)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_17(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_16(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 16)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_18(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_17(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 17)

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_19(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_18(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 18)

/* Individual fused multiply-add operations */
#define HYPRE_SMMFUSE_FFF(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi] * tptrs_##k##_2[fi]

#define HYPRE_SMMFUSE_FFC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi] * tptrs_##k##_2[ci]

#define HYPRE_SMMFUSE_FCC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[ci] * tptrs_##k##_2[ci]

#define HYPRE_SMMFUSE_FC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[ci]

#define HYPRE_SMMFUSE_CC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[ci] * tptrs_##k##_1[ci]

#define HYPRE_SMMFUSE_FF(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi]

#define HYPRE_SMMFUSE_F(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi]

#define HYPRE_SMMFUSE_C(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[ci]

/* Sequence macros for various fused multiply-add (FMA) operations */
#define HYPRE_SMMFUSE_FFF_UP_TO_1 \
   HYPRE_SMMFUSE_FFF(0)

#define HYPRE_SMMFUSE_FFF_UP_TO_2 \
   HYPRE_SMMFUSE_FFF(0); \
   HYPRE_SMMFUSE_FFF(1)

#define HYPRE_SMMFUSE_FFF_UP_TO_3 \
   HYPRE_SMMFUSE_FFF_UP_TO_2; \
   HYPRE_SMMFUSE_FFF(2)

#define HYPRE_SMMFUSE_FFF_UP_TO_4 \
   HYPRE_SMMFUSE_FFF_UP_TO_3; \
   HYPRE_SMMFUSE_FFF(3)

#define HYPRE_SMMFUSE_FFF_UP_TO_5 \
   HYPRE_SMMFUSE_FFF_UP_TO_4; \
   HYPRE_SMMFUSE_FFF(4)

#define HYPRE_SMMFUSE_FFF_UP_TO_6 \
   HYPRE_SMMFUSE_FFF_UP_TO_5; \
   HYPRE_SMMFUSE_FFF(5)

#define HYPRE_SMMFUSE_FFF_UP_TO_7 \
   HYPRE_SMMFUSE_FFF_UP_TO_6; \
   HYPRE_SMMFUSE_FFF(6)

#define HYPRE_SMMFUSE_FFF_UP_TO_8 \
   HYPRE_SMMFUSE_FFF_UP_TO_7; \
   HYPRE_SMMFUSE_FFF(7)

#define HYPRE_SMMFUSE_FFF_UP_TO_9 \
   HYPRE_SMMFUSE_FFF_UP_TO_8; \
   HYPRE_SMMFUSE_FFF(8)

#define HYPRE_SMMFUSE_FFF_UP_TO_10 \
   HYPRE_SMMFUSE_FFF_UP_TO_9; \
   HYPRE_SMMFUSE_FFF(9)

#define HYPRE_SMMFUSE_FFF_UP_TO_11 \
   HYPRE_SMMFUSE_FFF_UP_TO_10; \
   HYPRE_SMMFUSE_FFF(10)

#define HYPRE_SMMFUSE_FFF_UP_TO_12 \
   HYPRE_SMMFUSE_FFF_UP_TO_11; \
   HYPRE_SMMFUSE_FFF(11)

#define HYPRE_SMMFUSE_FFF_UP_TO_13 \
   HYPRE_SMMFUSE_FFF_UP_TO_12; \
   HYPRE_SMMFUSE_FFF(12)

#define HYPRE_SMMFUSE_FFF_UP_TO_14 \
   HYPRE_SMMFUSE_FFF_UP_TO_13; \
   HYPRE_SMMFUSE_FFF(13)

#define HYPRE_SMMFUSE_FFF_UP_TO_15 \
   HYPRE_SMMFUSE_FFF_UP_TO_14; \
   HYPRE_SMMFUSE_FFF(14)

#define HYPRE_SMMFUSE_FFF_UP_TO_16 \
   HYPRE_SMMFUSE_FFF_UP_TO_15; \
   HYPRE_SMMFUSE_FFF(15)

#define HYPRE_SMMFUSE_FFF_UP_TO_17 \
   HYPRE_SMMFUSE_FFF_UP_TO_16; \
   HYPRE_SMMFUSE_FFF(16)

#define HYPRE_SMMFUSE_FFF_UP_TO_18 \
   HYPRE_SMMFUSE_FFF_UP_TO_17; \
   HYPRE_SMMFUSE_FFF(17)

#define HYPRE_SMMFUSE_FFF_UP_TO_19 \
   HYPRE_SMMFUSE_FFF_UP_TO_18; \
   HYPRE_SMMFUSE_FFF(18)

#define HYPRE_SMMFUSE_FFC_UP_TO_1 \
   HYPRE_SMMFUSE_FFC(0)

#define HYPRE_SMMFUSE_FFC_UP_TO_2 \
   HYPRE_SMMFUSE_FFC(0); \
   HYPRE_SMMFUSE_FFC(1)

#define HYPRE_SMMFUSE_FFC_UP_TO_3 \
   HYPRE_SMMFUSE_FFC_UP_TO_2; \
   HYPRE_SMMFUSE_FFC(2)

#define HYPRE_SMMFUSE_FFC_UP_TO_4 \
   HYPRE_SMMFUSE_FFC_UP_TO_3; \
   HYPRE_SMMFUSE_FFC(3)

#define HYPRE_SMMFUSE_FFC_UP_TO_5 \
   HYPRE_SMMFUSE_FFC_UP_TO_4; \
   HYPRE_SMMFUSE_FFC(4)

#define HYPRE_SMMFUSE_FFC_UP_TO_6 \
   HYPRE_SMMFUSE_FFC_UP_TO_5; \
   HYPRE_SMMFUSE_FFC(5)

#define HYPRE_SMMFUSE_FFC_UP_TO_7 \
   HYPRE_SMMFUSE_FFC_UP_TO_6; \
   HYPRE_SMMFUSE_FFC(6)

#define HYPRE_SMMFUSE_FFC_UP_TO_8 \
   HYPRE_SMMFUSE_FFC_UP_TO_7; \
   HYPRE_SMMFUSE_FFC(7)

#define HYPRE_SMMFUSE_FFC_UP_TO_9 \
   HYPRE_SMMFUSE_FFC_UP_TO_8; \
   HYPRE_SMMFUSE_FFC(8)

#define HYPRE_SMMFUSE_FFC_UP_TO_10 \
   HYPRE_SMMFUSE_FFC_UP_TO_9; \
   HYPRE_SMMFUSE_FFC(9)

#define HYPRE_SMMFUSE_FFC_UP_TO_11 \
   HYPRE_SMMFUSE_FFC_UP_TO_10; \
   HYPRE_SMMFUSE_FFC(10)

#define HYPRE_SMMFUSE_FFC_UP_TO_12 \
   HYPRE_SMMFUSE_FFC_UP_TO_11; \
   HYPRE_SMMFUSE_FFC(11)

#define HYPRE_SMMFUSE_FFC_UP_TO_13 \
   HYPRE_SMMFUSE_FFC_UP_TO_12; \
   HYPRE_SMMFUSE_FFC(12)

#define HYPRE_SMMFUSE_FFC_UP_TO_14 \
   HYPRE_SMMFUSE_FFC_UP_TO_13; \
   HYPRE_SMMFUSE_FFC(13)

#define HYPRE_SMMFUSE_FFC_UP_TO_15 \
   HYPRE_SMMFUSE_FFC_UP_TO_14; \
   HYPRE_SMMFUSE_FFC(14)

#define HYPRE_SMMFUSE_FFC_UP_TO_16 \
   HYPRE_SMMFUSE_FFC_UP_TO_15; \
   HYPRE_SMMFUSE_FFC(15)

#define HYPRE_SMMFUSE_FFC_UP_TO_17 \
   HYPRE_SMMFUSE_FFC_UP_TO_16; \
   HYPRE_SMMFUSE_FFC(16)

#define HYPRE_SMMFUSE_FFC_UP_TO_18 \
   HYPRE_SMMFUSE_FFC_UP_TO_17; \
   HYPRE_SMMFUSE_FFC(17)

#define HYPRE_SMMFUSE_FFC_UP_TO_19 \
   HYPRE_SMMFUSE_FFC_UP_TO_18; \
   HYPRE_SMMFUSE_FFC(18)

#define HYPRE_SMMFUSE_FCC_UP_TO_1 \
   HYPRE_SMMFUSE_FCC(0)

#define HYPRE_SMMFUSE_FCC_UP_TO_2 \
   HYPRE_SMMFUSE_FCC(0); \
   HYPRE_SMMFUSE_FCC(1)

#define HYPRE_SMMFUSE_FCC_UP_TO_3 \
   HYPRE_SMMFUSE_FCC_UP_TO_2; \
   HYPRE_SMMFUSE_FCC(2)

#define HYPRE_SMMFUSE_FCC_UP_TO_4 \
   HYPRE_SMMFUSE_FCC_UP_TO_3; \
   HYPRE_SMMFUSE_FCC(3)

#define HYPRE_SMMFUSE_FCC_UP_TO_5 \
   HYPRE_SMMFUSE_FCC_UP_TO_4; \
   HYPRE_SMMFUSE_FCC(4)

#define HYPRE_SMMFUSE_FCC_UP_TO_6 \
   HYPRE_SMMFUSE_FCC_UP_TO_5; \
   HYPRE_SMMFUSE_FCC(5)

#define HYPRE_SMMFUSE_FCC_UP_TO_7 \
   HYPRE_SMMFUSE_FCC_UP_TO_6; \
   HYPRE_SMMFUSE_FCC(6)

#define HYPRE_SMMFUSE_FCC_UP_TO_8 \
   HYPRE_SMMFUSE_FCC_UP_TO_7; \
   HYPRE_SMMFUSE_FCC(7)

#define HYPRE_SMMFUSE_FCC_UP_TO_9 \
   HYPRE_SMMFUSE_FCC_UP_TO_8; \
   HYPRE_SMMFUSE_FCC(8)

#define HYPRE_SMMFUSE_FCC_UP_TO_10 \
   HYPRE_SMMFUSE_FCC_UP_TO_9; \
   HYPRE_SMMFUSE_FCC(9)

#define HYPRE_SMMFUSE_FCC_UP_TO_11 \
   HYPRE_SMMFUSE_FCC_UP_TO_10; \
   HYPRE_SMMFUSE_FCC(10)

#define HYPRE_SMMFUSE_FCC_UP_TO_12 \
   HYPRE_SMMFUSE_FCC_UP_TO_11; \
   HYPRE_SMMFUSE_FCC(11)

#define HYPRE_SMMFUSE_FCC_UP_TO_13 \
   HYPRE_SMMFUSE_FCC_UP_TO_12; \
   HYPRE_SMMFUSE_FCC(12)

#define HYPRE_SMMFUSE_FCC_UP_TO_14 \
   HYPRE_SMMFUSE_FCC_UP_TO_13; \
   HYPRE_SMMFUSE_FCC(13)

#define HYPRE_SMMFUSE_FCC_UP_TO_15 \
   HYPRE_SMMFUSE_FCC_UP_TO_14; \
   HYPRE_SMMFUSE_FCC(14)

#define HYPRE_SMMFUSE_FCC_UP_TO_16 \
   HYPRE_SMMFUSE_FCC_UP_TO_15; \
   HYPRE_SMMFUSE_FCC(15)

#define HYPRE_SMMFUSE_FCC_UP_TO_17 \
   HYPRE_SMMFUSE_FCC_UP_TO_16; \
   HYPRE_SMMFUSE_FCC(16)

#define HYPRE_SMMFUSE_FCC_UP_TO_18 \
   HYPRE_SMMFUSE_FCC_UP_TO_17; \
   HYPRE_SMMFUSE_FCC(17)

#define HYPRE_SMMFUSE_FCC_UP_TO_19 \
   HYPRE_SMMFUSE_FCC_UP_TO_18; \
   HYPRE_SMMFUSE_FCC(18)

#define HYPRE_SMMFUSE_FF_UP_TO_1 \
   HYPRE_SMMFUSE_FF(0)

#define HYPRE_SMMFUSE_FF_UP_TO_2 \
   HYPRE_SMMFUSE_FF(0); \
   HYPRE_SMMFUSE_FF(1)

#define HYPRE_SMMFUSE_FF_UP_TO_3 \
   HYPRE_SMMFUSE_FF_UP_TO_2; \
   HYPRE_SMMFUSE_FF(2)

#define HYPRE_SMMFUSE_FF_UP_TO_4 \
   HYPRE_SMMFUSE_FF_UP_TO_3; \
   HYPRE_SMMFUSE_FF(3)

#define HYPRE_SMMFUSE_FF_UP_TO_5 \
   HYPRE_SMMFUSE_FF_UP_TO_4; \
   HYPRE_SMMFUSE_FF(4)

#define HYPRE_SMMFUSE_FF_UP_TO_6 \
   HYPRE_SMMFUSE_FF_UP_TO_5; \
   HYPRE_SMMFUSE_FF(5)

#define HYPRE_SMMFUSE_FF_UP_TO_7 \
   HYPRE_SMMFUSE_FF_UP_TO_6; \
   HYPRE_SMMFUSE_FF(6)

#define HYPRE_SMMFUSE_FF_UP_TO_8 \
   HYPRE_SMMFUSE_FF_UP_TO_7; \
   HYPRE_SMMFUSE_FF(7)

#define HYPRE_SMMFUSE_FF_UP_TO_9 \
   HYPRE_SMMFUSE_FF_UP_TO_8; \
   HYPRE_SMMFUSE_FF(8)

#define HYPRE_SMMFUSE_FF_UP_TO_10 \
   HYPRE_SMMFUSE_FF_UP_TO_9; \
   HYPRE_SMMFUSE_FF(9)

#define HYPRE_SMMFUSE_FF_UP_TO_11 \
   HYPRE_SMMFUSE_FF_UP_TO_10; \
   HYPRE_SMMFUSE_FF(10)

#define HYPRE_SMMFUSE_FF_UP_TO_12 \
   HYPRE_SMMFUSE_FF_UP_TO_11; \
   HYPRE_SMMFUSE_FF(11)

#define HYPRE_SMMFUSE_FF_UP_TO_13 \
   HYPRE_SMMFUSE_FF_UP_TO_12; \
   HYPRE_SMMFUSE_FF(12)

#define HYPRE_SMMFUSE_FF_UP_TO_14 \
   HYPRE_SMMFUSE_FF_UP_TO_13; \
   HYPRE_SMMFUSE_FF(13)

#define HYPRE_SMMFUSE_FF_UP_TO_15 \
   HYPRE_SMMFUSE_FF_UP_TO_14; \
   HYPRE_SMMFUSE_FF(14)

#define HYPRE_SMMFUSE_FF_UP_TO_16 \
   HYPRE_SMMFUSE_FF_UP_TO_15; \
   HYPRE_SMMFUSE_FF(15)

#define HYPRE_SMMFUSE_FF_UP_TO_17 \
   HYPRE_SMMFUSE_FF_UP_TO_16; \
   HYPRE_SMMFUSE_FF(16)

#define HYPRE_SMMFUSE_FF_UP_TO_18 \
   HYPRE_SMMFUSE_FF_UP_TO_17; \
   HYPRE_SMMFUSE_FF(17)

#define HYPRE_SMMFUSE_FF_UP_TO_19 \
   HYPRE_SMMFUSE_FF_UP_TO_18; \
   HYPRE_SMMFUSE_FF(18)

#define HYPRE_SMMFUSE_FC_UP_TO_1 \
   HYPRE_SMMFUSE_FC(0)

#define HYPRE_SMMFUSE_FC_UP_TO_2 \
   HYPRE_SMMFUSE_FC(0); \
   HYPRE_SMMFUSE_FC(1)

#define HYPRE_SMMFUSE_FC_UP_TO_3 \
   HYPRE_SMMFUSE_FC_UP_TO_2; \
   HYPRE_SMMFUSE_FC(2)

#define HYPRE_SMMFUSE_FC_UP_TO_4 \
   HYPRE_SMMFUSE_FC_UP_TO_3; \
   HYPRE_SMMFUSE_FC(3)

#define HYPRE_SMMFUSE_FC_UP_TO_5 \
   HYPRE_SMMFUSE_FC_UP_TO_4; \
   HYPRE_SMMFUSE_FC(4)

#define HYPRE_SMMFUSE_FC_UP_TO_6 \
   HYPRE_SMMFUSE_FC_UP_TO_5; \
   HYPRE_SMMFUSE_FC(5)

#define HYPRE_SMMFUSE_FC_UP_TO_7 \
   HYPRE_SMMFUSE_FC_UP_TO_6; \
   HYPRE_SMMFUSE_FC(6)

#define HYPRE_SMMFUSE_FC_UP_TO_8 \
   HYPRE_SMMFUSE_FC_UP_TO_7; \
   HYPRE_SMMFUSE_FC(7)

#define HYPRE_SMMFUSE_FC_UP_TO_9 \
   HYPRE_SMMFUSE_FC_UP_TO_8; \
   HYPRE_SMMFUSE_FC(8)

#define HYPRE_SMMFUSE_FC_UP_TO_10 \
   HYPRE_SMMFUSE_FC_UP_TO_9; \
   HYPRE_SMMFUSE_FC(9)

#define HYPRE_SMMFUSE_FC_UP_TO_11 \
   HYPRE_SMMFUSE_FC_UP_TO_10; \
   HYPRE_SMMFUSE_FC(10)

#define HYPRE_SMMFUSE_FC_UP_TO_12 \
   HYPRE_SMMFUSE_FC_UP_TO_11; \
   HYPRE_SMMFUSE_FC(11)

#define HYPRE_SMMFUSE_FC_UP_TO_13 \
   HYPRE_SMMFUSE_FC_UP_TO_12; \
   HYPRE_SMMFUSE_FC(12)

#define HYPRE_SMMFUSE_FC_UP_TO_14 \
   HYPRE_SMMFUSE_FC_UP_TO_13; \
   HYPRE_SMMFUSE_FC(13)

#define HYPRE_SMMFUSE_FC_UP_TO_15 \
   HYPRE_SMMFUSE_FC_UP_TO_14; \
   HYPRE_SMMFUSE_FC(14)

#define HYPRE_SMMFUSE_FC_UP_TO_16 \
   HYPRE_SMMFUSE_FC_UP_TO_15; \
   HYPRE_SMMFUSE_FC(15)

#define HYPRE_SMMFUSE_FC_UP_TO_17 \
   HYPRE_SMMFUSE_FC_UP_TO_16; \
   HYPRE_SMMFUSE_FC(16)

#define HYPRE_SMMFUSE_FC_UP_TO_18 \
   HYPRE_SMMFUSE_FC_UP_TO_17; \
   HYPRE_SMMFUSE_FC(17)

#define HYPRE_SMMFUSE_FC_UP_TO_19 \
   HYPRE_SMMFUSE_FC_UP_TO_18; \
   HYPRE_SMMFUSE_FC(18)

#define HYPRE_SMMFUSE_CC_UP_TO_1 \
   HYPRE_SMMFUSE_CC(0)

#define HYPRE_SMMFUSE_CC_UP_TO_2 \
   HYPRE_SMMFUSE_CC(0); \
   HYPRE_SMMFUSE_CC(1)

#define HYPRE_SMMFUSE_CC_UP_TO_3 \
   HYPRE_SMMFUSE_CC_UP_TO_2; \
   HYPRE_SMMFUSE_CC(2)

#define HYPRE_SMMFUSE_CC_UP_TO_4 \
   HYPRE_SMMFUSE_CC_UP_TO_3; \
   HYPRE_SMMFUSE_CC(3)

#define HYPRE_SMMFUSE_CC_UP_TO_5 \
   HYPRE_SMMFUSE_CC_UP_TO_4; \
   HYPRE_SMMFUSE_CC(4)

#define HYPRE_SMMFUSE_CC_UP_TO_6 \
   HYPRE_SMMFUSE_CC_UP_TO_5; \
   HYPRE_SMMFUSE_CC(5)

#define HYPRE_SMMFUSE_CC_UP_TO_7 \
   HYPRE_SMMFUSE_CC_UP_TO_6; \
   HYPRE_SMMFUSE_CC(6)

#define HYPRE_SMMFUSE_CC_UP_TO_8 \
   HYPRE_SMMFUSE_CC_UP_TO_7; \
   HYPRE_SMMFUSE_CC(7)

#define HYPRE_SMMFUSE_CC_UP_TO_9 \
   HYPRE_SMMFUSE_CC_UP_TO_8; \
   HYPRE_SMMFUSE_CC(8)

#define HYPRE_SMMFUSE_CC_UP_TO_10 \
   HYPRE_SMMFUSE_CC_UP_TO_9; \
   HYPRE_SMMFUSE_CC(9)

#define HYPRE_SMMFUSE_CC_UP_TO_11 \
   HYPRE_SMMFUSE_CC_UP_TO_10; \
   HYPRE_SMMFUSE_CC(10)

#define HYPRE_SMMFUSE_CC_UP_TO_12 \
   HYPRE_SMMFUSE_CC_UP_TO_11; \
   HYPRE_SMMFUSE_CC(11)

#define HYPRE_SMMFUSE_CC_UP_TO_13 \
   HYPRE_SMMFUSE_CC_UP_TO_12; \
   HYPRE_SMMFUSE_CC(12)

#define HYPRE_SMMFUSE_CC_UP_TO_14 \
   HYPRE_SMMFUSE_CC_UP_TO_13; \
   HYPRE_SMMFUSE_CC(13)

#define HYPRE_SMMFUSE_CC_UP_TO_15 \
   HYPRE_SMMFUSE_CC_UP_TO_14; \
   HYPRE_SMMFUSE_CC(14)

#define HYPRE_SMMFUSE_CC_UP_TO_16 \
   HYPRE_SMMFUSE_CC_UP_TO_15; \
   HYPRE_SMMFUSE_CC(15)

#define HYPRE_SMMFUSE_CC_UP_TO_17 \
   HYPRE_SMMFUSE_CC_UP_TO_16; \
   HYPRE_SMMFUSE_CC(16)

#define HYPRE_SMMFUSE_CC_UP_TO_18 \
   HYPRE_SMMFUSE_CC_UP_TO_17; \
   HYPRE_SMMFUSE_CC(17)

#define HYPRE_SMMFUSE_CC_UP_TO_19 \
   HYPRE_SMMFUSE_CC_UP_TO_18; \
   HYPRE_SMMFUSE_CC(18)

#define HYPRE_SMMFUSE_F_UP_TO_1 \
   HYPRE_SMMFUSE_F(0)

#define HYPRE_SMMFUSE_F_UP_TO_2 \
   HYPRE_SMMFUSE_F(0); \
   HYPRE_SMMFUSE_F(1)

#define HYPRE_SMMFUSE_F_UP_TO_3 \
   HYPRE_SMMFUSE_F_UP_TO_2; \
   HYPRE_SMMFUSE_F(2)

#define HYPRE_SMMFUSE_F_UP_TO_4 \
   HYPRE_SMMFUSE_F_UP_TO_3; \
   HYPRE_SMMFUSE_F(3)

#define HYPRE_SMMFUSE_F_UP_TO_5 \
   HYPRE_SMMFUSE_F_UP_TO_4; \
   HYPRE_SMMFUSE_F(4)

#define HYPRE_SMMFUSE_F_UP_TO_6 \
   HYPRE_SMMFUSE_F_UP_TO_5; \
   HYPRE_SMMFUSE_F(5)

#define HYPRE_SMMFUSE_F_UP_TO_7 \
   HYPRE_SMMFUSE_F_UP_TO_6; \
   HYPRE_SMMFUSE_F(6)

#define HYPRE_SMMFUSE_F_UP_TO_8 \
   HYPRE_SMMFUSE_F_UP_TO_7; \
   HYPRE_SMMFUSE_F(7)

#define HYPRE_SMMFUSE_F_UP_TO_9 \
   HYPRE_SMMFUSE_F_UP_TO_8; \
   HYPRE_SMMFUSE_F(8)

#define HYPRE_SMMFUSE_F_UP_TO_10 \
   HYPRE_SMMFUSE_F_UP_TO_9; \
   HYPRE_SMMFUSE_F(9)

#define HYPRE_SMMFUSE_F_UP_TO_11 \
   HYPRE_SMMFUSE_F_UP_TO_10; \
   HYPRE_SMMFUSE_F(10)

#define HYPRE_SMMFUSE_F_UP_TO_12 \
   HYPRE_SMMFUSE_F_UP_TO_11; \
   HYPRE_SMMFUSE_F(11)

#define HYPRE_SMMFUSE_F_UP_TO_13 \
   HYPRE_SMMFUSE_F_UP_TO_12; \
   HYPRE_SMMFUSE_F(12)

#define HYPRE_SMMFUSE_F_UP_TO_14 \
   HYPRE_SMMFUSE_F_UP_TO_13; \
   HYPRE_SMMFUSE_F(13)

#define HYPRE_SMMFUSE_F_UP_TO_15 \
   HYPRE_SMMFUSE_F_UP_TO_14; \
   HYPRE_SMMFUSE_F(14)

#define HYPRE_SMMFUSE_F_UP_TO_16 \
   HYPRE_SMMFUSE_F_UP_TO_15; \
   HYPRE_SMMFUSE_F(15)

#define HYPRE_SMMFUSE_F_UP_TO_17 \
   HYPRE_SMMFUSE_F_UP_TO_16; \
   HYPRE_SMMFUSE_F(16)

#define HYPRE_SMMFUSE_F_UP_TO_18 \
   HYPRE_SMMFUSE_F_UP_TO_17; \
   HYPRE_SMMFUSE_F(17)

#define HYPRE_SMMFUSE_F_UP_TO_19 \
   HYPRE_SMMFUSE_F_UP_TO_18; \
   HYPRE_SMMFUSE_F(18)

#define HYPRE_SMMFUSE_C_UP_TO_1 \
   HYPRE_SMMFUSE_C(0)

#define HYPRE_SMMFUSE_C_UP_TO_2 \
   HYPRE_SMMFUSE_C(0); \
   HYPRE_SMMFUSE_C(1)

#define HYPRE_SMMFUSE_C_UP_TO_3 \
   HYPRE_SMMFUSE_C_UP_TO_2; \
   HYPRE_SMMFUSE_C(2)

#define HYPRE_SMMFUSE_C_UP_TO_4 \
   HYPRE_SMMFUSE_C_UP_TO_3; \
   HYPRE_SMMFUSE_C(3)

#define HYPRE_SMMFUSE_C_UP_TO_5 \
   HYPRE_SMMFUSE_C_UP_TO_4; \
   HYPRE_SMMFUSE_C(4)

#define HYPRE_SMMFUSE_C_UP_TO_6 \
   HYPRE_SMMFUSE_C_UP_TO_5; \
   HYPRE_SMMFUSE_C(5)

#define HYPRE_SMMFUSE_C_UP_TO_7 \
   HYPRE_SMMFUSE_C_UP_TO_6; \
   HYPRE_SMMFUSE_C(6)

#define HYPRE_SMMFUSE_C_UP_TO_8 \
   HYPRE_SMMFUSE_C_UP_TO_7; \
   HYPRE_SMMFUSE_C(7)

#define HYPRE_SMMFUSE_C_UP_TO_9 \
   HYPRE_SMMFUSE_C_UP_TO_8; \
   HYPRE_SMMFUSE_C(8)

#define HYPRE_SMMFUSE_C_UP_TO_10 \
   HYPRE_SMMFUSE_C_UP_TO_9; \
   HYPRE_SMMFUSE_C(9)

#define HYPRE_SMMFUSE_C_UP_TO_11 \
   HYPRE_SMMFUSE_C_UP_TO_10; \
   HYPRE_SMMFUSE_C(10)

#define HYPRE_SMMFUSE_C_UP_TO_12 \
   HYPRE_SMMFUSE_C_UP_TO_11; \
   HYPRE_SMMFUSE_C(11)

#define HYPRE_SMMFUSE_C_UP_TO_13 \
   HYPRE_SMMFUSE_C_UP_TO_12; \
   HYPRE_SMMFUSE_C(12)

#define HYPRE_SMMFUSE_C_UP_TO_14 \
   HYPRE_SMMFUSE_C_UP_TO_13; \
   HYPRE_SMMFUSE_C(13)

#define HYPRE_SMMFUSE_C_UP_TO_15 \
   HYPRE_SMMFUSE_C_UP_TO_14; \
   HYPRE_SMMFUSE_C(14)

#define HYPRE_SMMFUSE_C_UP_TO_16 \
   HYPRE_SMMFUSE_C_UP_TO_15; \
   HYPRE_SMMFUSE_C(15)

#define HYPRE_SMMFUSE_C_UP_TO_17 \
   HYPRE_SMMFUSE_C_UP_TO_16; \
   HYPRE_SMMFUSE_C(16)

#define HYPRE_SMMFUSE_C_UP_TO_18 \
   HYPRE_SMMFUSE_C_UP_TO_17; \
   HYPRE_SMMFUSE_C(17)

#define HYPRE_SMMFUSE_C_UP_TO_19 \
   HYPRE_SMMFUSE_C_UP_TO_18; \
   HYPRE_SMMFUSE_C(18)

/*--------------------------------------------------------------------------
 * Compute the fused product for FFF terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_fff( HYPRE_Int            nprod,
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
   HYPRE_SMMFUSE_DECLARE_19_5VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_FFF POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k][0]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][1] = %p\n",
                      k, (void*)tptrs[k][1]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][2] = %p\n",
                      k, (void*)tptrs[k][2]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "========================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("fff");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 19:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 19);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_19;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 18);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_18;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 17);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_17;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 16);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_16;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 15);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_15;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 14);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_14;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 13);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_13;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 12);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_12;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 11);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_11;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 10);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_10;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 9);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_9;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 8);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_8;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_7;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_6;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_5;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_4;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_3;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_2;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FFF_UP_TO_1;
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

/*--------------------------------------------------------------------------
 * Compute the fused product for FFC terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_ffc( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     hypre_3Cptrs        *tptrs,
                                     hypre_1Cptr         *mptrs,
                                     HYPRE_Int           *mentries,
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
   HYPRE_Int     k, depth;

   /* Variable declarations for up to HYPRE_FUSE_MAXDEPTH variables */
   HYPRE_SMMFUSE_DECLARE_19_5VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_FFC POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k][0]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][1] = %p\n",
                      k, (void*)tptrs[k][1]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][2] = %p\n",
                      k, (void*)tptrs[k][2]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "========================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ffc");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 19:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 19);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_19;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 18);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_18;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 17);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_17;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 16);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_16;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 15);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_15;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 14);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_14;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 13);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_13;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 12);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_12;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 11);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_11;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 10);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_10;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 9);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_9;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 8);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_8;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 7);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_7;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 6);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_6;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 5);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_5;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 4);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_4;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 3);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_3;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 2);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_2;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 1);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FFC_UP_TO_1;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
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

/*--------------------------------------------------------------------------
 * Compute the fused product for FCC terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_fcc( HYPRE_Int            nprod,
                                     HYPRE_Complex       *cprod,
                                     hypre_3Cptrs        *tptrs,
                                     hypre_1Cptr         *mptrs,
                                     HYPRE_Int           *mentries,
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
   HYPRE_Int     k, depth;

   /* Variable declarations for up to HYPRE_FUSE_MAXDEPTH variables */
   HYPRE_SMMFUSE_DECLARE_19_5VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_FCC POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k][0]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][1] = %p\n",
                      k, (void*)tptrs[k][1]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][2] = %p\n",
                      k, (void*)tptrs[k][2]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "========================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("fcc");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 19:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 19);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_19;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 18);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_18;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 17);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_17;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 16);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_16;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 15);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_15;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 14);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_14;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 13);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_13;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 12);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_12;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 11);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_11;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 10);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_10;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 9);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_9;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 8);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_8;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 7);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_7;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 6);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_6;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 5);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_5;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 4);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_4;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 3);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_3;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 2);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_2;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_5VARS_UP_TO(k, 1);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FCC_UP_TO_1;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
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

/*--------------------------------------------------------------------------
 * Compute the fused product for FF terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_ff( HYPRE_Int            nprod,
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
   HYPRE_SMMFUSE_DECLARE_19_4VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_FF POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k][0]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][1] = %p\n",
                      k, (void*)tptrs[k][1]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=====================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ff");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 19:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 19);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_19;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 18);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_18;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 17);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_17;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 16);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_16;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 15);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_15;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 14);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_14;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 13);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_13;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 12);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_12;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 11);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_11;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 10);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_10;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 9);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_9;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 8);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_8;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_7;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_6;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_5;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_4;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_3;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_2;
            }
            hypre_BoxLoop2End(Mi, fi);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi);
            {
               HYPRE_SMMFUSE_FF_UP_TO_1;
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

/*--------------------------------------------------------------------------

 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_fc( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    hypre_3Cptrs        *tptrs,
                                    hypre_1Cptr         *mptrs,
                                    HYPRE_Int           *mentries,
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
   HYPRE_Int     k, depth;

   /* Variable declarations for up to HYPRE_FUSE_MAXDEPTH variables */
   HYPRE_SMMFUSE_DECLARE_19_4VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_FC POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k][0]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][1] = %p\n",
                      k, (void*)tptrs[k][1]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "========================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("fc");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 19:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 19);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_19;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 18);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_18;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 17);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_17;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 16);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_16;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 15);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_15;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 14);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_14;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 13);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_13;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 12);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_12;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 11);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_11;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 10);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_10;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 9);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_9;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 8);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_8;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 7);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_7;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 6);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_6;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 5);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_5;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 4);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_4;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 3);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_3;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 2);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_2;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 1);
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                fdbox, fdstart, fdstride, fi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_FC_UP_TO_1;
            }
            hypre_BoxLoop3End(Mi, fi, ci);
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

/*--------------------------------------------------------------------------
 * Compute the fused product for CC terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_cc( HYPRE_Int            nprod,
                                    HYPRE_Complex       *cprod,
                                    hypre_3Cptrs        *tptrs,
                                    hypre_1Cptr         *mptrs,
                                    HYPRE_Int           *mentries,
                                    HYPRE_Int            ndim,
                                    hypre_Index          loop_size,
                                    hypre_Box           *cdbox,
                                    hypre_Index          cdstart,
                                    hypre_Index          cdstride,
                                    hypre_Box           *Mdbox,
                                    hypre_Index          Mdstart,
                                    hypre_Index          Mdstride )
{
   HYPRE_Int     k, depth;

   /* Variable declarations for up to HYPRE_FUSE_MAXDEPTH variables */
   HYPRE_SMMFUSE_DECLARE_19_4VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_CC POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k][0]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][1] = %p\n",
                      k, (void*)tptrs[k][1]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=====================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("cc");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 19:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 19);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_19;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 18);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_18;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 17);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_17;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 16);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_16;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 15);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_15;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 14);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_14;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 13);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_13;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 12);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_12;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 11);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_11;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 10);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_10;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 9);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_9;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 8);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_8;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_7;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_6;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_5;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_4;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_3;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_2;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_4VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_CC_UP_TO_1;
            }
            hypre_BoxLoop2End(Mi, ci);
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

/*--------------------------------------------------------------------------

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
   HYPRE_SMMFUSE_DECLARE_19_3VARS;

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

/*--------------------------------------------------------------------------
 * Compute the fused product for C terms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_fuse_c( HYPRE_Int            nprod,
                                   HYPRE_Complex       *cprod,
                                   hypre_3Cptrs        *tptrs,
                                   hypre_1Cptr         *mptrs,
                                   HYPRE_Int           *mentries,
                                   HYPRE_Int            ndim,
                                   hypre_Index          loop_size,
                                   hypre_Box           *cdbox,
                                   hypre_Index          cdstart,
                                   hypre_Index          cdstride,
                                   hypre_Box           *Mdbox,
                                   hypre_Index          Mdstart,
                                   hypre_Index          Mdstride )
{
   HYPRE_Int     k, depth;

   /* Variable declarations for up to HYPRE_FUSE_MAXDEPTH variables */
   HYPRE_SMMFUSE_DECLARE_19_3VARS;

   if (nprod < 1)
   {
      return hypre_error_flag;
   }

#if defined(DEBUG_MATMULT) && (DEBUG_MATMULT > 1)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== FUSE_C POINTERS ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of products (nprod): %d\n", nprod);
   for (k = 0; k < nprod; k++)
   {
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Product %d:\n", k);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  cprod[%d] = %p (value: %e)\n",
                      k, (void*)&cprod[k], cprod[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  mptrs[%d] = %p (entry: %d)\n",
                      k, (void*)mptrs[k], mentries[k]);
      hypre_ParPrintf(hypre_MPI_COMM_WORLD, "  tptrs[%d][0] = %p\n",
                      k, (void*)tptrs[k][0]);
   }
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=====================\n");
#else
   HYPRE_UNUSED_VAR(mentries);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("c");

   for (k = 0; k < nprod; k += HYPRE_FUSE_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_FUSE_MAXDEPTH, (nprod - k));

      switch (depth)
      {
         case 19:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 19);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_19;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 18:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 18);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_18;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 17:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 17);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_17;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 16:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 16);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_16;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 15:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 15);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_15;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 14:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 14);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_14;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 13:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 13);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_13;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 12:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 12);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_12;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 11:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 11);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_11;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 10:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 10);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_10;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 9:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 9);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_9;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 8:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 8);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_8;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 7:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 7);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_7;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 6:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 6);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_6;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 5:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 5);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_5;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 4:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 4);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_4;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 3:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 3);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_3;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 2:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 2);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_2;
            }
            hypre_BoxLoop2End(Mi, ci);
            break;

         case 1:
            HYPRE_SMMFUSE_LOAD_3VARS_UP_TO(k, 1);
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                cdbox, cdstart, cdstride, ci);
            {
               HYPRE_SMMFUSE_C_UP_TO_1;
            }
            hypre_BoxLoop2End(Mi, ci);
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

#if defined(USE_FUSE_SORT)
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
   HYPRE_Int       approach = 1;

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
   HYPRE_Int       nprod[8] = {0};
   HYPRE_Complex   cprod[8][na];
   hypre_3Cptrs    tptrs[8][na];
   hypre_1Cptr     mptrs[8][na];
   HYPRE_Int       mentries[8][na];

   HYPRE_Int       ptype = 0, nf, nc, nt;
   HYPRE_Int       i, k, t;

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

      /* Set array values for product k of product type ptype */
      k = nprod[ptype];
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

   /* Call fuse functions */
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

   hypre_StructMatmultCompute_fuse_fcc(nprod[2], cprod[2], tptrs[2],
                                       mptrs[2], mentries[2],
                                       ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       cdbox, cdstart, cdstride,
                                       Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_ff(nprod[3], cprod[3], tptrs[3],
                                      mptrs[3], mentries[3],
                                      ndim, loop_size,
                                      fdbox, fdstart, fdstride,
                                      Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_fc(nprod[4], cprod[4], tptrs[4],
                                      mptrs[4], mentries[4],
                                      ndim, loop_size,
                                      fdbox, fdstart, fdstride,
                                      cdbox, cdstart, cdstride,
                                      Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_cc(nprod[5], cprod[5], tptrs[5],
                                      mptrs[5], mentries[5],
                                      ndim, loop_size,
                                      cdbox, cdstart, cdstride,
                                      Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_fuse_f(nprod[6], cprod[6], tptrs[6],
                                     mptrs[6], mentries[6],
                                     ndim, loop_size,
                                     fdbox, fdstart, fdstride,
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
