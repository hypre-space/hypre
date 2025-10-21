/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Macros used for Struct Matrix/Matrix multiplication
 *--------------------------------------------------------------------------*/

#define HYPRE_FUSE_MAXDEPTH 20
//#define DEBUG_MATMULT 0
//#define USE_FUSE_SORT 1

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
 *    - HYPRE_SMMFUSE_DECLARE_20_*VARS: Declare up to 20 sets of variables for unrolling.
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
#define HYPRE_SMMFUSE_DECLARE_MPTR(k) \
   HYPRE_Complex *mptr_##k

#define HYPRE_SMMFUSE_DECLARE_3VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, cprod_##k, *mptr_##k

#define HYPRE_SMMFUSE_DECLARE_3P(k, p) \
   HYPRE_Complex *tptrs_##p##_##k##_0, cprod_##p##_##k, *mptr_##p##_##k

#define HYPRE_SMMFUSE_DECLARE_4VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, *tptrs_##k##_1, cprod_##k, *mptr_##k

#define HYPRE_SMMFUSE_DECLARE_4P(k, p) \
   HYPRE_Complex *tptrs_##p##_##k##_0, *tptrs_##p##_##k##_1; \
   HYPRE_Complex cprod_##p##_##k, *mptr_##p##_##k

#define HYPRE_SMMFUSE_DECLARE_5VARS(k) \
   HYPRE_Complex *tptrs_##k##_0, *tptrs_##k##_1, *tptrs_##k##_2, cprod_##k, *mptr_##k

#define HYPRE_SMMFUSE_DECLARE_5P(k, p) \
   HYPRE_Complex *tptrs_##p##_##k##_0, *tptrs_##p##_##k##_1, *tptrs_##p##_##k##_2; \
   HYPRE_Complex cprod_##p##_##k, *mptr_##p##_##k

#define HYPRE_SMMFUSE_DECLARE_MPTRS_UP_TO_27  \
   HYPRE_SMMFUSE_DECLARE_MPTR(0);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(1);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(2);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(3);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(4);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(5);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(6);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(7);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(8);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(9);       \
   HYPRE_SMMFUSE_DECLARE_MPTR(10);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(11);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(12);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(13);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(14);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(15);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(16);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(17);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(18);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(19);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(20);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(21);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(22);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(23);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(24);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(25);      \
   HYPRE_SMMFUSE_DECLARE_MPTR(26)

#define HYPRE_SMMFUSE_DECLARE_FCC_UP_TO_36  \
   HYPRE_SMMFUSE_DECLARE_5P(0, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(1, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(2, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(3, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(4, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(5, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(6, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(7, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(8, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(9, fcc);    \
   HYPRE_SMMFUSE_DECLARE_5P(10, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(11, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(12, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(13, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(14, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(15, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(16, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(17, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(18, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(19, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(20, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(21, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(22, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(23, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(24, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(25, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(26, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(27, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(28, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(29, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(30, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(31, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(32, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(33, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(34, fcc);   \
   HYPRE_SMMFUSE_DECLARE_5P(35, fcc)

#define HYPRE_SMMFUSE_DECLARE_FC_UP_TO_72 \
   HYPRE_SMMFUSE_DECLARE_4P(0, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(1, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(2, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(3, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(4, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(5, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(6, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(7, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(8, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(9, fc);     \
   HYPRE_SMMFUSE_DECLARE_4P(10, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(11, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(12, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(13, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(14, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(15, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(16, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(17, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(18, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(19, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(20, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(21, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(22, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(23, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(24, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(25, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(26, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(27, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(28, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(29, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(30, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(31, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(32, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(33, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(34, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(35, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(36, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(37, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(38, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(39, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(40, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(41, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(42, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(43, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(44, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(45, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(46, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(47, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(48, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(49, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(50, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(51, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(52, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(53, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(54, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(55, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(56, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(57, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(58, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(59, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(60, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(61, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(62, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(63, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(64, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(65, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(66, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(67, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(68, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(69, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(70, fc);    \
   HYPRE_SMMFUSE_DECLARE_4P(71, fc)

#define HYPRE_SMMFUSE_DECLARE_F_UP_TO_9 \
   HYPRE_SMMFUSE_DECLARE_3P(0, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(1, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(2, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(3, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(4, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(5, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(6, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(7, f);      \
   HYPRE_SMMFUSE_DECLARE_3P(8, f)

#define HYPRE_SMMFUSE_DECLARE_20_3VARS  \
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
   HYPRE_SMMFUSE_DECLARE_3VARS(18);     \
   HYPRE_SMMFUSE_DECLARE_3VARS(19);

#define HYPRE_SMMFUSE_DECLARE_20_4VARS  \
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
   HYPRE_SMMFUSE_DECLARE_4VARS(18);     \
   HYPRE_SMMFUSE_DECLARE_4VARS(19)

#define HYPRE_SMMFUSE_DECLARE_20_5VARS  \
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
   HYPRE_SMMFUSE_DECLARE_5VARS(18);     \
   HYPRE_SMMFUSE_DECLARE_5VARS(19)

/* Variable loading macros for different patterns */
#define HYPRE_SMMFUSE_LOAD_5VARS(k, m)   \
   cprod_##m     = cprod[(k + m)];       \
   mptr_##m      = mptrs[(k + m)];       \
   tptrs_##m##_0 = tptrs[(k + m)][0];    \
   tptrs_##m##_1 = tptrs[(k + m)][1];    \
   tptrs_##m##_2 = tptrs[(k + m)][2]

#define HYPRE_SMMFUSE_LOAD_5P(k, p)      \
   cprod_##p##_##k     = cprod_##p[k];     \
   mptr_##p##_##k      = mptrs_##p[k];     \
   tptrs_##p##_##k##_0 = tptrs_##p[k][0];  \
   tptrs_##p##_##k##_1 = tptrs_##p[k][1];  \
   tptrs_##p##_##k##_2 = tptrs_##p[k][2]

#define HYPRE_SMMFUSE_LOAD_4VARS(k, m)   \
   cprod_##m     = cprod[(k + m)];       \
   mptr_##m      = mptrs[(k + m)];       \
   tptrs_##m##_0 = tptrs[(k + m)][0];    \
   tptrs_##m##_1 = tptrs[(k + m)][1]

#define HYPRE_SMMFUSE_LOAD_4P(k, p)      \
   cprod_##p##_##k     = cprod_##p[k];     \
   mptr_##p##_##k      = mptrs_##p[k];     \
   tptrs_##p##_##k##_0 = tptrs_##p[k][0];  \
   tptrs_##p##_##k##_1 = tptrs_##p[k][1]

#define HYPRE_SMMFUSE_LOAD_3VARS(k, m)   \
   cprod_##m     = cprod[(k + m)];       \
   mptr_##m      = mptrs[(k + m)];       \
   tptrs_##m##_0 = tptrs[(k + m)][0]

#define HYPRE_SMMFUSE_LOAD_3P(k, p)      \
   cprod_##p##_##k     = cprod_##p[k];   \
   mptr_##p##_##k      = mptrs_##p[k];   \
   tptrs_##p##_##k##_0 = tptrs_##p[k][0]

#define HYPRE_SMMFUSE_LOAD_MPTR(k)       \
   mptr_##k      = mptrs[k]

#define HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_8  \
   HYPRE_SMMFUSE_LOAD_MPTR(0);            \
   HYPRE_SMMFUSE_LOAD_MPTR(1);            \
   HYPRE_SMMFUSE_LOAD_MPTR(2);            \
   HYPRE_SMMFUSE_LOAD_MPTR(3);            \
   HYPRE_SMMFUSE_LOAD_MPTR(4);            \
   HYPRE_SMMFUSE_LOAD_MPTR(5);            \
   HYPRE_SMMFUSE_LOAD_MPTR(6);            \
   HYPRE_SMMFUSE_LOAD_MPTR(7)

#define HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_14 \
   HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_8;      \
   HYPRE_SMMFUSE_LOAD_MPTR(8);            \
   HYPRE_SMMFUSE_LOAD_MPTR(9);            \
   HYPRE_SMMFUSE_LOAD_MPTR(10);           \
   HYPRE_SMMFUSE_LOAD_MPTR(11);           \
   HYPRE_SMMFUSE_LOAD_MPTR(12);           \
   HYPRE_SMMFUSE_LOAD_MPTR(13)

#define HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_15 \
   HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_14;     \
   HYPRE_SMMFUSE_LOAD_MPTR(14)

#define HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_27 \
   HYPRE_SMMFUSE_LOAD_MPTRS_UP_TO_15;     \
   HYPRE_SMMFUSE_LOAD_MPTR(15);           \
   HYPRE_SMMFUSE_LOAD_MPTR(16);           \
   HYPRE_SMMFUSE_LOAD_MPTR(17);           \
   HYPRE_SMMFUSE_LOAD_MPTR(18);           \
   HYPRE_SMMFUSE_LOAD_MPTR(19);           \
   HYPRE_SMMFUSE_LOAD_MPTR(20);           \
   HYPRE_SMMFUSE_LOAD_MPTR(21);           \
   HYPRE_SMMFUSE_LOAD_MPTR(22);           \
   HYPRE_SMMFUSE_LOAD_MPTR(23);           \
   HYPRE_SMMFUSE_LOAD_MPTR(24);           \
   HYPRE_SMMFUSE_LOAD_MPTR(25);           \
   HYPRE_SMMFUSE_LOAD_MPTR(26)

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

#define HYPRE_SMMFUSE_LOAD_3VARS_IMPL_20(k, ...) \
   HYPRE_SMMFUSE_LOAD_3VARS_IMPL_19(k); \
   HYPRE_SMMFUSE_LOAD_3VARS(k, 19)

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

#define HYPRE_SMMFUSE_LOAD_4VARS_IMPL_20(k, ...) \
   HYPRE_SMMFUSE_LOAD_4VARS_IMPL_19(k); \
   HYPRE_SMMFUSE_LOAD_4VARS(k, 19)

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

#define HYPRE_SMMFUSE_LOAD_5VARS_IMPL_20(k, ...) \
   HYPRE_SMMFUSE_LOAD_5VARS_IMPL_19(k); \
   HYPRE_SMMFUSE_LOAD_5VARS(k, 19)

#define HYPRE_SMMFUSE_LOAD_F_UP_TO_3    \
   HYPRE_SMMFUSE_LOAD_3P(0, f);         \
   HYPRE_SMMFUSE_LOAD_3P(1, f);         \
   HYPRE_SMMFUSE_LOAD_3P(2, f)

#define HYPRE_SMMFUSE_LOAD_F_UP_TO_5    \
   HYPRE_SMMFUSE_LOAD_F_UP_TO_3;        \
   HYPRE_SMMFUSE_LOAD_3P(3, f);         \
   HYPRE_SMMFUSE_LOAD_3P(4, f)

#define HYPRE_SMMFUSE_LOAD_F_UP_TO_9    \
   HYPRE_SMMFUSE_LOAD_F_UP_TO_5;        \
   HYPRE_SMMFUSE_LOAD_3P(5, f);         \
   HYPRE_SMMFUSE_LOAD_3P(6, f);         \
   HYPRE_SMMFUSE_LOAD_3P(7, f);         \
   HYPRE_SMMFUSE_LOAD_3P(8, f)

#define HYPRE_SMMFUSE_LOAD_FC_UP_TO_6    \
   HYPRE_SMMFUSE_LOAD_4P(0, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(1, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(2, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(3, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(4, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(5, fc)

#define HYPRE_SMMFUSE_LOAD_FC_UP_TO_8    \
   HYPRE_SMMFUSE_LOAD_FC_UP_TO_6;        \
   HYPRE_SMMFUSE_LOAD_4P(6, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(7, fc)

#define HYPRE_SMMFUSE_LOAD_FC_UP_TO_14   \
   HYPRE_SMMFUSE_LOAD_FC_UP_TO_8;        \
   HYPRE_SMMFUSE_LOAD_4P(8, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(9, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(10, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(11, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(12, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(13, fc)

#define HYPRE_SMMFUSE_LOAD_FC_UP_TO_24   \
   HYPRE_SMMFUSE_LOAD_FC_UP_TO_8;        \
   HYPRE_SMMFUSE_LOAD_4P(8, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(9, fc);         \
   HYPRE_SMMFUSE_LOAD_4P(10, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(11, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(12, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(13, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(14, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(15, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(16, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(17, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(18, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(19, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(20, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(21, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(22, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(23, fc)

#define HYPRE_SMMFUSE_LOAD_FC_UP_TO_38   \
   HYPRE_SMMFUSE_LOAD_FC_UP_TO_24;       \
   HYPRE_SMMFUSE_LOAD_4P(24, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(25, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(26, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(27, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(28, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(29, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(30, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(31, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(32, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(33, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(34, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(35, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(36, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(37, fc)

#define HYPRE_SMMFUSE_LOAD_FC_UP_TO_72   \
   HYPRE_SMMFUSE_LOAD_FC_UP_TO_38;       \
   HYPRE_SMMFUSE_LOAD_4P(38, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(39, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(40, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(41, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(42, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(43, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(44, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(45, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(46, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(47, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(48, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(49, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(50, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(51, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(52, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(53, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(54, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(55, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(56, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(57, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(58, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(59, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(60, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(61, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(62, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(63, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(64, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(65, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(66, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(67, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(68, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(69, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(70, fc);        \
   HYPRE_SMMFUSE_LOAD_4P(71, fc)

#define HYPRE_SMMFUSE_LOAD_FCC_UP_TO_11   \
   HYPRE_SMMFUSE_LOAD_5P(0, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(1, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(2, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(3, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(4, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(5, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(6, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(7, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(8, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(9, fcc);         \
   HYPRE_SMMFUSE_LOAD_5P(10, fcc)

#define HYPRE_SMMFUSE_LOAD_FCC_UP_TO_19   \
   HYPRE_SMMFUSE_LOAD_FCC_UP_TO_11;       \
   HYPRE_SMMFUSE_LOAD_5P(11, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(12, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(13, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(14, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(15, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(16, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(17, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(18, fcc)

#define HYPRE_SMMFUSE_LOAD_FCC_UP_TO_20   \
   HYPRE_SMMFUSE_LOAD_FCC_UP_TO_19;       \
   HYPRE_SMMFUSE_LOAD_5P(19, fcc)

#define HYPRE_SMMFUSE_LOAD_FCC_UP_TO_36   \
   HYPRE_SMMFUSE_LOAD_FCC_UP_TO_20;       \
   HYPRE_SMMFUSE_LOAD_5P(20, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(21, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(22, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(23, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(24, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(25, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(26, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(27, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(28, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(29, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(30, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(31, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(32, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(33, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(34, fcc);        \
   HYPRE_SMMFUSE_LOAD_5P(35, fcc)

/* Initialization macros */
#define HYPRE_SMMFUSE_INIT_MPTR(k)        \
   mptr_##k[Mi] = 0.0

#define HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_8  \
   HYPRE_SMMFUSE_INIT_MPTR(0);            \
   HYPRE_SMMFUSE_INIT_MPTR(1);            \
   HYPRE_SMMFUSE_INIT_MPTR(2);            \
   HYPRE_SMMFUSE_INIT_MPTR(3);            \
   HYPRE_SMMFUSE_INIT_MPTR(4);            \
   HYPRE_SMMFUSE_INIT_MPTR(5);            \
   HYPRE_SMMFUSE_INIT_MPTR(6);            \
   HYPRE_SMMFUSE_INIT_MPTR(7)

#define HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_14 \
   HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_8;      \
   HYPRE_SMMFUSE_INIT_MPTR(8);            \
   HYPRE_SMMFUSE_INIT_MPTR(9);            \
   HYPRE_SMMFUSE_INIT_MPTR(10);           \
   HYPRE_SMMFUSE_INIT_MPTR(11);           \
   HYPRE_SMMFUSE_INIT_MPTR(12);           \
   HYPRE_SMMFUSE_INIT_MPTR(13)

#define HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_15 \
   HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_14;     \
   HYPRE_SMMFUSE_INIT_MPTR(14)

#define HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_27 \
   HYPRE_SMMFUSE_INIT_MPTRS_UP_TO_15;     \
   HYPRE_SMMFUSE_INIT_MPTR(15);           \
   HYPRE_SMMFUSE_INIT_MPTR(16);           \
   HYPRE_SMMFUSE_INIT_MPTR(17);           \
   HYPRE_SMMFUSE_INIT_MPTR(18);           \
   HYPRE_SMMFUSE_INIT_MPTR(19);           \
   HYPRE_SMMFUSE_INIT_MPTR(20);           \
   HYPRE_SMMFUSE_INIT_MPTR(21);           \
   HYPRE_SMMFUSE_INIT_MPTR(22);           \
   HYPRE_SMMFUSE_INIT_MPTR(23);           \
   HYPRE_SMMFUSE_INIT_MPTR(24);           \
   HYPRE_SMMFUSE_INIT_MPTR(25);           \
   HYPRE_SMMFUSE_INIT_MPTR(26)

/* Individual fused multiply-add operations */
#define HYPRE_SMMFUSE_FFF(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi] * tptrs_##k##_2[fi]

#define HYPRE_SMMFUSE_FFC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi] * tptrs_##k##_2[ci]

#define HYPRE_SMMFUSE_FCC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[ci] * tptrs_##k##_2[ci]

#define HYPRE_SMMFUSE_FCC_P(k) \
   mptr_fcc_##k[Mi] += cprod_fcc_##k * tptrs_fcc_##k##_0[fi] * tptrs_fcc_##k##_1[ci] * tptrs_fcc_##k##_2[ci]

#define HYPRE_SMMFUSE_FC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[ci]

#define HYPRE_SMMFUSE_FC_P(k) \
   mptr_fc_##k[Mi] += cprod_fc_##k * tptrs_fc_##k##_0[fi] * tptrs_fc_##k##_1[ci]

#define HYPRE_SMMFUSE_CC(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[ci] * tptrs_##k##_1[ci]

#define HYPRE_SMMFUSE_FF(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi] * tptrs_##k##_1[fi]

#define HYPRE_SMMFUSE_F(k) \
   mptr_##k[Mi] += cprod_##k * tptrs_##k##_0[fi]

#define HYPRE_SMMFUSE_F_P(k) \
   mptr_f_##k[Mi] += cprod_f_##k * tptrs_f_##k##_0[fi]

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

#define HYPRE_SMMFUSE_FFF_UP_TO_20 \
   HYPRE_SMMFUSE_FFF_UP_TO_19; \
   HYPRE_SMMFUSE_FFF(19)

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

#define HYPRE_SMMFUSE_FFC_UP_TO_20 \
   HYPRE_SMMFUSE_FFC_UP_TO_19; \
   HYPRE_SMMFUSE_FFC(19)

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

#define HYPRE_SMMFUSE_FCC_UP_TO_20 \
   HYPRE_SMMFUSE_FCC_UP_TO_19; \
   HYPRE_SMMFUSE_FCC(19)

#define HYPRE_SMMFUSE_CALC_F_UP_TO_3  \
   HYPRE_SMMFUSE_F(f_0);              \
   HYPRE_SMMFUSE_F(f_1);              \
   HYPRE_SMMFUSE_F(f_2)

#define HYPRE_SMMFUSE_CALC_F_UP_TO_5  \
   HYPRE_SMMFUSE_CALC_F_UP_TO_3;      \
   HYPRE_SMMFUSE_F(f_3);              \
   HYPRE_SMMFUSE_F(f_4)

#define HYPRE_SMMFUSE_CALC_F_UP_TO_9  \
   HYPRE_SMMFUSE_CALC_F_UP_TO_5;      \
   HYPRE_SMMFUSE_F(f_5);              \
   HYPRE_SMMFUSE_F(f_6);              \
   HYPRE_SMMFUSE_F(f_7);              \
   HYPRE_SMMFUSE_F(f_8)

#define HYPRE_SMMFUSE_CALC_FC_UP_TO_6  \
   HYPRE_SMMFUSE_FC(fc_0);             \
   HYPRE_SMMFUSE_FC(fc_1);             \
   HYPRE_SMMFUSE_FC(fc_2);             \
   HYPRE_SMMFUSE_FC(fc_3);             \
   HYPRE_SMMFUSE_FC(fc_4);             \
   HYPRE_SMMFUSE_FC(fc_5)

#define HYPRE_SMMFUSE_CALC_FC_UP_TO_8  \
   HYPRE_SMMFUSE_CALC_FC_UP_TO_6;      \
   HYPRE_SMMFUSE_FC(fc_6);             \
   HYPRE_SMMFUSE_FC(fc_7)

#define HYPRE_SMMFUSE_CALC_FC_UP_TO_14 \
   HYPRE_SMMFUSE_CALC_FC_UP_TO_8;      \
   HYPRE_SMMFUSE_FC(fc_8);             \
   HYPRE_SMMFUSE_FC(fc_9);             \
   HYPRE_SMMFUSE_FC(fc_10);            \
   HYPRE_SMMFUSE_FC(fc_11);            \
   HYPRE_SMMFUSE_FC(fc_12);            \
   HYPRE_SMMFUSE_FC(fc_13)

#define HYPRE_SMMFUSE_CALC_FC_UP_TO_24 \
   HYPRE_SMMFUSE_CALC_FC_UP_TO_14;     \
   HYPRE_SMMFUSE_FC(fc_14);            \
   HYPRE_SMMFUSE_FC(fc_15);            \
   HYPRE_SMMFUSE_FC(fc_16);            \
   HYPRE_SMMFUSE_FC(fc_17);            \
   HYPRE_SMMFUSE_FC(fc_18);            \
   HYPRE_SMMFUSE_FC(fc_19);            \
   HYPRE_SMMFUSE_FC(fc_20);            \
   HYPRE_SMMFUSE_FC(fc_21);            \
   HYPRE_SMMFUSE_FC(fc_22);            \
   HYPRE_SMMFUSE_FC(fc_23)

#define HYPRE_SMMFUSE_CALC_FC_UP_TO_38 \
   HYPRE_SMMFUSE_CALC_FC_UP_TO_24;     \
   HYPRE_SMMFUSE_FC(fc_24);            \
   HYPRE_SMMFUSE_FC(fc_25);            \
   HYPRE_SMMFUSE_FC(fc_26);            \
   HYPRE_SMMFUSE_FC(fc_27);            \
   HYPRE_SMMFUSE_FC(fc_28);            \
   HYPRE_SMMFUSE_FC(fc_29);            \
   HYPRE_SMMFUSE_FC(fc_30);            \
   HYPRE_SMMFUSE_FC(fc_31);            \
   HYPRE_SMMFUSE_FC(fc_32);            \
   HYPRE_SMMFUSE_FC(fc_33);            \
   HYPRE_SMMFUSE_FC(fc_34);            \
   HYPRE_SMMFUSE_FC(fc_35);            \
   HYPRE_SMMFUSE_FC(fc_36);            \
   HYPRE_SMMFUSE_FC(fc_37)

#define HYPRE_SMMFUSE_CALC_FC_UP_TO_72 \
   HYPRE_SMMFUSE_CALC_FC_UP_TO_38;     \
   HYPRE_SMMFUSE_FC(fc_38);            \
   HYPRE_SMMFUSE_FC(fc_39);            \
   HYPRE_SMMFUSE_FC(fc_40);            \
   HYPRE_SMMFUSE_FC(fc_41);            \
   HYPRE_SMMFUSE_FC(fc_42);            \
   HYPRE_SMMFUSE_FC(fc_43);            \
   HYPRE_SMMFUSE_FC(fc_44);            \
   HYPRE_SMMFUSE_FC(fc_45);            \
   HYPRE_SMMFUSE_FC(fc_46);            \
   HYPRE_SMMFUSE_FC(fc_47);            \
   HYPRE_SMMFUSE_FC(fc_48);            \
   HYPRE_SMMFUSE_FC(fc_49);            \
   HYPRE_SMMFUSE_FC(fc_50);            \
   HYPRE_SMMFUSE_FC(fc_51);            \
   HYPRE_SMMFUSE_FC(fc_52);            \
   HYPRE_SMMFUSE_FC(fc_53);            \
   HYPRE_SMMFUSE_FC(fc_54);            \
   HYPRE_SMMFUSE_FC(fc_55);            \
   HYPRE_SMMFUSE_FC(fc_56);            \
   HYPRE_SMMFUSE_FC(fc_57);            \
   HYPRE_SMMFUSE_FC(fc_58);            \
   HYPRE_SMMFUSE_FC(fc_59);            \
   HYPRE_SMMFUSE_FC(fc_60);            \
   HYPRE_SMMFUSE_FC(fc_61);            \
   HYPRE_SMMFUSE_FC(fc_62);            \
   HYPRE_SMMFUSE_FC(fc_63);            \
   HYPRE_SMMFUSE_FC(fc_64);            \
   HYPRE_SMMFUSE_FC(fc_65);            \
   HYPRE_SMMFUSE_FC(fc_66);            \
   HYPRE_SMMFUSE_FC(fc_67);            \
   HYPRE_SMMFUSE_FC(fc_68);            \
   HYPRE_SMMFUSE_FC(fc_69);            \
   HYPRE_SMMFUSE_FC(fc_70);            \
   HYPRE_SMMFUSE_FC(fc_71)

#define HYPRE_SMMFUSE_CALC_FCC_UP_TO_11 \
   HYPRE_SMMFUSE_FCC(fcc_0);            \
   HYPRE_SMMFUSE_FCC(fcc_1);            \
   HYPRE_SMMFUSE_FCC(fcc_2);            \
   HYPRE_SMMFUSE_FCC(fcc_3);            \
   HYPRE_SMMFUSE_FCC(fcc_4);            \
   HYPRE_SMMFUSE_FCC(fcc_5);            \
   HYPRE_SMMFUSE_FCC(fcc_6);            \
   HYPRE_SMMFUSE_FCC(fcc_7);            \
   HYPRE_SMMFUSE_FCC(fcc_8);            \
   HYPRE_SMMFUSE_FCC(fcc_9);            \
   HYPRE_SMMFUSE_FCC(fcc_10)

#define HYPRE_SMMFUSE_CALC_FCC_UP_TO_19 \
   HYPRE_SMMFUSE_CALC_FCC_UP_TO_11;     \
   HYPRE_SMMFUSE_FCC(fcc_11);           \
   HYPRE_SMMFUSE_FCC(fcc_12);           \
   HYPRE_SMMFUSE_FCC(fcc_13);           \
   HYPRE_SMMFUSE_FCC(fcc_14);           \
   HYPRE_SMMFUSE_FCC(fcc_15);           \
   HYPRE_SMMFUSE_FCC(fcc_16);           \
   HYPRE_SMMFUSE_FCC(fcc_17);           \
   HYPRE_SMMFUSE_FCC(fcc_18)

#define HYPRE_SMMFUSE_CALC_FCC_UP_TO_20 \
   HYPRE_SMMFUSE_CALC_FCC_UP_TO_19;     \
   HYPRE_SMMFUSE_FCC(fcc_19)

#define HYPRE_SMMFUSE_CALC_FCC_UP_TO_36 \
   HYPRE_SMMFUSE_CALC_FCC_UP_TO_20;     \
   HYPRE_SMMFUSE_FCC(fcc_20);           \
   HYPRE_SMMFUSE_FCC(fcc_21);           \
   HYPRE_SMMFUSE_FCC(fcc_22);           \
   HYPRE_SMMFUSE_FCC(fcc_23);           \
   HYPRE_SMMFUSE_FCC(fcc_24);           \
   HYPRE_SMMFUSE_FCC(fcc_25);           \
   HYPRE_SMMFUSE_FCC(fcc_26);           \
   HYPRE_SMMFUSE_FCC(fcc_27);           \
   HYPRE_SMMFUSE_FCC(fcc_28);           \
   HYPRE_SMMFUSE_FCC(fcc_29);           \
   HYPRE_SMMFUSE_FCC(fcc_30);           \
   HYPRE_SMMFUSE_FCC(fcc_31);           \
   HYPRE_SMMFUSE_FCC(fcc_32);           \
   HYPRE_SMMFUSE_FCC(fcc_33);           \
   HYPRE_SMMFUSE_FCC(fcc_34);           \
   HYPRE_SMMFUSE_FCC(fcc_35)

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

#define HYPRE_SMMFUSE_FF_UP_TO_20 \
   HYPRE_SMMFUSE_FF_UP_TO_19; \
   HYPRE_SMMFUSE_FF(19)

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

#define HYPRE_SMMFUSE_FC_UP_TO_20 \
   HYPRE_SMMFUSE_FC_UP_TO_19; \
   HYPRE_SMMFUSE_FC(19)

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

#define HYPRE_SMMFUSE_CC_UP_TO_20 \
   HYPRE_SMMFUSE_CC_UP_TO_19; \
   HYPRE_SMMFUSE_CC(19)

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

#define HYPRE_SMMFUSE_F_UP_TO_20 \
   HYPRE_SMMFUSE_F_UP_TO_19; \
   HYPRE_SMMFUSE_F(19)

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

#define HYPRE_SMMFUSE_C_UP_TO_20 \
   HYPRE_SMMFUSE_C_UP_TO_19; \
   HYPRE_SMMFUSE_C(19)
