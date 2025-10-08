/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

#if !defined(HYPRE_UNROLL_MAXDEPTH) ||\
     defined(HYPRE_UNROLL_MAXDEPTH) && HYPRE_UNROLL_MAXDEPTH > 27
#if defined(HYPRE_USING_GPU)
#define HYPRE_UNROLL_MAXDEPTH 27
#else
#define HYPRE_UNROLL_MAXDEPTH 21
#endif
#endif

/*--------------------------------------------------------------------------
 * Macros used in the kernel loops below
 *--------------------------------------------------------------------------*/

#define HYPRE_DECLARE_OFFSETS(n)   \
   HYPRE_Complex *Ap##n   = NULL;  \
   HYPRE_Int      xoff##n = 0

#define HYPRE_MAP_A_OFFSET(offset)                       \
   hypre_StructMatrixMapDataIndex(A, offset);            \
   hypre_SubtractIndexes(offset, Adstart, ndim, offset);

#define HYPRE_MAP_X_OFFSET(offset)                       \
   hypre_MapToFineIndex(offset, NULL, xfstride, ndim);   \
   hypre_StructVectorMapDataIndex(x, offset);            \
   hypre_SubtractIndexes(offset, xdstart, ndim, offset);

#define HYPRE_SET_CAX(Ap, xoff, entry)                               \
   Ap = hypre_StructMatrixBoxData(A, Ab, entry);                     \
   hypre_AddIndexes(start, stencil_shape[entry], ndim, offset);      \
   HYPRE_MAP_X_OFFSET(offset);                                       \
   xoff = hypre_BoxOffsetDistance(x_data_box, offset);

#define HYPRE_SET_CAX_TRANS(Ap, xoff, entry)                         \
   Ap = hypre_StructMatrixBoxData(A, Ab, entry);                     \
   hypre_SubtractIndexes(start, stencil_shape[entry], ndim, offset); \
   HYPRE_MAP_X_OFFSET(offset);                                       \
   xoff = hypre_BoxOffsetDistance(x_data_box, offset);

#define HYPRE_SET_AX(Ap, xoff, entry)                                \
   HYPRE_SET_CAX(Ap, xoff, entry)

#define HYPRE_SET_AX_TRANS(Ap, xoff, entry)                          \
   HYPRE_SET_CAX_TRANS(Ap, xoff, entry)                              \
   hypre_SubtractIndexes(start, stencil_shape[entry], ndim, offset); \
   HYPRE_MAP_A_OFFSET(offset);                                       \
   Ap += hypre_BoxOffsetDistance(A_data_box, offset);

#define HYPRE_LOAD_CAX(transpose, n)                                 \
   if (transpose) {                                                  \
      HYPRE_SET_CAX_TRANS(Ap##n, xoff##n, entries[si + n])           \
   } else {                                                          \
      HYPRE_SET_CAX(Ap##n, xoff##n, entries[si + n])                 \
   }

#define HYPRE_LOAD_AX(transpose, n)                                  \
   if (transpose) {                                                  \
      HYPRE_SET_AX_TRANS(Ap##n, xoff##n, entries[si + n])            \
   } else {                                                          \
      HYPRE_SET_AX(Ap##n, xoff##n, entries[si + n])                  \
   }

#define HYPRE_CALC_AX(n)        \
   Ap##n[Ai] * xp[xi + xoff##n]

#define HYPRE_CALC_AX_ADD(n)    \
   Ap##n[Ai] * xp[xi + xoff##n] +

#define HYPRE_CALC_CAX(n)        \
   Ap##n[0] * xp[xi + xoff##n]

#define HYPRE_CALC_CAX_ADD(n)    \
   Ap##n[0] * xp[xi + xoff##n] +

/* Sequence macros for declaring offset variables */
#define HYPRE_DECLARE_OFFSETS_UP_TO_27 \
   HYPRE_DECLARE_OFFSETS(0);     \
   HYPRE_DECLARE_OFFSETS(1);     \
   HYPRE_DECLARE_OFFSETS(2);     \
   HYPRE_DECLARE_OFFSETS(3);     \
   HYPRE_DECLARE_OFFSETS(4);     \
   HYPRE_DECLARE_OFFSETS(5);     \
   HYPRE_DECLARE_OFFSETS(6);     \
   HYPRE_DECLARE_OFFSETS(7);     \
   HYPRE_DECLARE_OFFSETS(8);     \
   HYPRE_DECLARE_OFFSETS(9);     \
   HYPRE_DECLARE_OFFSETS(10);    \
   HYPRE_DECLARE_OFFSETS(11);    \
   HYPRE_DECLARE_OFFSETS(12);    \
   HYPRE_DECLARE_OFFSETS(13);    \
   HYPRE_DECLARE_OFFSETS(14);    \
   HYPRE_DECLARE_OFFSETS(15);    \
   HYPRE_DECLARE_OFFSETS(16);    \
   HYPRE_DECLARE_OFFSETS(17);    \
   HYPRE_DECLARE_OFFSETS(18);    \
   HYPRE_DECLARE_OFFSETS(19);    \
   HYPRE_DECLARE_OFFSETS(20);    \
   HYPRE_DECLARE_OFFSETS(21);    \
   HYPRE_DECLARE_OFFSETS(22);    \
   HYPRE_DECLARE_OFFSETS(23);    \
   HYPRE_DECLARE_OFFSETS(24);    \
   HYPRE_DECLARE_OFFSETS(25);    \
   HYPRE_DECLARE_OFFSETS(26)

/* Sequence macros for loading matrix entries */
#define HYPRE_LOAD_AX_UP_TO_1(t) \
   HYPRE_LOAD_AX(t, 0) \

#define HYPRE_LOAD_AX_UP_TO_2(t) \
   HYPRE_LOAD_AX(t, 1) \
   HYPRE_LOAD_AX_UP_TO_1(t)

#define HYPRE_LOAD_AX_UP_TO_3(t) \
   HYPRE_LOAD_AX(t, 2) \
   HYPRE_LOAD_AX_UP_TO_2(t)

#define HYPRE_LOAD_AX_UP_TO_4(t) \
   HYPRE_LOAD_AX(t, 3) \
   HYPRE_LOAD_AX_UP_TO_3(t)

#define HYPRE_LOAD_AX_UP_TO_5(t) \
   HYPRE_LOAD_AX(t, 4) \
   HYPRE_LOAD_AX_UP_TO_4(t)

#define HYPRE_LOAD_AX_UP_TO_6(t) \
   HYPRE_LOAD_AX(t, 5) \
   HYPRE_LOAD_AX_UP_TO_5(t)

#define HYPRE_LOAD_AX_UP_TO_7(t) \
   HYPRE_LOAD_AX(t, 6) \
   HYPRE_LOAD_AX_UP_TO_6(t)

#define HYPRE_LOAD_AX_UP_TO_8(t) \
   HYPRE_LOAD_AX(t, 7) \
   HYPRE_LOAD_AX_UP_TO_7(t)

#define HYPRE_LOAD_AX_UP_TO_9(t) \
   HYPRE_LOAD_AX(t, 8) \
   HYPRE_LOAD_AX_UP_TO_8(t)

#define HYPRE_LOAD_AX_UP_TO_10(t) \
   HYPRE_LOAD_AX(t, 9) \
   HYPRE_LOAD_AX_UP_TO_9(t)

#define HYPRE_LOAD_AX_UP_TO_11(t) \
   HYPRE_LOAD_AX(t, 10) \
   HYPRE_LOAD_AX_UP_TO_10(t)

#define HYPRE_LOAD_AX_UP_TO_12(t) \
   HYPRE_LOAD_AX(t, 11) \
   HYPRE_LOAD_AX_UP_TO_11(t)

#define HYPRE_LOAD_AX_UP_TO_13(t) \
   HYPRE_LOAD_AX(t, 12) \
   HYPRE_LOAD_AX_UP_TO_12(t)

#define HYPRE_LOAD_AX_UP_TO_14(t) \
   HYPRE_LOAD_AX(t, 13) \
   HYPRE_LOAD_AX_UP_TO_13(t)

#define HYPRE_LOAD_AX_UP_TO_15(t) \
   HYPRE_LOAD_AX(t, 14) \
   HYPRE_LOAD_AX_UP_TO_14(t)

#define HYPRE_LOAD_AX_UP_TO_16(t) \
   HYPRE_LOAD_AX(t, 15) \
   HYPRE_LOAD_AX_UP_TO_15(t)

#define HYPRE_LOAD_AX_UP_TO_17(t) \
   HYPRE_LOAD_AX(t, 16) \
   HYPRE_LOAD_AX_UP_TO_16(t)

#define HYPRE_LOAD_AX_UP_TO_18(t) \
   HYPRE_LOAD_AX(t, 17) \
   HYPRE_LOAD_AX_UP_TO_17(t)

#define HYPRE_LOAD_AX_UP_TO_19(t) \
   HYPRE_LOAD_AX(t, 18) \
   HYPRE_LOAD_AX_UP_TO_18(t)

#define HYPRE_LOAD_AX_UP_TO_20(t) \
   HYPRE_LOAD_AX(t, 19) \
   HYPRE_LOAD_AX_UP_TO_19(t)

#define HYPRE_LOAD_AX_UP_TO_21(t) \
   HYPRE_LOAD_AX(t, 20) \
   HYPRE_LOAD_AX_UP_TO_20(t)

#define HYPRE_LOAD_AX_UP_TO_22(t) \
   HYPRE_LOAD_AX(t, 21) \
   HYPRE_LOAD_AX_UP_TO_21(t)

#define HYPRE_LOAD_AX_UP_TO_23(t) \
   HYPRE_LOAD_AX(t, 22) \
   HYPRE_LOAD_AX_UP_TO_22(t)

#define HYPRE_LOAD_AX_UP_TO_24(t) \
   HYPRE_LOAD_AX(t, 23) \
   HYPRE_LOAD_AX_UP_TO_23(t)

#define HYPRE_LOAD_AX_UP_TO_25(t) \
   HYPRE_LOAD_AX(t, 24) \
   HYPRE_LOAD_AX_UP_TO_24(t)

#define HYPRE_LOAD_AX_UP_TO_26(t) \
   HYPRE_LOAD_AX(t, 25) \
   HYPRE_LOAD_AX_UP_TO_25(t)

#define HYPRE_LOAD_AX_UP_TO_27(t) \
   HYPRE_LOAD_AX(t, 26) \
   HYPRE_LOAD_AX_UP_TO_26(t)

#define HYPRE_LOAD_CAX_UP_TO_1(t) \
   HYPRE_LOAD_CAX(t, 0) \

#define HYPRE_LOAD_CAX_UP_TO_2(t) \
   HYPRE_LOAD_CAX(t, 1) \
   HYPRE_LOAD_CAX_UP_TO_1(t)

#define HYPRE_LOAD_CAX_UP_TO_3(t) \
   HYPRE_LOAD_CAX(t, 2) \
   HYPRE_LOAD_CAX_UP_TO_2(t)

#define HYPRE_LOAD_CAX_UP_TO_4(t) \
   HYPRE_LOAD_CAX(t, 3) \
   HYPRE_LOAD_CAX_UP_TO_3(t)

#define HYPRE_LOAD_CAX_UP_TO_5(t) \
   HYPRE_LOAD_CAX(t, 4) \
   HYPRE_LOAD_CAX_UP_TO_4(t)

#define HYPRE_LOAD_CAX_UP_TO_6(t) \
   HYPRE_LOAD_CAX(t, 5) \
   HYPRE_LOAD_CAX_UP_TO_5(t)

#define HYPRE_LOAD_CAX_UP_TO_7(t) \
   HYPRE_LOAD_CAX(t, 6) \
   HYPRE_LOAD_CAX_UP_TO_6(t)

#define HYPRE_LOAD_CAX_UP_TO_8(t) \
   HYPRE_LOAD_CAX(t, 7) \
   HYPRE_LOAD_CAX_UP_TO_7(t)

#define HYPRE_LOAD_CAX_UP_TO_9(t) \
   HYPRE_LOAD_CAX(t, 8) \
   HYPRE_LOAD_CAX_UP_TO_8(t)

#define HYPRE_LOAD_CAX_UP_TO_10(t) \
   HYPRE_LOAD_CAX(t, 9) \
   HYPRE_LOAD_CAX_UP_TO_9(t)

#define HYPRE_LOAD_CAX_UP_TO_11(t) \
   HYPRE_LOAD_CAX(t, 10) \
   HYPRE_LOAD_CAX_UP_TO_10(t)

#define HYPRE_LOAD_CAX_UP_TO_12(t) \
   HYPRE_LOAD_CAX(t, 11) \
   HYPRE_LOAD_CAX_UP_TO_11(t)

#define HYPRE_LOAD_CAX_UP_TO_13(t) \
   HYPRE_LOAD_CAX(t, 12) \
   HYPRE_LOAD_CAX_UP_TO_12(t)

#define HYPRE_LOAD_CAX_UP_TO_14(t) \
   HYPRE_LOAD_CAX(t, 13) \
   HYPRE_LOAD_CAX_UP_TO_13(t)

#define HYPRE_LOAD_CAX_UP_TO_15(t) \
   HYPRE_LOAD_CAX(t, 14) \
   HYPRE_LOAD_CAX_UP_TO_14(t)

#define HYPRE_LOAD_CAX_UP_TO_16(t) \
   HYPRE_LOAD_CAX(t, 15) \
   HYPRE_LOAD_CAX_UP_TO_15(t)

#define HYPRE_LOAD_CAX_UP_TO_17(t) \
   HYPRE_LOAD_CAX(t, 16) \
   HYPRE_LOAD_CAX_UP_TO_16(t)

#define HYPRE_LOAD_CAX_UP_TO_18(t) \
   HYPRE_LOAD_CAX(t, 17) \
   HYPRE_LOAD_CAX_UP_TO_17(t)

#define HYPRE_LOAD_CAX_UP_TO_19(t) \
   HYPRE_LOAD_CAX(t, 18) \
   HYPRE_LOAD_CAX_UP_TO_18(t)

#define HYPRE_LOAD_CAX_UP_TO_20(t) \
   HYPRE_LOAD_CAX(t, 19) \
   HYPRE_LOAD_CAX_UP_TO_19(t)

#define HYPRE_LOAD_CAX_UP_TO_21(t) \
   HYPRE_LOAD_CAX(t, 20) \
   HYPRE_LOAD_CAX_UP_TO_20(t)

#define HYPRE_LOAD_CAX_UP_TO_22(t) \
   HYPRE_LOAD_CAX(t, 21) \
   HYPRE_LOAD_CAX_UP_TO_21(t)

#define HYPRE_LOAD_CAX_UP_TO_23(t) \
   HYPRE_LOAD_CAX(t, 22) \
   HYPRE_LOAD_CAX_UP_TO_22(t)

#define HYPRE_LOAD_CAX_UP_TO_24(t) \
   HYPRE_LOAD_CAX(t, 23) \
   HYPRE_LOAD_CAX_UP_TO_23(t)

#define HYPRE_LOAD_CAX_UP_TO_25(t) \
   HYPRE_LOAD_CAX(t, 24) \
   HYPRE_LOAD_CAX_UP_TO_24(t)

#define HYPRE_LOAD_CAX_UP_TO_26(t) \
   HYPRE_LOAD_CAX(t, 25) \
   HYPRE_LOAD_CAX_UP_TO_25(t)

#define HYPRE_LOAD_CAX_UP_TO_27(t) \
   HYPRE_LOAD_CAX(t, 26) \
   HYPRE_LOAD_CAX_UP_TO_26(t)

/* Sequence macros for various matrix/vector multiplication components */
#define HYPRE_CALC_AX_ADD_UP_TO_1 \
   HYPRE_CALC_AX(0) \

#define HYPRE_CALC_AX_ADD_UP_TO_2 \
   HYPRE_CALC_AX_ADD(1) \
   HYPRE_CALC_AX_ADD_UP_TO_1

#define HYPRE_CALC_AX_ADD_UP_TO_3 \
   HYPRE_CALC_AX_ADD(2) \
   HYPRE_CALC_AX_ADD_UP_TO_2

#define HYPRE_CALC_AX_ADD_UP_TO_4 \
   HYPRE_CALC_AX_ADD(3) \
   HYPRE_CALC_AX_ADD_UP_TO_3

#define HYPRE_CALC_AX_ADD_UP_TO_5 \
   HYPRE_CALC_AX_ADD(4) \
   HYPRE_CALC_AX_ADD_UP_TO_4

#define HYPRE_CALC_AX_ADD_UP_TO_6 \
   HYPRE_CALC_AX_ADD(5) \
   HYPRE_CALC_AX_ADD_UP_TO_5

#define HYPRE_CALC_AX_ADD_UP_TO_7 \
   HYPRE_CALC_AX_ADD(6) \
   HYPRE_CALC_AX_ADD_UP_TO_6

#define HYPRE_CALC_AX_ADD_UP_TO_8 \
   HYPRE_CALC_AX_ADD(7) \
   HYPRE_CALC_AX_ADD_UP_TO_7

#define HYPRE_CALC_AX_ADD_UP_TO_9 \
   HYPRE_CALC_AX_ADD(8) \
   HYPRE_CALC_AX_ADD_UP_TO_8

#define HYPRE_CALC_AX_ADD_UP_TO_10 \
   HYPRE_CALC_AX_ADD(9) \
   HYPRE_CALC_AX_ADD_UP_TO_9

#define HYPRE_CALC_AX_ADD_UP_TO_11 \
   HYPRE_CALC_AX_ADD(10) \
   HYPRE_CALC_AX_ADD_UP_TO_10

#define HYPRE_CALC_AX_ADD_UP_TO_12 \
   HYPRE_CALC_AX_ADD(11) \
   HYPRE_CALC_AX_ADD_UP_TO_11

#define HYPRE_CALC_AX_ADD_UP_TO_13 \
   HYPRE_CALC_AX_ADD(12) \
   HYPRE_CALC_AX_ADD_UP_TO_12

#define HYPRE_CALC_AX_ADD_UP_TO_14 \
   HYPRE_CALC_AX_ADD(13) \
   HYPRE_CALC_AX_ADD_UP_TO_13

#define HYPRE_CALC_AX_ADD_UP_TO_15 \
   HYPRE_CALC_AX_ADD(14) \
   HYPRE_CALC_AX_ADD_UP_TO_14

#define HYPRE_CALC_AX_ADD_UP_TO_16 \
   HYPRE_CALC_AX_ADD(15) \
   HYPRE_CALC_AX_ADD_UP_TO_15

#define HYPRE_CALC_AX_ADD_UP_TO_17 \
   HYPRE_CALC_AX_ADD(16) \
   HYPRE_CALC_AX_ADD_UP_TO_16

#define HYPRE_CALC_AX_ADD_UP_TO_18 \
   HYPRE_CALC_AX_ADD(17) \
   HYPRE_CALC_AX_ADD_UP_TO_17

#define HYPRE_CALC_AX_ADD_UP_TO_19 \
   HYPRE_CALC_AX_ADD(18) \
   HYPRE_CALC_AX_ADD_UP_TO_18

#define HYPRE_CALC_AX_ADD_UP_TO_20 \
   HYPRE_CALC_AX_ADD(19) \
   HYPRE_CALC_AX_ADD_UP_TO_19

#define HYPRE_CALC_AX_ADD_UP_TO_21 \
   HYPRE_CALC_AX_ADD(20) \
   HYPRE_CALC_AX_ADD_UP_TO_20

#define HYPRE_CALC_AX_ADD_UP_TO_22 \
   HYPRE_CALC_AX_ADD(21) \
   HYPRE_CALC_AX_ADD_UP_TO_21

#define HYPRE_CALC_AX_ADD_UP_TO_23 \
   HYPRE_CALC_AX_ADD(22) \
   HYPRE_CALC_AX_ADD_UP_TO_22

#define HYPRE_CALC_AX_ADD_UP_TO_24 \
   HYPRE_CALC_AX_ADD(23) \
   HYPRE_CALC_AX_ADD_UP_TO_23

#define HYPRE_CALC_AX_ADD_UP_TO_25 \
   HYPRE_CALC_AX_ADD(24) \
   HYPRE_CALC_AX_ADD_UP_TO_24

#define HYPRE_CALC_AX_ADD_UP_TO_26 \
   HYPRE_CALC_AX_ADD(25) \
   HYPRE_CALC_AX_ADD_UP_TO_25

#define HYPRE_CALC_AX_ADD_UP_TO_27 \
   HYPRE_CALC_AX_ADD(26) \
   HYPRE_CALC_AX_ADD_UP_TO_26

#define HYPRE_CALC_CAX_ADD_UP_TO_1 \
   HYPRE_CALC_CAX(0) \

#define HYPRE_CALC_CAX_ADD_UP_TO_2 \
   HYPRE_CALC_CAX_ADD(1) \
   HYPRE_CALC_CAX_ADD_UP_TO_1

#define HYPRE_CALC_CAX_ADD_UP_TO_3 \
   HYPRE_CALC_CAX_ADD(2) \
   HYPRE_CALC_CAX_ADD_UP_TO_2

#define HYPRE_CALC_CAX_ADD_UP_TO_4 \
   HYPRE_CALC_CAX_ADD(3) \
   HYPRE_CALC_CAX_ADD_UP_TO_3

#define HYPRE_CALC_CAX_ADD_UP_TO_5 \
   HYPRE_CALC_CAX_ADD(4) \
   HYPRE_CALC_CAX_ADD_UP_TO_4

#define HYPRE_CALC_CAX_ADD_UP_TO_6 \
   HYPRE_CALC_CAX_ADD(5) \
   HYPRE_CALC_CAX_ADD_UP_TO_5

#define HYPRE_CALC_CAX_ADD_UP_TO_7 \
   HYPRE_CALC_CAX_ADD(6) \
   HYPRE_CALC_CAX_ADD_UP_TO_6

#define HYPRE_CALC_CAX_ADD_UP_TO_8 \
   HYPRE_CALC_CAX_ADD(7) \
   HYPRE_CALC_CAX_ADD_UP_TO_7

#define HYPRE_CALC_CAX_ADD_UP_TO_9 \
   HYPRE_CALC_CAX_ADD(8) \
   HYPRE_CALC_CAX_ADD_UP_TO_8

#define HYPRE_CALC_CAX_ADD_UP_TO_10 \
   HYPRE_CALC_CAX_ADD(9) \
   HYPRE_CALC_CAX_ADD_UP_TO_9

#define HYPRE_CALC_CAX_ADD_UP_TO_11 \
   HYPRE_CALC_CAX_ADD(10) \
   HYPRE_CALC_CAX_ADD_UP_TO_10

#define HYPRE_CALC_CAX_ADD_UP_TO_12 \
   HYPRE_CALC_CAX_ADD(11) \
   HYPRE_CALC_CAX_ADD_UP_TO_11

#define HYPRE_CALC_CAX_ADD_UP_TO_13 \
   HYPRE_CALC_CAX_ADD(12) \
   HYPRE_CALC_CAX_ADD_UP_TO_12

#define HYPRE_CALC_CAX_ADD_UP_TO_14 \
   HYPRE_CALC_CAX_ADD(13) \
   HYPRE_CALC_CAX_ADD_UP_TO_13

#define HYPRE_CALC_CAX_ADD_UP_TO_15 \
   HYPRE_CALC_CAX_ADD(14) \
   HYPRE_CALC_CAX_ADD_UP_TO_14

#define HYPRE_CALC_CAX_ADD_UP_TO_16 \
   HYPRE_CALC_CAX_ADD(15) \
   HYPRE_CALC_CAX_ADD_UP_TO_15

#define HYPRE_CALC_CAX_ADD_UP_TO_17 \
   HYPRE_CALC_CAX_ADD(16) \
   HYPRE_CALC_CAX_ADD_UP_TO_16

#define HYPRE_CALC_CAX_ADD_UP_TO_18 \
   HYPRE_CALC_CAX_ADD(17) \
   HYPRE_CALC_CAX_ADD_UP_TO_17

#define HYPRE_CALC_CAX_ADD_UP_TO_19 \
   HYPRE_CALC_CAX_ADD(18) \
   HYPRE_CALC_CAX_ADD_UP_TO_18

#define HYPRE_CALC_CAX_ADD_UP_TO_20 \
   HYPRE_CALC_CAX_ADD(19) \
   HYPRE_CALC_CAX_ADD_UP_TO_19

#define HYPRE_CALC_CAX_ADD_UP_TO_21 \
   HYPRE_CALC_CAX_ADD(20) \
   HYPRE_CALC_CAX_ADD_UP_TO_20

#define HYPRE_CALC_CAX_ADD_UP_TO_22 \
   HYPRE_CALC_CAX_ADD(21) \
   HYPRE_CALC_CAX_ADD_UP_TO_21

#define HYPRE_CALC_CAX_ADD_UP_TO_23 \
   HYPRE_CALC_CAX_ADD(22) \
   HYPRE_CALC_CAX_ADD_UP_TO_22

#define HYPRE_CALC_CAX_ADD_UP_TO_24 \
   HYPRE_CALC_CAX_ADD(23) \
   HYPRE_CALC_CAX_ADD_UP_TO_23

#define HYPRE_CALC_CAX_ADD_UP_TO_25 \
   HYPRE_CALC_CAX_ADD(24) \
   HYPRE_CALC_CAX_ADD_UP_TO_24

#define HYPRE_CALC_CAX_ADD_UP_TO_26 \
   HYPRE_CALC_CAX_ADD(25) \
   HYPRE_CALC_CAX_ADD_UP_TO_25

#define HYPRE_CALC_CAX_ADD_UP_TO_27 \
   HYPRE_CALC_CAX_ADD(26) \
   HYPRE_CALC_CAX_ADD_UP_TO_26

/*--------------------------------------------------------------------------
 * z = beta * y + alpha * A*x
 *
 * StructMatrix/Vector multiplication core function for constant coeficients.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute_core_CC( HYPRE_Complex       alpha,
                                   hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   HYPRE_Complex       beta,
                                   hypre_StructVector *y,
                                   hypre_StructVector *z,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
                                   HYPRE_Int           zb,
                                   HYPRE_Int           transpose,
                                   HYPRE_Int           nentries,
                                   HYPRE_Int          *entries,
                                   hypre_IndexRef      start,
                                   hypre_IndexRef      stride,
                                   hypre_IndexRef      loop_size,
                                   hypre_IndexRef      xfstride,
                                   hypre_IndexRef      ran_stride,
                                   hypre_IndexRef      xdstride,
                                   hypre_IndexRef      ydstride,
                                   hypre_IndexRef      zdstride,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box,
                                   hypre_Box          *z_data_box)
{
#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,Ap13,Ap14,Ap15,Ap16,Ap17,Ap18,Ap19,Ap20,Ap21,Ap22,Ap23,Ap24,Ap25,Ap26)
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           xdstart, ydstart, zdstart;
   hypre_Index           offset;
   HYPRE_Int             si = 0, depth;
   HYPRE_Complex        *xp, *yp, *zp;
   HYPRE_DECLARE_OFFSETS_UP_TO_27;

   /* Exit early if no stencil entries fall in this category */
   if (!nentries)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("CC");

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);
   zp = hypre_StructVectorBoxData(z, zb);

   hypre_CopyToIndex(start, ndim, xdstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that xdstart is aligned with the vector data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because an offset will be used to index
    * into the vector x (xoff = index - xdstart). */
   hypre_SnapIndexNeg(xdstart, NULL, stride, ndim);
   hypre_MapToFineIndex(xdstart, NULL, xfstride, ndim);
   hypre_StructVectorMapDataIndex(x, xdstart);
   hypre_CopyToIndex(start, ndim, ydstart);
   hypre_MapToCoarseIndex(ydstart, NULL, ran_stride, ndim);
   hypre_CopyToIndex(start, ndim, zdstart);
   hypre_MapToCoarseIndex(zdstart, NULL, ran_stride, ndim);

   /* Initialize output vector in the first pass */
#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                                     \
   case n:                                                                     \
      HYPRE_LOAD_CAX_UP_TO_##n(transpose);                                     \
      hypre_BoxLoop3Begin(ndim, loop_size,                                     \
                          x_data_box, xdstart, xdstride, xi,                   \
                          y_data_box, ydstart, ydstride, yi,                   \
                          z_data_box, zdstart, zdstride, zi);                  \
      {                                                                        \
         zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_##n);      \
      }                                                                        \
      hypre_BoxLoop3End(xi, yi, zi);                                           \
      break;

   depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, nentries);
   switch (depth)
   {
         HYPRE_CORE_CASE(27);
         HYPRE_CORE_CASE(26);
         HYPRE_CORE_CASE(25);
         HYPRE_CORE_CASE(24);
         HYPRE_CORE_CASE(23);
         HYPRE_CORE_CASE(22);
         HYPRE_CORE_CASE(21);
         HYPRE_CORE_CASE(20);
         HYPRE_CORE_CASE(19);
         HYPRE_CORE_CASE(18);
         HYPRE_CORE_CASE(17);
         HYPRE_CORE_CASE(16);
         HYPRE_CORE_CASE(15);
         HYPRE_CORE_CASE(14);
         HYPRE_CORE_CASE(13);
         HYPRE_CORE_CASE(12);
         HYPRE_CORE_CASE(11);
         HYPRE_CORE_CASE(10);
         HYPRE_CORE_CASE(9);
         HYPRE_CORE_CASE(8);
         HYPRE_CORE_CASE(7);
         HYPRE_CORE_CASE(6);
         HYPRE_CORE_CASE(5);
         HYPRE_CORE_CASE(4);
         HYPRE_CORE_CASE(3);
         HYPRE_CORE_CASE(2);
         HYPRE_CORE_CASE(1);

      case 0:
         break;
   }

   /* Update output vector with remaining A*x components if any */
#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                                     \
   case n:                                                                     \
      HYPRE_LOAD_CAX_UP_TO_##n(transpose);                                     \
      hypre_BoxLoop2Begin(ndim, loop_size,                                     \
                          x_data_box, xdstart, xdstride, xi,                   \
                          z_data_box, zdstart, zdstride, zi);                  \
      {                                                                        \
         zp[zi] += alpha * (HYPRE_CALC_CAX_ADD_UP_TO_##n);                     \
      }                                                                        \
      hypre_BoxLoop2End(xi, zi);                                               \
      break;

   for (si = depth; si < nentries; si += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nentries - si));

      switch (depth)
      {
            HYPRE_CORE_CASE(27);
            HYPRE_CORE_CASE(26);
            HYPRE_CORE_CASE(25);
            HYPRE_CORE_CASE(24);
            HYPRE_CORE_CASE(23);
            HYPRE_CORE_CASE(22);
            HYPRE_CORE_CASE(21);
            HYPRE_CORE_CASE(20);
            HYPRE_CORE_CASE(19);
            HYPRE_CORE_CASE(18);
            HYPRE_CORE_CASE(17);
            HYPRE_CORE_CASE(16);
            HYPRE_CORE_CASE(15);
            HYPRE_CORE_CASE(14);
            HYPRE_CORE_CASE(13);
            HYPRE_CORE_CASE(12);
            HYPRE_CORE_CASE(11);
            HYPRE_CORE_CASE(10);
            HYPRE_CORE_CASE(9);
            HYPRE_CORE_CASE(8);
            HYPRE_CORE_CASE(7);
            HYPRE_CORE_CASE(6);
            HYPRE_CORE_CASE(5);
            HYPRE_CORE_CASE(4);
            HYPRE_CORE_CASE(3);
            HYPRE_CORE_CASE(2);
            HYPRE_CORE_CASE(1);

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * z = beta * y + alpha * A*x
 *
 * StructMatrix/Vector multiplication core routine for variable coeficients.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute_core_VC( HYPRE_Complex       alpha,
                                   hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   HYPRE_Complex       beta,
                                   hypre_StructVector *y,
                                   hypre_StructVector *z,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
                                   HYPRE_Int           zb,
                                   HYPRE_Int           transpose,
                                   HYPRE_Int           only_Ax,
                                   HYPRE_Int           nentries,
                                   HYPRE_Int          *entries,
                                   hypre_IndexRef      start,
                                   hypre_IndexRef      stride,
                                   hypre_IndexRef      loop_size,
                                   hypre_IndexRef      xfstride,
                                   hypre_IndexRef      ran_stride,
                                   hypre_IndexRef      Adstride,
                                   hypre_IndexRef      xdstride,
                                   hypre_IndexRef      ydstride,
                                   hypre_IndexRef      zdstride,
                                   hypre_Box          *A_data_box,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box,
                                   hypre_Box          *z_data_box )
{
#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,Ap13,Ap14,Ap15,Ap16,Ap17,Ap18,Ap19,Ap20,Ap21,Ap22,Ap23,Ap24,Ap25,Ap26)
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           Adstart, xdstart, ydstart, zdstart;
   hypre_Index           offset;
   HYPRE_Int             si = 0, depth;
   HYPRE_Complex        *xp, *yp, *zp;
   HYPRE_DECLARE_OFFSETS_UP_TO_27;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("VC");

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);
   zp = hypre_StructVectorBoxData(z, zb);

   hypre_CopyToIndex(start, ndim, Adstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that Adstart is aligned with the matrix data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because the coefficient pointer will be
    * offset (Ap += BoxOffsetDistance(index - Adstart)). */
   hypre_SnapIndexNeg(Adstart, NULL, stride, ndim);
   hypre_StructMatrixMapDataIndex(A, Adstart);
   hypre_CopyToIndex(start, ndim, xdstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that xdstart is aligned with the vector data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because an offset will be used to index
    * into the vector x (xoff = index - xdstart). */
   hypre_SnapIndexNeg(xdstart, NULL, stride, ndim);
   hypre_MapToFineIndex(xdstart, NULL, xfstride, ndim);
   hypre_StructVectorMapDataIndex(x, xdstart);
   hypre_CopyToIndex(start, ndim, ydstart);
   hypre_MapToCoarseIndex(ydstart, NULL, ran_stride, ndim);
   hypre_CopyToIndex(start, ndim, zdstart);
   hypre_MapToCoarseIndex(zdstart, NULL, ran_stride, ndim);

   /* Initialize output vector in the first pass */
#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                                     \
   case n:                                                                     \
      HYPRE_LOAD_AX_UP_TO_##n(transpose);                                      \
      hypre_BoxLoop4Begin(ndim, loop_size,                                     \
                          A_data_box, Adstart, Adstride, Ai,                   \
                          x_data_box, xdstart, xdstride, xi,                   \
                          y_data_box, ydstart, ydstride, yi,                   \
                          z_data_box, zdstart, zdstride, zi)                   \
      {                                                                        \
         zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_##n);       \
      }                                                                        \
      hypre_BoxLoop4End(Ai, xi, yi, zi)                                        \
      break;

   /* The flag "only_Ax" indicates that "beta * yp" has been computed previously (by the CC kernel) */
   depth = only_Ax ? 0 : hypre_min(HYPRE_UNROLL_MAXDEPTH, nentries);
   switch (depth)
   {
         HYPRE_CORE_CASE(27);
         HYPRE_CORE_CASE(26);
         HYPRE_CORE_CASE(25);
         HYPRE_CORE_CASE(24);
         HYPRE_CORE_CASE(23);
         HYPRE_CORE_CASE(22);
         HYPRE_CORE_CASE(21);
         HYPRE_CORE_CASE(20);
         HYPRE_CORE_CASE(19);
         HYPRE_CORE_CASE(18);
         HYPRE_CORE_CASE(17);
         HYPRE_CORE_CASE(16);
         HYPRE_CORE_CASE(15);
         HYPRE_CORE_CASE(14);
         HYPRE_CORE_CASE(13);
         HYPRE_CORE_CASE(12);
         HYPRE_CORE_CASE(11);
         HYPRE_CORE_CASE(10);
         HYPRE_CORE_CASE(9);
         HYPRE_CORE_CASE(8);
         HYPRE_CORE_CASE(7);
         HYPRE_CORE_CASE(6);
         HYPRE_CORE_CASE(5);
         HYPRE_CORE_CASE(4);
         HYPRE_CORE_CASE(3);
         HYPRE_CORE_CASE(2);
         HYPRE_CORE_CASE(1);

      case 0:
         break;
   }

   /* Update output vector with remaining A*x components if any */
#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                                     \
   case n:                                                                     \
      HYPRE_LOAD_AX_UP_TO_##n(transpose);                                      \
      hypre_BoxLoop3Begin(ndim, loop_size,                                     \
                          A_data_box, Adstart, Adstride, Ai,                   \
                          x_data_box, xdstart, xdstride, xi,                   \
                          z_data_box, zdstart, zdstride, zi);                  \
      {                                                                        \
         zp[zi] += alpha * (HYPRE_CALC_AX_ADD_UP_TO_##n);                      \
      }                                                                        \
      hypre_BoxLoop3End(Ai, xi, zi);                                           \
      break;

   for (si = depth; si < nentries; si += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nentries - si));

      switch (depth)
      {
            HYPRE_CORE_CASE(27);
            HYPRE_CORE_CASE(26);
            HYPRE_CORE_CASE(25);
            HYPRE_CORE_CASE(24);
            HYPRE_CORE_CASE(23);
            HYPRE_CORE_CASE(22);
            HYPRE_CORE_CASE(21);
            HYPRE_CORE_CASE(20);
            HYPRE_CORE_CASE(19);
            HYPRE_CORE_CASE(18);
            HYPRE_CORE_CASE(17);
            HYPRE_CORE_CASE(16);
            HYPRE_CORE_CASE(15);
            HYPRE_CORE_CASE(14);
            HYPRE_CORE_CASE(13);
            HYPRE_CORE_CASE(12);
            HYPRE_CORE_CASE(11);
            HYPRE_CORE_CASE(10);
            HYPRE_CORE_CASE(9);
            HYPRE_CORE_CASE(8);
            HYPRE_CORE_CASE(7);
            HYPRE_CORE_CASE(6);
            HYPRE_CORE_CASE(5);
            HYPRE_CORE_CASE(4);
            HYPRE_CORE_CASE(3);
            HYPRE_CORE_CASE(2);
            HYPRE_CORE_CASE(1);

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * z = beta * y + alpha * A*x
 *
 * StructMatrix/Vector multiplication core routine for one constant
 * and an arbitrary number of variable coefficients
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute_core_VCC( HYPRE_Complex       alpha,
                                    hypre_StructMatrix *A,
                                    hypre_StructVector *x,
                                    HYPRE_Complex       beta,
                                    hypre_StructVector *y,
                                    hypre_StructVector *z,
                                    HYPRE_Int           Ab,
                                    HYPRE_Int           xb,
                                    HYPRE_Int           yb,
                                    HYPRE_Int           zb,
                                    HYPRE_Int           transpose,
                                    HYPRE_Int           centry,
                                    HYPRE_Int           nentries,
                                    HYPRE_Int          *entries,
                                    hypre_IndexRef      start,
                                    hypre_IndexRef      stride,
                                    hypre_IndexRef      loop_size,
                                    hypre_IndexRef      xfstride,
                                    hypre_IndexRef      ran_stride,
                                    hypre_IndexRef      Adstride,
                                    hypre_IndexRef      xdstride,
                                    hypre_IndexRef      ydstride,
                                    hypre_IndexRef      zdstride,
                                    hypre_Box          *A_data_box,
                                    hypre_Box          *x_data_box,
                                    hypre_Box          *y_data_box,
                                    hypre_Box          *z_data_box )
{
#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,Ap13,Ap14,Ap15,Ap16,Ap17,Ap18,Ap19,Ap20,Ap21,Ap22,Ap23,Ap24,Ap25,Ap26)
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           Adstart, xdstart, ydstart, zdstart;
   hypre_Index           offset;
   HYPRE_Int             si = 0, depth;
   HYPRE_Complex        *xp, *yp, *zp;
   HYPRE_DECLARE_OFFSETS_UP_TO_27;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("VCC");

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);
   zp = hypre_StructVectorBoxData(z, zb);

   hypre_CopyToIndex(start, ndim, Adstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that Adstart is aligned with the matrix data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because the coefficient pointer will be
    * offset (Ap += BoxOffsetDistance(index - Adstart)). */
   hypre_SnapIndexNeg(Adstart, NULL, stride, ndim);
   hypre_StructMatrixMapDataIndex(A, Adstart);
   hypre_CopyToIndex(start, ndim, xdstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that xdstart is aligned with the vector data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because an offset will be used to index
    * into the vector x (xoff = index - xdstart). */
   hypre_SnapIndexNeg(xdstart, NULL, stride, ndim);
   hypre_MapToFineIndex(xdstart, NULL, xfstride, ndim);
   hypre_StructVectorMapDataIndex(x, xdstart);
   hypre_CopyToIndex(start, ndim, ydstart);
   hypre_MapToCoarseIndex(ydstart, NULL, ran_stride, ndim);
   hypre_CopyToIndex(start, ndim, zdstart);
   hypre_MapToCoarseIndex(zdstart, NULL, ran_stride, ndim);

   /* Initialize output vector in the first pass
       - Ap26 refers to the constant coefficient entries
       - Ap[0-25] refer to the variable coefficient entries */
#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                                     \
   case n:                                                                     \
      if (transpose) {                                                         \
         HYPRE_SET_AX_TRANS(Ap26, xoff26, centry)                              \
      } else {                                                                 \
         HYPRE_SET_AX(Ap26, xoff26, centry)                                    \
      }                                                                        \
      HYPRE_LOAD_AX_UP_TO_##n(transpose);                                      \
      hypre_BoxLoop4Begin(ndim, loop_size,                                     \
                          A_data_box, Adstart, Adstride, Ai,                   \
                          x_data_box, xdstart, xdstride, xi,                   \
                          y_data_box, ydstart, ydstride, yi,                   \
                          z_data_box, zdstart, zdstride, zi)                   \
      {                                                                        \
         zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD(26) + HYPRE_CALC_AX_ADD_UP_TO_##n); \
      }                                                                        \
      hypre_BoxLoop4End(Ai, xi, yi, zi)                                        \
      break;

   /* We use "-1" to reserve space for the constant coefficient entry */
   depth = hypre_min(HYPRE_UNROLL_MAXDEPTH - 1, nentries);
   switch (depth)
   {
         HYPRE_CORE_CASE(26);
         HYPRE_CORE_CASE(25);
         HYPRE_CORE_CASE(24);
         HYPRE_CORE_CASE(23);
         HYPRE_CORE_CASE(22);
         HYPRE_CORE_CASE(21);
         HYPRE_CORE_CASE(20);
         HYPRE_CORE_CASE(19);
         HYPRE_CORE_CASE(18);
         HYPRE_CORE_CASE(17);
         HYPRE_CORE_CASE(16);
         HYPRE_CORE_CASE(15);
         HYPRE_CORE_CASE(14);
         HYPRE_CORE_CASE(13);
         HYPRE_CORE_CASE(12);
         HYPRE_CORE_CASE(11);
         HYPRE_CORE_CASE(10);
         HYPRE_CORE_CASE(9);
         HYPRE_CORE_CASE(8);
         HYPRE_CORE_CASE(7);
         HYPRE_CORE_CASE(6);
         HYPRE_CORE_CASE(5);
         HYPRE_CORE_CASE(4);
         HYPRE_CORE_CASE(3);
         HYPRE_CORE_CASE(2);
         HYPRE_CORE_CASE(1);

      case 0:
         break;
   }

   /* Update output vector with remaining A*x components if any */
#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                                     \
   case n:                                                                     \
      HYPRE_LOAD_AX_UP_TO_##n(transpose);                                      \
      hypre_BoxLoop3Begin(ndim, loop_size,                                     \
                          A_data_box, Adstart, Adstride, Ai,                   \
                          x_data_box, xdstart, xdstride, xi,                   \
                          z_data_box, zdstart, zdstride, zi);                  \
      {                                                                        \
         zp[zi] += alpha * (HYPRE_CALC_AX_ADD_UP_TO_##n);                      \
      }                                                                        \
      hypre_BoxLoop3End(Ai, xi, zi);                                           \
      break;

   for (si = depth; si < nentries; si += (HYPRE_UNROLL_MAXDEPTH - 1))
   {
      depth = hypre_min((HYPRE_UNROLL_MAXDEPTH - 1), (nentries - si));

      switch (depth)
      {
            HYPRE_CORE_CASE(26);
            HYPRE_CORE_CASE(25);
            HYPRE_CORE_CASE(24);
            HYPRE_CORE_CASE(23);
            HYPRE_CORE_CASE(22);
            HYPRE_CORE_CASE(21);
            HYPRE_CORE_CASE(20);
            HYPRE_CORE_CASE(19);
            HYPRE_CORE_CASE(18);
            HYPRE_CORE_CASE(17);
            HYPRE_CORE_CASE(16);
            HYPRE_CORE_CASE(15);
            HYPRE_CORE_CASE(14);
            HYPRE_CORE_CASE(13);
            HYPRE_CORE_CASE(12);
            HYPRE_CORE_CASE(11);
            HYPRE_CORE_CASE(10);
            HYPRE_CORE_CASE(9);
            HYPRE_CORE_CASE(8);
            HYPRE_CORE_CASE(7);
            HYPRE_CORE_CASE(6);
            HYPRE_CORE_CASE(5);
            HYPRE_CORE_CASE(4);
            HYPRE_CORE_CASE(3);
            HYPRE_CORE_CASE(2);
            HYPRE_CORE_CASE(1);

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
