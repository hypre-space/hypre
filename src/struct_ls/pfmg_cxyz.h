/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#if defined(HYPRE_UNROLL_MAXDEPTH)
#undef HYPRE_UNROLL_MAXDEPTH
#endif
#define HYPRE_UNROLL_MAXDEPTH 18

#define HYPRE_CXYZ_DEFINE_SIGN         \
  HYPRE_Real sign = diag_is_constant ? \
    (A_diag[0]  < 0.0 ? 1.0 : -1.0) :  \
    (A_diag[Ai] < 0.0 ? 1.0 : -1.0)

#define HYPRE_AP_DECLARE(n)     \
  HYPRE_Real *Ap##n##_d = NULL, *Ap##n##_0 = NULL, *Ap##n##_1 = NULL, *Ap##n##_2 = NULL

#define HYPRE_AP_LOAD(n, d)     \
  Ap##n##_##d = hypre_StructMatrixBoxData(A, Ab, entries[d][k + n])

#define HYPRE_CAP_LOAD(n, d)    \
  Ap##n##_##d = hypre_StructMatrixConstData(A, entries[d][k + n])

#define HYPRE_AP_EVAL(n, d)     \
  Ap##n##_##d[Ai]

#define HYPRE_CAP_EVAL(n, d)    \
  Ap##n##_##d[0]

#define HYPRE_AP_DECLARE_UP_TO_9  \
  HYPRE_AP_DECLARE(0);            \
  HYPRE_AP_DECLARE(1);            \
  HYPRE_AP_DECLARE(2);            \
  HYPRE_AP_DECLARE(3);            \
  HYPRE_AP_DECLARE(4);            \
  HYPRE_AP_DECLARE(5);            \
  HYPRE_AP_DECLARE(6);            \
  HYPRE_AP_DECLARE(7);            \
  HYPRE_AP_DECLARE(8)

#define HYPRE_AP_DECLARE_UP_TO_10 \
  HYPRE_AP_DECLARE_UP_TO_9;       \
  HYPRE_AP_DECLARE(9)

#define HYPRE_AP_DECLARE_UP_TO_18 \
  HYPRE_AP_DECLARE_UP_TO_10;      \
  HYPRE_AP_DECLARE(10);           \
  HYPRE_AP_DECLARE(11);           \
  HYPRE_AP_DECLARE(12);           \
  HYPRE_AP_DECLARE(13);           \
  HYPRE_AP_DECLARE(14);           \
  HYPRE_AP_DECLARE(15);           \
  HYPRE_AP_DECLARE(16);           \
  HYPRE_AP_DECLARE(17)

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

#define HYPRE_AP_LOAD_UP_TO_10(d) \
  HYPRE_AP_LOAD_UP_TO_9(d);       \
  HYPRE_AP_LOAD(9, d)

#define HYPRE_AP_LOAD_UP_TO_11(d) \
  HYPRE_AP_LOAD_UP_TO_10(d);      \
  HYPRE_AP_LOAD(10, d)

#define HYPRE_AP_LOAD_UP_TO_12(d) \
  HYPRE_AP_LOAD_UP_TO_11(d);      \
  HYPRE_AP_LOAD(11, d)

#define HYPRE_AP_LOAD_UP_TO_13(d) \
  HYPRE_AP_LOAD_UP_TO_12(d);      \
  HYPRE_AP_LOAD(12, d)

#define HYPRE_AP_LOAD_UP_TO_14(d) \
  HYPRE_AP_LOAD_UP_TO_13(d);      \
  HYPRE_AP_LOAD(13, d)

#define HYPRE_AP_LOAD_UP_TO_15(d) \
  HYPRE_AP_LOAD_UP_TO_14(d);      \
  HYPRE_AP_LOAD(14, d)

#define HYPRE_AP_LOAD_UP_TO_16(d) \
  HYPRE_AP_LOAD_UP_TO_15(d);      \
  HYPRE_AP_LOAD(15, d)

#define HYPRE_AP_LOAD_UP_TO_17(d) \
  HYPRE_AP_LOAD_UP_TO_16(d);      \
  HYPRE_AP_LOAD(16, d)

#define HYPRE_AP_LOAD_UP_TO_18(d) \
  HYPRE_AP_LOAD_UP_TO_17(d);      \
  HYPRE_AP_LOAD(17, d)

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

#define HYPRE_CAP_LOAD_UP_TO_10(d) \
  HYPRE_CAP_LOAD_UP_TO_9(d);       \
  HYPRE_CAP_LOAD(9, d)

#define HYPRE_CAP_LOAD_UP_TO_11(d) \
  HYPRE_CAP_LOAD_UP_TO_10(d);      \
  HYPRE_CAP_LOAD(10, d)

#define HYPRE_CAP_LOAD_UP_TO_12(d) \
  HYPRE_CAP_LOAD_UP_TO_11(d);      \
  HYPRE_CAP_LOAD(11, d)

#define HYPRE_CAP_LOAD_UP_TO_13(d) \
  HYPRE_CAP_LOAD_UP_TO_12(d);      \
  HYPRE_CAP_LOAD(12, d)

#define HYPRE_CAP_LOAD_UP_TO_14(d) \
  HYPRE_CAP_LOAD_UP_TO_13(d);      \
  HYPRE_CAP_LOAD(13, d)

#define HYPRE_CAP_LOAD_UP_TO_15(d) \
  HYPRE_CAP_LOAD_UP_TO_14(d);      \
  HYPRE_CAP_LOAD(14, d)

#define HYPRE_CAP_LOAD_UP_TO_16(d) \
  HYPRE_CAP_LOAD_UP_TO_15(d);      \
  HYPRE_CAP_LOAD(15, d)

#define HYPRE_CAP_LOAD_UP_TO_17(d) \
  HYPRE_CAP_LOAD_UP_TO_16(d);      \
  HYPRE_CAP_LOAD(16, d)

#define HYPRE_CAP_LOAD_UP_TO_18(d) \
  HYPRE_CAP_LOAD_UP_TO_17(d);      \
  HYPRE_CAP_LOAD(17, d)

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

#define HYPRE_AP_SUM_UP_TO_10(d) \
  HYPRE_AP_SUM_UP_TO_9(d) +      \
  HYPRE_AP_EVAL(9, d)

#define HYPRE_AP_SUM_UP_TO_11(d) \
  HYPRE_AP_SUM_UP_TO_10(d) +     \
  HYPRE_AP_EVAL(10, d)

#define HYPRE_AP_SUM_UP_TO_12(d) \
  HYPRE_AP_SUM_UP_TO_11(d) +     \
  HYPRE_AP_EVAL(11, d)

#define HYPRE_AP_SUM_UP_TO_13(d) \
  HYPRE_AP_SUM_UP_TO_12(d) +     \
  HYPRE_AP_EVAL(12, d)

#define HYPRE_AP_SUM_UP_TO_14(d) \
  HYPRE_AP_SUM_UP_TO_13(d) +     \
  HYPRE_AP_EVAL(13, d)

#define HYPRE_AP_SUM_UP_TO_15(d) \
  HYPRE_AP_SUM_UP_TO_14(d) +     \
  HYPRE_AP_EVAL(14, d)

#define HYPRE_AP_SUM_UP_TO_16(d) \
  HYPRE_AP_SUM_UP_TO_15(d) +     \
  HYPRE_AP_EVAL(15, d)

#define HYPRE_AP_SUM_UP_TO_17(d) \
  HYPRE_AP_SUM_UP_TO_16(d) +     \
  HYPRE_AP_EVAL(16, d)

#define HYPRE_AP_SUM_UP_TO_18(d) \
  HYPRE_AP_SUM_UP_TO_17(d) +     \
  HYPRE_AP_EVAL(17, d)

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

#define HYPRE_CAP_SUM_UP_TO_10(d) \
  HYPRE_CAP_SUM_UP_TO_9(d) +      \
  HYPRE_CAP_EVAL(9, d)

#define HYPRE_CAP_SUM_UP_TO_11(d) \
  HYPRE_CAP_SUM_UP_TO_10(d) +     \
  HYPRE_CAP_EVAL(10, d)

#define HYPRE_CAP_SUM_UP_TO_12(d) \
  HYPRE_CAP_SUM_UP_TO_11(d) +     \
  HYPRE_CAP_EVAL(11, d)

#define HYPRE_CAP_SUM_UP_TO_13(d) \
  HYPRE_CAP_SUM_UP_TO_12(d) +     \
  HYPRE_CAP_EVAL(12, d)

#define HYPRE_CAP_SUM_UP_TO_14(d) \
  HYPRE_CAP_SUM_UP_TO_13(d) +     \
  HYPRE_CAP_EVAL(13, d)

#define HYPRE_CAP_SUM_UP_TO_15(d) \
  HYPRE_CAP_SUM_UP_TO_14(d) +     \
  HYPRE_CAP_EVAL(14, d)

#define HYPRE_CAP_SUM_UP_TO_16(d) \
  HYPRE_CAP_SUM_UP_TO_15(d) +     \
  HYPRE_CAP_EVAL(15, d)

#define HYPRE_CAP_SUM_UP_TO_17(d) \
  HYPRE_CAP_SUM_UP_TO_16(d) +     \
  HYPRE_CAP_EVAL(16, d)

#define HYPRE_CAP_SUM_UP_TO_18(d) \
  HYPRE_CAP_SUM_UP_TO_17(d) +     \
  HYPRE_CAP_EVAL(17, d)
