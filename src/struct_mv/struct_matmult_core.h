/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the struct matrix-matrix multiplication core functions
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MATMULT_CORE_HEADER
#define hypre_STRUCT_MATMULT_CORE_HEADER

#define HYPRE_SMMCORE_1T(i, k) \
   a[i[k]].mptr[Mi] +=         \
   a[i[k]].cprod*              \
   a[i[k]].tptrs[0][gi]*       \
   a[i[k]].tptrs[1][gi]*       \
   a[i[k]].tptrs[2][gi];

#define HYPRE_SMMCORE_1TB(i, o, k)                              \
   a[i[k]].mptr[Mi] +=                                          \
   a[i[k]].cprod*                                               \
   a[i[k]].tptrs[o[k][0]][gi]*                                  \
   a[i[k]].tptrs[o[k][1]][gi]*                                  \
   ((((HYPRE_Int) a[i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_1TBB(i, o, k)                             \
   a[i[k]].mptr[Mi] +=                                          \
   a[i[k]].cprod*                                               \
   a[i[k]].tptrs[o[k][0]][gi]*                                  \
   ((((HYPRE_Int) a[i[k]].tptrs[o[k][1]][gi]) >> o[k][1]) & 1)* \
   ((((HYPRE_Int) a[i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_1TBBB(i, k)                    \
    a[i[k]].mptr[Mi] +=                              \
    a[i[k]].cprod*                                   \
    ((((HYPRE_Int) a[i[k]].tptrs[0][gi]) >> 0) & 1)* \
    ((((HYPRE_Int) a[i[k]].tptrs[1][gi]) >> 1) & 1)* \
    ((((HYPRE_Int) a[i[k]].tptrs[2][gi]) >> 2) & 1);

#define HYPRE_SMMCORE_2T(i, o, k) \
    a[i[k]].mptr[Mi] +=           \
    a[i[k]].cprod*                \
    a[i[k]].tptrs[o[k][0]][gi]*   \
    a[i[k]].tptrs[o[k][1]][gi]*   \
    a[i[k]].tptrs[o[k][2]][hi];

#define HYPRE_SMMCORE_2T_V2(c, k)      \
    aptr = &a[k];                    \
    o0 = aptr->offsets[0];           \
    o1 = aptr->offsets[1];           \
    o2 = aptr->offsets[2];           \
    aptr->mptr[Mi] += c[k]* \
                      dptrs[0][o0 + gi]* \
                      dptrs[2][o2 + gi]* \
                      dptrs[1][o1 + hi];

#define HYPRE_SMMCORE_2TB(i, o, k)                               \
    a[i[k]].mptr[Mi] +=                                          \
    a[i[k]].cprod*                                               \
    a[i[k]].tptrs[o[k][0]][gi]*                                  \
    a[i[k]].tptrs[o[k][1]][hi]*                                  \
    ((((HYPRE_Int) a[i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_2ETB(i, o, k)                              \
    a[i[k]].mptr[Mi] +=                                          \
    a[i[k]].cprod*                                               \
    a[i[k]].tptrs[o[k][0]][hi]*                                  \
    a[i[k]].tptrs[o[k][1]][hi]*                                  \
    ((((HYPRE_Int) a[i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_2TBB(i, o, k)                              \
    a[i[k]].mptr[Mi] +=                                          \
    a[i[k]].cprod*                                               \
    a[i[k]].tptrs[o[k][0]][hi]*                                  \
    ((((HYPRE_Int) a[i[k]].tptrs[o[k][1]][gi]) >> o[k][1]) & 1)* \
    ((((HYPRE_Int) a[i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#endif
