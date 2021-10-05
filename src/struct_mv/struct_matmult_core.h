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

#define HYPRE_SMMCORE_1T(e, i, k) \
   a[e[k]][i[k]].mptr[Mi] +=      \
   a[e[k]][i[k]].cprod*           \
   a[e[k]][i[k]].tptrs[0][gi]*    \
   a[e[k]][i[k]].tptrs[1][gi]*    \
   a[e[k]][i[k]].tptrs[2][gi];

#define HYPRE_SMMCORE_1TB(e, i, o, k)                                 \
   a[e[k]][i[k]].mptr[Mi] +=                                          \
   a[e[k]][i[k]].cprod*                                               \
   a[e[k]][i[k]].tptrs[o[k][0]][gi]*                                  \
   a[e[k]][i[k]].tptrs[o[k][1]][gi]*                                  \
   ((((HYPRE_Int) a[e[k]][i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_1TBB(e, i, o, k)                                \
   a[e[k]][i[k]].mptr[Mi] +=                                          \
   a[e[k]][i[k]].cprod*                                               \
   a[e[k]][i[k]].tptrs[o[k][0]][gi]*                                  \
   ((((HYPRE_Int) a[e[k]][i[k]].tptrs[o[k][1]][gi]) >> o[k][1]) & 1)* \
   ((((HYPRE_Int) a[e[k]][i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_1TBBB(e, i, k)                       \
    a[e[k]][i[k]].mptr[Mi] +=                              \
    a[e[k]][i[k]].cprod*                                   \
    ((((HYPRE_Int) a[e[k]][i[k]].tptrs[0][gi]) >> 0) & 1)* \
    ((((HYPRE_Int) a[e[k]][i[k]].tptrs[1][gi]) >> 1) & 1)* \
    ((((HYPRE_Int) a[e[k]][i[k]].tptrs[2][gi]) >> 2) & 1);

#define HYPRE_SMMCORE_2T(e, i, o, k)    \
    a[e[k]][i[k]].mptr[Mi] +=           \
    a[e[k]][i[k]].cprod*                \
    a[e[k]][i[k]].tptrs[o[k][0]][gi]*   \
    a[e[k]][i[k]].tptrs[o[k][1]][gi]*   \
    a[e[k]][i[k]].tptrs[o[k][2]][hi];

#define HYPRE_SMMCORE_2TB(e, i, o, k)                                  \
    a[e[k]][i[k]].mptr[Mi] +=                                          \
    a[e[k]][i[k]].cprod*                                               \
    a[e[k]][i[k]].tptrs[o[k][0]][gi]*                                  \
    a[e[k]][i[k]].tptrs[o[k][1]][hi]*                                  \
    ((((HYPRE_Int) a[e[k]][i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_2ETB(e, i, o, k)                                 \
    a[e[k]][i[k]].mptr[Mi] +=                                          \
    a[e[k]][i[k]].cprod*                                               \
    a[e[k]][i[k]].tptrs[o[k][0]][hi]*                                  \
    a[e[k]][i[k]].tptrs[o[k][1]][hi]*                                  \
    ((((HYPRE_Int) a[e[k]][i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#define HYPRE_SMMCORE_2TBB(e, i, o, k)                                 \
    a[e[k]][i[k]].mptr[Mi] +=                                          \
    a[e[k]][i[k]].cprod*                                               \
    a[e[k]][i[k]].tptrs[o[k][0]][hi]*                                  \
    ((((HYPRE_Int) a[e[k]][i[k]].tptrs[o[k][1]][gi]) >> o[k][1]) & 1)* \
    ((((HYPRE_Int) a[e[k]][i[k]].tptrs[o[k][2]][gi]) >> o[k][2]) & 1);

#endif
