/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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

#define HYPRE_SMMCORE_1D(k)      \
   cprod[k]*         \
   tptrs[k][0][gi]*  \
   tptrs[k][1][gi]

#define HYPRE_SMMCORE_1DB(k) \
   cprod[k]*                 \
   tptrs[k][0][gi]*          \
   ((((HYPRE_Int) tptrs[k][1][gi]) >> order[k][1]) & 1)

#define HYPRE_SMMCORE_1DBB(k) \
   cprod[k]*                  \
   ((((HYPRE_Int) tptrs[k][0][gi]) >> 0) & 1)* \
   ((((HYPRE_Int) tptrs[k][1][gi]) >> 1) & 1)

#define HYPRE_SMMCORE_2D(k) \
    cprod[k]*               \
    tptrs[k][0][gi]*        \
    tptrs[k][1][gi]

#define HYPRE_SMMCORE_2DB(k) \
    cprod[k]*                \
    tptrs[k][0][hi]*         \
    ((((HYPRE_Int) tptrs[k][1][gi]) >> order[k][1]) & 1)

#define HYPRE_SMMCORE_1T(k)      \
   cprod[k]*         \
   tptrs[k][0][gi]*  \
   tptrs[k][1][gi]*  \
   tptrs[k][2][gi]

#define HYPRE_SMMCORE_1TB(k) \
   cprod[k]*                 \
   tptrs[k][0][gi]*          \
   tptrs[k][1][gi]*          \
   ((((HYPRE_Int) tptrs[k][2][gi]) >> order[k][2]) & 1)

#define HYPRE_SMMCORE_1TBB(k) \
   cprod[k]*                  \
   tptrs[k][0][gi]*           \
   ((((HYPRE_Int) tptrs[k][1][gi]) >> order[k][1]) & 1)* \
   ((((HYPRE_Int) tptrs[k][2][gi]) >> order[k][2]) & 1)

#define HYPRE_SMMCORE_1TBBB(k) \
    cprod[k]*                  \
    ((((HYPRE_Int) tptrs[k][0][gi]) >> 0) & 1)* \
    ((((HYPRE_Int) tptrs[k][1][gi]) >> 1) & 1)* \
    ((((HYPRE_Int) tptrs[k][2][gi]) >> 2) & 1)

#define HYPRE_SMMCORE_2T(k) \
    cprod[k]*               \
    tptrs[k][0][gi]*        \
    tptrs[k][1][gi]*        \
    tptrs[k][2][hi]

#define HYPRE_SMMCORE_2TB(k) \
    cprod[k]*                \
    tptrs[k][0][gi]*         \
    tptrs[k][1][hi]*         \
    ((((HYPRE_Int) tptrs[k][2][gi]) >> order[k][2]) & 1)

#define HYPRE_SMMCORE_2ETB(k) \
    cprod[k]*                 \
    tptrs[k][0][hi]*          \
    tptrs[k][1][hi]*          \
    ((((HYPRE_Int) tptrs[k][2][gi]) >> order[k][2]) & 1)

#define HYPRE_SMMCORE_2TBB(k) \
    cprod[k]*                 \
    tptrs[k][0][hi]*          \
    ((((HYPRE_Int) tptrs[k][1][gi]) >> order[k][1]) & 1)* \
    ((((HYPRE_Int) tptrs[k][2][gi]) >> order[k][2]) & 1)

#endif