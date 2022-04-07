/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef GET_ROW_DH
#define GET_ROW_DH

/* #include "euclid_common.h" */

/* "row" refers to global row number */

extern void EuclidGetDimensions(void *A, HYPRE_Int *beg_row, HYPRE_Int *rowsLocal, HYPRE_Int *rowsGlobal);
extern void EuclidGetRow(void *A, HYPRE_Int row, HYPRE_Int *len, HYPRE_Int **ind, HYPRE_Real **val);
extern void EuclidRestoreRow(void *A, HYPRE_Int row, HYPRE_Int *len, HYPRE_Int **ind, HYPRE_Real **val);

extern HYPRE_Int EuclidReadLocalNz(void *A);

extern void PrintMatUsingGetRow(void* A, HYPRE_Int beg_row, HYPRE_Int m,
                          HYPRE_Int *n2o_row, HYPRE_Int *n2o_col, char *filename);


#endif

