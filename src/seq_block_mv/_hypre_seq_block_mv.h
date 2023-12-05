/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*** DO NOT EDIT THIS FILE DIRECTLY (use 'headers' to generate) ***/

#ifndef hypre_SEQ_BLOCK_MV_HEADER
#define hypre_SEQ_BLOCK_MV_HEADER

#include <HYPRE_config.h>
#include "seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef hypre_DENSE_BLOCK_MATRIX_HEADER
#define hypre_DENSE_BLOCK_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Uniformly blocked dense matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_DenseBlockMatrix_struct
{
   HYPRE_Int             row_major;          /* Flag indicating storage format (false: col major)*/
   HYPRE_Int             num_rows;           /* Number of rows of entire matrix */
   HYPRE_Int             num_cols;           /* Number of columns of entire matrix */
   HYPRE_Int             num_coefs;          /* Number of coefficients of entire matrix */
   HYPRE_Int             num_blocks;         /* Number of sub-matrices (blocks) */

   /* Local info for a individual block (sub-matrix) */
   HYPRE_Int             num_rows_block;     /* Number of rows per block */
   HYPRE_Int             num_cols_block;     /* Number of columns per block */
   HYPRE_Int             num_coefs_block;    /* Number of coefficients per block */

   /* Matrix coefficients array */
   HYPRE_Int             owns_data;          /* Flag indicating ownership of the data array */
   HYPRE_Complex        *data;               /* Matrix coefficients */
   HYPRE_Complex       **data_aop;           /* Array of pointers to data */
   HYPRE_MemoryLocation  memory_location;    /* Memory location of data array */
} hypre_DenseBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the uniform batched matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_DenseBlockMatrixRowMajor(matrix)        ((matrix) -> row_major)
#define hypre_DenseBlockMatrixNumRows(matrix)         ((matrix) -> num_rows)
#define hypre_DenseBlockMatrixNumCols(matrix)         ((matrix) -> num_cols)
#define hypre_DenseBlockMatrixNumCoefs(matrix)        ((matrix) -> num_coefs)
#define hypre_DenseBlockMatrixNumBlocks(matrix)       ((matrix) -> num_blocks)

#define hypre_DenseBlockMatrixNumRowsBlock(matrix)    ((matrix) -> num_rows_block)
#define hypre_DenseBlockMatrixNumColsBlock(matrix)    ((matrix) -> num_cols_block)
#define hypre_DenseBlockMatrixNumCoefsBlock(matrix)   ((matrix) -> num_coefs_block)

#define hypre_DenseBlockMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_DenseBlockMatrixData(matrix)            ((matrix) -> data)
#define hypre_DenseBlockMatrixDataAOP(matrix)         ((matrix) -> data_aop)
#define hypre_DenseBlockMatrixMemoryLocation(matrix)  ((matrix) -> memory_location)

#endif

/* dense_block_matrix.c */
hypre_DenseBlockMatrix* hypre_DenseBlockMatrixCreate(HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int);
hypre_DenseBlockMatrix* hypre_DenseBlockMatrixCreateByBlock(HYPRE_Int, HYPRE_Int, HYPRE_Int);
HYPRE_Int hypre_DenseBlockMatrixDestroy(hypre_DenseBlockMatrix*);
HYPRE_Int hypre_DenseBlockMatrixInitializeOn(hypre_DenseBlockMatrix*, HYPRE_MemoryLocation);
HYPRE_Int hypre_DenseBlockMatrixInitialize(hypre_DenseBlockMatrix*);
HYPRE_Int hypre_DenseBlockMatrixCopy(hypre_DenseBlockMatrix*, hypre_DenseBlockMatrix*);
hypre_DenseBlockMatrix* hypre_DenseBlockMatrixClone(hypre_DenseBlockMatrix*, HYPRE_Int);

#ifdef __cplusplus
}
#endif

#endif
