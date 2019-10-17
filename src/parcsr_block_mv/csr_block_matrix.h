/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for CSR Block Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 * Note: everything is in terms of blocks (ie. num_rows is the number
 *       of block rows)
 *
 *****************************************************************************/

#ifndef hypre_CSR_BLOCK_MATRIX_HEADER
#define hypre_CSR_BLOCK_MATRIX_HEADER

#include "seq_mv.h"
#include "_hypre_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * CSR Block Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
  HYPRE_Complex    *data;
  HYPRE_Int        *i;
  HYPRE_Int        *j;
  HYPRE_BigInt     *big_j;
  HYPRE_Int         block_size;
  HYPRE_Int         num_rows;
  HYPRE_Int         num_cols;
  HYPRE_Int         num_nonzeros;
  HYPRE_Int         owns_data;

} hypre_CSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBlockMatrixData(matrix)         ((matrix) -> data)
#define hypre_CSRBlockMatrixI(matrix)            ((matrix) -> i)
#define hypre_CSRBlockMatrixJ(matrix)            ((matrix) -> j)
#define hypre_CSRBlockMatrixBigJ(matrix)         ((matrix) -> big_j)
#define hypre_CSRBlockMatrixBlockSize(matrix)    ((matrix) -> block_size)
#define hypre_CSRBlockMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define hypre_CSRBlockMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define hypre_CSRBlockMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define hypre_CSRBlockMatrixOwnsData(matrix)     ((matrix) -> owns_data)

/*--------------------------------------------------------------------------
 * other functions for the CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix 
      *hypre_CSRBlockMatrixCreate(HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixDestroy(hypre_CSRBlockMatrix *);
HYPRE_Int hypre_CSRBlockMatrixInitialize(hypre_CSRBlockMatrix *);
HYPRE_Int hypre_CSRBlockMatrixBigInitialize(hypre_CSRBlockMatrix *);
HYPRE_Int hypre_CSRBlockMatrixSetDataOwner(hypre_CSRBlockMatrix *, HYPRE_Int);
hypre_CSRMatrix 
      *hypre_CSRBlockMatrixCompress(hypre_CSRBlockMatrix *);
hypre_CSRMatrix 
      *hypre_CSRBlockMatrixConvertToCSRMatrix(hypre_CSRBlockMatrix *);
hypre_CSRBlockMatrix
      *hypre_CSRBlockMatrixConvertFromCSRMatrix(hypre_CSRMatrix *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAdd(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Complex*, HYPRE_Int);

HYPRE_Int hypre_CSRBlockMatrixBlockMultAdd(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Complex, HYPRE_Complex *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiag(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Complex, HYPRE_Complex *, HYPRE_Int);
HYPRE_Int
hypre_CSRBlockMatrixBlockMultAddDiag2(HYPRE_Complex* i1, HYPRE_Complex* i2, HYPRE_Complex beta, 
                                      HYPRE_Complex* o, HYPRE_Int block_size);
HYPRE_Int
hypre_CSRBlockMatrixBlockMultAddDiag3(HYPRE_Complex* i1, HYPRE_Complex* i2, HYPRE_Complex beta, 
                                      HYPRE_Complex* o, HYPRE_Int block_size);
   

HYPRE_Int hypre_CSRBlockMatrixBlockInvMult(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Complex *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockInvMultDiag(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Complex *, HYPRE_Int);

HYPRE_Int
hypre_CSRBlockMatrixBlockInvMultDiag2(HYPRE_Complex* i1, HYPRE_Complex* i2, HYPRE_Complex* o, HYPRE_Int block_size);
   
HYPRE_Int
hypre_CSRBlockMatrixBlockInvMultDiag3(HYPRE_Complex* i1, HYPRE_Complex* i2, HYPRE_Complex* o, HYPRE_Int block_size);
   



HYPRE_Int hypre_CSRBlockMatrixBlockMultInv(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Complex *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockTranspose(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Int);

HYPRE_Int hypre_CSRBlockMatrixTranspose(hypre_CSRBlockMatrix *A,
                                  hypre_CSRBlockMatrix **AT, HYPRE_Int data);

HYPRE_Int hypre_CSRBlockMatrixBlockCopyData(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockCopyDataDiag(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex, HYPRE_Int);

HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulate(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulateDiag(HYPRE_Complex* i1, HYPRE_Complex* o, HYPRE_Int block_size);
   


HYPRE_Int
hypre_CSRBlockMatrixMatvec(HYPRE_Complex alpha, hypre_CSRBlockMatrix *A,
                           hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y);
   

HYPRE_Int
hypre_CSRBlockMatrixMatvecT( HYPRE_Complex alpha, hypre_CSRBlockMatrix *A, hypre_Vector  *x,
                             HYPRE_Complex beta, hypre_Vector *y );

HYPRE_Int
hypre_CSRBlockMatrixBlockInvMatvec(HYPRE_Complex* mat, HYPRE_Complex* v, 
                                   HYPRE_Complex* ov, HYPRE_Int block_size);
   
HYPRE_Int 
hypre_CSRBlockMatrixBlockMatvec(HYPRE_Complex alpha, HYPRE_Complex* mat, HYPRE_Complex* v, HYPRE_Complex beta, 
                                HYPRE_Complex* ov, HYPRE_Int block_size);
   

HYPRE_Int hypre_CSRBlockMatrixBlockNorm(HYPRE_Int norm_type, HYPRE_Complex* data, HYPRE_Real* out, HYPRE_Int block_size);
   
HYPRE_Int hypre_CSRBlockMatrixBlockSetScalar(HYPRE_Complex* o, HYPRE_Complex beta, HYPRE_Int block_size);
   
HYPRE_Int hypre_CSRBlockMatrixComputeSign(HYPRE_Complex *i1, HYPRE_Complex *o, HYPRE_Int block_size);
HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(HYPRE_Complex* i1, HYPRE_Complex* o, HYPRE_Int block_size, HYPRE_Real *sign);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(HYPRE_Complex* i1, HYPRE_Complex* i2, HYPRE_Complex beta, HYPRE_Complex* o, HYPRE_Int block_size, HYPRE_Real *sign);

#ifdef __cplusplus
}
#endif
#endif
