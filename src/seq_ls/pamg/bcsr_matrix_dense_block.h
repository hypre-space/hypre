/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/

/*****************************************************************************
 *
 * This code implements a class for a dense block of a compressed sparse row
 * matrix.
 *
 *****************************************************************************/

#ifndef hypre_BCSR_MATRIX_DENSE_BLOCK_HEADER
#define hypre_BCSR_MATRIX_DENSE_BLOCK_HEADER

typedef struct {
  double* data;
  HYPRE_Int num_rows;
  HYPRE_Int num_cols;
} hypre_BCSRMatrixDenseBlock;

/*****************************************************************************
 *
 * Prototypes
 *
 *****************************************************************************/

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCreate(HYPRE_Int num_rows, HYPRE_Int num_cols);

HYPRE_Int
hypre_BCSRMatrixDenseBlockDestroy(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockInitialise(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockFillData(hypre_BCSRMatrixDenseBlock* A,
				   double* data);

HYPRE_Int
hypre_BCSRMatrixDenseBlockGetData(hypre_BCSRMatrixDenseBlock* A,
				   double* data);

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCopy(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockAdd(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);

HYPRE_Int
hypre_BCSRMatrixDenseBlockMultiply(hypre_BCSRMatrixDenseBlock* A,
				   hypre_BCSRMatrixDenseBlock* B);

HYPRE_Int
hypre_BCSRMatrixDenseBlockNeg(hypre_BCSRMatrixDenseBlock* A);

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockDiag(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockMulInv(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);

HYPRE_Int
hypre_BCSRMatrixDenseBlockMultiplyInverse2(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);


HYPRE_Int
hypre_BCSRMatrixDenseBlockTranspose(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixBlockMatvec(double alpha, hypre_BCSRMatrixDenseBlock* A,
			    double* x_data, double beta, double* y_data);

HYPRE_Int
hypre_BCSRMatrixBlockMatvecT(double alpha, hypre_BCSRMatrixDenseBlock* A,
			     double* x_data, double beta, double* y_data);

double
hypre_BCSRMatrixDenseBlockNorm(hypre_BCSRMatrixDenseBlock* A,
			       const char* norm);

HYPRE_Int
hypre_BCSRMatrixDenseBlockPrint(hypre_BCSRMatrixDenseBlock* A,
				FILE* out_file);

#ifdef hypre_BCSR_MATRIX_USE_DENSE_BLOCKS

#define hypre_BCSRMatrixBlock hypre_BCSRMatrixDenseBlock
#define hypre_BCSRMatrixBlockCreate hypre_BCSRMatrixDenseBlockCreate
#define hypre_BCSRMatrixBlockDestroy hypre_BCSRMatrixDenseBlockDestroy
#define hypre_BCSRMatrixBlockInitialise hypre_BCSRMatrixDenseBlockInitialise
#define hypre_BCSRMatrixBlockFillData hypre_BCSRMatrixDenseBlockFillData
#define hypre_BCSRMatrixBlockGetData hypre_BCSRMatrixDenseBlockGetData
#define hypre_BCSRMatrixBlockCopy hypre_BCSRMatrixDenseBlockCopy
#define hypre_BCSRMatrixBlockAdd hypre_BCSRMatrixDenseBlockAdd
#define hypre_BCSRMatrixBlockMultiply hypre_BCSRMatrixDenseBlockMultiply
#define hypre_BCSRMatrixBlockNeg hypre_BCSRMatrixDenseBlockNeg
#define hypre_BCSRMatrixBlockDiag hypre_BCSRMatrixDenseBlockDiag
#define hypre_BCSRMatrixBlockMulInv hypre_BCSRMatrixDenseBlockMulInv
#define hypre_BCSRMatrixBlockMultiplyInverse2 hypre_BCSRMatrixDenseBlockMultiplyInverse2
#define hypre_BCSRMatrixBlockTranspose hypre_BCSRMatrixDenseBlockTranspose
#define hypre_BCSRMatrixBlockMatvec hypre_BCSRMatrixDenseBlockMatvec
#define hypre_BCSRMatrixBlockMatvecT hypre_BCSRMatrixDenseBlockMatvecT
#define hypre_BCSRMatrixBlockNorm hypre_BCSRMatrixDenseBlockNorm
#define hypre_BCSRMatrixBlockPrint hypre_BCSRMatrixDenseBlockPrint

#endif

#endif
