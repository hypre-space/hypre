/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.13 $
 ***********************************************************************EHEADER*/





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
  double	        *data;
  HYPRE_Int                   *i;
  HYPRE_Int                   *j;
  HYPRE_Int                   block_size;
  HYPRE_Int     		num_rows;
  HYPRE_Int     		num_cols;
  HYPRE_Int                   num_nonzeros;

  HYPRE_Int                   owns_data;

} hypre_CSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBlockMatrixData(matrix)         ((matrix) -> data)
#define hypre_CSRBlockMatrixI(matrix)            ((matrix) -> i)
#define hypre_CSRBlockMatrixJ(matrix)            ((matrix) -> j)
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
HYPRE_Int hypre_CSRBlockMatrixSetDataOwner(hypre_CSRBlockMatrix *, HYPRE_Int);
hypre_CSRMatrix 
      *hypre_CSRBlockMatrixCompress(hypre_CSRBlockMatrix *);
hypre_CSRMatrix 
      *hypre_CSRBlockMatrixConvertToCSRMatrix(hypre_CSRBlockMatrix *);
hypre_CSRBlockMatrix
      *hypre_CSRBlockMatrixConvertFromCSRMatrix(hypre_CSRMatrix *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAdd(double *, double *, double*, HYPRE_Int);

HYPRE_Int hypre_CSRBlockMatrixBlockMultAdd(double *, double *, double, double *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiag(double *, double *, double, double *, HYPRE_Int);
HYPRE_Int
hypre_CSRBlockMatrixBlockMultAddDiag2(double* i1, double* i2, double beta, 
                                      double* o, HYPRE_Int block_size);
HYPRE_Int
hypre_CSRBlockMatrixBlockMultAddDiag3(double* i1, double* i2, double beta, 
                                      double* o, HYPRE_Int block_size);
   

HYPRE_Int hypre_CSRBlockMatrixBlockInvMult(double *, double *, double *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockInvMultDiag(double *, double *, double *, HYPRE_Int);

HYPRE_Int
hypre_CSRBlockMatrixBlockInvMultDiag2(double* i1, double* i2, double* o, HYPRE_Int block_size);
   
HYPRE_Int
hypre_CSRBlockMatrixBlockInvMultDiag3(double* i1, double* i2, double* o, HYPRE_Int block_size);
   



HYPRE_Int hypre_CSRBlockMatrixBlockMultInv(double *, double *, double *, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockTranspose(double *, double *, HYPRE_Int);

HYPRE_Int hypre_CSRBlockMatrixTranspose(hypre_CSRBlockMatrix *A,
                                  hypre_CSRBlockMatrix **AT, HYPRE_Int data);

HYPRE_Int hypre_CSRBlockMatrixBlockCopyData(double*, double*, double, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockCopyDataDiag(double*, double*, double, HYPRE_Int);

HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulate(double*, double*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulateDiag(double* i1, double* o, HYPRE_Int block_size);
   


HYPRE_Int
hypre_CSRBlockMatrixMatvec(double alpha, hypre_CSRBlockMatrix *A,
                           hypre_Vector *x, double beta, hypre_Vector *y);
   

HYPRE_Int
hypre_CSRBlockMatrixMatvecT( double alpha, hypre_CSRBlockMatrix *A, hypre_Vector  *x,
                             double beta, hypre_Vector *y );

HYPRE_Int
hypre_CSRBlockMatrixBlockInvMatvec(double* mat, double* v, 
                                   double* ov, HYPRE_Int block_size);
   
HYPRE_Int 
hypre_CSRBlockMatrixBlockMatvec(double alpha, double* mat, double* v, double beta, 
                                double* ov, HYPRE_Int block_size);
   

HYPRE_Int hypre_CSRBlockMatrixBlockNorm(HYPRE_Int norm_type, double* data, double* out, HYPRE_Int block_size);
   
HYPRE_Int hypre_CSRBlockMatrixBlockSetScalar(double* o, double beta, HYPRE_Int block_size);
   
HYPRE_Int hypre_CSRBlockMatrixComputeSign(double *i1, double *o, HYPRE_Int block_size);
HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(double* i1, double* o, HYPRE_Int block_size, double *sign);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(double* i1, double* i2, double beta, 
                                              double* o, HYPRE_Int block_size, double *sign);
   




#ifdef __cplusplus
}
#endif
#endif
