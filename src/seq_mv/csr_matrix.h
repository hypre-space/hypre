/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int           *i;
   HYPRE_Int           *j;
   HYPRE_BigInt        *big_j;
   HYPRE_Int            num_rows;
   HYPRE_Int            num_cols;
   HYPRE_Int            num_nonzeros;
   hypre_int           *i_short;
   hypre_int           *j_short;
   HYPRE_Int            owns_data;       /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   HYPRE_Complex       *data;
   HYPRE_Int           *rownnz;          /* for compressing rows in matrix multiplication  */
   HYPRE_Int            num_rownnz;
   HYPRE_MemoryLocation memory_location; /* memory location of arrays i, j, data */

#if defined(HYPRE_USING_CUDA)
   /* Data structures for sparse triangular solves */
   void * L; // For now this is opaque. If used, it will be allocated/cast to a hypre_CSRMatrix
   void * L_cusparse_data; // For now this is opaque. If used, it will be allocated/cast to a hypre_CudaSpTriMatrixData
   void * U; // For now this is opaque. If used, it will be allocated/cast to a hypre_CSRMatrix
   void * U_cusparse_data; // For now this is opaque. If used, it will be allocated/cast to a hypre_CudaSpTriMatrixData
   HYPRE_Complex  *D; // separate out the diagonal
   HYPRE_Complex  *work_vector;
   HYPRE_Complex  *work_vector2;
   HYPRE_Int rebuildTriMats; // Every time amg setup is called, this flag is set to 1. After D, L and U are built, the flag is set to 0
   HYPRE_Int rebuildTriSolves; // Every time amg setup is called, this flag is set to 1. After Solve Data for L and/or U are built, the flag is set to 0
#endif

} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)           ((matrix) -> data)
#define hypre_CSRMatrixI(matrix)              ((matrix) -> i)
#define hypre_CSRMatrixJ(matrix)              ((matrix) -> j)
#define hypre_CSRMatrixBigJ(matrix)           ((matrix) -> big_j)
#define hypre_CSRMatrixNumRows(matrix)        ((matrix) -> num_rows)
#define hypre_CSRMatrixNumCols(matrix)        ((matrix) -> num_cols)
#define hypre_CSRMatrixNumNonzeros(matrix)    ((matrix) -> num_nonzeros)
#define hypre_CSRMatrixRownnz(matrix)         ((matrix) -> rownnz)
#define hypre_CSRMatrixNumRownnz(matrix)      ((matrix) -> num_rownnz)
#define hypre_CSRMatrixOwnsData(matrix)       ((matrix) -> owns_data)
#define hypre_CSRMatrixMemoryLocation(matrix) ((matrix) -> memory_location)

#if defined(HYPRE_USING_CUDA)
/* Accessors for sparse triangular solve */
#define hypre_CSRMatrixLower(matrix)                ((matrix) -> L)
#define hypre_CSRMatrixCusparseDataLower(matrix)    ((matrix) -> L_cusparse_data)
#define hypre_CSRMatrixUpper(matrix)                ((matrix) -> U)
#define hypre_CSRMatrixCusparseDataUpper(matrix)    ((matrix) -> U_cusparse_data)
#define hypre_CSRMatrixDiagonal(matrix)             ((matrix) -> D)
#define hypre_CSRMatrixWorkVector(matrix)           ((matrix) -> work_vector)
#define hypre_CSRMatrixWorkVector2(matrix)          ((matrix) -> work_vector2)
#define hypre_CSRMatrixRebuildTriMats(matrix)       ((matrix) -> rebuildTriMats)
#define hypre_CSRMatrixRebuildTriSolves(matrix)     ((matrix) -> rebuildTriSolves)
#endif

HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd( hypre_CSRMatrix *A );

/*--------------------------------------------------------------------------
 * CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int    *i;
   HYPRE_Int    *j;
   HYPRE_BigInt *big_j;
   HYPRE_Int     num_rows;
   HYPRE_Int     num_cols;
   HYPRE_Int     num_nonzeros;
   HYPRE_Int     owns_data;

} hypre_CSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBooleanMatrix_Get_I(matrix)        ((matrix)->i)
#define hypre_CSRBooleanMatrix_Get_J(matrix)        ((matrix)->j)
#define hypre_CSRBooleanMatrix_Get_BigJ(matrix)     ((matrix)->big_j)
#define hypre_CSRBooleanMatrix_Get_NRows(matrix)    ((matrix)->num_rows)
#define hypre_CSRBooleanMatrix_Get_NCols(matrix)    ((matrix)->num_cols)
#define hypre_CSRBooleanMatrix_Get_NNZ(matrix)      ((matrix)->num_nonzeros)
#define hypre_CSRBooleanMatrix_Get_OwnsData(matrix) ((matrix)->owns_data)

#endif

