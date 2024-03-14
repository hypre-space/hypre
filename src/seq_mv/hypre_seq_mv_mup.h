/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef HYPRE_SEQ_MV_MUP_HEADER
#define HYPRE_SEQ_MV_MUP_HEADER

#if defined (HYPRE_MIXED_PRECISION)

/* csr_matop.c */
HYPRE_Int hypre_CSRMatrixAddFirstPass_dbl ( HYPRE_Int firstrow, HYPRE_Int lastrow, HYPRE_Int *marker,
                                            HYPRE_Int *twspace, HYPRE_Int *map_A2C, HYPRE_Int *map_B2C,
                                            hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                            HYPRE_Int nnzrows_C, HYPRE_Int nrows_C, HYPRE_Int ncols_C,
                                            HYPRE_Int *rownnz_C,
                                            HYPRE_MemoryLocation memory_location_C,
                                            HYPRE_Int *C_i, hypre_CSRMatrix **C_ptr );
HYPRE_Int hypre_CSRMatrixAddFirstPass_flt ( HYPRE_Int firstrow, HYPRE_Int lastrow, HYPRE_Int *marker,
                                            HYPRE_Int *twspace, HYPRE_Int *map_A2C, HYPRE_Int *map_B2C,
                                            hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                            HYPRE_Int nnzrows_C, HYPRE_Int nrows_C, HYPRE_Int ncols_C,
                                            HYPRE_Int *rownnz_C,
                                            HYPRE_MemoryLocation memory_location_C,
                                            HYPRE_Int *C_i, hypre_CSRMatrix **C_ptr );
HYPRE_Int hypre_CSRMatrixAddFirstPass_long_dbl ( HYPRE_Int firstrow, HYPRE_Int lastrow, HYPRE_Int *marker,
                                                 HYPRE_Int *twspace, HYPRE_Int *map_A2C, HYPRE_Int *map_B2C,
                                                 hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                 HYPRE_Int nnzrows_C, HYPRE_Int nrows_C, HYPRE_Int ncols_C,
                                                 HYPRE_Int *rownnz_C,
                                                 HYPRE_MemoryLocation memory_location_C,
                                                 HYPRE_Int *C_i, hypre_CSRMatrix **C_ptr );
HYPRE_Int hypre_CSRMatrixAddSecondPass_dbl ( HYPRE_Int firstrow, HYPRE_Int lastrow, HYPRE_Int *marker,
                                             HYPRE_Int *map_A2C, HYPRE_Int *map_B2C,
                                             HYPRE_Int *rownnz_C, HYPRE_Complex alpha,
                                             HYPRE_Complex beta, hypre_CSRMatrix *A,
                                             hypre_CSRMatrix *B, hypre_CSRMatrix *C);
HYPRE_Int hypre_CSRMatrixAddSecondPass_flt ( HYPRE_Int firstrow, HYPRE_Int lastrow, HYPRE_Int *marker,
                                             HYPRE_Int *map_A2C, HYPRE_Int *map_B2C,
                                             HYPRE_Int *rownnz_C, HYPRE_Complex alpha,
                                             HYPRE_Complex beta, hypre_CSRMatrix *A,
                                             hypre_CSRMatrix *B, hypre_CSRMatrix *C);
HYPRE_Int hypre_CSRMatrixAddSecondPass_long_dbl ( HYPRE_Int firstrow, HYPRE_Int lastrow, HYPRE_Int *marker,
                                             HYPRE_Int *map_A2C, HYPRE_Int *map_B2C,
                                             HYPRE_Int *rownnz_C, HYPRE_Complex alpha,
                                             HYPRE_Complex beta, hypre_CSRMatrix *A,
                                             hypre_CSRMatrix *B, hypre_CSRMatrix *C);
hypre_CSRMatrix *hypre_CSRMatrixAddHost_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                              HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAddHost_flt ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                              HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAddHost_longdbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                              HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAdd_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, HYPRE_Complex beta,
                                          hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAdd_flt ( HYPRE_Complex alpha, hypre_CSRMatrix *A, HYPRE_Complex beta,
                                          hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAdd_long_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, HYPRE_Complex beta,
                                               hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixBigAdd_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixBigAdd_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixBigAdd_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyHost_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyHost_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyHost_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiply_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiply_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiply_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros_dbl ( hypre_CSRMatrix *A, HYPRE_Real tol );
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros_flt ( hypre_CSRMatrix *A, HYPRE_Real tol );
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros_long_dbl ( hypre_CSRMatrix *A, HYPRE_Real tol );
HYPRE_Int hypre_CSRMatrixTransposeHost_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTransposeHost_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTransposeHost_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTranspose_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTranspose_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTranspose_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT, HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixReorder_dbl ( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixReorder_flt ( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixReorder_long_dbl ( hypre_CSRMatrix *A );
HYPRE_Complex hypre_CSRMatrixSumElts_dbl ( hypre_CSRMatrix *A );
HYPRE_Complex hypre_CSRMatrixSumElts_flt ( hypre_CSRMatrix *A );
HYPRE_Complex hypre_CSRMatrixSumElts_long_dbl ( hypre_CSRMatrix *A );
HYPRE_Real hypre_CSRMatrixFnorm_dbl( hypre_CSRMatrix *A );
HYPRE_Real hypre_CSRMatrixFnorm_flt( hypre_CSRMatrix *A );
HYPRE_Real hypre_CSRMatrixFnorm_long_dbl( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixSplit_dbl(hypre_CSRMatrix *Bs_ext, HYPRE_BigInt first_col_diag_B,
                                   HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                   HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr, hypre_CSRMatrix **Bext_diag_ptr,
                                   hypre_CSRMatrix **Bext_offd_ptr);
HYPRE_Int hypre_CSRMatrixSplit_flt(hypre_CSRMatrix *Bs_ext, HYPRE_BigInt first_col_diag_B,
                                   HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                   HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr, hypre_CSRMatrix **Bext_diag_ptr,
                                   hypre_CSRMatrix **Bext_offd_ptr);
HYPRE_Int hypre_CSRMatrixSplit_long_dbl(hypre_CSRMatrix *Bs_ext, HYPRE_BigInt first_col_diag_B,
                                   HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                   HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr, hypre_CSRMatrix **Bext_diag_ptr,
                                   hypre_CSRMatrix **Bext_offd_ptr);
hypre_CSRMatrix * hypre_CSRMatrixAddPartial_dbl( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                 HYPRE_Int *row_nums);
hypre_CSRMatrix * hypre_CSRMatrixAddPartial_flt( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                 HYPRE_Int *row_nums);
hypre_CSRMatrix * hypre_CSRMatrixAddPartial_long_dbl( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                 HYPRE_Int *row_nums);
void hypre_CSRMatrixComputeRowSumHost_dbl( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                           HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixComputeRowSumHost_flt( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                           HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixComputeRowSumHost_long_dbl( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                           HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixComputeRowSum_dbl( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                       HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixComputeRowSum_flt( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                       HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixComputeRowSum_long_dbl( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                       HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixExtractDiagonal_dbl( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
void hypre_CSRMatrixExtractDiagonal_flt( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
void hypre_CSRMatrixExtractDiagonal_long_dbl( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
void hypre_CSRMatrixExtractDiagonalHost_dbl( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
void hypre_CSRMatrixExtractDiagonalHost_flt( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
void hypre_CSRMatrixExtractDiagonalHost_long_dbl( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
HYPRE_Int hypre_CSRMatrixScale_dbl(hypre_CSRMatrix *A, HYPRE_Complex scalar);
HYPRE_Int hypre_CSRMatrixScale_flt(hypre_CSRMatrix *A, HYPRE_Complex scalar);
HYPRE_Int hypre_CSRMatrixScale_long_dbl(hypre_CSRMatrix *A, HYPRE_Complex scalar);
HYPRE_Int hypre_CSRMatrixSetConstantValues_dbl( hypre_CSRMatrix *A, HYPRE_Complex value);
HYPRE_Int hypre_CSRMatrixSetConstantValues_flt( hypre_CSRMatrix *A, HYPRE_Complex value);
HYPRE_Int hypre_CSRMatrixSetConstantValues_long_dbl( hypre_CSRMatrix *A, HYPRE_Complex value);
HYPRE_Int hypre_CSRMatrixDiagScale_dbl( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);
HYPRE_Int hypre_CSRMatrixDiagScale_flt( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);
HYPRE_Int hypre_CSRMatrixDiagScale_long_dbl( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);

/* csr_matop_device.c */
hypre_CSRMatrix *hypre_CSRMatrixAddDevice_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                                HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAddDevice_flt ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                                HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAddDevice_long_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                                HYPRE_Complex beta, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyDevice_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyDevice_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyDevice_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixTripleMultiplyDevice_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                           hypre_CSRMatrix *C );
hypre_CSRMatrix *hypre_CSRMatrixTripleMultiplyDevice_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                           hypre_CSRMatrix *C );
hypre_CSRMatrix *hypre_CSRMatrixTripleMultiplyDevice_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                           hypre_CSRMatrix *C );
HYPRE_Int hypre_CSRMatrixMergeColMapOffd_dbl( HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                              HYPRE_Int B_ext_offd_nnz, HYPRE_BigInt *B_ext_offd_bigj, HYPRE_Int *num_cols_offd_C_ptr,
                                              HYPRE_BigInt **col_map_offd_C_ptr, HYPRE_Int **map_B_to_C_ptr );
HYPRE_Int hypre_CSRMatrixMergeColMapOffd_flt( HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                              HYPRE_Int B_ext_offd_nnz, HYPRE_BigInt *B_ext_offd_bigj, HYPRE_Int *num_cols_offd_C_ptr,
                                              HYPRE_BigInt **col_map_offd_C_ptr, HYPRE_Int **map_B_to_C_ptr );
HYPRE_Int hypre_CSRMatrixMergeColMapOffd_long_dbl( HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                              HYPRE_Int B_ext_offd_nnz, HYPRE_BigInt *B_ext_offd_bigj, HYPRE_Int *num_cols_offd_C_ptr,
                                              HYPRE_BigInt **col_map_offd_C_ptr, HYPRE_Int **map_B_to_C_ptr );
HYPRE_Int hypre_CSRMatrixSplitDevice_core_dbl( HYPRE_Int job, HYPRE_Int num_rows, HYPRE_Int B_ext_nnz,
                                               HYPRE_Int *B_ext_ii, HYPRE_BigInt *B_ext_bigj, HYPRE_Complex *B_ext_data, char *B_ext_xata,
                                               HYPRE_BigInt first_col_diag_B, HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B,
                                               HYPRE_BigInt *col_map_offd_B, HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr,
                                               HYPRE_BigInt **col_map_offd_C_ptr, HYPRE_Int *B_ext_diag_nnz_ptr, HYPRE_Int *B_ext_diag_ii,
                                               HYPRE_Int *B_ext_diag_j, HYPRE_Complex *B_ext_diag_data, char *B_ext_diag_xata,
                                               HYPRE_Int *B_ext_offd_nnz_ptr, HYPRE_Int *B_ext_offd_ii, HYPRE_Int *B_ext_offd_j,
                                               HYPRE_Complex *B_ext_offd_data, char *B_ext_offd_xata );
HYPRE_Int hypre_CSRMatrixSplitDevice_core_flt( HYPRE_Int job, HYPRE_Int num_rows, HYPRE_Int B_ext_nnz,
                                               HYPRE_Int *B_ext_ii, HYPRE_BigInt *B_ext_bigj, HYPRE_Complex *B_ext_data, char *B_ext_xata,
                                               HYPRE_BigInt first_col_diag_B, HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B,
                                               HYPRE_BigInt *col_map_offd_B, HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr,
                                               HYPRE_BigInt **col_map_offd_C_ptr, HYPRE_Int *B_ext_diag_nnz_ptr, HYPRE_Int *B_ext_diag_ii,
                                               HYPRE_Int *B_ext_diag_j, HYPRE_Complex *B_ext_diag_data, char *B_ext_diag_xata,
                                               HYPRE_Int *B_ext_offd_nnz_ptr, HYPRE_Int *B_ext_offd_ii, HYPRE_Int *B_ext_offd_j,
                                               HYPRE_Complex *B_ext_offd_data, char *B_ext_offd_xata );
HYPRE_Int hypre_CSRMatrixSplitDevice_core_long_dbl( HYPRE_Int job, HYPRE_Int num_rows, HYPRE_Int B_ext_nnz,
                                               HYPRE_Int *B_ext_ii, HYPRE_BigInt *B_ext_bigj, HYPRE_Complex *B_ext_data, char *B_ext_xata,
                                               HYPRE_BigInt first_col_diag_B, HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B,
                                               HYPRE_BigInt *col_map_offd_B, HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr,
                                               HYPRE_BigInt **col_map_offd_C_ptr, HYPRE_Int *B_ext_diag_nnz_ptr, HYPRE_Int *B_ext_diag_ii,
                                               HYPRE_Int *B_ext_diag_j, HYPRE_Complex *B_ext_diag_data, char *B_ext_diag_xata,
                                               HYPRE_Int *B_ext_offd_nnz_ptr, HYPRE_Int *B_ext_offd_ii, HYPRE_Int *B_ext_offd_j,
                                               HYPRE_Complex *B_ext_offd_data, char *B_ext_offd_xata );
HYPRE_Int hypre_CSRMatrixSplitDevice_dbl(hypre_CSRMatrix *B_ext, HYPRE_BigInt first_col_diag_B,
                                         HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                         HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr,
                                         hypre_CSRMatrix **B_ext_diag_ptr, hypre_CSRMatrix **B_ext_offd_ptr);
HYPRE_Int hypre_CSRMatrixSplitDevice_flt(hypre_CSRMatrix *B_ext, HYPRE_BigInt first_col_diag_B,
                                         HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                         HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr,
                                         hypre_CSRMatrix **B_ext_diag_ptr, hypre_CSRMatrix **B_ext_offd_ptr);
HYPRE_Int hypre_CSRMatrixSplitDevice_long_dbl(hypre_CSRMatrix *B_ext, HYPRE_BigInt first_col_diag_B,
                                         HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                         HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr,
                                         hypre_CSRMatrix **B_ext_diag_ptr, hypre_CSRMatrix **B_ext_offd_ptr);
HYPRE_Int hypre_CSRMatrixTransposeDevice_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT,
                                               HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTransposeDevice_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT,
                                               HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTransposeDevice_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix **AT,
                                               HYPRE_Int data );
hypre_CSRMatrix* hypre_CSRMatrixAddPartialDevice_dbl( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                      HYPRE_Int *row_nums);
hypre_CSRMatrix* hypre_CSRMatrixAddPartialDevice_flt( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                      HYPRE_Int *row_nums);
hypre_CSRMatrix* hypre_CSRMatrixAddPartialDevice_long_dbl( hypre_CSRMatrix *A, hypre_CSRMatrix *B,
                                                      HYPRE_Int *row_nums);
HYPRE_Int hypre_CSRMatrixColNNzRealDevice_dbl( hypre_CSRMatrix *A, HYPRE_Real *colnnz);
HYPRE_Int hypre_CSRMatrixColNNzRealDevice_flt( hypre_CSRMatrix *A, HYPRE_Real *colnnz);
HYPRE_Int hypre_CSRMatrixColNNzRealDevice_long_dbl( hypre_CSRMatrix *A, HYPRE_Real *colnnz);
HYPRE_Int hypre_CSRMatrixMoveDiagFirstDevice_dbl( hypre_CSRMatrix  *A );
HYPRE_Int hypre_CSRMatrixMoveDiagFirstDevice_flt( hypre_CSRMatrix  *A );
HYPRE_Int hypre_CSRMatrixMoveDiagFirstDevice_long_dbl( hypre_CSRMatrix  *A );
HYPRE_Int hypre_CSRMatrixCheckDiagFirstDevice_dbl( hypre_CSRMatrix  *A );
HYPRE_Int hypre_CSRMatrixCheckDiagFirstDevice_flt( hypre_CSRMatrix  *A );
HYPRE_Int hypre_CSRMatrixCheckDiagFirstDevice_long_dbl( hypre_CSRMatrix  *A );
HYPRE_Int hypre_CSRMatrixCheckForMissingDiagonal_dbl( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixCheckForMissingDiagonal_flt( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixCheckForMissingDiagonal_long_dbl( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixReplaceDiagDevice_dbl( hypre_CSRMatrix *A, HYPRE_Complex *new_diag,
                                                HYPRE_Complex v, HYPRE_Real tol );
HYPRE_Int hypre_CSRMatrixReplaceDiagDevice_flt( hypre_CSRMatrix *A, HYPRE_Complex *new_diag,
                                                HYPRE_Complex v, HYPRE_Real tol );
HYPRE_Int hypre_CSRMatrixReplaceDiagDevice_long_dbl( hypre_CSRMatrix *A, HYPRE_Complex *new_diag,
                                                HYPRE_Complex v, HYPRE_Real tol );
HYPRE_Int hypre_CSRMatrixComputeRowSumDevice_dbl( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                                  HYPRE_Complex *row_sum, HYPRE_Int type,
                                                  HYPRE_Complex scal, const char *set_or_add );
HYPRE_Int hypre_CSRMatrixComputeRowSumDevice_flt( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                                  HYPRE_Complex *row_sum, HYPRE_Int type,
                                                  HYPRE_Complex scal, const char *set_or_add );
HYPRE_Int hypre_CSRMatrixComputeRowSumDevice_long_dbl( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j,
                                                  HYPRE_Complex *row_sum, HYPRE_Int type,
                                                  HYPRE_Complex scal, const char *set_or_add );
HYPRE_Int hypre_CSRMatrixExtractDiagonalDevice_dbl( hypre_CSRMatrix *A, HYPRE_Complex *d,
                                                    HYPRE_Int type );
HYPRE_Int hypre_CSRMatrixExtractDiagonalDevice_flt( hypre_CSRMatrix *A, HYPRE_Complex *d,
                                                    HYPRE_Int type );
HYPRE_Int hypre_CSRMatrixExtractDiagonalDevice_long_dbl( hypre_CSRMatrix *A, HYPRE_Complex *d,
                                                    HYPRE_Int type );
hypre_CSRMatrix* hypre_CSRMatrixStack2Device_dbl(hypre_CSRMatrix *A, hypre_CSRMatrix *B);
hypre_CSRMatrix* hypre_CSRMatrixStack2Device_flt(hypre_CSRMatrix *A, hypre_CSRMatrix *B);
hypre_CSRMatrix* hypre_CSRMatrixStack2Device_long_dbl(hypre_CSRMatrix *A, hypre_CSRMatrix *B);
hypre_CSRMatrix* hypre_CSRMatrixIdentityDevice_dbl(HYPRE_Int n, HYPRE_Complex alp);
hypre_CSRMatrix* hypre_CSRMatrixIdentityDevice_flt(HYPRE_Int n, HYPRE_Complex alp);
hypre_CSRMatrix* hypre_CSRMatrixIdentityDevice_long_dbl(HYPRE_Int n, HYPRE_Complex alp);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromVectorDevice_dbl(HYPRE_Int n, HYPRE_Complex *v);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromVectorDevice_flt(HYPRE_Int n, HYPRE_Complex *v);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromVectorDevice_long_dbl(HYPRE_Int n, HYPRE_Complex *v);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromMatrixDevice_dbl(hypre_CSRMatrix *A, HYPRE_Int type);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromMatrixDevice_flt(hypre_CSRMatrix *A, HYPRE_Int type);
hypre_CSRMatrix* hypre_CSRMatrixDiagMatrixFromMatrixDevice_long_dbl(hypre_CSRMatrix *A, HYPRE_Int type);
HYPRE_Int hypre_CSRMatrixRemoveDiagonalDevice_dbl(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixRemoveDiagonalDevice_flt(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixRemoveDiagonalDevice_long_dbl(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixDropSmallEntriesDevice_dbl( hypre_CSRMatrix *A, HYPRE_Real tol,
                                                     HYPRE_Real *elmt_tols);
HYPRE_Int hypre_CSRMatrixDropSmallEntriesDevice_flt( hypre_CSRMatrix *A, HYPRE_Real tol,
                                                     HYPRE_Real *elmt_tols);
HYPRE_Int hypre_CSRMatrixDropSmallEntriesDevice_long_dbl( hypre_CSRMatrix *A, HYPRE_Real tol,
                                                     HYPRE_Real *elmt_tols);
HYPRE_Int hypre_CSRMatrixPermuteDevice_dbl( hypre_CSRMatrix *A, HYPRE_Int *perm,
                                            HYPRE_Int *rqperm, hypre_CSRMatrix *B );
HYPRE_Int hypre_CSRMatrixPermuteDevice_flt( hypre_CSRMatrix *A, HYPRE_Int *perm,
                                            HYPRE_Int *rqperm, hypre_CSRMatrix *B );
HYPRE_Int hypre_CSRMatrixPermuteDevice_long_dbl( hypre_CSRMatrix *A, HYPRE_Int *perm,
                                            HYPRE_Int *rqperm, hypre_CSRMatrix *B );
HYPRE_Int hypre_CSRMatrixSortRow_dbl(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixSortRow_flt(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixSortRow_long_dbl(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixSortRowOutOfPlace_dbl(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixSortRowOutOfPlace_flt(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixSortRowOutOfPlace_long_dbl(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveDevice_core_dbl(char uplo, HYPRE_Int unit_diag,
                                                           hypre_CSRMatrix *A, HYPRE_Real *l1_norms, hypre_Vector *f, HYPRE_Int offset_f, hypre_Vector *u,
                                                           HYPRE_Int offset_u);
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveDevice_core_flt(char uplo, HYPRE_Int unit_diag,
                                                           hypre_CSRMatrix *A, HYPRE_Real *l1_norms, hypre_Vector *f, HYPRE_Int offset_f, hypre_Vector *u,
                                                           HYPRE_Int offset_u);
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveDevice_core_long_dbl(char uplo, HYPRE_Int unit_diag,
                                                           hypre_CSRMatrix *A, HYPRE_Real *l1_norms, hypre_Vector *f, HYPRE_Int offset_f, hypre_Vector *u,
                                                           HYPRE_Int offset_u);
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveDevice_dbl(char uplo, HYPRE_Int unit_diag,
                                                      hypre_CSRMatrix *A, HYPRE_Real *l1_norms, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveDevice_flt(char uplo, HYPRE_Int unit_diag,
                                                      hypre_CSRMatrix *A, HYPRE_Real *l1_norms, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveDevice_long_dbl(char uplo, HYPRE_Int unit_diag,
                                                      hypre_CSRMatrix *A, HYPRE_Real *l1_norms, hypre_Vector *f, hypre_Vector *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveRocsparse_dbl(char uplo, HYPRE_Int unit_diag,
                                                         hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveRocsparse_flt(char uplo, HYPRE_Int unit_diag,
                                                         hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveRocsparse_long_dbl(char uplo, HYPRE_Int unit_diag,
                                                         hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveCusparse_dbl(char uplo, HYPRE_Int unit_diag,
                                                        hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveCusparse_flt(char uplo, HYPRE_Int unit_diag,
                                                        hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveCusparse_long_dbl(char uplo, HYPRE_Int unit_diag,
                                                        hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveOnemklsparse_dbl(char uplo, HYPRE_Int unit_diag,
                                                            hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveOnemklsparse_flt(char uplo, HYPRE_Int unit_diag,
                                                            hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixTriLowerUpperSolveOnemklsparse_long_dbl(char uplo, HYPRE_Int unit_diag,
                                                            hypre_CSRMatrix *A, HYPRE_Real *l1_norms, HYPRE_Complex *f, HYPRE_Complex *u );
HYPRE_Int hypre_CSRMatrixIntersectPattern_dbl(hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int *markA,
                                              HYPRE_Int diag_option);
HYPRE_Int hypre_CSRMatrixIntersectPattern_flt(hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int *markA,
                                              HYPRE_Int diag_option);
HYPRE_Int hypre_CSRMatrixIntersectPattern_long_dbl(hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int *markA,
                                              HYPRE_Int diag_option);
HYPRE_Int hypre_CSRMatrixDiagScaleDevice_dbl( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);
HYPRE_Int hypre_CSRMatrixDiagScaleDevice_flt( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);
HYPRE_Int hypre_CSRMatrixDiagScaleDevice_long_dbl( hypre_CSRMatrix *A, hypre_Vector *ld, hypre_Vector *rd);
HYPRE_Int hypre_CSRMatrixCompressColumnsDevice_dbl(hypre_CSRMatrix *A, HYPRE_BigInt *col_map,
                                                   HYPRE_Int **col_idx_new_ptr, HYPRE_BigInt **col_map_new_ptr);
HYPRE_Int hypre_CSRMatrixCompressColumnsDevice_flt(hypre_CSRMatrix *A, HYPRE_BigInt *col_map,
                                                   HYPRE_Int **col_idx_new_ptr, HYPRE_BigInt **col_map_new_ptr);
HYPRE_Int hypre_CSRMatrixCompressColumnsDevice_long_dbl(hypre_CSRMatrix *A, HYPRE_BigInt *col_map,
                                                   HYPRE_Int **col_idx_new_ptr, HYPRE_BigInt **col_map_new_ptr);
HYPRE_Int hypre_CSRMatrixILU0_dbl(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixILU0_flt(hypre_CSRMatrix *A);
HYPRE_Int hypre_CSRMatrixILU0_long_dbl(hypre_CSRMatrix *A);

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CSRMatrixCreate_dbl ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                             HYPRE_Int num_nonzeros );
hypre_CSRMatrix *hypre_CSRMatrixCreate_flt ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                             HYPRE_Int num_nonzeros );
hypre_CSRMatrix *hypre_CSRMatrixCreate_long_dbl ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                             HYPRE_Int num_nonzeros );
HYPRE_Int hypre_CSRMatrixDestroy_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixDestroy_flt ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixDestroy_long_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixInitialize_v2_dbl( hypre_CSRMatrix *matrix, HYPRE_Int bigInit,
                                            HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_CSRMatrixInitialize_v2_flt( hypre_CSRMatrix *matrix, HYPRE_Int bigInit,
                                            HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_CSRMatrixInitialize_v2_long_dbl( hypre_CSRMatrix *matrix, HYPRE_Int bigInit,
                                            HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_CSRMatrixInitialize_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixInitialize_flt ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixInitialize_long_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigInitialize_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigInitialize_flt ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigInitialize_long_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigJtoJ_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigJtoJ_flt ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigJtoJ_long_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixJtoBigJ_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixJtoBigJ_flt ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixJtoBigJ_long_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixSetDataOwner_dbl ( hypre_CSRMatrix *matrix, HYPRE_Int owns_data );
HYPRE_Int hypre_CSRMatrixSetDataOwner_flt ( hypre_CSRMatrix *matrix, HYPRE_Int owns_data );
HYPRE_Int hypre_CSRMatrixSetDataOwner_long_dbl ( hypre_CSRMatrix *matrix, HYPRE_Int owns_data );
HYPRE_Int hypre_CSRMatrixSetPatternOnly_dbl( hypre_CSRMatrix *matrix, HYPRE_Int pattern_only );
HYPRE_Int hypre_CSRMatrixSetPatternOnly_flt( hypre_CSRMatrix *matrix, HYPRE_Int pattern_only );
HYPRE_Int hypre_CSRMatrixSetPatternOnly_long_dbl( hypre_CSRMatrix *matrix, HYPRE_Int pattern_only );
HYPRE_Int hypre_CSRMatrixSetRownnz_dbl ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixSetRownnz_flt ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixSetRownnz_long_dbl ( hypre_CSRMatrix *matrix );
hypre_CSRMatrix *hypre_CSRMatrixRead_dbl ( char *file_name );
hypre_CSRMatrix *hypre_CSRMatrixRead_flt ( char *file_name );
hypre_CSRMatrix *hypre_CSRMatrixRead_long_dbl ( char *file_name );
HYPRE_Int hypre_CSRMatrixPrint_dbl ( hypre_CSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_CSRMatrixPrint_flt ( hypre_CSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_CSRMatrixPrint_long_dbl ( hypre_CSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_CSRMatrixPrintIJ_dbl( hypre_CSRMatrix *matrix, HYPRE_Int base_i,
                                      HYPRE_Int base_j, char *filename );
HYPRE_Int hypre_CSRMatrixPrintIJ_flt( hypre_CSRMatrix *matrix, HYPRE_Int base_i,
                                      HYPRE_Int base_j, char *filename );
HYPRE_Int hypre_CSRMatrixPrintIJ_long_dbl( hypre_CSRMatrix *matrix, HYPRE_Int base_i,
                                      HYPRE_Int base_j, char *filename );
HYPRE_Int hypre_CSRMatrixPrintHB_dbl ( hypre_CSRMatrix *matrix_input, char *file_name );
HYPRE_Int hypre_CSRMatrixPrintHB_flt ( hypre_CSRMatrix *matrix_input, char *file_name );
HYPRE_Int hypre_CSRMatrixPrintHB_long_dbl ( hypre_CSRMatrix *matrix_input, char *file_name );
HYPRE_Int hypre_CSRMatrixPrintMM_dbl( hypre_CSRMatrix *matrix, HYPRE_Int basei, HYPRE_Int basej,
                                      HYPRE_Int trans, const char *file_name );
HYPRE_Int hypre_CSRMatrixPrintMM_flt( hypre_CSRMatrix *matrix, HYPRE_Int basei, HYPRE_Int basej,
                                      HYPRE_Int trans, const char *file_name );
HYPRE_Int hypre_CSRMatrixPrintMM_long_dbl( hypre_CSRMatrix *matrix, HYPRE_Int basei, HYPRE_Int basej,
                                      HYPRE_Int trans, const char *file_name );
HYPRE_Int hypre_CSRMatrixCopy_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int copy_data );
HYPRE_Int hypre_CSRMatrixCopy_flt ( hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int copy_data );
HYPRE_Int hypre_CSRMatrixCopy_long_dbl ( hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int copy_data );
HYPRE_Int hypre_CSRMatrixMigrate_dbl( hypre_CSRMatrix *A, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_CSRMatrixMigrate_flt( hypre_CSRMatrix *A, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_CSRMatrixMigrate_long_dbl( hypre_CSRMatrix *A, HYPRE_MemoryLocation memory_location );
hypre_CSRMatrix *hypre_CSRMatrixClone_dbl ( hypre_CSRMatrix *A, HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone_flt ( hypre_CSRMatrix *A, HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone_long_dbl ( hypre_CSRMatrix *A, HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone_v2_dbl( hypre_CSRMatrix *A, HYPRE_Int copy_data,
                                              HYPRE_MemoryLocation memory_location );
hypre_CSRMatrix *hypre_CSRMatrixClone_v2_flt( hypre_CSRMatrix *A, HYPRE_Int copy_data,
                                              HYPRE_MemoryLocation memory_location );
hypre_CSRMatrix *hypre_CSRMatrixClone_v2_long_dbl( hypre_CSRMatrix *A, HYPRE_Int copy_data,
                                              HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_CSRMatrixPermute_dbl( hypre_CSRMatrix *A, HYPRE_Int *perm,
                                      HYPRE_Int *rqperm, hypre_CSRMatrix **B_ptr );
HYPRE_Int hypre_CSRMatrixPermute_flt( hypre_CSRMatrix *A, HYPRE_Int *perm,
                                      HYPRE_Int *rqperm, hypre_CSRMatrix **B_ptr );
HYPRE_Int hypre_CSRMatrixPermute_long_dbl( hypre_CSRMatrix *A, HYPRE_Int *perm,
                                      HYPRE_Int *rqperm, hypre_CSRMatrix **B_ptr );
hypre_CSRMatrix *hypre_CSRMatrixUnion_dbl( hypre_CSRMatrix *A,
                                           hypre_CSRMatrix *B,
                                           HYPRE_BigInt *col_map_offd_A,
                                           HYPRE_BigInt *col_map_offd_B,
                                           HYPRE_BigInt **col_map_offd_C );
hypre_CSRMatrix *hypre_CSRMatrixUnion_flt( hypre_CSRMatrix *A,
                                           hypre_CSRMatrix *B,
                                           HYPRE_BigInt *col_map_offd_A,
                                           HYPRE_BigInt *col_map_offd_B,
                                           HYPRE_BigInt **col_map_offd_C );
hypre_CSRMatrix *hypre_CSRMatrixUnion_long_dbl( hypre_CSRMatrix *A,
                                           hypre_CSRMatrix *B,
                                           HYPRE_BigInt *col_map_offd_A,
                                           HYPRE_BigInt *col_map_offd_B,
                                           HYPRE_BigInt **col_map_offd_C );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin_dbl( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin_flt( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin_long_dbl( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd_dbl( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd_flt( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd_long_dbl( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixPrefetch_dbl( hypre_CSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_CSRMatrixPrefetch_flt( hypre_CSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_CSRMatrixPrefetch_long_dbl( hypre_CSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_CSRMatrixCheckSetNumNonzeros_dbl( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixCheckSetNumNonzeros_flt( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixCheckSetNumNonzeros_long_dbl( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixResize_dbl( hypre_CSRMatrix *matrix, HYPRE_Int new_num_rows,
                                     HYPRE_Int new_num_cols, HYPRE_Int new_num_nonzeros );
HYPRE_Int hypre_CSRMatrixResize_flt( hypre_CSRMatrix *matrix, HYPRE_Int new_num_rows,
                                     HYPRE_Int new_num_cols, HYPRE_Int new_num_nonzeros );
HYPRE_Int hypre_CSRMatrixResize_long_dbl( hypre_CSRMatrix *matrix, HYPRE_Int new_num_rows,
                                     HYPRE_Int new_num_cols, HYPRE_Int new_num_nonzeros );

/* csr_matvec.c */
// y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end]
HYPRE_Int hypre_CSRMatrixMatvecOutOfPlace_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                                hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOutOfPlace_flt ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                                hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOutOfPlace_long_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                                hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
// y = alpha*A + beta*y
HYPRE_Int hypre_CSRMatrixMatvec_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                      HYPRE_Complex beta, hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvec_flt ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                      HYPRE_Complex beta, hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvec_long_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                      HYPRE_Complex beta, hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvecT_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                       HYPRE_Complex beta, hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvecT_flt ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                       HYPRE_Complex beta, hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvecT_long_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                       HYPRE_Complex beta, hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvec_FF_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                         HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int *CF_marker_x, HYPRE_Int *CF_marker_y,
                                         HYPRE_Int fpt );
HYPRE_Int hypre_CSRMatrixMatvec_FF_flt ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                         HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int *CF_marker_x, HYPRE_Int *CF_marker_y,
                                         HYPRE_Int fpt );
HYPRE_Int hypre_CSRMatrixMatvec_FF_long_dbl ( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x,
                                         HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int *CF_marker_x, HYPRE_Int *CF_marker_y,
                                         HYPRE_Int fpt );

/* csr_matvec_device.c */
HYPRE_Int hypre_CSRMatrixMatvecDevice_dbl(HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                          hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecDevice_flt(HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                          hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecDevice_long_dbl(HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                          hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparseNewAPI_dbl( HYPRE_Int trans, HYPRE_Complex alpha,
                                                   hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparseNewAPI_flt( HYPRE_Int trans, HYPRE_Complex alpha,
                                                   hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparseNewAPI_long_dbl( HYPRE_Int trans, HYPRE_Complex alpha,
                                                   hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparseOldAPI_dbl( HYPRE_Int trans, HYPRE_Complex alpha,
                                                   hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparseOldAPI_flt( HYPRE_Int trans, HYPRE_Complex alpha,
                                                   hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparseOldAPI_long_dbl( HYPRE_Int trans, HYPRE_Complex alpha,
                                                   hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparse_dbl( HYPRE_Int trans, HYPRE_Complex alpha,
                                             hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparse_flt( HYPRE_Int trans, HYPRE_Complex alpha,
                                             hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecCusparse_long_dbl( HYPRE_Int trans, HYPRE_Complex alpha,
                                             hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOMPOffload_dbl (HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                               hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOMPOffload_flt (HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                               hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOMPOffload_long_dbl (HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                               hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecRocsparse_dbl (HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                              hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecRocsparse_flt (HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                              hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecRocsparse_long_dbl (HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                              hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOnemklsparse_dbl (HYPRE_Int trans, HYPRE_Complex alpha,
                                                 hypre_CSRMatrix *A,
                                                 hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOnemklsparse_flt (HYPRE_Int trans, HYPRE_Complex alpha,
                                                 hypre_CSRMatrix *A,
                                                 hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOnemklsparse_long_dbl (HYPRE_Int trans, HYPRE_Complex alpha,
                                                 hypre_CSRMatrix *A,
                                                 hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int offset );

/* genpart.c */
HYPRE_Int hypre_GeneratePartitioning_dbl ( HYPRE_BigInt length, HYPRE_Int num_procs,
                                           HYPRE_BigInt **part_ptr );
HYPRE_Int hypre_GeneratePartitioning_flt ( HYPRE_BigInt length, HYPRE_Int num_procs,
                                           HYPRE_BigInt **part_ptr );
HYPRE_Int hypre_GeneratePartitioning_long_dbl ( HYPRE_BigInt length, HYPRE_Int num_procs,
                                           HYPRE_BigInt **part_ptr );
HYPRE_Int hypre_GenerateLocalPartitioning_dbl ( HYPRE_BigInt length, HYPRE_Int num_procs,
                                                HYPRE_Int myid, HYPRE_BigInt *part );
HYPRE_Int hypre_GenerateLocalPartitioning_flt ( HYPRE_BigInt length, HYPRE_Int num_procs,
                                                HYPRE_Int myid, HYPRE_BigInt *part );
HYPRE_Int hypre_GenerateLocalPartitioning_long_dbl ( HYPRE_BigInt length, HYPRE_Int num_procs,
                                                HYPRE_Int myid, HYPRE_BigInt *part );

/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate_dbl ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                            HYPRE_Int *row_sizes );
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate_flt ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                            HYPRE_Int *row_sizes );
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate_long_dbl ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                            HYPRE_Int *row_sizes );
HYPRE_Int HYPRE_CSRMatrixDestroy_dbl ( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixDestroy_flt ( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixDestroy_long_dbl ( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixInitialize_dbl ( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixInitialize_flt ( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixInitialize_long_dbl ( HYPRE_CSRMatrix matrix );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead_dbl ( char *file_name );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead_flt ( char *file_name );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead_long_dbl ( char *file_name );
void HYPRE_CSRMatrixPrint_dbl ( HYPRE_CSRMatrix matrix, char *file_name );
void HYPRE_CSRMatrixPrint_flt ( HYPRE_CSRMatrix matrix, char *file_name );
void HYPRE_CSRMatrixPrint_long_dbl ( HYPRE_CSRMatrix matrix, char *file_name );
HYPRE_Int HYPRE_CSRMatrixGetNumRows_dbl ( HYPRE_CSRMatrix matrix, HYPRE_Int *num_rows );
HYPRE_Int HYPRE_CSRMatrixGetNumRows_flt ( HYPRE_CSRMatrix matrix, HYPRE_Int *num_rows );
HYPRE_Int HYPRE_CSRMatrixGetNumRows_long_dbl ( HYPRE_CSRMatrix matrix, HYPRE_Int *num_rows );

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate_dbl ( void );
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate_flt ( void );
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate_long_dbl ( void );
HYPRE_Int HYPRE_MappedMatrixDestroy_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixDestroy_flt ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixDestroy_long_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixLimitedDestroy_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixLimitedDestroy_flt ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixLimitedDestroy_long_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixInitialize_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixInitialize_flt ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixInitialize_long_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixAssemble_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixAssemble_flt ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixAssemble_long_dbl ( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint_dbl ( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint_flt ( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint_long_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixGetColIndex_dbl ( HYPRE_MappedMatrix matrix, HYPRE_Int j );
HYPRE_Int HYPRE_MappedMatrixGetColIndex_flt ( HYPRE_MappedMatrix matrix, HYPRE_Int j );
HYPRE_Int HYPRE_MappedMatrixGetColIndex_long_dbl ( HYPRE_MappedMatrix matrix, HYPRE_Int j );
void *HYPRE_MappedMatrixGetMatrix_dbl ( HYPRE_MappedMatrix matrix );
void *HYPRE_MappedMatrixGetMatrix_flt ( HYPRE_MappedMatrix matrix );
void *HYPRE_MappedMatrixGetMatrix_long_dbl ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixSetMatrix_dbl ( HYPRE_MappedMatrix matrix, void *matrix_data );
HYPRE_Int HYPRE_MappedMatrixSetMatrix_flt ( HYPRE_MappedMatrix matrix, void *matrix_data );
HYPRE_Int HYPRE_MappedMatrixSetMatrix_long_dbl ( HYPRE_MappedMatrix matrix, void *matrix_data );
HYPRE_Int HYPRE_MappedMatrixSetColMap_dbl ( HYPRE_MappedMatrix matrix, HYPRE_Int (*ColMap )(HYPRE_Int, void *));
HYPRE_Int HYPRE_MappedMatrixSetColMap_flt ( HYPRE_MappedMatrix matrix, HYPRE_Int (*ColMap )(HYPRE_Int, void *));
HYPRE_Int HYPRE_MappedMatrixSetColMap_long_dbl ( HYPRE_MappedMatrix matrix, HYPRE_Int (*ColMap )(HYPRE_Int, void *));
HYPRE_Int HYPRE_MappedMatrixSetMapData_dbl ( HYPRE_MappedMatrix matrix, void *MapData );
HYPRE_Int HYPRE_MappedMatrixSetMapData_flt ( HYPRE_MappedMatrix matrix, void *MapData );
HYPRE_Int HYPRE_MappedMatrixSetMapData_long_dbl ( HYPRE_MappedMatrix matrix, void *MapData );

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate_dbl ( void );
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate_flt ( void );
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate_long_dbl ( void );
HYPRE_Int HYPRE_MultiblockMatrixDestroy_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixDestroy_flt ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixDestroy_long_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixLimitedDestroy_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixLimitedDestroy_flt ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixLimitedDestroy_long_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixInitialize_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixInitialize_flt ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixInitialize_long_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixAssemble_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixAssemble_flt ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixAssemble_long_dbl ( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint_dbl ( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint_flt ( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint_long_dbl ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixSetNumSubmatrices_dbl ( HYPRE_MultiblockMatrix matrix, HYPRE_Int n );
HYPRE_Int HYPRE_MultiblockMatrixSetNumSubmatrices_flt ( HYPRE_MultiblockMatrix matrix, HYPRE_Int n );
HYPRE_Int HYPRE_MultiblockMatrixSetNumSubmatrices_long_dbl ( HYPRE_MultiblockMatrix matrix, HYPRE_Int n );
HYPRE_Int HYPRE_MultiblockMatrixSetSubmatrixType_dbl ( HYPRE_MultiblockMatrix matrix, HYPRE_Int j,
                                                       HYPRE_Int type );
HYPRE_Int HYPRE_MultiblockMatrixSetSubmatrixType_flt ( HYPRE_MultiblockMatrix matrix, HYPRE_Int j,
                                                       HYPRE_Int type );
HYPRE_Int HYPRE_MultiblockMatrixSetSubmatrixType_long_dbl ( HYPRE_MultiblockMatrix matrix, HYPRE_Int j,
                                                       HYPRE_Int type );

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate_dbl ( HYPRE_Int size );
HYPRE_Vector HYPRE_VectorCreate_flt ( HYPRE_Int size );
HYPRE_Vector HYPRE_VectorCreate_long_dbl ( HYPRE_Int size );
HYPRE_Int HYPRE_VectorDestroy_dbl ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorDestroy_flt ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorDestroy_long_dbl ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorInitialize_dbl ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorInitialize_flt ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorInitialize_long_dbl ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorPrint_dbl ( HYPRE_Vector vector, char *file_name );
HYPRE_Int HYPRE_VectorPrint_flt ( HYPRE_Vector vector, char *file_name );
HYPRE_Int HYPRE_VectorPrint_long_dbl ( HYPRE_Vector vector, char *file_name );
HYPRE_Vector HYPRE_VectorRead_dbl ( char *file_name );
HYPRE_Vector HYPRE_VectorRead_flt ( char *file_name );
HYPRE_Vector HYPRE_VectorRead_long_dbl ( char *file_name );

/* mapped_matrix.c */
hypre_MappedMatrix *hypre_MappedMatrixCreate_dbl ( void );
hypre_MappedMatrix *hypre_MappedMatrixCreate_flt ( void );
hypre_MappedMatrix *hypre_MappedMatrixCreate_long_dbl ( void );
HYPRE_Int hypre_MappedMatrixDestroy_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixDestroy_flt ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixDestroy_long_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixLimitedDestroy_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixLimitedDestroy_flt ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixLimitedDestroy_long_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixInitialize_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixInitialize_flt ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixInitialize_long_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixAssemble_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixAssemble_flt ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixAssemble_long_dbl ( hypre_MappedMatrix *matrix );
void hypre_MappedMatrixPrint_dbl ( hypre_MappedMatrix *matrix );
void hypre_MappedMatrixPrint_flt ( hypre_MappedMatrix *matrix );
void hypre_MappedMatrixPrint_long_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixGetColIndex_dbl ( hypre_MappedMatrix *matrix, HYPRE_Int j );
HYPRE_Int hypre_MappedMatrixGetColIndex_flt ( hypre_MappedMatrix *matrix, HYPRE_Int j );
HYPRE_Int hypre_MappedMatrixGetColIndex_long_dbl ( hypre_MappedMatrix *matrix, HYPRE_Int j );
void *hypre_MappedMatrixGetMatrix_dbl ( hypre_MappedMatrix *matrix );
void *hypre_MappedMatrixGetMatrix_flt ( hypre_MappedMatrix *matrix );
void *hypre_MappedMatrixGetMatrix_long_dbl ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixSetMatrix_dbl ( hypre_MappedMatrix *matrix, void *matrix_data );
HYPRE_Int hypre_MappedMatrixSetMatrix_flt ( hypre_MappedMatrix *matrix, void *matrix_data );
HYPRE_Int hypre_MappedMatrixSetMatrix_long_dbl ( hypre_MappedMatrix *matrix, void *matrix_data );
HYPRE_Int hypre_MappedMatrixSetColMap_dbl ( hypre_MappedMatrix *matrix, HYPRE_Int (*ColMap )(HYPRE_Int, void *));
HYPRE_Int hypre_MappedMatrixSetColMap_flt ( hypre_MappedMatrix *matrix, HYPRE_Int (*ColMap )(HYPRE_Int, void *));
HYPRE_Int hypre_MappedMatrixSetColMap_long_dbl ( hypre_MappedMatrix *matrix, HYPRE_Int (*ColMap )(HYPRE_Int, void *));
HYPRE_Int hypre_MappedMatrixSetMapData_dbl ( hypre_MappedMatrix *matrix, void *map_data );
HYPRE_Int hypre_MappedMatrixSetMapData_flt ( hypre_MappedMatrix *matrix, void *map_data );
HYPRE_Int hypre_MappedMatrixSetMapData_long_dbl ( hypre_MappedMatrix *matrix, void *map_data );

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate_dbl ( void );
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate_flt ( void );
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate_long_dbl ( void );
HYPRE_Int hypre_MultiblockMatrixDestroy_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixDestroy_flt ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixDestroy_long_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixLimitedDestroy_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixLimitedDestroy_flt ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixLimitedDestroy_long_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixInitialize_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixInitialize_flt ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixInitialize_long_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixAssemble_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixAssemble_flt ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixAssemble_long_dbl ( hypre_MultiblockMatrix *matrix );
void hypre_MultiblockMatrixPrint_dbl ( hypre_MultiblockMatrix *matrix );
void hypre_MultiblockMatrixPrint_flt ( hypre_MultiblockMatrix *matrix );
void hypre_MultiblockMatrixPrint_long_dbl ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixSetNumSubmatrices_dbl ( hypre_MultiblockMatrix *matrix, HYPRE_Int n );
HYPRE_Int hypre_MultiblockMatrixSetNumSubmatrices_flt ( hypre_MultiblockMatrix *matrix, HYPRE_Int n );
HYPRE_Int hypre_MultiblockMatrixSetNumSubmatrices_long_dbl ( hypre_MultiblockMatrix *matrix, HYPRE_Int n );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrixType_dbl ( hypre_MultiblockMatrix *matrix, HYPRE_Int j,
                                                       HYPRE_Int type );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrixType_flt ( hypre_MultiblockMatrix *matrix, HYPRE_Int j,
                                                       HYPRE_Int type );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrixType_long_dbl ( hypre_MultiblockMatrix *matrix, HYPRE_Int j,
                                                       HYPRE_Int type );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrix_dbl ( hypre_MultiblockMatrix *matrix, HYPRE_Int j,
                                                   void *submatrix );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrix_flt ( hypre_MultiblockMatrix *matrix, HYPRE_Int j,
                                                   void *submatrix );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrix_long_dbl ( hypre_MultiblockMatrix *matrix, HYPRE_Int j,
                                                   void *submatrix );

/* vector.c */
hypre_Vector *hypre_SeqVectorCreate_dbl ( HYPRE_Int size );
hypre_Vector *hypre_SeqVectorCreate_flt ( HYPRE_Int size );
hypre_Vector *hypre_SeqVectorCreate_long_dbl ( HYPRE_Int size );
hypre_Vector *hypre_SeqMultiVectorCreate_dbl ( HYPRE_Int size, HYPRE_Int num_vectors );
hypre_Vector *hypre_SeqMultiVectorCreate_flt ( HYPRE_Int size, HYPRE_Int num_vectors );
hypre_Vector *hypre_SeqMultiVectorCreate_long_dbl ( HYPRE_Int size, HYPRE_Int num_vectors );
HYPRE_Int hypre_SeqVectorDestroy_dbl ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorDestroy_flt ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorDestroy_long_dbl ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorInitialize_v2_dbl( hypre_Vector *vector,
                                            HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_SeqVectorInitialize_v2_flt( hypre_Vector *vector,
                                            HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_SeqVectorInitialize_v2_long_dbl( hypre_Vector *vector,
                                            HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_SeqVectorInitialize_dbl ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorInitialize_flt ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorInitialize_long_dbl ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorSetDataOwner_dbl ( hypre_Vector *vector, HYPRE_Int owns_data );
HYPRE_Int hypre_SeqVectorSetDataOwner_flt ( hypre_Vector *vector, HYPRE_Int owns_data );
HYPRE_Int hypre_SeqVectorSetDataOwner_long_dbl ( hypre_Vector *vector, HYPRE_Int owns_data );
HYPRE_Int hypre_SeqVectorSetSize_dbl ( hypre_Vector *vector, HYPRE_Int size );
HYPRE_Int hypre_SeqVectorSetSize_flt ( hypre_Vector *vector, HYPRE_Int size );
HYPRE_Int hypre_SeqVectorSetSize_long_dbl ( hypre_Vector *vector, HYPRE_Int size );
HYPRE_Int hypre_SeqVectorResize_dbl ( hypre_Vector *vector, HYPRE_Int num_vectors_in );
HYPRE_Int hypre_SeqVectorResize_flt ( hypre_Vector *vector, HYPRE_Int num_vectors_in );
HYPRE_Int hypre_SeqVectorResize_long_dbl ( hypre_Vector *vector, HYPRE_Int num_vectors_in );
hypre_Vector *hypre_SeqVectorRead_dbl ( char *file_name );
hypre_Vector *hypre_SeqVectorRead_flt ( char *file_name );
hypre_Vector *hypre_SeqVectorRead_long_dbl ( char *file_name );
HYPRE_Int hypre_SeqVectorPrint_dbl ( hypre_Vector *vector, char *file_name );
HYPRE_Int hypre_SeqVectorPrint_flt ( hypre_Vector *vector, char *file_name );
HYPRE_Int hypre_SeqVectorPrint_long_dbl ( hypre_Vector *vector, char *file_name );
HYPRE_Int hypre_SeqVectorSetConstantValues_dbl ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValues_flt ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValues_long_dbl ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValuesHost_dbl ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValuesHost_flt ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValuesHost_long_dbl ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValuesDevice_dbl ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValuesDevice_flt ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetConstantValuesDevice_long_dbl ( hypre_Vector *v, HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetRandomValues_dbl ( hypre_Vector *v, HYPRE_Int seed );
HYPRE_Int hypre_SeqVectorSetRandomValues_flt ( hypre_Vector *v, HYPRE_Int seed );
HYPRE_Int hypre_SeqVectorSetRandomValues_long_dbl ( hypre_Vector *v, HYPRE_Int seed );
HYPRE_Int hypre_SeqVectorCopy_dbl ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorCopy_flt ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorCopy_long_dbl ( hypre_Vector *x, hypre_Vector *y );
hypre_Vector *hypre_SeqVectorCloneDeep_dbl ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneDeep_flt ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneDeep_long_dbl ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneDeep_v2_dbl( hypre_Vector *x, HYPRE_MemoryLocation memory_location );
hypre_Vector *hypre_SeqVectorCloneDeep_v2_flt( hypre_Vector *x, HYPRE_MemoryLocation memory_location );
hypre_Vector *hypre_SeqVectorCloneDeep_v2_long_dbl( hypre_Vector *x, HYPRE_MemoryLocation memory_location );
hypre_Vector *hypre_SeqVectorCloneShallow_dbl ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneShallow_flt ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneShallow_long_dbl ( hypre_Vector *x );
HYPRE_Int hypre_SeqVectorMigrate_dbl( hypre_Vector *x, HYPRE_MemoryLocation  memory_location );
HYPRE_Int hypre_SeqVectorMigrate_flt( hypre_Vector *x, HYPRE_MemoryLocation  memory_location );
HYPRE_Int hypre_SeqVectorMigrate_long_dbl( hypre_Vector *x, HYPRE_MemoryLocation  memory_location );
HYPRE_Int hypre_SeqVectorScale_dbl( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScale_flt( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScale_long_dbl( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScaleHost_dbl( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScaleHost_flt( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScaleHost_long_dbl( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScaleDevice_dbl( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScaleDevice_flt( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorScaleDevice_long_dbl( HYPRE_Complex alpha, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpy_dbl ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpy_flt ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpy_long_dbl ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpyHost_dbl ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpyHost_flt ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpyHost_long_dbl ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpyDevice_dbl ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpyDevice_flt ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpyDevice_long_dbl ( HYPRE_Complex alpha, hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpyz_dbl ( HYPRE_Complex alpha, hypre_Vector *x,
                                     HYPRE_Complex beta, hypre_Vector *y,
                                     hypre_Vector *z );
HYPRE_Int hypre_SeqVectorAxpyz_flt ( HYPRE_Complex alpha, hypre_Vector *x,
                                     HYPRE_Complex beta, hypre_Vector *y,
                                     hypre_Vector *z );
HYPRE_Int hypre_SeqVectorAxpyz_long_dbl ( HYPRE_Complex alpha, hypre_Vector *x,
                                     HYPRE_Complex beta, hypre_Vector *y,
                                     hypre_Vector *z );
HYPRE_Int hypre_SeqVectorAxpyzDevice_dbl ( HYPRE_Complex alpha, hypre_Vector *x,
                                           HYPRE_Complex beta, hypre_Vector *y,
                                           hypre_Vector *z );
HYPRE_Int hypre_SeqVectorAxpyzDevice_flt ( HYPRE_Complex alpha, hypre_Vector *x,
                                           HYPRE_Complex beta, hypre_Vector *y,
                                           hypre_Vector *z );
HYPRE_Int hypre_SeqVectorAxpyzDevice_long_dbl ( HYPRE_Complex alpha, hypre_Vector *x,
                                           HYPRE_Complex beta, hypre_Vector *y,
                                           hypre_Vector *z );
HYPRE_Real hypre_SeqVectorInnerProd_dbl ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProd_flt ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProd_long_dbl ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProdHost_dbl ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProdHost_flt ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProdHost_long_dbl ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProdDevice_dbl ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProdDevice_flt ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProdDevice_long_dbl ( hypre_Vector *x, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorMassInnerProd_dbl(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                           HYPRE_Int unroll, HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd_flt(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                           HYPRE_Int unroll, HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd_long_dbl(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                           HYPRE_Int unroll, HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd4_dbl(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                            HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd4_flt(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                            HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd4_long_dbl(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                            HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd8_dbl(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                            HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd8_flt(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                            HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd8_long_dbl(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,
                                            HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassDotpTwo_dbl(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                         HYPRE_Int k, HYPRE_Int unroll,  HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo_flt(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                         HYPRE_Int k, HYPRE_Int unroll,  HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo_long_dbl(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                         HYPRE_Int k, HYPRE_Int unroll,  HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo4_dbl(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                          HYPRE_Int k, HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo4_flt(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                          HYPRE_Int k, HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo4_long_dbl(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                          HYPRE_Int k, HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo8_dbl(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                          HYPRE_Int k,  HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo8_flt(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                          HYPRE_Int k,  HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo8_long_dbl(hypre_Vector *x, hypre_Vector *y, hypre_Vector **z,
                                          HYPRE_Int k,  HYPRE_Real *result_x, HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassAxpy_dbl(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                      HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_SeqVectorMassAxpy_flt(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                      HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_SeqVectorMassAxpy_long_dbl(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                      HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_SeqVectorMassAxpy4_dbl(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                       HYPRE_Int k);
HYPRE_Int hypre_SeqVectorMassAxpy4_flt(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                       HYPRE_Int k);
HYPRE_Int hypre_SeqVectorMassAxpy4_long_dbl(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                       HYPRE_Int k);
HYPRE_Int hypre_SeqVectorMassAxpy8_dbl(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                       HYPRE_Int k);
HYPRE_Int hypre_SeqVectorMassAxpy8_flt(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                       HYPRE_Int k);
HYPRE_Int hypre_SeqVectorMassAxpy8_long_dbl(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y,
                                       HYPRE_Int k);
HYPRE_Complex hypre_SeqVectorSumElts_dbl ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumElts_flt ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumElts_long_dbl ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumEltsHost_dbl ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumEltsHost_flt ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumEltsHost_long_dbl ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumEltsDevice_dbl ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumEltsDevice_flt ( hypre_Vector *vector );
HYPRE_Complex hypre_SeqVectorSumEltsDevice_long_dbl ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorPrefetch_dbl(hypre_Vector *x, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_SeqVectorPrefetch_flt(hypre_Vector *x, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_SeqVectorPrefetch_long_dbl(hypre_Vector *x, HYPRE_MemoryLocation memory_location);
//HYPRE_Int hypre_SeqVectorMax( HYPRE_Complex alpha, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y );

HYPRE_Int hypreDevice_CSRSpAdd_dbl(HYPRE_Int ma, HYPRE_Int mb, HYPRE_Int n, HYPRE_Int nnzA,
                                   HYPRE_Int nnzB, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex alpha, HYPRE_Complex *d_aa,
                                   HYPRE_Int *d_ja_map, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex beta, HYPRE_Complex *d_ab,
                                   HYPRE_Int *d_jb_map, HYPRE_Int *d_num_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,
                                   HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out);
HYPRE_Int hypreDevice_CSRSpAdd_flt(HYPRE_Int ma, HYPRE_Int mb, HYPRE_Int n, HYPRE_Int nnzA,
                                   HYPRE_Int nnzB, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex alpha, HYPRE_Complex *d_aa,
                                   HYPRE_Int *d_ja_map, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex beta, HYPRE_Complex *d_ab,
                                   HYPRE_Int *d_jb_map, HYPRE_Int *d_num_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,
                                   HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out);
HYPRE_Int hypreDevice_CSRSpAdd_long_dbl(HYPRE_Int ma, HYPRE_Int mb, HYPRE_Int n, HYPRE_Int nnzA,
                                   HYPRE_Int nnzB, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex alpha, HYPRE_Complex *d_aa,
                                   HYPRE_Int *d_ja_map, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex beta, HYPRE_Complex *d_ab,
                                   HYPRE_Int *d_jb_map, HYPRE_Int *d_num_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,
                                   HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out);

HYPRE_Int hypreDevice_CSRSpTrans_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                     HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                     HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTrans_flt(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                     HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                     HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTrans_long_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                     HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                     HYPRE_Complex **d_ac_out, HYPRE_Int want_data);

HYPRE_Int hypreDevice_CSRSpTransCusparse_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                             HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                             HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTransCusparse_flt(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                             HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                             HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTransCusparse_long_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                             HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                             HYPRE_Complex **d_ac_out, HYPRE_Int want_data);

HYPRE_Int hypreDevice_CSRSpTransRocsparse_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                              HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                              HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTransRocsparse_flt(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                              HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                              HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTransRocsparse_long_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia,
                                              HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                              HYPRE_Complex **d_ac_out, HYPRE_Int want_data);

HYPRE_Int hypreDevice_CSRSpTransOnemklsparse_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA,
                                                 HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                                 HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTransOnemklsparse_flt(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA,
                                                 HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                                 HYPRE_Complex **d_ac_out, HYPRE_Int want_data);
HYPRE_Int hypreDevice_CSRSpTransOnemklsparse_long_dbl(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA,
                                                 HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                                 HYPRE_Complex **d_ac_out, HYPRE_Int want_data);

HYPRE_Int hypreDevice_CSRSpGemm_dbl(hypre_CSRMatrix *A, hypre_CSRMatrix *B, hypre_CSRMatrix **C_ptr);
HYPRE_Int hypreDevice_CSRSpGemm_flt(hypre_CSRMatrix *A, hypre_CSRMatrix *B, hypre_CSRMatrix **C_ptr);
HYPRE_Int hypreDevice_CSRSpGemm_long_dbl(hypre_CSRMatrix *A, hypre_CSRMatrix *B, hypre_CSRMatrix **C_ptr);

HYPRE_Int hypreDevice_CSRSpGemmCusparseGenericAPI_dbl(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                                      HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int nnzB,
                                                      HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,
                                                      HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out);
HYPRE_Int hypreDevice_CSRSpGemmCusparseGenericAPI_flt(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                                      HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int nnzB,
                                                      HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,
                                                      HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out);
HYPRE_Int hypreDevice_CSRSpGemmCusparseGenericAPI_long_dbl(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                                      HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int nnzB,
                                                      HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,
                                                      HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out);

HYPRE_Int hypre_SeqVectorElmdivpy_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorElmdivpy_flt( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorElmdivpy_long_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y );
HYPRE_Int hypre_SeqVectorElmdivpyMarked_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                             HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyMarked_flt( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                             HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyMarked_long_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                             HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyHost_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                           HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyHost_flt( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                           HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyHost_long_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                           HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyDevice_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                             HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyDevice_flt( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                             HYPRE_Int *marker, HYPRE_Int marker_val );
HYPRE_Int hypre_SeqVectorElmdivpyDevice_long_dbl( hypre_Vector *x, hypre_Vector *b, hypre_Vector *y,
                                             HYPRE_Int *marker, HYPRE_Int marker_val );

HYPRE_Int hypre_CSRMatrixSpMVDevice_dbl( HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                         hypre_Vector *x,
                                         HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int fill );
HYPRE_Int hypre_CSRMatrixSpMVDevice_flt( HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                         hypre_Vector *x,
                                         HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int fill );
HYPRE_Int hypre_CSRMatrixSpMVDevice_long_dbl( HYPRE_Int trans, HYPRE_Complex alpha, hypre_CSRMatrix *A,
                                         hypre_Vector *x,
                                         HYPRE_Complex beta, hypre_Vector *y, HYPRE_Int fill );

HYPRE_Int hypre_CSRMatrixIntSpMVDevice_dbl( HYPRE_Int num_rows, HYPRE_Int num_nonzeros,
                                            HYPRE_Int alpha, HYPRE_Int *d_ia, HYPRE_Int *d_ja,
                                            HYPRE_Int *d_a, HYPRE_Int *d_x, HYPRE_Int beta,
                                            HYPRE_Int *d_y );
HYPRE_Int hypre_CSRMatrixIntSpMVDevice_flt( HYPRE_Int num_rows, HYPRE_Int num_nonzeros,
                                            HYPRE_Int alpha, HYPRE_Int *d_ia, HYPRE_Int *d_ja,
                                            HYPRE_Int *d_a, HYPRE_Int *d_x, HYPRE_Int beta,
                                            HYPRE_Int *d_y );
HYPRE_Int hypre_CSRMatrixIntSpMVDevice_long_dbl( HYPRE_Int num_rows, HYPRE_Int num_nonzeros,
                                            HYPRE_Int alpha, HYPRE_Int *d_ia, HYPRE_Int *d_ja,
                                            HYPRE_Int *d_a, HYPRE_Int *d_x, HYPRE_Int beta,
                                            HYPRE_Int *d_y );

#if defined(HYPRE_USING_CUSPARSE)  ||\
    defined(HYPRE_USING_ROCSPARSE) ||\
    defined(HYPRE_USING_ONEMKLSPARSE)
/*hypre_CsrsvData* hypre_CsrsvDataCreate();
HYPRE_Int hypre_CsrsvDataDestroy(hypre_CsrsvData *data);
hypre_GpuMatData* hypre_GpuMatDataCreate();
HYPRE_Int hypre_GPUMatDataSetCSRData(hypre_CSRMatrix *matrix);
HYPRE_Int hypre_GpuMatDataDestroy(hypre_GpuMatData *data);
hypre_GpuMatData* hypre_CSRMatrixGetGPUMatData(hypre_CSRMatrix *matrix);*/

#define hypre_CSRMatrixGPUMatDescr(matrix)       ( hypre_GpuMatDataMatDescr(hypre_CSRMatrixGetGPUMatData(matrix)) )
#define hypre_CSRMatrixGPUMatInfo(matrix)        ( hypre_GpuMatDataMatInfo (hypre_CSRMatrixGetGPUMatData(matrix)) )
#define hypre_CSRMatrixGPUMatHandle(matrix)      ( hypre_GpuMatDataMatHandle (hypre_CSRMatrixGetGPUMatData(matrix)) )
#define hypre_CSRMatrixGPUMatSpMVBuffer(matrix)  ( hypre_GpuMatDataSpMVBuffer (hypre_CSRMatrixGetGPUMatData(matrix)) )
#endif

HYPRE_Int hypre_CSRMatrixSpMVAnalysisDevice_dbl(hypre_CSRMatrix *matrix);
HYPRE_Int hypre_CSRMatrixSpMVAnalysisDevice_flt(hypre_CSRMatrix *matrix);
HYPRE_Int hypre_CSRMatrixSpMVAnalysisDevice_long_dbl(hypre_CSRMatrix *matrix);

#endif

#endif
