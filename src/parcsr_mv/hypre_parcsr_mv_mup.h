
/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * Header file of multiprecision function prototypes.
 * This is needed for mixed-precision algorithm development.
 *****************************************************************************/

#ifndef HYPRE_PARCSR_MV_MUP_HEADER
#define HYPRE_PARCSR_MV_MUP_HEADER

#include "_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined (HYPRE_MIXED_PRECISION)

HYPRE_Int hypre_MatTCommPkgCreate_flt  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatTCommPkgCreate_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatTCommPkgCreate_long_dbl  ( hypre_ParCSRMatrix *A );
void hypre_MatTCommPkgCreate_core_flt  ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_rows_diag,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd, HYPRE_BigInt *row_starts,
                                    HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag, HYPRE_Int *mat_j_diag,
                                    HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd, HYPRE_Int data, HYPRE_Int *p_num_recvs,
                                    HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends,
                                    HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
void hypre_MatTCommPkgCreate_core_dbl  ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_rows_diag,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd, HYPRE_BigInt *row_starts,
                                    HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag, HYPRE_Int *mat_j_diag,
                                    HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd, HYPRE_Int data, HYPRE_Int *p_num_recvs,
                                    HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends,
                                    HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
void hypre_MatTCommPkgCreate_core_long_dbl  ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_rows_diag,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd, HYPRE_BigInt *row_starts,
                                    HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag, HYPRE_Int *mat_j_diag,
                                    HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd, HYPRE_Int data, HYPRE_Int *p_num_recvs,
                                    HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends,
                                    HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
void hypre_RowsWithColumn_flt  ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                            HYPRE_Int num_rows_diag, HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag,
                            HYPRE_Int *mat_j_diag, HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd );
void hypre_RowsWithColumn_dbl  ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                            HYPRE_Int num_rows_diag, HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag,
                            HYPRE_Int *mat_j_diag, HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd );
void hypre_RowsWithColumn_long_dbl  ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                            HYPRE_Int num_rows_diag, HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag,
                            HYPRE_Int *mat_j_diag, HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd );
void hypre_RowsWithColumn_original_flt  ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                                     hypre_ParCSRMatrix *A );
void hypre_RowsWithColumn_original_dbl  ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                                     hypre_ParCSRMatrix *A );
void hypre_RowsWithColumn_original_long_dbl  ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                                     hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                          hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                          hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                          hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC3_flt (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC3_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC3_long_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCD3_flt (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                           hypre_ParCSRMatrix **A_FF_ptr, hypre_float **D_lambda_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCD3_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                           hypre_ParCSRMatrix **A_FF_ptr, hypre_double **D_lambda_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCD3_long_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                           hypre_ParCSRMatrix **A_FF_ptr, hypre_long_double **D_lambda_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCHost_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                              hypre_ParCSRMatrix **A_FC_ptr,
                                              hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCHost_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                              hypre_ParCSRMatrix **A_FC_ptr,
                                              hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCHost_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                              hypre_ParCSRMatrix **A_FC_ptr,
                                              hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_flt  ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                          HYPRE_BigInt *row_partitioning, HYPRE_BigInt *col_partitioning, HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_dbl  ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                          HYPRE_BigInt *row_partitioning, HYPRE_BigInt *col_partitioning, HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_long_dbl  ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                          HYPRE_BigInt *row_partitioning, HYPRE_BigInt *col_partitioning, HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning_flt  ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                                              HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning_dbl  ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                                              HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning_long_dbl  ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                                              HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixCreate_flt  ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                     HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts,
                                     HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag, HYPRE_Int num_nonzeros_offd,
                                     HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                     HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts,
                                     HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag, HYPRE_Int num_nonzeros_offd,
                                     HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                     HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts,
                                     HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag, HYPRE_Int num_nonzeros_offd,
                                     HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy_flt  ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy_dbl  ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy_long_dbl  ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning_flt  ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning_dbl  ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning_long_dbl  ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetComm_flt  ( HYPRE_ParCSRMatrix matrix, MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetComm_dbl  ( HYPRE_ParCSRMatrix matrix, MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetComm_long_dbl  ( HYPRE_ParCSRMatrix matrix, MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetDims_flt  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *M, HYPRE_BigInt *N );
HYPRE_Int HYPRE_ParCSRMatrixGetDims_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *M, HYPRE_BigInt *N );
HYPRE_Int HYPRE_ParCSRMatrixGetDims_long_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *M, HYPRE_BigInt *N );
HYPRE_Int HYPRE_ParCSRMatrixGetGlobalRowPartitioning_flt  ( HYPRE_ParCSRMatrix matrix,
                                                       HYPRE_Int all_procs, HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetGlobalRowPartitioning_dbl  ( HYPRE_ParCSRMatrix matrix,
                                                       HYPRE_Int all_procs, HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetGlobalRowPartitioning_long_dbl  ( HYPRE_ParCSRMatrix matrix,
                                                       HYPRE_Int all_procs, HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange_flt  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange_long_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetRow_flt  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, hypre_float **values );
HYPRE_Int HYPRE_ParCSRMatrixGetRow_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, hypre_double **values );
HYPRE_Int HYPRE_ParCSRMatrixGetRow_long_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, hypre_long_double **values );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning_flt  ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning_dbl  ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning_long_dbl  ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixInitialize_flt  ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixInitialize_dbl  ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixInitialize_long_dbl  ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixMatvec_flt  ( hypre_float alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                     hypre_float beta, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvec_dbl  ( hypre_double alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                     hypre_double beta, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvec_long_dbl  ( hypre_long_double alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                     hypre_long_double beta, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT_flt  ( hypre_float alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                      hypre_float beta, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT_dbl  ( hypre_double alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                      hypre_double beta, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT_long_dbl  ( hypre_long_double alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                      hypre_long_double beta, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixPrint_flt  ( HYPRE_ParCSRMatrix matrix, const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixPrint_dbl  ( HYPRE_ParCSRMatrix matrix, const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixPrint_long_dbl  ( HYPRE_ParCSRMatrix matrix, const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixRead_flt  ( MPI_Comm comm, const char *file_name,
                                   HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixRead_dbl  ( MPI_Comm comm, const char *file_name,
                                   HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixRead_long_dbl  ( MPI_Comm comm, const char *file_name,
                                   HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow_flt  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, hypre_float **values );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, hypre_double **values );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow_long_dbl  ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, hypre_long_double **values );
HYPRE_Int HYPRE_ParMultiVectorCreate_flt  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                       HYPRE_BigInt *partitioning, HYPRE_Int number_vectors, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParMultiVectorCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                       HYPRE_BigInt *partitioning, HYPRE_Int number_vectors, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParMultiVectorCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                       HYPRE_BigInt *partitioning, HYPRE_Int number_vectors, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorAxpy_flt  ( hypre_float alpha, HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorAxpy_dbl  ( hypre_double alpha, HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorAxpy_long_dbl  ( hypre_long_double alpha, HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_ParVector HYPRE_ParVectorCloneShallow_flt  ( HYPRE_ParVector x );
HYPRE_ParVector HYPRE_ParVectorCloneShallow_dbl  ( HYPRE_ParVector x );
HYPRE_ParVector HYPRE_ParVectorCloneShallow_long_dbl  ( HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorCopy_flt  ( HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorCopy_dbl  ( HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorCopy_long_dbl  ( HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorCreate_flt  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                  HYPRE_BigInt *partitioning, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                  HYPRE_BigInt *partitioning, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                  HYPRE_BigInt *partitioning, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorDestroy_flt  ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorDestroy_dbl  ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorDestroy_long_dbl  ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorGetValues_flt  ( HYPRE_ParVector vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, hypre_float *values);
HYPRE_Int HYPRE_ParVectorGetValues_dbl  ( HYPRE_ParVector vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, hypre_double *values);
HYPRE_Int HYPRE_ParVectorGetValues_long_dbl  ( HYPRE_ParVector vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, hypre_long_double *values);
HYPRE_Int HYPRE_ParVectorInitialize_flt  ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInitialize_dbl  ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInitialize_long_dbl  ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInnerProd_flt  ( HYPRE_ParVector x, HYPRE_ParVector y, hypre_float *prod );
HYPRE_Int HYPRE_ParVectorInnerProd_dbl  ( HYPRE_ParVector x, HYPRE_ParVector y, hypre_double *prod );
HYPRE_Int HYPRE_ParVectorInnerProd_long_dbl  ( HYPRE_ParVector x, HYPRE_ParVector y, hypre_long_double *prod );
HYPRE_Int HYPRE_ParVectorPrint_flt  ( HYPRE_ParVector vector, const char *file_name );
HYPRE_Int HYPRE_ParVectorPrint_dbl  ( HYPRE_ParVector vector, const char *file_name );
HYPRE_Int HYPRE_ParVectorPrint_long_dbl  ( HYPRE_ParVector vector, const char *file_name );
HYPRE_Int HYPRE_ParVectorRead_flt  ( MPI_Comm comm, const char *file_name, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorRead_dbl  ( MPI_Comm comm, const char *file_name, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorRead_long_dbl  ( MPI_Comm comm, const char *file_name, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorScale_flt  ( hypre_float value, HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorScale_dbl  ( hypre_double value, HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorScale_long_dbl  ( hypre_long_double value, HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorSetConstantValues_flt  ( HYPRE_ParVector vector, hypre_float value );
HYPRE_Int HYPRE_ParVectorSetConstantValues_dbl  ( HYPRE_ParVector vector, hypre_double value );
HYPRE_Int HYPRE_ParVectorSetConstantValues_long_dbl  ( HYPRE_ParVector vector, hypre_long_double value );
HYPRE_Int HYPRE_ParVectorSetRandomValues_flt  ( HYPRE_ParVector vector, HYPRE_Int seed );
HYPRE_Int HYPRE_ParVectorSetRandomValues_dbl  ( HYPRE_ParVector vector, HYPRE_Int seed );
HYPRE_Int HYPRE_ParVectorSetRandomValues_long_dbl  ( HYPRE_ParVector vector, HYPRE_Int seed );
HYPRE_Int HYPRE_VectorToParVector_flt  ( MPI_Comm comm, HYPRE_Vector b, HYPRE_BigInt *partitioning,
                                    HYPRE_ParVector *vector );
HYPRE_Int HYPRE_VectorToParVector_dbl  ( MPI_Comm comm, HYPRE_Vector b, HYPRE_BigInt *partitioning,
                                    HYPRE_ParVector *vector );
HYPRE_Int HYPRE_VectorToParVector_long_dbl  ( MPI_Comm comm, HYPRE_Vector b, HYPRE_BigInt *partitioning,
                                    HYPRE_ParVector *vector );
HYPRE_Int hypre_FillResponseIJDetermineSendProcs_flt  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                   HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                   HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseIJDetermineSendProcs_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                   HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                   HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseIJDetermineSendProcs_long_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                   HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                   HYPRE_Int *response_message_size );
HYPRE_Int hypre_NewCommPkgDestroy_flt  ( hypre_ParCSRMatrix *parcsr_A );
HYPRE_Int hypre_NewCommPkgDestroy_dbl  ( hypre_ParCSRMatrix *parcsr_A );
HYPRE_Int hypre_NewCommPkgDestroy_long_dbl  ( hypre_ParCSRMatrix *parcsr_A );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_flt  ( MPI_Comm  comm, HYPRE_BigInt *col_map_off_d,
                                           HYPRE_BigInt  first_col_diag, HYPRE_Int  num_cols_off_d, HYPRE_BigInt  global_num_cols,
                                           hypre_IJAssumedPart *apart, hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_dbl  ( MPI_Comm  comm, HYPRE_BigInt *col_map_off_d,
                                           HYPRE_BigInt  first_col_diag, HYPRE_Int  num_cols_off_d, HYPRE_BigInt  global_num_cols,
                                           hypre_IJAssumedPart *apart, hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_long_dbl  ( MPI_Comm  comm, HYPRE_BigInt *col_map_off_d,
                                           HYPRE_BigInt  first_col_diag, HYPRE_Int  num_cols_off_d, HYPRE_BigInt  global_num_cols,
                                           hypre_IJAssumedPart *apart, hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_core_flt  ( MPI_Comm comm, HYPRE_BigInt *col_map_off_d,
                                                HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_off_d, HYPRE_BigInt global_num_cols,
                                                HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts,
                                                HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts,
                                                HYPRE_Int **p_send_map_elements, hypre_IJAssumedPart *apart );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_core_dbl  ( MPI_Comm comm, HYPRE_BigInt *col_map_off_d,
                                                HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_off_d, HYPRE_BigInt global_num_cols,
                                                HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts,
                                                HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts,
                                                HYPRE_Int **p_send_map_elements, hypre_IJAssumedPart *apart );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_core_long_dbl  ( MPI_Comm comm, HYPRE_BigInt *col_map_off_d,
                                                HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_off_d, HYPRE_BigInt global_num_cols,
                                                HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts,
                                                HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts,
                                                HYPRE_Int **p_send_map_elements, hypre_IJAssumedPart *apart );
HYPRE_Int hypre_PrintCommpkg_flt  ( hypre_ParCSRMatrix *A, const char *file_name );
HYPRE_Int hypre_PrintCommpkg_dbl  ( hypre_ParCSRMatrix *A, const char *file_name );
HYPRE_Int hypre_PrintCommpkg_long_dbl  ( hypre_ParCSRMatrix *A, const char *file_name );
HYPRE_Int hypre_RangeFillResponseIJDetermineRecvProcs_flt  ( void *p_recv_contact_buf,
                                                        HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                        HYPRE_Int *response_message_size );
HYPRE_Int hypre_RangeFillResponseIJDetermineRecvProcs_dbl  ( void *p_recv_contact_buf,
                                                        HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                        HYPRE_Int *response_message_size );
HYPRE_Int hypre_RangeFillResponseIJDetermineRecvProcs_long_dbl  ( void *p_recv_contact_buf,
                                                        HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                        HYPRE_Int *response_message_size );
HYPRE_Int *hypre_NumbersArray_flt  ( hypre_NumbersNode *node );
HYPRE_Int *hypre_NumbersArray_dbl  ( hypre_NumbersNode *node );
HYPRE_Int *hypre_NumbersArray_long_dbl  ( hypre_NumbersNode *node );
void hypre_NumbersDeleteNode_flt  ( hypre_NumbersNode *node );
void hypre_NumbersDeleteNode_dbl  ( hypre_NumbersNode *node );
void hypre_NumbersDeleteNode_long_dbl  ( hypre_NumbersNode *node );
HYPRE_Int hypre_NumbersEnter_flt  ( hypre_NumbersNode *node, const HYPRE_Int n );
HYPRE_Int hypre_NumbersEnter_dbl  ( hypre_NumbersNode *node, const HYPRE_Int n );
HYPRE_Int hypre_NumbersEnter_long_dbl  ( hypre_NumbersNode *node, const HYPRE_Int n );
HYPRE_Int hypre_NumbersNEntered_flt  ( hypre_NumbersNode *node );
HYPRE_Int hypre_NumbersNEntered_dbl  ( hypre_NumbersNode *node );
HYPRE_Int hypre_NumbersNEntered_long_dbl  ( hypre_NumbersNode *node );
hypre_NumbersNode *hypre_NumbersNewNode_flt  ( void );
hypre_NumbersNode *hypre_NumbersNewNode_dbl  ( void );
hypre_NumbersNode *hypre_NumbersNewNode_long_dbl  ( void );
HYPRE_Int hypre_NumbersQuery_flt  ( hypre_NumbersNode *node, const HYPRE_Int n );
HYPRE_Int hypre_NumbersQuery_dbl  ( hypre_NumbersNode *node, const HYPRE_Int n );
HYPRE_Int hypre_NumbersQuery_long_dbl  ( hypre_NumbersNode *node, const HYPRE_Int n );
void hypre_ParAat_RowSizes_flt  ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i, HYPRE_Int *B_marker,
                             HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                             HYPRE_BigInt *A_col_map_offd, HYPRE_Int *A_ext_i, HYPRE_BigInt *A_ext_j,
                             HYPRE_BigInt *A_ext_row_map, HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
                             HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A, HYPRE_Int num_rows_A_ext,
                             HYPRE_BigInt first_col_diag_A, HYPRE_BigInt first_row_index_A );
void hypre_ParAat_RowSizes_dbl  ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i, HYPRE_Int *B_marker,
                             HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                             HYPRE_BigInt *A_col_map_offd, HYPRE_Int *A_ext_i, HYPRE_BigInt *A_ext_j,
                             HYPRE_BigInt *A_ext_row_map, HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
                             HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A, HYPRE_Int num_rows_A_ext,
                             HYPRE_BigInt first_col_diag_A, HYPRE_BigInt first_row_index_A );
void hypre_ParAat_RowSizes_long_dbl  ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i, HYPRE_Int *B_marker,
                             HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                             HYPRE_BigInt *A_col_map_offd, HYPRE_Int *A_ext_i, HYPRE_BigInt *A_ext_j,
                             HYPRE_BigInt *A_ext_row_map, HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
                             HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A, HYPRE_Int num_rows_A_ext,
                             HYPRE_BigInt first_col_diag_A, HYPRE_BigInt first_row_index_A );
hypre_ParCSRMatrix *hypre_ParCSRAAt_flt  ( hypre_ParCSRMatrix *A );
hypre_ParCSRMatrix *hypre_ParCSRAAt_dbl  ( hypre_ParCSRMatrix *A );
hypre_ParCSRMatrix *hypre_ParCSRAAt_long_dbl  ( hypre_ParCSRMatrix *A );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int data,
                                                 HYPRE_BigInt **pA_ext_row_map );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int data,
                                                 HYPRE_BigInt **pA_ext_row_map );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int data,
                                                 HYPRE_BigInt **pA_ext_row_map );
hypre_IJAssumedPart *hypre_AssumedPartitionCreate_flt  ( MPI_Comm comm, HYPRE_BigInt global_num,
                                                    HYPRE_BigInt start, HYPRE_BigInt end );
hypre_IJAssumedPart *hypre_AssumedPartitionCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt global_num,
                                                    HYPRE_BigInt start, HYPRE_BigInt end );
hypre_IJAssumedPart *hypre_AssumedPartitionCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt global_num,
                                                    HYPRE_BigInt start, HYPRE_BigInt end );
HYPRE_Int hypre_AssumedPartitionDestroy_flt  ( hypre_IJAssumedPart *apart );
HYPRE_Int hypre_AssumedPartitionDestroy_dbl  ( hypre_IJAssumedPart *apart );
HYPRE_Int hypre_AssumedPartitionDestroy_long_dbl  ( hypre_IJAssumedPart *apart );
HYPRE_Int hypre_GetAssumedPartitionProcFromRow_flt  ( MPI_Comm comm, HYPRE_BigInt row,
                                                 HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_Int *proc_id );
HYPRE_Int hypre_GetAssumedPartitionProcFromRow_dbl  ( MPI_Comm comm, HYPRE_BigInt row,
                                                 HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_Int *proc_id );
HYPRE_Int hypre_GetAssumedPartitionProcFromRow_long_dbl  ( MPI_Comm comm, HYPRE_BigInt row,
                                                 HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_Int *proc_id );
HYPRE_Int hypre_GetAssumedPartitionRowRange_flt  ( MPI_Comm comm, HYPRE_Int proc_id,
                                              HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_BigInt *row_start,
                                              HYPRE_BigInt *row_end );
HYPRE_Int hypre_GetAssumedPartitionRowRange_dbl  ( MPI_Comm comm, HYPRE_Int proc_id,
                                              HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_BigInt *row_start,
                                              HYPRE_BigInt *row_end );
HYPRE_Int hypre_GetAssumedPartitionRowRange_long_dbl  ( MPI_Comm comm, HYPRE_Int proc_id,
                                              HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_BigInt *row_start,
                                              HYPRE_BigInt *row_end );
HYPRE_Int hypre_LocateAssumedPartition_flt  ( MPI_Comm comm, HYPRE_BigInt row_start,
                                         HYPRE_BigInt row_end, HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows,
                                         hypre_IJAssumedPart *part, HYPRE_Int myid );
HYPRE_Int hypre_LocateAssumedPartition_dbl  ( MPI_Comm comm, HYPRE_BigInt row_start,
                                         HYPRE_BigInt row_end, HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows,
                                         hypre_IJAssumedPart *part, HYPRE_Int myid );
HYPRE_Int hypre_LocateAssumedPartition_long_dbl  ( MPI_Comm comm, HYPRE_BigInt row_start,
                                         HYPRE_BigInt row_end, HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows,
                                         hypre_IJAssumedPart *part, HYPRE_Int myid );
HYPRE_Int hypre_ParCSRMatrixCreateAssumedPartition_flt  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixCreateAssumedPartition_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixCreateAssumedPartition_long_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParVectorCreateAssumedPartition_flt  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorCreateAssumedPartition_dbl  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorCreateAssumedPartition_long_dbl  ( hypre_ParVector *vector );
HYPRE_Int hypre_BooleanMatTCommPkgCreate_flt  ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatTCommPkgCreate_dbl  ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatTCommPkgCreate_long_dbl  ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatvecCommPkgCreate_flt  ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatvecCommPkgCreate_dbl  ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatvecCommPkgCreate_long_dbl  ( hypre_ParCSRBooleanMatrix *A );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt_flt  ( hypre_ParCSRBooleanMatrix *A );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt_dbl  ( hypre_ParCSRBooleanMatrix *A );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt_long_dbl  ( hypre_ParCSRBooleanMatrix *A );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul_flt  ( hypre_ParCSRBooleanMatrix *A,
                                                    hypre_ParCSRBooleanMatrix *B );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul_dbl  ( hypre_ParCSRBooleanMatrix *A,
                                                    hypre_ParCSRBooleanMatrix *B );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul_long_dbl  ( hypre_ParCSRBooleanMatrix *A,
                                                    hypre_ParCSRBooleanMatrix *B );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt_flt  ( hypre_ParCSRBooleanMatrix *A,
                                                               HYPRE_BigInt **pA_ext_row_map );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt_dbl  ( hypre_ParCSRBooleanMatrix *A,
                                                               HYPRE_BigInt **pA_ext_row_map );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt_long_dbl  ( hypre_ParCSRBooleanMatrix *A,
                                                               HYPRE_BigInt **pA_ext_row_map );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt_flt  ( hypre_ParCSRBooleanMatrix *B,
                                                               hypre_ParCSRBooleanMatrix *A );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt_dbl  ( hypre_ParCSRBooleanMatrix *B,
                                                               hypre_ParCSRBooleanMatrix *A );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt_long_dbl  ( hypre_ParCSRBooleanMatrix *B,
                                                               hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanGenerateDiagAndOffd_flt  ( hypre_CSRBooleanMatrix *A,
                                             hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );
HYPRE_Int hypre_BooleanGenerateDiagAndOffd_dbl  ( hypre_CSRBooleanMatrix *A,
                                             hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );
HYPRE_Int hypre_BooleanGenerateDiagAndOffd_long_dbl  ( hypre_CSRBooleanMatrix *A,
                                             hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );
HYPRE_Int hypre_BuildCSRBooleanMatrixMPIDataType_flt  ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                                   HYPRE_Int *a_i, HYPRE_Int *a_j, hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_BuildCSRBooleanMatrixMPIDataType_dbl  ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                                   HYPRE_Int *a_i, HYPRE_Int *a_j, hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_BuildCSRBooleanMatrixMPIDataType_long_dbl  ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                                   HYPRE_Int *a_i, HYPRE_Int *a_j, hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_CSRBooleanMatrixBigInitialize_flt  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixBigInitialize_dbl  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixBigInitialize_long_dbl  ( hypre_CSRBooleanMatrix *matrix );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate_flt  ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                                       HYPRE_Int num_nonzeros );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate_dbl  ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                                       HYPRE_Int num_nonzeros );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate_long_dbl  ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                                       HYPRE_Int num_nonzeros );
HYPRE_Int hypre_CSRBooleanMatrixDestroy_flt  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixDestroy_dbl  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixDestroy_long_dbl  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixInitialize_flt  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixInitialize_dbl  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixInitialize_long_dbl  ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixPrint_flt  ( hypre_CSRBooleanMatrix *matrix, const char *file_name );
HYPRE_Int hypre_CSRBooleanMatrixPrint_dbl  ( hypre_CSRBooleanMatrix *matrix, const char *file_name );
HYPRE_Int hypre_CSRBooleanMatrixPrint_long_dbl  ( hypre_CSRBooleanMatrix *matrix, const char *file_name );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead_flt  ( const char *file_name );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead_dbl  ( const char *file_name );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead_long_dbl  ( const char *file_name );
HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner_flt  ( hypre_CSRBooleanMatrix *matrix,
                                               HYPRE_Int owns_data );
HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner_dbl  ( hypre_CSRBooleanMatrix *matrix,
                                               HYPRE_Int owns_data );
HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner_long_dbl  ( hypre_CSRBooleanMatrix *matrix,
                                               HYPRE_Int owns_data );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix_flt  ( MPI_Comm comm,
                                                                         hypre_CSRBooleanMatrix *A, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix_dbl  ( MPI_Comm comm,
                                                                         hypre_CSRBooleanMatrix *A, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix_long_dbl  ( MPI_Comm comm,
                                                                         hypre_CSRBooleanMatrix *A, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate_flt  ( MPI_Comm comm,
                                                             HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts,
                                                             HYPRE_BigInt *col_starts, HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag,
                                                             HYPRE_Int num_nonzeros_offd );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate_dbl  ( MPI_Comm comm,
                                                             HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts,
                                                             HYPRE_BigInt *col_starts, HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag,
                                                             HYPRE_Int num_nonzeros_offd );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate_long_dbl  ( MPI_Comm comm,
                                                             HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts,
                                                             HYPRE_BigInt *col_starts, HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag,
                                                             HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRBooleanMatrixDestroy_flt  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixDestroy_dbl  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixDestroy_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange_flt  ( hypre_ParCSRBooleanMatrix *matrix,
                                                   HYPRE_BigInt *row_start, HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                   HYPRE_BigInt *row_start, HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                   HYPRE_BigInt *row_start, HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRBooleanMatrixGetRow_flt  ( hypre_ParCSRBooleanMatrix *mat, HYPRE_BigInt row,
                                            HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixGetRow_dbl  ( hypre_ParCSRBooleanMatrix *mat, HYPRE_BigInt row,
                                            HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixGetRow_long_dbl  ( hypre_ParCSRBooleanMatrix *mat, HYPRE_BigInt row,
                                            HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixInitialize_flt  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixInitialize_dbl  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixInitialize_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixPrint_flt  ( hypre_ParCSRBooleanMatrix *matrix,
                                           const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrint_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                           const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrint_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                           const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ_flt  ( hypre_ParCSRBooleanMatrix *matrix,
                                             const char *filename );
HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                             const char *filename );
HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                             const char *filename );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead_flt  ( MPI_Comm comm, const char *file_name );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead_dbl  ( MPI_Comm comm, const char *file_name );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead_long_dbl  ( MPI_Comm comm, const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow_flt  ( hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt row,
                                                HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow_dbl  ( hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt row,
                                                HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt row,
                                                HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner_flt  ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_col_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_col_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_col_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner_flt  ( hypre_ParCSRBooleanMatrix *matrix,
                                                  HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                  HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                  HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ_flt  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ_dbl  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner_flt  ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_row_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_row_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner_long_dbl  ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_row_starts );
HYPRE_Int hypre_BuildCSRJDataType_flt  ( HYPRE_Int num_nonzeros, hypre_float *a_data, HYPRE_Int *a_j,
                                    hypre_MPI_Datatype *csr_jdata_datatype );
HYPRE_Int hypre_BuildCSRJDataType_dbl  ( HYPRE_Int num_nonzeros, hypre_double *a_data, HYPRE_Int *a_j,
                                    hypre_MPI_Datatype *csr_jdata_datatype );
HYPRE_Int hypre_BuildCSRJDataType_long_dbl  ( HYPRE_Int num_nonzeros, hypre_long_double *a_data, HYPRE_Int *a_j,
                                    hypre_MPI_Datatype *csr_jdata_datatype );
HYPRE_Int hypre_BuildCSRMatrixMPIDataType_flt  ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                            hypre_float *a_data, HYPRE_Int *a_i, HYPRE_Int *a_j,
                                            hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_BuildCSRMatrixMPIDataType_dbl  ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                            hypre_double *a_data, HYPRE_Int *a_i, HYPRE_Int *a_j,
                                            hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_BuildCSRMatrixMPIDataType_long_dbl  ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                            hypre_long_double *a_data, HYPRE_Int *a_i, HYPRE_Int *a_j,
                                            hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_MatvecCommPkgCreate_flt  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatvecCommPkgCreate_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatvecCommPkgCreate_long_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatvecCommPkgDestroy_flt  ( hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_MatvecCommPkgDestroy_dbl  ( hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_MatvecCommPkgDestroy_long_dbl  ( hypre_ParCSRCommPkg *comm_pkg );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_flt  ( HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg,
                                                       void *send_data, void *recv_data );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_dbl  ( HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg,
                                                       void *send_data, void *recv_data );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_long_dbl  ( HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg,
                                                       void *send_data, void *recv_data );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_v2_flt  ( HYPRE_Int job,
                                                          hypre_ParCSRCommPkg *comm_pkg,
                                                          HYPRE_MemoryLocation send_memory_location,
                                                          void *send_data_in,
                                                          HYPRE_MemoryLocation recv_memory_location,
                                                          void *recv_data_in );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_v2_dbl  ( HYPRE_Int job,
                                                          hypre_ParCSRCommPkg *comm_pkg,
                                                          HYPRE_MemoryLocation send_memory_location,
                                                          void *send_data_in,
                                                          HYPRE_MemoryLocation recv_memory_location,
                                                          void *recv_data_in );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_v2_long_dbl  ( HYPRE_Int job,
                                                          hypre_ParCSRCommPkg *comm_pkg,
                                                          HYPRE_MemoryLocation send_memory_location,
                                                          void *send_data_in,
                                                          HYPRE_MemoryLocation recv_memory_location,
                                                          void *recv_data_in );
HYPRE_Int hypre_ParCSRCommHandleDestroy_flt  ( hypre_ParCSRCommHandle *comm_handle );
HYPRE_Int hypre_ParCSRCommHandleDestroy_dbl  ( hypre_ParCSRCommHandle *comm_handle );
HYPRE_Int hypre_ParCSRCommHandleDestroy_long_dbl  ( hypre_ParCSRCommHandle *comm_handle );
HYPRE_Int hypre_ParCSRCommPkgCreate_flt (MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd,
                                    hypre_ParCSRCommPkg *comm_pkg);
HYPRE_Int hypre_ParCSRCommPkgCreate_dbl (MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd,
                                    hypre_ParCSRCommPkg *comm_pkg);
HYPRE_Int hypre_ParCSRCommPkgCreate_long_dbl (MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd,
                                    hypre_ParCSRCommPkg *comm_pkg);
HYPRE_Int hypre_ParCSRCommPkgCreateAndFill_flt  ( MPI_Comm comm, HYPRE_Int num_recvs,
                                             HYPRE_Int *recv_procs, HYPRE_Int *recv_vec_starts,
                                             HYPRE_Int num_sends, HYPRE_Int *send_procs,
                                             HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                             hypre_ParCSRCommPkg **comm_pkg_ptr );
HYPRE_Int hypre_ParCSRCommPkgCreateAndFill_dbl  ( MPI_Comm comm, HYPRE_Int num_recvs,
                                             HYPRE_Int *recv_procs, HYPRE_Int *recv_vec_starts,
                                             HYPRE_Int num_sends, HYPRE_Int *send_procs,
                                             HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                             hypre_ParCSRCommPkg **comm_pkg_ptr );
HYPRE_Int hypre_ParCSRCommPkgCreateAndFill_long_dbl  ( MPI_Comm comm, HYPRE_Int num_recvs,
                                             HYPRE_Int *recv_procs, HYPRE_Int *recv_vec_starts,
                                             HYPRE_Int num_sends, HYPRE_Int *send_procs,
                                             HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                             hypre_ParCSRCommPkg **comm_pkg_ptr );
void hypre_ParCSRCommPkgCreate_core_flt  ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                      HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_cols_diag,
                                      HYPRE_Int num_cols_offd, HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs,
                                      HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs,
                                      HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
void hypre_ParCSRCommPkgCreate_core_dbl  ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                      HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_cols_diag,
                                      HYPRE_Int num_cols_offd, HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs,
                                      HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs,
                                      HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
void hypre_ParCSRCommPkgCreate_core_long_dbl  ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                      HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_cols_diag,
                                      HYPRE_Int num_cols_offd, HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs,
                                      HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs,
                                      HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
HYPRE_Int hypre_ParCSRCommPkgUpdateVecStarts_flt  ( hypre_ParCSRCommPkg *comm_pkg, hypre_ParVector *x );
HYPRE_Int hypre_ParCSRCommPkgUpdateVecStarts_dbl  ( hypre_ParCSRCommPkg *comm_pkg, hypre_ParVector *x );
HYPRE_Int hypre_ParCSRCommPkgUpdateVecStarts_long_dbl  ( hypre_ParCSRCommPkg *comm_pkg, hypre_ParVector *x );
HYPRE_Int hypre_ParCSRFindExtendCommPkg_flt (MPI_Comm comm, HYPRE_BigInt global_num_cols,
                                        HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_diag, HYPRE_BigInt *col_starts,
                                        hypre_IJAssumedPart *apart, HYPRE_Int indices_len, HYPRE_BigInt *indices,
                                        hypre_ParCSRCommPkg **extend_comm_pkg);
HYPRE_Int hypre_ParCSRFindExtendCommPkg_dbl (MPI_Comm comm, HYPRE_BigInt global_num_cols,
                                        HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_diag, HYPRE_BigInt *col_starts,
                                        hypre_IJAssumedPart *apart, HYPRE_Int indices_len, HYPRE_BigInt *indices,
                                        hypre_ParCSRCommPkg **extend_comm_pkg);
HYPRE_Int hypre_ParCSRFindExtendCommPkg_long_dbl (MPI_Comm comm, HYPRE_BigInt global_num_cols,
                                        HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_diag, HYPRE_BigInt *col_starts,
                                        hypre_IJAssumedPart *apart, HYPRE_Int indices_len, HYPRE_BigInt *indices,
                                        hypre_ParCSRCommPkg **extend_comm_pkg);
void hypre_ParCSRMatrixCopy_C_flt  ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixCopy_C_dbl  ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixCopy_C_long_dbl  ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixZero_F_flt  ( hypre_ParCSRMatrix *P, HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixZero_F_dbl  ( hypre_ParCSRMatrix *P, HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixZero_F_long_dbl  ( hypre_ParCSRMatrix *P, HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F_flt  ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                          HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F_dbl  ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                          HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F_long_dbl  ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                          HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatmul_FC_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                         HYPRE_Int *CF_marker, HYPRE_Int *dof_func,
                                         HYPRE_Int *dof_func_offd );
hypre_ParCSRMatrix *hypre_ParMatmul_FC_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                         HYPRE_Int *CF_marker, HYPRE_Int *dof_func,
                                         HYPRE_Int *dof_func_offd );
hypre_ParCSRMatrix *hypre_ParMatmul_FC_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                         HYPRE_Int *CF_marker, HYPRE_Int *dof_func,
                                         HYPRE_Int *dof_func_offd );
void hypre_ParMatmul_RowSizes_Marked_flt  ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i,
                                       HYPRE_Int **B_marker, HYPRE_Int *A_diag_i,
                                       HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i,
                                       HYPRE_Int *A_offd_j, HYPRE_Int *B_diag_i,
                                       HYPRE_Int *B_diag_j, HYPRE_Int *B_offd_i,
                                       HYPRE_Int *B_offd_j, HYPRE_Int *B_ext_diag_i,
                                       HYPRE_Int *B_ext_diag_j, HYPRE_Int *B_ext_offd_i,
                                       HYPRE_Int *B_ext_offd_j, HYPRE_Int *map_B_to_C,
                                       HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
                                       HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A,
                                       HYPRE_Int allsquare, HYPRE_Int num_cols_diag_B,
                                       HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C,
                                       HYPRE_Int *CF_marker, HYPRE_Int *dof_func,
                                       HYPRE_Int *dof_func_offd );
void hypre_ParMatmul_RowSizes_Marked_dbl  ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i,
                                       HYPRE_Int **B_marker, HYPRE_Int *A_diag_i,
                                       HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i,
                                       HYPRE_Int *A_offd_j, HYPRE_Int *B_diag_i,
                                       HYPRE_Int *B_diag_j, HYPRE_Int *B_offd_i,
                                       HYPRE_Int *B_offd_j, HYPRE_Int *B_ext_diag_i,
                                       HYPRE_Int *B_ext_diag_j, HYPRE_Int *B_ext_offd_i,
                                       HYPRE_Int *B_ext_offd_j, HYPRE_Int *map_B_to_C,
                                       HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
                                       HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A,
                                       HYPRE_Int allsquare, HYPRE_Int num_cols_diag_B,
                                       HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C,
                                       HYPRE_Int *CF_marker, HYPRE_Int *dof_func,
                                       HYPRE_Int *dof_func_offd );
void hypre_ParMatmul_RowSizes_Marked_long_dbl  ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i,
                                       HYPRE_Int **B_marker, HYPRE_Int *A_diag_i,
                                       HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i,
                                       HYPRE_Int *A_offd_j, HYPRE_Int *B_diag_i,
                                       HYPRE_Int *B_diag_j, HYPRE_Int *B_offd_i,
                                       HYPRE_Int *B_offd_j, HYPRE_Int *B_ext_diag_i,
                                       HYPRE_Int *B_ext_diag_j, HYPRE_Int *B_ext_offd_i,
                                       HYPRE_Int *B_ext_offd_j, HYPRE_Int *map_B_to_C,
                                       HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
                                       HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A,
                                       HYPRE_Int allsquare, HYPRE_Int num_cols_diag_B,
                                       HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C,
                                       HYPRE_Int *CF_marker, HYPRE_Int *dof_func,
                                       HYPRE_Int *dof_func_offd );
void hypre_ParMatScaleDiagInv_F_flt  ( hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *A,
                                  hypre_float weight, HYPRE_Int *CF_marker );
void hypre_ParMatScaleDiagInv_F_dbl  ( hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *A,
                                  hypre_double weight, HYPRE_Int *CF_marker );
void hypre_ParMatScaleDiagInv_F_long_dbl  ( hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *A,
                                  hypre_long_double weight, HYPRE_Int *CF_marker );
HYPRE_Int hypre_ExchangeExternalRowsInit_flt ( hypre_CSRMatrix *B_ext, hypre_ParCSRCommPkg *comm_pkg_A,
                                          void **request_ptr);
HYPRE_Int hypre_ExchangeExternalRowsInit_dbl ( hypre_CSRMatrix *B_ext, hypre_ParCSRCommPkg *comm_pkg_A,
                                          void **request_ptr);
HYPRE_Int hypre_ExchangeExternalRowsInit_long_dbl ( hypre_CSRMatrix *B_ext, hypre_ParCSRCommPkg *comm_pkg_A,
                                          void **request_ptr);
hypre_CSRMatrix* hypre_ExchangeExternalRowsWait_flt (void *vequest);
hypre_CSRMatrix* hypre_ExchangeExternalRowsWait_dbl (void *vequest);
hypre_CSRMatrix* hypre_ExchangeExternalRowsWait_long_dbl (void *vequest);
HYPRE_Int hypre_ParcsrBdiagInvScal_flt ( hypre_ParCSRMatrix *A, HYPRE_Int blockSize,
                                    hypre_ParCSRMatrix **As);
HYPRE_Int hypre_ParcsrBdiagInvScal_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blockSize,
                                    hypre_ParCSRMatrix **As);
HYPRE_Int hypre_ParcsrBdiagInvScal_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blockSize,
                                    hypre_ParCSRMatrix **As);
HYPRE_Int hypre_ParCSRDiagScaleVector_flt ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                       hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRDiagScaleVector_dbl ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                       hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRDiagScaleVector_long_dbl ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                       hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRDiagScaleVectorHost_flt ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                           hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRDiagScaleVectorHost_dbl ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                           hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRDiagScaleVectorHost_long_dbl ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                           hypre_ParVector *par_x );
HYPRE_Int hypre_ParcsrGetExternalRowsInit_flt ( hypre_ParCSRMatrix *A, HYPRE_Int indices_len,
                                           HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int want_data, void **request_ptr);
HYPRE_Int hypre_ParcsrGetExternalRowsInit_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int indices_len,
                                           HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int want_data, void **request_ptr);
HYPRE_Int hypre_ParcsrGetExternalRowsInit_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int indices_len,
                                           HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsWait_flt (void *vrequest);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsWait_dbl (void *vrequest);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsWait_long_dbl (void *vrequest);
HYPRE_Int hypre_ParCSRMatrixAdd_flt ( hypre_float alpha, hypre_ParCSRMatrix *A, hypre_float beta,
                                 hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAdd_dbl ( hypre_double alpha, hypre_ParCSRMatrix *A, hypre_double beta,
                                 hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAdd_long_dbl ( hypre_long_double alpha, hypre_ParCSRMatrix *A, hypre_long_double beta,
                                 hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAddHost_flt ( hypre_float alpha, hypre_ParCSRMatrix *A,
                                     hypre_float beta, hypre_ParCSRMatrix *B,
                                     hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAddHost_dbl ( hypre_double alpha, hypre_ParCSRMatrix *A,
                                     hypre_double beta, hypre_ParCSRMatrix *B,
                                     hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAddHost_long_dbl ( hypre_long_double alpha, hypre_ParCSRMatrix *A,
                                     hypre_long_double beta, hypre_ParCSRMatrix *B,
                                     hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAminvDB_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                      hypre_float *d, hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_ParCSRMatrixAminvDB_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                      hypre_double *d, hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_ParCSRMatrixAminvDB_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                      hypre_long_double *d, hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_ParCSRMatrixCompressOffdMap_flt (hypre_ParCSRMatrix *A);
HYPRE_Int hypre_ParCSRMatrixCompressOffdMap_dbl (hypre_ParCSRMatrix *A);
HYPRE_Int hypre_ParCSRMatrixCompressOffdMap_long_dbl (hypre_ParCSRMatrix *A);
HYPRE_Int hypre_ParCSRMatrixDiagScale_flt ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_ld,
                                       hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixDiagScale_dbl ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_ld,
                                       hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixDiagScale_long_dbl ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_ld,
                                       hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixDiagScaleHost_flt ( hypre_ParCSRMatrix *par_A,  hypre_ParVector *par_ld,
                                       hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixDiagScaleHost_dbl ( hypre_ParCSRMatrix *par_A,  hypre_ParVector *par_ld,
                                       hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixDiagScaleHost_long_dbl ( hypre_ParCSRMatrix *par_A,  hypre_ParVector *par_ld,
                                       hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixDropSmallEntries_flt ( hypre_ParCSRMatrix *A, hypre_float tol,
                                              HYPRE_Int type);
HYPRE_Int hypre_ParCSRMatrixDropSmallEntries_dbl ( hypre_ParCSRMatrix *A, hypre_double tol,
                                              HYPRE_Int type);
HYPRE_Int hypre_ParCSRMatrixDropSmallEntries_long_dbl ( hypre_ParCSRMatrix *A, hypre_long_double tol,
                                              HYPRE_Int type);
HYPRE_Int hypre_ParCSRMatrixDropSmallEntriesHost_flt ( hypre_ParCSRMatrix *A, hypre_float tol,
                                                  HYPRE_Int type);
HYPRE_Int hypre_ParCSRMatrixDropSmallEntriesHost_dbl ( hypre_ParCSRMatrix *A, hypre_double tol,
                                                  HYPRE_Int type);
HYPRE_Int hypre_ParCSRMatrixDropSmallEntriesHost_long_dbl ( hypre_ParCSRMatrix *A, hypre_long_double tol,
                                                  HYPRE_Int type);
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_flt  ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                 HYPRE_Int data );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_dbl  ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                 HYPRE_Int data );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_long_dbl  ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                 HYPRE_Int data );
void hypre_ParCSRMatrixExtractBExt_Arrays_flt  ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                            hypre_float **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                            HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                            HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                            HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                            HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                            HYPRE_BigInt *col_map_offd, hypre_float *diag_data, hypre_float *offd_data );
void hypre_ParCSRMatrixExtractBExt_Arrays_dbl  ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                            hypre_double **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                            HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                            HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                            HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                            HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                            HYPRE_BigInt *col_map_offd, hypre_double *diag_data, hypre_double *offd_data );
void hypre_ParCSRMatrixExtractBExt_Arrays_long_dbl  ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                            hypre_long_double **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                            HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                            HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                            HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                            HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                            HYPRE_BigInt *col_map_offd, hypre_long_double *diag_data, hypre_long_double *offd_data );
void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap_flt  ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                                    hypre_float **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                                    HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                                    HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                                    HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                                    HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                                    HYPRE_BigInt *col_map_offd, hypre_float *diag_data, hypre_float *offd_data,
                                                    hypre_ParCSRCommHandle **comm_handle_idx, hypre_ParCSRCommHandle **comm_handle_data,
                                                    HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap_dbl  ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                                    hypre_double **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                                    HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                                    HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                                    HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                                    HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                                    HYPRE_BigInt *col_map_offd, hypre_double *diag_data, hypre_double *offd_data,
                                                    hypre_ParCSRCommHandle **comm_handle_idx, hypre_ParCSRCommHandle **comm_handle_data,
                                                    HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap_long_dbl  ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                                    hypre_long_double **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                                    HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                                    HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                                    HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                                    HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                                    HYPRE_BigInt *col_map_offd, hypre_long_double *diag_data, hypre_long_double *offd_data,
                                                    hypre_ParCSRCommHandle **comm_handle_idx, hypre_ParCSRCommHandle **comm_handle_data,
                                                    HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_Overlap_flt  ( hypre_ParCSRMatrix *B,
                                                         hypre_ParCSRMatrix *A, HYPRE_Int data, hypre_ParCSRCommHandle **comm_handle_idx,
                                                         hypre_ParCSRCommHandle **comm_handle_data, HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
                                                         HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_Overlap_dbl  ( hypre_ParCSRMatrix *B,
                                                         hypre_ParCSRMatrix *A, HYPRE_Int data, hypre_ParCSRCommHandle **comm_handle_idx,
                                                         hypre_ParCSRCommHandle **comm_handle_data, HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
                                                         HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_Overlap_long_dbl  ( hypre_ParCSRMatrix *B,
                                                         hypre_ParCSRMatrix *A, HYPRE_Int data, hypre_ParCSRCommHandle **comm_handle_idx,
                                                         hypre_ParCSRCommHandle **comm_handle_data, HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
                                                         HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
void hypre_ParCSRMatrixExtractRowSubmatrices_flt  ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                               hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractRowSubmatrices_dbl  ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                               hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractRowSubmatrices_long_dbl  ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                               hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractSubmatrices_flt  ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                            hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractSubmatrices_dbl  ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                            hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractSubmatrices_long_dbl  ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                            hypre_ParCSRMatrix ***submatrices );
HYPRE_Int hypre_ParCSRMatrixExtractSubmatrixFC_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, const char *job,
                                                hypre_ParCSRMatrix **B_ptr,
                                                hypre_float strength_thresh);
HYPRE_Int hypre_ParCSRMatrixExtractSubmatrixFC_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, const char *job,
                                                hypre_ParCSRMatrix **B_ptr,
                                                hypre_double strength_thresh);
HYPRE_Int hypre_ParCSRMatrixExtractSubmatrixFC_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, const char *job,
                                                hypre_ParCSRMatrix **B_ptr,
                                                hypre_long_double strength_thresh);
hypre_float hypre_ParCSRMatrixFnorm_flt ( hypre_ParCSRMatrix *A );
hypre_double hypre_ParCSRMatrixFnorm_dbl ( hypre_ParCSRMatrix *A );
hypre_long_double hypre_ParCSRMatrixFnorm_long_dbl ( hypre_ParCSRMatrix *A );
void hypre_ParCSRMatrixGenSpanningTree_flt  ( hypre_ParCSRMatrix *G_csr, HYPRE_Int **indices,
                                         HYPRE_Int G_type );
void hypre_ParCSRMatrixGenSpanningTree_dbl  ( hypre_ParCSRMatrix *G_csr, HYPRE_Int **indices,
                                         HYPRE_Int G_type );
void hypre_ParCSRMatrixGenSpanningTree_long_dbl  ( hypre_ParCSRMatrix *G_csr, HYPRE_Int **indices,
                                         HYPRE_Int G_type );
HYPRE_Int hypre_ParCSRMatrixInfNorm_flt  ( hypre_ParCSRMatrix *A, hypre_float *norm );
HYPRE_Int hypre_ParCSRMatrixInfNorm_dbl  ( hypre_ParCSRMatrix *A, hypre_double *norm );
HYPRE_Int hypre_ParCSRMatrixInfNorm_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double *norm );
hypre_float hypre_ParCSRMatrixLocalSumElts_flt  ( hypre_ParCSRMatrix *A );
hypre_double hypre_ParCSRMatrixLocalSumElts_dbl  ( hypre_ParCSRMatrix *A );
hypre_long_double hypre_ParCSRMatrixLocalSumElts_long_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixLocalTranspose_flt ( hypre_ParCSRMatrix  *A );
HYPRE_Int hypre_ParCSRMatrixLocalTranspose_dbl ( hypre_ParCSRMatrix  *A );
HYPRE_Int hypre_ParCSRMatrixLocalTranspose_long_dbl ( hypre_ParCSRMatrix  *A );
HYPRE_Int hypre_ParCSRMatrixReorder_flt  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixReorder_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixReorder_long_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixScale_flt (hypre_ParCSRMatrix *A, hypre_float scalar);
HYPRE_Int hypre_ParCSRMatrixScale_dbl (hypre_ParCSRMatrix *A, hypre_double scalar);
HYPRE_Int hypre_ParCSRMatrixScale_long_dbl (hypre_ParCSRMatrix *A, hypre_long_double scalar);
HYPRE_Int hypre_ParCSRMatrixTranspose_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                        HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTranspose_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                        HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTranspose_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                        HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTransposeHost_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                            HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTransposeHost_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                            HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTransposeHost_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                            HYPRE_Int data );
hypre_ParCSRMatrix *hypre_ParMatmul_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix *hypre_ParMatmul_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix *hypre_ParMatmul_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
void hypre_ParMatmul_RowSizes_flt  ( HYPRE_MemoryLocation memory_location, HYPRE_Int **C_diag_i,
                                HYPRE_Int **C_offd_i, HYPRE_Int *rownnz_A, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j, HYPRE_Int *B_diag_i, HYPRE_Int *B_diag_j,
                                HYPRE_Int *B_offd_i, HYPRE_Int *B_offd_j, HYPRE_Int *B_ext_diag_i, HYPRE_Int *B_ext_diag_j,
                                HYPRE_Int *B_ext_offd_i, HYPRE_Int *B_ext_offd_j, HYPRE_Int *map_B_to_C, HYPRE_Int *C_diag_size,
                                HYPRE_Int *C_offd_size, HYPRE_Int num_rownnz_A, HYPRE_Int num_rows_diag_A,
                                HYPRE_Int num_cols_offd_A, HYPRE_Int  allsquare, HYPRE_Int num_cols_diag_B,
                                HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C );
void hypre_ParMatmul_RowSizes_dbl  ( HYPRE_MemoryLocation memory_location, HYPRE_Int **C_diag_i,
                                HYPRE_Int **C_offd_i, HYPRE_Int *rownnz_A, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j, HYPRE_Int *B_diag_i, HYPRE_Int *B_diag_j,
                                HYPRE_Int *B_offd_i, HYPRE_Int *B_offd_j, HYPRE_Int *B_ext_diag_i, HYPRE_Int *B_ext_diag_j,
                                HYPRE_Int *B_ext_offd_i, HYPRE_Int *B_ext_offd_j, HYPRE_Int *map_B_to_C, HYPRE_Int *C_diag_size,
                                HYPRE_Int *C_offd_size, HYPRE_Int num_rownnz_A, HYPRE_Int num_rows_diag_A,
                                HYPRE_Int num_cols_offd_A, HYPRE_Int  allsquare, HYPRE_Int num_cols_diag_B,
                                HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C );
void hypre_ParMatmul_RowSizes_long_dbl  ( HYPRE_MemoryLocation memory_location, HYPRE_Int **C_diag_i,
                                HYPRE_Int **C_offd_i, HYPRE_Int *rownnz_A, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j, HYPRE_Int *B_diag_i, HYPRE_Int *B_diag_j,
                                HYPRE_Int *B_offd_i, HYPRE_Int *B_offd_j, HYPRE_Int *B_ext_diag_i, HYPRE_Int *B_ext_diag_j,
                                HYPRE_Int *B_ext_offd_i, HYPRE_Int *B_ext_offd_j, HYPRE_Int *map_B_to_C, HYPRE_Int *C_diag_size,
                                HYPRE_Int *C_offd_size, HYPRE_Int num_rownnz_A, HYPRE_Int num_rows_diag_A,
                                HYPRE_Int num_cols_offd_A, HYPRE_Int  allsquare, HYPRE_Int num_cols_diag_B,
                                HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C );
hypre_ParCSRMatrix *hypre_ParTMatmul_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix *hypre_ParTMatmul_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix *hypre_ParTMatmul_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
HYPRE_Int hypre_ParvecBdiagInvScal_flt ( hypre_ParVector *b, HYPRE_Int blockSize, hypre_ParVector **bs,
                                    hypre_ParCSRMatrix *A);
HYPRE_Int hypre_ParvecBdiagInvScal_dbl ( hypre_ParVector *b, HYPRE_Int blockSize, hypre_ParVector **bs,
                                    hypre_ParCSRMatrix *A);
HYPRE_Int hypre_ParvecBdiagInvScal_long_dbl ( hypre_ParVector *b, HYPRE_Int blockSize, hypre_ParVector **bs,
                                    hypre_ParCSRMatrix *A);
HYPRE_Int GenerateDiagAndOffd_flt  ( hypre_CSRMatrix *A, hypre_ParCSRMatrix *matrix,
                                HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );
HYPRE_Int GenerateDiagAndOffd_dbl  ( hypre_CSRMatrix *A, hypre_ParCSRMatrix *matrix,
                                HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );
HYPRE_Int GenerateDiagAndOffd_long_dbl  ( hypre_CSRMatrix *A, hypre_ParCSRMatrix *matrix,
                                HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix_flt  ( MPI_Comm comm, hypre_CSRMatrix *A,
                                                    HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix_dbl  ( MPI_Comm comm, hypre_CSRMatrix *A,
                                                    HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix_long_dbl  ( MPI_Comm comm, hypre_CSRMatrix *A,
                                                    HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
HYPRE_Int hypre_FillResponseParToCSRMatrix_flt  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseParToCSRMatrix_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseParToCSRMatrix_long_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
hypre_CSRMatrix *hypre_MergeDiagAndOffd_flt  ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_MergeDiagAndOffd_dbl  ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_MergeDiagAndOffd_long_dbl  ( hypre_ParCSRMatrix *par_matrix );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_v2_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data,
                                                 HYPRE_MemoryLocation memory_location );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_v2_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data,
                                                 HYPRE_MemoryLocation memory_location );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_v2_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data,
                                                 HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParCSRMatrixCopy_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                   HYPRE_Int copy_data );
HYPRE_Int hypre_ParCSRMatrixCopy_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                   HYPRE_Int copy_data );
HYPRE_Int hypre_ParCSRMatrixCopy_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                   HYPRE_Int copy_data );
void hypre_ParCSRMatrixCopyColMapOffdToDevice_flt (hypre_ParCSRMatrix *A);
void hypre_ParCSRMatrixCopyColMapOffdToDevice_dbl (hypre_ParCSRMatrix *A);
void hypre_ParCSRMatrixCopyColMapOffdToDevice_long_dbl (hypre_ParCSRMatrix *A);
void hypre_ParCSRMatrixCopyColMapOffdToHost_flt (hypre_ParCSRMatrix *A);
void hypre_ParCSRMatrixCopyColMapOffdToHost_dbl (hypre_ParCSRMatrix *A);
void hypre_ParCSRMatrixCopyColMapOffdToHost_long_dbl (hypre_ParCSRMatrix *A);
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate_flt  ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                               HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts_in, HYPRE_BigInt *col_starts_in,
                                               HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag, HYPRE_Int num_nonzeros_offd );
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                               HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts_in, HYPRE_BigInt *col_starts_in,
                                               HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag, HYPRE_Int num_nonzeros_offd );
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                               HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts_in, HYPRE_BigInt *col_starts_in,
                                               HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag, HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRMatrixDestroy_flt  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixDestroy_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixDestroy_long_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixGetLocalRange_flt  ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRMatrixGetLocalRange_dbl  ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRMatrixGetLocalRange_long_dbl  ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRMatrixGetRow_flt  ( hypre_ParCSRMatrix *mat, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, hypre_float **values );
HYPRE_Int hypre_ParCSRMatrixGetRow_dbl  ( hypre_ParCSRMatrix *mat, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, hypre_double **values );
HYPRE_Int hypre_ParCSRMatrixGetRow_long_dbl  ( hypre_ParCSRMatrix *mat, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, hypre_long_double **values );
HYPRE_Int hypre_ParCSRMatrixGetRowHost_flt ( hypre_ParCSRMatrix  *mat, HYPRE_BigInt row,
                                     HYPRE_Int *size, HYPRE_BigInt **col_ind,
                                     hypre_float **values );
HYPRE_Int hypre_ParCSRMatrixGetRowHost_dbl ( hypre_ParCSRMatrix  *mat, HYPRE_BigInt row,
                                     HYPRE_Int *size, HYPRE_BigInt **col_ind,
                                     hypre_double **values );
HYPRE_Int hypre_ParCSRMatrixGetRowHost_long_dbl ( hypre_ParCSRMatrix  *mat, HYPRE_BigInt row,
                                     HYPRE_Int *size, HYPRE_BigInt **col_ind,
                                     hypre_long_double **values );
HYPRE_Int hypre_ParCSRMatrixInitialize_flt  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixInitialize_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixInitialize_long_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixInitialize_v2_flt ( hypre_ParCSRMatrix *matrix,
                                           HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParCSRMatrixInitialize_v2_dbl ( hypre_ParCSRMatrix *matrix,
                                           HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParCSRMatrixInitialize_v2_long_dbl ( hypre_ParCSRMatrix *matrix,
                                           HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParCSRMatrixMigrate_flt (hypre_ParCSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParCSRMatrixMigrate_dbl (hypre_ParCSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParCSRMatrixMigrate_long_dbl (hypre_ParCSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParCSRMatrixPrint_flt  ( hypre_ParCSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_ParCSRMatrixPrint_dbl  ( hypre_ParCSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_ParCSRMatrixPrint_long_dbl  ( hypre_ParCSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_ParCSRMatrixPrintIJ_flt  ( const hypre_ParCSRMatrix *matrix, const HYPRE_Int base_i,
                                      const HYPRE_Int base_j, const char *filename );
HYPRE_Int hypre_ParCSRMatrixPrintIJ_dbl  ( const hypre_ParCSRMatrix *matrix, const HYPRE_Int base_i,
                                      const HYPRE_Int base_j, const char *filename );
HYPRE_Int hypre_ParCSRMatrixPrintIJ_long_dbl  ( const hypre_ParCSRMatrix *matrix, const HYPRE_Int base_i,
                                      const HYPRE_Int base_j, const char *filename );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead_flt  ( MPI_Comm comm, const char *file_name );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead_dbl  ( MPI_Comm comm, const char *file_name );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead_long_dbl  ( MPI_Comm comm, const char *file_name );
HYPRE_Int hypre_ParCSRMatrixReadIJ_flt  ( MPI_Comm comm, const char *filename, HYPRE_Int *base_i_ptr,
                                     HYPRE_Int *base_j_ptr, hypre_ParCSRMatrix **matrix_ptr );
HYPRE_Int hypre_ParCSRMatrixReadIJ_dbl  ( MPI_Comm comm, const char *filename, HYPRE_Int *base_i_ptr,
                                     HYPRE_Int *base_j_ptr, hypre_ParCSRMatrix **matrix_ptr );
HYPRE_Int hypre_ParCSRMatrixReadIJ_long_dbl  ( MPI_Comm comm, const char *filename, HYPRE_Int *base_i_ptr,
                                     HYPRE_Int *base_j_ptr, hypre_ParCSRMatrix **matrix_ptr );
HYPRE_Int hypre_ParCSRMatrixRestoreRow_flt  ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, hypre_float **values );
HYPRE_Int hypre_ParCSRMatrixRestoreRow_dbl  ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, hypre_double **values );
HYPRE_Int hypre_ParCSRMatrixRestoreRow_long_dbl  ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, hypre_long_double **values );
HYPRE_Int hypre_ParCSRMatrixSetConstantValues_flt ( hypre_ParCSRMatrix *A, hypre_float value );
HYPRE_Int hypre_ParCSRMatrixSetConstantValues_dbl ( hypre_ParCSRMatrix *A, hypre_double value );
HYPRE_Int hypre_ParCSRMatrixSetConstantValues_long_dbl ( hypre_ParCSRMatrix *A, hypre_long_double value );
HYPRE_Int hypre_ParCSRMatrixSetDataOwner_flt  ( hypre_ParCSRMatrix *matrix, HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRMatrixSetDataOwner_dbl  ( hypre_ParCSRMatrix *matrix, HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRMatrixSetDataOwner_long_dbl  ( hypre_ParCSRMatrix *matrix, HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRMatrixSetDNumNonzeros_flt  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetDNumNonzeros_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetDNumNonzeros_long_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros_flt  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros_long_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros_core_flt ( hypre_ParCSRMatrix *matrix, const char* format );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros_core_dbl ( hypre_ParCSRMatrix *matrix, const char* format );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros_core_long_dbl ( hypre_ParCSRMatrix *matrix, const char* format );
HYPRE_Int hypre_ParCSRMatrixSetNumRownnz_flt  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumRownnz_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumRownnz_long_dbl  ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetPatternOnly_flt ( hypre_ParCSRMatrix *matrix, HYPRE_Int pattern_only);
HYPRE_Int hypre_ParCSRMatrixSetPatternOnly_dbl ( hypre_ParCSRMatrix *matrix, HYPRE_Int pattern_only);
HYPRE_Int hypre_ParCSRMatrixSetPatternOnly_long_dbl ( hypre_ParCSRMatrix *matrix, HYPRE_Int pattern_only);
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll_flt  ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll_dbl  ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll_long_dbl  ( hypre_ParCSRMatrix *par_matrix );
HYPRE_Int hypre_ParCSRMatrixTruncate_flt (hypre_ParCSRMatrix *A, hypre_float tol,
                                     HYPRE_Int max_row_elmts, HYPRE_Int rescale,
                                     HYPRE_Int nrm_type);
HYPRE_Int hypre_ParCSRMatrixTruncate_dbl (hypre_ParCSRMatrix *A, hypre_double tol,
                                     HYPRE_Int max_row_elmts, HYPRE_Int rescale,
                                     HYPRE_Int nrm_type);
HYPRE_Int hypre_ParCSRMatrixTruncate_long_dbl (hypre_ParCSRMatrix *A, hypre_long_double tol,
                                     HYPRE_Int max_row_elmts, HYPRE_Int rescale,
                                     HYPRE_Int nrm_type);
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
HYPRE_Int hypre_ParCSRMatrixMatvec_flt  ( hypre_float alpha, hypre_ParCSRMatrix *A, hypre_ParVector *x,
                                     hypre_float beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvec_dbl  ( hypre_double alpha, hypre_ParCSRMatrix *A, hypre_ParVector *x,
                                     hypre_double beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvec_long_dbl  ( hypre_long_double alpha, hypre_ParCSRMatrix *A, hypre_ParVector *x,
                                     hypre_long_double beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvec_FF_flt  ( hypre_float alpha, hypre_ParCSRMatrix *A,
                                        hypre_ParVector *x, hypre_float beta, hypre_ParVector *y,
                                        HYPRE_Int *CF_marker, HYPRE_Int fpt );
HYPRE_Int hypre_ParCSRMatrixMatvec_FF_dbl  ( hypre_double alpha, hypre_ParCSRMatrix *A,
                                        hypre_ParVector *x, hypre_double beta, hypre_ParVector *y,
                                        HYPRE_Int *CF_marker, HYPRE_Int fpt );
HYPRE_Int hypre_ParCSRMatrixMatvec_FF_long_dbl  ( hypre_long_double alpha, hypre_ParCSRMatrix *A,
                                        hypre_ParVector *x, hypre_long_double beta, hypre_ParVector *y,
                                        HYPRE_Int *CF_marker, HYPRE_Int fpt );
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlace_flt  ( hypre_float alpha, hypre_ParCSRMatrix *A,
                                               hypre_ParVector *x, hypre_float beta,
                                               hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlace_dbl  ( hypre_double alpha, hypre_ParCSRMatrix *A,
                                               hypre_ParVector *x, hypre_double beta,
                                               hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlace_long_dbl  ( hypre_long_double alpha, hypre_ParCSRMatrix *A,
                                               hypre_ParVector *x, hypre_long_double beta,
                                               hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlaceHost_flt ( hypre_float       alpha,
                                        hypre_ParCSRMatrix *A,
                                        hypre_ParVector    *x,
                                        hypre_float       beta,
                                        hypre_ParVector    *b,
                                        hypre_ParVector    *y );
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlaceHost_dbl ( hypre_double       alpha,
                                        hypre_ParCSRMatrix *A,
                                        hypre_ParVector    *x,
                                        hypre_double       beta,
                                        hypre_ParVector    *b,
                                        hypre_ParVector    *y );
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlaceHost_long_dbl ( hypre_long_double       alpha,
                                        hypre_ParCSRMatrix *A,
                                        hypre_ParVector    *x,
                                        hypre_long_double       beta,
                                        hypre_ParVector    *b,
                                        hypre_ParVector    *y );
HYPRE_Int hypre_ParCSRMatrixMatvecT_flt  ( hypre_float alpha, hypre_ParCSRMatrix *A,
                                      hypre_ParVector *x, hypre_float beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecT_dbl  ( hypre_double alpha, hypre_ParCSRMatrix *A,
                                      hypre_ParVector *x, hypre_double beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecT_long_dbl  ( hypre_long_double alpha, hypre_ParCSRMatrix *A,
                                      hypre_ParVector *x, hypre_long_double beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecTHost_flt ( hypre_float alpha, hypre_ParCSRMatrix *A,
                               		    hypre_ParVector    *x, hypre_float beta,
                                            hypre_ParVector    *y );
HYPRE_Int hypre_ParCSRMatrixMatvecTHost_dbl ( hypre_double alpha, hypre_ParCSRMatrix *A,
                               		    hypre_ParVector    *x, hypre_double beta,
                                            hypre_ParVector    *y );
HYPRE_Int hypre_ParCSRMatrixMatvecTHost_long_dbl ( hypre_long_double alpha, hypre_ParCSRMatrix *A,
                               		    hypre_ParVector    *x, hypre_long_double beta,
                                            hypre_ParVector    *y );
hypre_ParCSRMatrix *hypre_ParCSRMatMat_flt ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMat_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMat_long_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatHost_flt ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatHost_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatHost_long_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAP_flt ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                           hypre_ParCSRMatrix  *P );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAP_dbl ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                           hypre_ParCSRMatrix  *P );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAP_long_dbl ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                           hypre_ParCSRMatrix  *P );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAPKT_flt ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                             hypre_ParCSRMatrix  *P, HYPRE_Int keepTranspose );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAPKT_dbl ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                             hypre_ParCSRMatrix  *P, HYPRE_Int keepTranspose );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAPKT_long_dbl ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                             hypre_ParCSRMatrix  *P, HYPRE_Int keepTranspose );
hypre_ParCSRMatrix* hypre_ParCSRMatrixRAPKTHost_flt ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keep_transpose );
hypre_ParCSRMatrix* hypre_ParCSRMatrixRAPKTHost_dbl ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keep_transpose );
hypre_ParCSRMatrix* hypre_ParCSRMatrixRAPKTHost_long_dbl ( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keep_transpose );
hypre_ParCSRMatrix *hypre_ParCSRTMatMat_flt ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B);
hypre_ParCSRMatrix *hypre_ParCSRTMatMat_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B);
hypre_ParCSRMatrix *hypre_ParCSRTMatMat_long_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKT_flt ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                           HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKT_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                           HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKT_long_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                           HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTHost_flt ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                               HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTHost_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                               HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTHost_long_dbl ( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                               HYPRE_Int keep_transpose);
HYPRE_Int HYPRE_Destroy2DSystem_flt  ( HYPRE_ParCSR_System_Problem *sys_prob );
HYPRE_Int HYPRE_Destroy2DSystem_dbl  ( HYPRE_ParCSR_System_Problem *sys_prob );
HYPRE_Int HYPRE_Destroy2DSystem_long_dbl  ( HYPRE_ParCSR_System_Problem *sys_prob );
HYPRE_ParCSR_System_Problem *HYPRE_Generate2DSystem_flt  ( HYPRE_ParCSRMatrix H_L1,
                                                      HYPRE_ParCSRMatrix H_L2, HYPRE_ParVector H_b1, HYPRE_ParVector H_b2, HYPRE_ParVector H_x1,
                                                      HYPRE_ParVector H_x2, hypre_float *M_vals );
HYPRE_ParCSR_System_Problem *HYPRE_Generate2DSystem_dbl  ( HYPRE_ParCSRMatrix H_L1,
                                                      HYPRE_ParCSRMatrix H_L2, HYPRE_ParVector H_b1, HYPRE_ParVector H_b2, HYPRE_ParVector H_x1,
                                                      HYPRE_ParVector H_x2, hypre_double *M_vals );
HYPRE_ParCSR_System_Problem *HYPRE_Generate2DSystem_long_dbl  ( HYPRE_ParCSRMatrix H_L1,
                                                      HYPRE_ParCSRMatrix H_L2, HYPRE_ParVector H_b1, HYPRE_ParVector H_b2, HYPRE_ParVector H_x1,
                                                      HYPRE_ParVector H_x2, hypre_long_double *M_vals );
HYPRE_Int hypre_ParVectorMassAxpy_flt  ( hypre_float *alpha, hypre_ParVector **x, hypre_ParVector *y,
                                    HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_ParVectorMassAxpy_dbl  ( hypre_double *alpha, hypre_ParVector **x, hypre_ParVector *y,
                                    HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_ParVectorMassAxpy_long_dbl  ( hypre_long_double *alpha, hypre_ParVector **x, hypre_ParVector *y,
                                    HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_ParVectorMassDotpTwo_flt  ( hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector **z,
                                       HYPRE_Int k, HYPRE_Int unroll, hypre_float *prod_x, hypre_float *prod_y );
HYPRE_Int hypre_ParVectorMassDotpTwo_dbl  ( hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector **z,
                                       HYPRE_Int k, HYPRE_Int unroll, hypre_double *prod_x, hypre_double *prod_y );
HYPRE_Int hypre_ParVectorMassDotpTwo_long_dbl  ( hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector **z,
                                       HYPRE_Int k, HYPRE_Int unroll, hypre_long_double *prod_x, hypre_long_double *prod_y );
HYPRE_Int hypre_ParVectorMassInnerProd_flt  ( hypre_ParVector *x, hypre_ParVector **y, HYPRE_Int k,
                                         HYPRE_Int unroll, hypre_float *prod );
HYPRE_Int hypre_ParVectorMassInnerProd_dbl  ( hypre_ParVector *x, hypre_ParVector **y, HYPRE_Int k,
                                         HYPRE_Int unroll, hypre_double *prod );
HYPRE_Int hypre_ParVectorMassInnerProd_long_dbl  ( hypre_ParVector *x, hypre_ParVector **y, HYPRE_Int k,
                                         HYPRE_Int unroll, hypre_long_double *prod );
HYPRE_Int hypre_FillResponseParToVectorAll_flt  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseParToVectorAll_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseParToVectorAll_long_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
hypre_ParVector *hypre_ParMultiVectorCreate_flt  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                              HYPRE_BigInt *partitioning, HYPRE_Int num_vectors );
hypre_ParVector *hypre_ParMultiVectorCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                              HYPRE_BigInt *partitioning, HYPRE_Int num_vectors );
hypre_ParVector *hypre_ParMultiVectorCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                              HYPRE_BigInt *partitioning, HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorAxpy_flt  ( hypre_float alpha, hypre_ParVector *x, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorAxpy_dbl  ( hypre_double alpha, hypre_ParVector *x, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorAxpy_long_dbl  ( hypre_long_double alpha, hypre_ParVector *x, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorAxpyz_flt  ( hypre_float alpha, hypre_ParVector *x,
                                 hypre_float beta, hypre_ParVector *y,
                                 hypre_ParVector *z );
HYPRE_Int hypre_ParVectorAxpyz_dbl  ( hypre_double alpha, hypre_ParVector *x,
                                 hypre_double beta, hypre_ParVector *y,
                                 hypre_ParVector *z );
HYPRE_Int hypre_ParVectorAxpyz_long_dbl  ( hypre_long_double alpha, hypre_ParVector *x,
                                 hypre_long_double beta, hypre_ParVector *y,
                                 hypre_ParVector *z );
hypre_ParVector *hypre_ParVectorCloneDeep_v2_flt ( hypre_ParVector *x,
                                              HYPRE_MemoryLocation memory_location );
hypre_ParVector *hypre_ParVectorCloneDeep_v2_dbl ( hypre_ParVector *x,
                                              HYPRE_MemoryLocation memory_location );
hypre_ParVector *hypre_ParVectorCloneDeep_v2_long_dbl ( hypre_ParVector *x,
                                              HYPRE_MemoryLocation memory_location );
hypre_ParVector *hypre_ParVectorCloneShallow_flt  ( hypre_ParVector *x );
hypre_ParVector *hypre_ParVectorCloneShallow_dbl  ( hypre_ParVector *x );
hypre_ParVector *hypre_ParVectorCloneShallow_long_dbl  ( hypre_ParVector *x );
HYPRE_Int hypre_ParVectorCopy_flt  ( hypre_ParVector *x, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorCopy_dbl  ( hypre_ParVector *x, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorCopy_long_dbl  ( hypre_ParVector *x, hypre_ParVector *y );
hypre_ParVector *hypre_ParVectorCreate_flt  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                         HYPRE_BigInt *partitioning_in );
hypre_ParVector *hypre_ParVectorCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                         HYPRE_BigInt *partitioning_in );
hypre_ParVector *hypre_ParVectorCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt global_size,
                                         HYPRE_BigInt *partitioning_in );
HYPRE_Int hypre_ParVectorDestroy_flt  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorDestroy_dbl  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorDestroy_long_dbl  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorElmdivpy_flt ( hypre_ParVector *x, hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorElmdivpy_dbl ( hypre_ParVector *x, hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorElmdivpy_long_dbl ( hypre_ParVector *x, hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorElmdivpyMarked_flt ( hypre_ParVector *x, hypre_ParVector *b,
                                         hypre_ParVector *y, HYPRE_Int *marker,
                                         HYPRE_Int marker_val );
HYPRE_Int hypre_ParVectorElmdivpyMarked_dbl ( hypre_ParVector *x, hypre_ParVector *b,
                                         hypre_ParVector *y, HYPRE_Int *marker,
                                         HYPRE_Int marker_val );
HYPRE_Int hypre_ParVectorElmdivpyMarked_long_dbl ( hypre_ParVector *x, hypre_ParVector *b,
                                         hypre_ParVector *y, HYPRE_Int *marker,
                                         HYPRE_Int marker_val );
HYPRE_Int hypre_ParVectorGetValues_flt  ( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, hypre_float *values);
HYPRE_Int hypre_ParVectorGetValues_dbl  ( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, hypre_double *values);
HYPRE_Int hypre_ParVectorGetValues_long_dbl  ( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, hypre_long_double *values);
HYPRE_Int hypre_ParVectorGetValues2_flt ( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, HYPRE_BigInt base, hypre_float *values );
HYPRE_Int hypre_ParVectorGetValues2_dbl ( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, HYPRE_BigInt base, hypre_double *values );
HYPRE_Int hypre_ParVectorGetValues2_long_dbl ( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, HYPRE_BigInt base, hypre_long_double *values );
HYPRE_Int hypre_ParVectorGetValuesHost_flt (hypre_ParVector *vector, HYPRE_Int num_values,
                                       HYPRE_BigInt *indices, HYPRE_BigInt base, hypre_float *values);
HYPRE_Int hypre_ParVectorGetValuesHost_dbl (hypre_ParVector *vector, HYPRE_Int num_values,
                                       HYPRE_BigInt *indices, HYPRE_BigInt base, hypre_double *values);
HYPRE_Int hypre_ParVectorGetValuesHost_long_dbl (hypre_ParVector *vector, HYPRE_Int num_values,
                                       HYPRE_BigInt *indices, HYPRE_BigInt base, hypre_long_double *values);
HYPRE_Int hypre_ParVectorInitialize_flt  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorInitialize_dbl  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorInitialize_long_dbl  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorInitialize_v2_flt ( hypre_ParVector *vector,
                                        HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParVectorInitialize_v2_dbl ( hypre_ParVector *vector,
                                        HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParVectorInitialize_v2_long_dbl ( hypre_ParVector *vector,
                                        HYPRE_MemoryLocation memory_location );
hypre_float hypre_ParVectorInnerProd_flt  ( hypre_ParVector *x, hypre_ParVector *y );
hypre_double hypre_ParVectorInnerProd_dbl  ( hypre_ParVector *x, hypre_ParVector *y );
hypre_long_double hypre_ParVectorInnerProd_long_dbl  ( hypre_ParVector *x, hypre_ParVector *y );
hypre_float hypre_ParVectorLocalSumElts_flt  ( hypre_ParVector *vector );
hypre_double hypre_ParVectorLocalSumElts_dbl  ( hypre_ParVector *vector );
hypre_long_double hypre_ParVectorLocalSumElts_long_dbl  ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorMigrate_flt (hypre_ParVector *x, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParVectorMigrate_dbl (hypre_ParVector *x, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParVectorMigrate_long_dbl (hypre_ParVector *x, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParVectorPrint_flt  ( hypre_ParVector *vector, const char *file_name );
HYPRE_Int hypre_ParVectorPrint_dbl  ( hypre_ParVector *vector, const char *file_name );
HYPRE_Int hypre_ParVectorPrint_long_dbl  ( hypre_ParVector *vector, const char *file_name );
HYPRE_Int hypre_ParVectorPrintIJ_flt  ( hypre_ParVector *vector, HYPRE_Int base_j,
                                   const char *filename );
HYPRE_Int hypre_ParVectorPrintIJ_dbl  ( hypre_ParVector *vector, HYPRE_Int base_j,
                                   const char *filename );
HYPRE_Int hypre_ParVectorPrintIJ_long_dbl  ( hypre_ParVector *vector, HYPRE_Int base_j,
                                   const char *filename );
hypre_ParVector *hypre_ParVectorRead_flt  ( MPI_Comm comm, const char *file_name );
hypre_ParVector *hypre_ParVectorRead_dbl  ( MPI_Comm comm, const char *file_name );
hypre_ParVector *hypre_ParVectorRead_long_dbl  ( MPI_Comm comm, const char *file_name );
HYPRE_Int hypre_ParVectorReadIJ_flt  ( MPI_Comm comm, const char *filename, HYPRE_Int *base_j_ptr,
                                  hypre_ParVector **vector_ptr );
HYPRE_Int hypre_ParVectorReadIJ_dbl  ( MPI_Comm comm, const char *filename, HYPRE_Int *base_j_ptr,
                                  hypre_ParVector **vector_ptr );
HYPRE_Int hypre_ParVectorReadIJ_long_dbl  ( MPI_Comm comm, const char *filename, HYPRE_Int *base_j_ptr,
                                  hypre_ParVector **vector_ptr );
HYPRE_Int hypre_ParVectorResize_flt  ( hypre_ParVector *vector, HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorResize_dbl  ( hypre_ParVector *vector, HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorResize_long_dbl  ( hypre_ParVector *vector, HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorScale_flt  ( hypre_float alpha, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorScale_dbl  ( hypre_double alpha, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorScale_long_dbl  ( hypre_long_double alpha, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorSetComponent_flt  ( hypre_ParVector *vector, HYPRE_Int component );
HYPRE_Int hypre_ParVectorSetComponent_dbl  ( hypre_ParVector *vector, HYPRE_Int component );
HYPRE_Int hypre_ParVectorSetComponent_long_dbl  ( hypre_ParVector *vector, HYPRE_Int component );
HYPRE_Int hypre_ParVectorSetConstantValues_flt  ( hypre_ParVector *v, hypre_float value );
HYPRE_Int hypre_ParVectorSetConstantValues_dbl  ( hypre_ParVector *v, hypre_double value );
HYPRE_Int hypre_ParVectorSetConstantValues_long_dbl  ( hypre_ParVector *v, hypre_long_double value );
HYPRE_Int hypre_ParVectorSetDataOwner_flt  ( hypre_ParVector *vector, HYPRE_Int owns_data );
HYPRE_Int hypre_ParVectorSetDataOwner_dbl  ( hypre_ParVector *vector, HYPRE_Int owns_data );
HYPRE_Int hypre_ParVectorSetDataOwner_long_dbl  ( hypre_ParVector *vector, HYPRE_Int owns_data );
HYPRE_Int hypre_ParVectorSetLocalSize_flt  ( hypre_ParVector *vector, HYPRE_Int local_size );
HYPRE_Int hypre_ParVectorSetLocalSize_dbl  ( hypre_ParVector *vector, HYPRE_Int local_size );
HYPRE_Int hypre_ParVectorSetLocalSize_long_dbl  ( hypre_ParVector *vector, HYPRE_Int local_size );
HYPRE_Int hypre_ParVectorSetRandomValues_flt  ( hypre_ParVector *v, HYPRE_Int seed );
HYPRE_Int hypre_ParVectorSetRandomValues_dbl  ( hypre_ParVector *v, HYPRE_Int seed );
HYPRE_Int hypre_ParVectorSetRandomValues_long_dbl  ( hypre_ParVector *v, HYPRE_Int seed );
HYPRE_Int hypre_ParVectorSetZeros_flt ( hypre_ParVector *v );
HYPRE_Int hypre_ParVectorSetZeros_dbl ( hypre_ParVector *v );
HYPRE_Int hypre_ParVectorSetZeros_long_dbl ( hypre_ParVector *v );
hypre_Vector *hypre_ParVectorToVectorAll_flt  ( hypre_ParVector *par_v );
hypre_Vector *hypre_ParVectorToVectorAll_dbl  ( hypre_ParVector *par_v );
hypre_Vector *hypre_ParVectorToVectorAll_long_dbl  ( hypre_ParVector *par_v );
hypre_ParVector *hypre_VectorToParVector_flt  ( MPI_Comm comm, hypre_Vector *v,
                                           HYPRE_BigInt *vec_starts );
hypre_ParVector *hypre_VectorToParVector_dbl  ( MPI_Comm comm, hypre_Vector *v,
                                           HYPRE_BigInt *vec_starts );
hypre_ParVector *hypre_VectorToParVector_long_dbl  ( MPI_Comm comm, hypre_Vector *v,
                                           HYPRE_BigInt *vec_starts );

#endif

#ifdef __cplusplus
}
#endif

#endif
