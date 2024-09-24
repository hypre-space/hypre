
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

#ifndef HYPRE_IJ_MV_MUP_HEADER
#define HYPRE_IJ_MV_MUP_HEADER

#include "_hypre_IJ_mv.h"

#if defined (HYPRE_MIXED_PRECISION)

HYPRE_Int hypre_AuxParCSRMatrixCreate_flt  ( hypre_AuxParCSRMatrix **aux_matrix,
                                        HYPRE_Int local_num_rows, HYPRE_Int local_num_cols, HYPRE_Int *sizes );
HYPRE_Int hypre_AuxParCSRMatrixCreate_dbl  ( hypre_AuxParCSRMatrix **aux_matrix,
                                        HYPRE_Int local_num_rows, HYPRE_Int local_num_cols, HYPRE_Int *sizes );
HYPRE_Int hypre_AuxParCSRMatrixCreate_long_dbl  ( hypre_AuxParCSRMatrix **aux_matrix,
                                        HYPRE_Int local_num_rows, HYPRE_Int local_num_cols, HYPRE_Int *sizes );
HYPRE_Int hypre_AuxParCSRMatrixDestroy_flt  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixDestroy_dbl  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixDestroy_long_dbl  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_flt  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_dbl  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_long_dbl  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_v2_flt ( hypre_AuxParCSRMatrix *matrix,
                                              HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_v2_dbl ( hypre_AuxParCSRMatrix *matrix,
                                              HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_v2_long_dbl ( hypre_AuxParCSRMatrix *matrix,
                                              HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_AuxParCSRMatrixSetRownnz_flt  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixSetRownnz_dbl  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixSetRownnz_long_dbl  ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParVectorCreate_flt  ( hypre_AuxParVector **aux_vector );
HYPRE_Int hypre_AuxParVectorCreate_dbl  ( hypre_AuxParVector **aux_vector );
HYPRE_Int hypre_AuxParVectorCreate_long_dbl  ( hypre_AuxParVector **aux_vector );
HYPRE_Int hypre_AuxParVectorDestroy_flt  ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorDestroy_dbl  ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorDestroy_long_dbl  ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorInitialize_v2_flt ( hypre_AuxParVector *vector,
                                           HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_AuxParVectorInitialize_v2_dbl ( hypre_AuxParVector *vector,
                                           HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_AuxParVectorInitialize_v2_long_dbl ( hypre_AuxParVector *vector,
                                           HYPRE_MemoryLocation memory_location );
HYPRE_Int HYPRE_IJMatrixAdd_flt  ( hypre_float alpha, HYPRE_IJMatrix matrix_A, hypre_float beta,
                              HYPRE_IJMatrix matrix_B, HYPRE_IJMatrix *matrix_C );
HYPRE_Int HYPRE_IJMatrixAdd_dbl  ( hypre_double alpha, HYPRE_IJMatrix matrix_A, hypre_double beta,
                              HYPRE_IJMatrix matrix_B, HYPRE_IJMatrix *matrix_C );
HYPRE_Int HYPRE_IJMatrixAdd_long_dbl  ( hypre_long_double alpha, HYPRE_IJMatrix matrix_A, hypre_long_double beta,
                              HYPRE_IJMatrix matrix_B, HYPRE_IJMatrix *matrix_C );
HYPRE_Int HYPRE_IJMatrixAddToValues_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                      const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const hypre_float *values );
HYPRE_Int HYPRE_IJMatrixAddToValues_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                      const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const hypre_double *values );
HYPRE_Int HYPRE_IJMatrixAddToValues_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                      const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const hypre_long_double *values );
HYPRE_Int HYPRE_IJMatrixAssemble_flt  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixAssemble_dbl  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixAssemble_long_dbl  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixCreate_flt  ( MPI_Comm comm, HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                                 HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_IJMatrix *matrix );
HYPRE_Int HYPRE_IJMatrixCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                                 HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_IJMatrix *matrix );
HYPRE_Int HYPRE_IJMatrixCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                                 HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_IJMatrix *matrix );
HYPRE_Int HYPRE_IJMatrixDestroy_flt  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixDestroy_dbl  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixDestroy_long_dbl  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixGetLocalRange_flt  ( HYPRE_IJMatrix matrix, HYPRE_BigInt *ilower,
                                        HYPRE_BigInt *iupper, HYPRE_BigInt *jlower, HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJMatrixGetLocalRange_dbl  ( HYPRE_IJMatrix matrix, HYPRE_BigInt *ilower,
                                        HYPRE_BigInt *iupper, HYPRE_BigInt *jlower, HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJMatrixGetLocalRange_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_BigInt *ilower,
                                        HYPRE_BigInt *iupper, HYPRE_BigInt *jlower, HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJMatrixGetObject_flt  ( HYPRE_IJMatrix matrix, void **object );
HYPRE_Int HYPRE_IJMatrixGetObject_dbl  ( HYPRE_IJMatrix matrix, void **object );
HYPRE_Int HYPRE_IJMatrixGetObject_long_dbl  ( HYPRE_IJMatrix matrix, void **object );
HYPRE_Int HYPRE_IJMatrixGetObjectType_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int *type );
HYPRE_Int HYPRE_IJMatrixGetObjectType_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int *type );
HYPRE_Int HYPRE_IJMatrixGetObjectType_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int *type );
HYPRE_Int HYPRE_IJMatrixGetRowCounts_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_BigInt *rows,
                                       HYPRE_Int *ncols );
HYPRE_Int HYPRE_IJMatrixGetRowCounts_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_BigInt *rows,
                                       HYPRE_Int *ncols );
HYPRE_Int HYPRE_IJMatrixGetRowCounts_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_BigInt *rows,
                                       HYPRE_Int *ncols );
HYPRE_Int HYPRE_IJMatrixGetValues_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    HYPRE_BigInt *rows, HYPRE_BigInt *cols, hypre_float *values );
HYPRE_Int HYPRE_IJMatrixGetValues_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    HYPRE_BigInt *rows, HYPRE_BigInt *cols, hypre_double *values );
HYPRE_Int HYPRE_IJMatrixGetValues_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    HYPRE_BigInt *rows, HYPRE_BigInt *cols, hypre_long_double *values );
HYPRE_Int HYPRE_IJMatrixInitialize_flt  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixInitialize_dbl  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixInitialize_long_dbl  ( HYPRE_IJMatrix matrix );
HYPRE_Int HYPRE_IJMatrixNorm_flt  ( HYPRE_IJMatrix matrix, hypre_float *norm );
HYPRE_Int HYPRE_IJMatrixNorm_dbl  ( HYPRE_IJMatrix matrix, hypre_double *norm );
HYPRE_Int HYPRE_IJMatrixNorm_long_dbl  ( HYPRE_IJMatrix matrix, hypre_long_double *norm );
HYPRE_Int HYPRE_IJMatrixPrint_flt  ( HYPRE_IJMatrix matrix, const char *filename );
HYPRE_Int HYPRE_IJMatrixPrint_dbl  ( HYPRE_IJMatrix matrix, const char *filename );
HYPRE_Int HYPRE_IJMatrixPrint_long_dbl  ( HYPRE_IJMatrix matrix, const char *filename );
HYPRE_Int HYPRE_IJMatrixRead_flt  ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixRead_dbl  ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixRead_long_dbl  ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixReadMM_flt ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                                HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixReadMM_dbl ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                                HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixReadMM_long_dbl ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                                HYPRE_IJMatrix *matrix_ptr );
HYPRE_Int HYPRE_IJMatrixSetConstantValues_flt  ( HYPRE_IJMatrix matrix, hypre_float value );
HYPRE_Int HYPRE_IJMatrixSetConstantValues_dbl  ( HYPRE_IJMatrix matrix, hypre_double value );
HYPRE_Int HYPRE_IJMatrixSetConstantValues_long_dbl  ( HYPRE_IJMatrix matrix, hypre_long_double value );
HYPRE_Int HYPRE_IJMatrixSetDiagOffdSizes_flt  ( HYPRE_IJMatrix matrix, const HYPRE_Int *diag_sizes,
                                           const HYPRE_Int *offdiag_sizes );
HYPRE_Int HYPRE_IJMatrixSetDiagOffdSizes_dbl  ( HYPRE_IJMatrix matrix, const HYPRE_Int *diag_sizes,
                                           const HYPRE_Int *offdiag_sizes );
HYPRE_Int HYPRE_IJMatrixSetDiagOffdSizes_long_dbl  ( HYPRE_IJMatrix matrix, const HYPRE_Int *diag_sizes,
                                           const HYPRE_Int *offdiag_sizes );
HYPRE_Int HYPRE_IJMatrixSetMaxOffProcElmts_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJMatrixSetMaxOffProcElmts_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJMatrixSetMaxOffProcElmts_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJMatrixSetObjectType_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int type );
HYPRE_Int HYPRE_IJMatrixSetObjectType_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int type );
HYPRE_Int HYPRE_IJMatrixSetObjectType_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int type );
HYPRE_Int HYPRE_IJMatrixSetOMPFlag_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int omp_flag );
HYPRE_Int HYPRE_IJMatrixSetOMPFlag_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int omp_flag );
HYPRE_Int HYPRE_IJMatrixSetOMPFlag_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int omp_flag );
HYPRE_Int HYPRE_IJMatrixSetPrintLevel_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJMatrixSetPrintLevel_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJMatrixSetPrintLevel_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJMatrixSetRowSizes_flt  ( HYPRE_IJMatrix matrix, const HYPRE_Int *sizes );
HYPRE_Int HYPRE_IJMatrixSetRowSizes_dbl  ( HYPRE_IJMatrix matrix, const HYPRE_Int *sizes );
HYPRE_Int HYPRE_IJMatrixSetRowSizes_long_dbl  ( HYPRE_IJMatrix matrix, const HYPRE_Int *sizes );
HYPRE_Int HYPRE_IJMatrixSetValues_flt  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const hypre_float *values );
HYPRE_Int HYPRE_IJMatrixSetValues_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const hypre_double *values );
HYPRE_Int HYPRE_IJMatrixSetValues_long_dbl  ( HYPRE_IJMatrix matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                    const HYPRE_BigInt *rows, const HYPRE_BigInt *cols, const hypre_long_double *values );
HYPRE_Int HYPRE_IJMatrixTranspose_flt  ( HYPRE_IJMatrix  matrix_A, HYPRE_IJMatrix *matrix_AT );
HYPRE_Int HYPRE_IJMatrixTranspose_dbl  ( HYPRE_IJMatrix  matrix_A, HYPRE_IJMatrix *matrix_AT );
HYPRE_Int HYPRE_IJMatrixTranspose_long_dbl  ( HYPRE_IJMatrix  matrix_A, HYPRE_IJMatrix *matrix_AT );
HYPRE_Int HYPRE_IJVectorAddToValues_flt  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                      const HYPRE_BigInt *indices, const hypre_float *values );
HYPRE_Int HYPRE_IJVectorAddToValues_dbl  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                      const HYPRE_BigInt *indices, const hypre_double *values );
HYPRE_Int HYPRE_IJVectorAddToValues_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                      const HYPRE_BigInt *indices, const hypre_long_double *values );
HYPRE_Int HYPRE_IJVectorAssemble_flt  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorAssemble_dbl  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorAssemble_long_dbl  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorCreate_flt  ( MPI_Comm comm, HYPRE_BigInt jlower, HYPRE_BigInt jupper,
                                 HYPRE_IJVector *vector );
HYPRE_Int HYPRE_IJVectorCreate_dbl  ( MPI_Comm comm, HYPRE_BigInt jlower, HYPRE_BigInt jupper,
                                 HYPRE_IJVector *vector );
HYPRE_Int HYPRE_IJVectorCreate_long_dbl  ( MPI_Comm comm, HYPRE_BigInt jlower, HYPRE_BigInt jupper,
                                 HYPRE_IJVector *vector );
HYPRE_Int HYPRE_IJVectorDestroy_flt  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorDestroy_dbl  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorDestroy_long_dbl  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorGetLocalRange_flt  ( HYPRE_IJVector vector, HYPRE_BigInt *jlower,
                                        HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJVectorGetLocalRange_dbl  ( HYPRE_IJVector vector, HYPRE_BigInt *jlower,
                                        HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJVectorGetLocalRange_long_dbl  ( HYPRE_IJVector vector, HYPRE_BigInt *jlower,
                                        HYPRE_BigInt *jupper );
HYPRE_Int HYPRE_IJVectorGetObject_flt  ( HYPRE_IJVector vector, void **object );
HYPRE_Int HYPRE_IJVectorGetObject_dbl  ( HYPRE_IJVector vector, void **object );
HYPRE_Int HYPRE_IJVectorGetObject_long_dbl  ( HYPRE_IJVector vector, void **object );
HYPRE_Int HYPRE_IJVectorGetObjectType_flt  ( HYPRE_IJVector vector, HYPRE_Int *type );
HYPRE_Int HYPRE_IJVectorGetObjectType_dbl  ( HYPRE_IJVector vector, HYPRE_Int *type );
HYPRE_Int HYPRE_IJVectorGetObjectType_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int *type );
HYPRE_Int HYPRE_IJVectorGetValues_flt  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, hypre_float *values );
HYPRE_Int HYPRE_IJVectorGetValues_dbl  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, hypre_double *values );
HYPRE_Int HYPRE_IJVectorGetValues_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, hypre_long_double *values );
HYPRE_Int HYPRE_IJVectorInitialize_flt  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorInitialize_dbl  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorInitialize_long_dbl  ( HYPRE_IJVector vector );
HYPRE_Int HYPRE_IJVectorInnerProd_flt  ( HYPRE_IJVector x, HYPRE_IJVector y, hypre_float *prod );
HYPRE_Int HYPRE_IJVectorInnerProd_dbl  ( HYPRE_IJVector x, HYPRE_IJVector y, hypre_double *prod );
HYPRE_Int HYPRE_IJVectorInnerProd_long_dbl  ( HYPRE_IJVector x, HYPRE_IJVector y, hypre_long_double *prod );
HYPRE_Int HYPRE_IJVectorPrint_flt  ( HYPRE_IJVector vector, const char *filename );
HYPRE_Int HYPRE_IJVectorPrint_dbl  ( HYPRE_IJVector vector, const char *filename );
HYPRE_Int HYPRE_IJVectorPrint_long_dbl  ( HYPRE_IJVector vector, const char *filename );
HYPRE_Int HYPRE_IJVectorRead_flt  ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJVector *vector_ptr );
HYPRE_Int HYPRE_IJVectorRead_dbl  ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJVector *vector_ptr );
HYPRE_Int HYPRE_IJVectorRead_long_dbl  ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                               HYPRE_IJVector *vector_ptr );
HYPRE_Int HYPRE_IJVectorSetComponent_flt  ( HYPRE_IJVector vector, HYPRE_Int component );
HYPRE_Int HYPRE_IJVectorSetComponent_dbl  ( HYPRE_IJVector vector, HYPRE_Int component );
HYPRE_Int HYPRE_IJVectorSetComponent_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int component );
HYPRE_Int HYPRE_IJVectorSetMaxOffProcElmts_flt  ( HYPRE_IJVector vector, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJVectorSetMaxOffProcElmts_dbl  ( HYPRE_IJVector vector, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJVectorSetMaxOffProcElmts_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int max_off_proc_elmts );
HYPRE_Int HYPRE_IJVectorSetNumComponents_flt  ( HYPRE_IJVector vector, HYPRE_Int num_components );
HYPRE_Int HYPRE_IJVectorSetNumComponents_dbl  ( HYPRE_IJVector vector, HYPRE_Int num_components );
HYPRE_Int HYPRE_IJVectorSetNumComponents_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int num_components );
HYPRE_Int HYPRE_IJVectorSetObjectType_flt  ( HYPRE_IJVector vector, HYPRE_Int type );
HYPRE_Int HYPRE_IJVectorSetObjectType_dbl  ( HYPRE_IJVector vector, HYPRE_Int type );
HYPRE_Int HYPRE_IJVectorSetObjectType_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int type );
HYPRE_Int HYPRE_IJVectorSetPrintLevel_flt  ( HYPRE_IJVector vector, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJVectorSetPrintLevel_dbl  ( HYPRE_IJVector vector, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJVectorSetPrintLevel_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int print_level );
HYPRE_Int HYPRE_IJVectorSetValues_flt  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, const hypre_float *values );
HYPRE_Int HYPRE_IJVectorSetValues_dbl  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, const hypre_double *values );
HYPRE_Int HYPRE_IJVectorSetValues_long_dbl  ( HYPRE_IJVector vector, HYPRE_Int nvalues,
                                    const HYPRE_BigInt *indices, const hypre_long_double *values );
HYPRE_Int hypre_IJMatrixCreateAssumedPartition_flt  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixCreateAssumedPartition_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixCreateAssumedPartition_long_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJVectorCreateAssumedPartition_flt  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorCreateAssumedPartition_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorCreateAssumedPartition_long_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJMatrixGetColPartitioning_flt  ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **col_partitioning );
HYPRE_Int hypre_IJMatrixGetColPartitioning_dbl  ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **col_partitioning );
HYPRE_Int hypre_IJMatrixGetColPartitioning_long_dbl  ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **col_partitioning );
HYPRE_Int hypre_IJMatrixGetRowPartitioning_flt  ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **row_partitioning );
HYPRE_Int hypre_IJMatrixGetRowPartitioning_dbl  ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **row_partitioning );
HYPRE_Int hypre_IJMatrixGetRowPartitioning_long_dbl  ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **row_partitioning );
HYPRE_Int hypre_IJMatrixRead_flt ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                              HYPRE_IJMatrix *matrix_ptr, HYPRE_Int is_mm );
HYPRE_Int hypre_IJMatrixRead_dbl ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                              HYPRE_IJMatrix *matrix_ptr, HYPRE_Int is_mm );
HYPRE_Int hypre_IJMatrixRead_long_dbl ( const char *filename, MPI_Comm comm, HYPRE_Int type,
                              HYPRE_IJMatrix *matrix_ptr, HYPRE_Int is_mm );
HYPRE_Int hypre_IJMatrixSetObject_flt  ( HYPRE_IJMatrix matrix, void *object );
HYPRE_Int hypre_IJMatrixSetObject_dbl  ( HYPRE_IJMatrix matrix, void *object );
HYPRE_Int hypre_IJMatrixSetObject_long_dbl  ( HYPRE_IJMatrix matrix, void *object );
HYPRE_Int hypre_FillResponseIJOffProcVals_flt  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                            HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                            HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseIJOffProcVals_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                            HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                            HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseIJOffProcVals_long_dbl  ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                            HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                            HYPRE_Int *response_message_size );
HYPRE_Int hypre_FindProc_flt  ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_FindProc_dbl  ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_FindProc_long_dbl  ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_IJMatrixAddParCSR_flt  ( hypre_float alpha, hypre_IJMatrix *matrix_A,
                                    hypre_float beta, hypre_IJMatrix *matrix_B, hypre_IJMatrix *matrix_C );
HYPRE_Int hypre_IJMatrixAddParCSR_dbl  ( hypre_double alpha, hypre_IJMatrix *matrix_A,
                                    hypre_double beta, hypre_IJMatrix *matrix_B, hypre_IJMatrix *matrix_C );
HYPRE_Int hypre_IJMatrixAddParCSR_long_dbl  ( hypre_long_double alpha, hypre_IJMatrix *matrix_A,
                                    hypre_long_double beta, hypre_IJMatrix *matrix_B, hypre_IJMatrix *matrix_C );
HYPRE_Int hypre_IJMatrixAddToValuesOMPParCSR_flt  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                               HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                               const hypre_float *values );
HYPRE_Int hypre_IJMatrixAddToValuesOMPParCSR_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                               HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                               const hypre_double *values );
HYPRE_Int hypre_IJMatrixAddToValuesOMPParCSR_long_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                               HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                               const hypre_long_double *values );
HYPRE_Int hypre_IJMatrixAddToValuesParCSR_flt  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                            HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                            const hypre_float *values );
HYPRE_Int hypre_IJMatrixAddToValuesParCSR_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                            HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                            const hypre_double *values );
HYPRE_Int hypre_IJMatrixAddToValuesParCSR_long_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                            HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                            const hypre_long_double *values );
HYPRE_Int hypre_IJMatrixAssembleOffProcValsParCSR_flt  ( hypre_IJMatrix *matrix,
                                                    HYPRE_Int off_proc_i_indx, HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts,
                                                    HYPRE_MemoryLocation memory_location, HYPRE_BigInt *off_proc_i, HYPRE_BigInt *off_proc_j,
                                                    hypre_float *off_proc_data );
HYPRE_Int hypre_IJMatrixAssembleOffProcValsParCSR_dbl  ( hypre_IJMatrix *matrix,
                                                    HYPRE_Int off_proc_i_indx, HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts,
                                                    HYPRE_MemoryLocation memory_location, HYPRE_BigInt *off_proc_i, HYPRE_BigInt *off_proc_j,
                                                    hypre_double *off_proc_data );
HYPRE_Int hypre_IJMatrixAssembleOffProcValsParCSR_long_dbl  ( hypre_IJMatrix *matrix,
                                                    HYPRE_Int off_proc_i_indx, HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts,
                                                    HYPRE_MemoryLocation memory_location, HYPRE_BigInt *off_proc_i, HYPRE_BigInt *off_proc_j,
                                                    hypre_long_double *off_proc_data );
HYPRE_Int hypre_IJMatrixAssembleParCSR_flt  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixAssembleParCSR_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixAssembleParCSR_long_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixCreateParCSR_flt  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixCreateParCSR_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixCreateParCSR_long_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDestroyParCSR_flt  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDestroyParCSR_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDestroyParCSR_long_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixGetRowCountsParCSR_flt  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_BigInt *rows, HYPRE_Int *ncols );
HYPRE_Int hypre_IJMatrixGetRowCountsParCSR_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_BigInt *rows, HYPRE_Int *ncols );
HYPRE_Int hypre_IJMatrixGetRowCountsParCSR_long_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_BigInt *rows, HYPRE_Int *ncols );
HYPRE_Int hypre_IJMatrixGetValuesParCSR_flt  ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          HYPRE_BigInt *rows, HYPRE_BigInt *cols, hypre_float *values );
HYPRE_Int hypre_IJMatrixGetValuesParCSR_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          HYPRE_BigInt *rows, HYPRE_BigInt *cols, hypre_double *values );
HYPRE_Int hypre_IJMatrixGetValuesParCSR_long_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          HYPRE_BigInt *rows, HYPRE_BigInt *cols, hypre_long_double *values );
HYPRE_Int hypre_IJMatrixInitializeParCSR_flt  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInitializeParCSR_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInitializeParCSR_long_dbl  ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInitializeParCSR_v2_flt (hypre_IJMatrix *matrix,
                                            HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJMatrixInitializeParCSR_v2_dbl (hypre_IJMatrix *matrix,
                                            HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJMatrixInitializeParCSR_v2_long_dbl (hypre_IJMatrix *matrix,
                                            HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJMatrixNormParCSR_flt  ( hypre_IJMatrix *matrix, hypre_float *norm );
HYPRE_Int hypre_IJMatrixNormParCSR_dbl  ( hypre_IJMatrix *matrix, hypre_double *norm );
HYPRE_Int hypre_IJMatrixNormParCSR_long_dbl  ( hypre_IJMatrix *matrix, hypre_long_double *norm );
HYPRE_Int hypre_IJMatrixSetConstantValuesParCSR_flt  ( hypre_IJMatrix *matrix, hypre_float value );
HYPRE_Int hypre_IJMatrixSetConstantValuesParCSR_dbl  ( hypre_IJMatrix *matrix, hypre_double value );
HYPRE_Int hypre_IJMatrixSetConstantValuesParCSR_long_dbl  ( hypre_IJMatrix *matrix, hypre_long_double value );
HYPRE_Int hypre_IJMatrixSetDiagOffdSizesParCSR_flt  ( hypre_IJMatrix *matrix,
                                                 const HYPRE_Int *diag_sizes, const HYPRE_Int *offdiag_sizes );
HYPRE_Int hypre_IJMatrixSetDiagOffdSizesParCSR_dbl  ( hypre_IJMatrix *matrix,
                                                 const HYPRE_Int *diag_sizes, const HYPRE_Int *offdiag_sizes );
HYPRE_Int hypre_IJMatrixSetDiagOffdSizesParCSR_long_dbl  ( hypre_IJMatrix *matrix,
                                                 const HYPRE_Int *diag_sizes, const HYPRE_Int *offdiag_sizes );
HYPRE_Int hypre_IJMatrixSetMaxOffProcElmtsParCSR_flt  ( hypre_IJMatrix *matrix,
                                                   HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJMatrixSetMaxOffProcElmtsParCSR_dbl  ( hypre_IJMatrix *matrix,
                                                   HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJMatrixSetMaxOffProcElmtsParCSR_long_dbl  ( hypre_IJMatrix *matrix,
                                                   HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJMatrixSetRowSizesParCSR_flt  ( hypre_IJMatrix *matrix, const HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetRowSizesParCSR_dbl  ( hypre_IJMatrix *matrix, const HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetRowSizesParCSR_long_dbl  ( hypre_IJMatrix *matrix, const HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetValuesOMPParCSR_flt  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                             const hypre_float *values );
HYPRE_Int hypre_IJMatrixSetValuesOMPParCSR_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                             const hypre_double *values );
HYPRE_Int hypre_IJMatrixSetValuesOMPParCSR_long_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                             const hypre_long_double *values );
HYPRE_Int hypre_IJMatrixSetValuesParCSR_flt  ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                          const hypre_float *values );
HYPRE_Int hypre_IJMatrixSetValuesParCSR_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                          const hypre_double *values );
HYPRE_Int hypre_IJMatrixSetValuesParCSR_long_dbl  ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                          const hypre_long_double *values );
HYPRE_Int hypre_IJMatrixTransposeParCSR_flt  ( hypre_IJMatrix  *matrix_A, hypre_IJMatrix *matrix_AT );
HYPRE_Int hypre_IJMatrixTransposeParCSR_dbl  ( hypre_IJMatrix  *matrix_A, hypre_IJMatrix *matrix_AT );
HYPRE_Int hypre_IJMatrixTransposeParCSR_long_dbl  ( hypre_IJMatrix  *matrix_A, hypre_IJMatrix *matrix_AT );
HYPRE_Int hypre_IJVectorDistribute_flt  ( HYPRE_IJVector vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorDistribute_dbl  ( HYPRE_IJVector vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorDistribute_long_dbl  ( HYPRE_IJVector vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorZeroValues_flt  ( HYPRE_IJVector vector );
HYPRE_Int hypre_IJVectorZeroValues_dbl  ( HYPRE_IJVector vector );
HYPRE_Int hypre_IJVectorZeroValues_long_dbl  ( HYPRE_IJVector vector );
HYPRE_Int hypre_IJVectorAddToValuesPar_flt  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                         const HYPRE_BigInt *indices, const hypre_float *values );
HYPRE_Int hypre_IJVectorAddToValuesPar_dbl  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                         const HYPRE_BigInt *indices, const hypre_double *values );
HYPRE_Int hypre_IJVectorAddToValuesPar_long_dbl  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                         const HYPRE_BigInt *indices, const hypre_long_double *values );
HYPRE_Int hypre_IJVectorAssembleOffProcValsPar_flt  ( hypre_IJVector *vector,
                                                 HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts, HYPRE_MemoryLocation memory_location,
                                                 HYPRE_BigInt *off_proc_i, hypre_float *off_proc_data );
HYPRE_Int hypre_IJVectorAssembleOffProcValsPar_dbl  ( hypre_IJVector *vector,
                                                 HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts, HYPRE_MemoryLocation memory_location,
                                                 HYPRE_BigInt *off_proc_i, hypre_double *off_proc_data );
HYPRE_Int hypre_IJVectorAssembleOffProcValsPar_long_dbl  ( hypre_IJVector *vector,
                                                 HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts, HYPRE_MemoryLocation memory_location,
                                                 HYPRE_BigInt *off_proc_i, hypre_long_double *off_proc_data );
HYPRE_Int hypre_IJVectorAssemblePar_flt  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorAssemblePar_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorAssemblePar_long_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorCreatePar_flt  ( hypre_IJVector *vector, HYPRE_BigInt *IJpartitioning );
HYPRE_Int hypre_IJVectorCreatePar_dbl  ( hypre_IJVector *vector, HYPRE_BigInt *IJpartitioning );
HYPRE_Int hypre_IJVectorCreatePar_long_dbl  ( hypre_IJVector *vector, HYPRE_BigInt *IJpartitioning );
HYPRE_Int hypre_IJVectorDestroyPar_flt  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorDestroyPar_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorDestroyPar_long_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorDistributePar_flt  ( hypre_IJVector *vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorDistributePar_dbl  ( hypre_IJVector *vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorDistributePar_long_dbl  ( hypre_IJVector *vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorGetValuesPar_flt  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, hypre_float *values );
HYPRE_Int hypre_IJVectorGetValuesPar_dbl  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, hypre_double *values );
HYPRE_Int hypre_IJVectorGetValuesPar_long_dbl  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, hypre_long_double *values );
HYPRE_Int hypre_IJVectorInitializePar_flt  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializePar_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializePar_long_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializePar_v2_flt (hypre_IJVector *vector,
                                         HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJVectorInitializePar_v2_dbl (hypre_IJVector *vector,
                                         HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJVectorInitializePar_v2_long_dbl (hypre_IJVector *vector,
                                         HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJVectorSetComponentPar_flt  ( hypre_IJVector *vector, HYPRE_Int component);
HYPRE_Int hypre_IJVectorSetComponentPar_dbl  ( hypre_IJVector *vector, HYPRE_Int component);
HYPRE_Int hypre_IJVectorSetComponentPar_long_dbl  ( hypre_IJVector *vector, HYPRE_Int component);
HYPRE_Int hypre_IJVectorSetMaxOffProcElmtsPar_flt  ( hypre_IJVector *vector,
                                                HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJVectorSetMaxOffProcElmtsPar_dbl  ( hypre_IJVector *vector,
                                                HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJVectorSetMaxOffProcElmtsPar_long_dbl  ( hypre_IJVector *vector,
                                                HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJVectorSetValuesPar_flt  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, const hypre_float *values );
HYPRE_Int hypre_IJVectorSetValuesPar_dbl  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, const hypre_double *values );
HYPRE_Int hypre_IJVectorSetValuesPar_long_dbl  ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, const hypre_long_double *values );
HYPRE_Int hypre_IJVectorZeroValuesPar_flt  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorZeroValuesPar_dbl  ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorZeroValuesPar_long_dbl  ( hypre_IJVector *vector );

#endif

#endif
