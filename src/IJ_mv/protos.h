/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* aux_parcsr_matrix.c */
HYPRE_Int hypre_AuxParCSRMatrixCreate ( hypre_AuxParCSRMatrix **aux_matrix,
                                        HYPRE_Int local_num_rows, HYPRE_Int local_num_cols, HYPRE_Int *sizes );
HYPRE_Int hypre_AuxParCSRMatrixDestroy ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixSetRownnz ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize ( hypre_AuxParCSRMatrix *matrix );
HYPRE_Int hypre_AuxParCSRMatrixInitialize_v2( hypre_AuxParCSRMatrix *matrix,
                                              HYPRE_MemoryLocation memory_location );

/* aux_par_vector.c */
HYPRE_Int hypre_AuxParVectorCreate ( hypre_AuxParVector **aux_vector );
HYPRE_Int hypre_AuxParVectorDestroy ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorInitialize ( hypre_AuxParVector *vector );
HYPRE_Int hypre_AuxParVectorInitialize_v2( hypre_AuxParVector *vector,
                                           HYPRE_MemoryLocation memory_location );

/* IJ_assumed_part.c */
HYPRE_Int hypre_IJMatrixCreateAssumedPartition ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJVectorCreateAssumedPartition ( hypre_IJVector *vector );

/* IJMatrix.c */
HYPRE_Int hypre_IJMatrixGetRowPartitioning ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **row_partitioning );
HYPRE_Int hypre_IJMatrixGetColPartitioning ( HYPRE_IJMatrix matrix,
                                             HYPRE_BigInt **col_partitioning );
HYPRE_Int hypre_IJMatrixSetObject ( HYPRE_IJMatrix matrix, void *object );
HYPRE_Int hypre_IJMatrixRead( const char *filename, MPI_Comm comm, HYPRE_Int type,
                              HYPRE_IJMatrix *matrix_ptr, HYPRE_Int is_mm );
HYPRE_Int hypre_IJMatrixReadBinary( const char *prefixname, MPI_Comm comm,
                                    HYPRE_Int type, HYPRE_IJMatrix *matrix_ptr );

/* IJMatrix_isis.c */
HYPRE_Int hypre_IJMatrixSetLocalSizeISIS ( hypre_IJMatrix *matrix, HYPRE_Int local_m,
                                           HYPRE_Int local_n );
HYPRE_Int hypre_IJMatrixCreateISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesISIS ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagRowSizesISIS ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesISIS ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixInitializeISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInsertBlockISIS ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                          HYPRE_Int *rows, HYPRE_Int *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToBlockISIS ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                         HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixInsertRowISIS ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                        HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToRowISIS ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                       HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAssembleISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDistributeISIS ( hypre_IJMatrix *matrix, HYPRE_BigInt *row_starts,
                                         HYPRE_BigInt *col_starts );
HYPRE_Int hypre_IJMatrixApplyISIS ( hypre_IJMatrix *matrix, hypre_ParVector *x,
                                    hypre_ParVector *b );
HYPRE_Int hypre_IJMatrixDestroyISIS ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetTotalSizeISIS ( hypre_IJMatrix *matrix, HYPRE_Int size );

/* IJMatrix_parcsr.c */
HYPRE_Int hypre_IJMatrixCreateParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesParCSR ( hypre_IJMatrix *matrix, const HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagOffdSizesParCSR ( hypre_IJMatrix *matrix,
                                                 const HYPRE_Int *diag_sizes, const HYPRE_Int *offdiag_sizes );
HYPRE_Int hypre_IJMatrixSetMaxOffProcElmtsParCSR ( hypre_IJMatrix *matrix,
                                                   HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJMatrixSetInitAllocationParCSR(hypre_IJMatrix *matrix,
                                                HYPRE_Int       factor);
HYPRE_Int hypre_IJMatrixSetEarlyAssembleParCSR(hypre_IJMatrix *matrix,
                                               HYPRE_Int       early_assemble);
HYPRE_Int hypre_IJMatrixSetGrowFactorParCSR(hypre_IJMatrix *matrix,
                                            HYPRE_Real      factor);
HYPRE_Int hypre_IJMatrixInitializeParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixGetRowCountsParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_BigInt *rows, HYPRE_Int *ncols );
HYPRE_Int hypre_IJMatrixGetValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          HYPRE_BigInt *rows,
                                          HYPRE_Int *row_indexes, HYPRE_BigInt *cols, HYPRE_Complex *values, HYPRE_Int zero_out );
HYPRE_Int hypre_IJMatrixSetValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows, HYPRE_Int *ncols,
                                          const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                          const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixSetConstantValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Complex value );
HYPRE_Int hypre_IJMatrixAddToValuesParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                            HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                            const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixDestroyParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixTransposeParCSR ( hypre_IJMatrix  *matrix_A, hypre_IJMatrix *matrix_AT );
HYPRE_Int hypre_IJMatrixNormParCSR ( hypre_IJMatrix *matrix, HYPRE_Real *norm );
HYPRE_Int hypre_IJMatrixAddParCSR ( HYPRE_Complex alpha, hypre_IJMatrix *matrix_A,
                                    HYPRE_Complex beta, hypre_IJMatrix *matrix_B, hypre_IJMatrix *matrix_C );
HYPRE_Int hypre_IJMatrixAssembleOffProcValsParCSR ( hypre_IJMatrix *matrix,
                                                    HYPRE_Int off_proc_i_indx, HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts,
                                                    HYPRE_MemoryLocation memory_location, HYPRE_BigInt *off_proc_i, HYPRE_BigInt *off_proc_j,
                                                    HYPRE_Complex *off_proc_data );
HYPRE_Int hypre_FillResponseIJOffProcVals ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                            HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                            HYPRE_Int *response_message_size );
HYPRE_Int hypre_FindProc ( HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length );
HYPRE_Int hypre_IJMatrixAssembleParCSR ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetValuesOMPParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                             HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                             const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixAddToValuesOMPParCSR ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                               HYPRE_Int *ncols, const HYPRE_BigInt *rows, const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                               const HYPRE_Complex *values );
HYPRE_Int hypre_IJMatrixInitializeParCSR_v2(hypre_IJMatrix *matrix,
                                            HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJMatrixMigrateParCSR(hypre_IJMatrix *matrix, HYPRE_MemoryLocation memory_location);

HYPRE_Int hypre_IJMatrixAssembleCommunicate(hypre_IJMatrix *matrix);

/* IJMatrix_parcsr_device.c */
HYPRE_Int hypre_IJMatrixSetConstantValuesParCSRDevice( hypre_IJMatrix *matrix,
                                                       HYPRE_Complex value );
HYPRE_Int hypre_IJMatrixSetAddValuesParCSRDevice ( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                                   HYPRE_Int *ncols, const HYPRE_BigInt *rows,
                                                   const HYPRE_Int *row_indexes, const HYPRE_BigInt *cols,
                                                   const HYPRE_Complex *values, const char *action );
HYPRE_Int hypre_IJMatrixGetValuesParCSRDevice( hypre_IJMatrix *matrix, HYPRE_Int nrows,
                                               HYPRE_Int *ncols, HYPRE_BigInt *rows,
                                               HYPRE_Int *row_indexes, HYPRE_BigInt *cols,
                                               HYPRE_Complex *values, HYPRE_Int zero_out );
HYPRE_Int hypre_IJMatrixAssembleCompressDevice(hypre_IJMatrix *matrix, HYPRE_Int reduce_stack_size);
HYPRE_Int hypre_IJMatrixAssembleParCSRDevice(hypre_IJMatrix *matrix);

/* IJMatrix_petsc.c */
HYPRE_Int hypre_IJMatrixSetLocalSizePETSc ( hypre_IJMatrix *matrix, HYPRE_Int local_m,
                                            HYPRE_Int local_n );
HYPRE_Int hypre_IJMatrixCreatePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetRowSizesPETSc ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetDiagRowSizesPETSc ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixSetOffDiagRowSizesPETSc ( hypre_IJMatrix *matrix, HYPRE_Int *sizes );
HYPRE_Int hypre_IJMatrixInitializePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixInsertBlockPETSc ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                           HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToBlockPETSc ( hypre_IJMatrix *matrix, HYPRE_Int m, HYPRE_Int n,
                                          HYPRE_Int *rows, HYPRE_Int *cols, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixInsertRowPETSc ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                         HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAddToRowPETSc ( hypre_IJMatrix *matrix, HYPRE_Int n, HYPRE_BigInt row,
                                        HYPRE_BigInt *indices, HYPRE_Complex *coeffs );
HYPRE_Int hypre_IJMatrixAssemblePETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixDistributePETSc ( hypre_IJMatrix *matrix, HYPRE_BigInt *row_starts,
                                          HYPRE_BigInt *col_starts );
HYPRE_Int hypre_IJMatrixApplyPETSc ( hypre_IJMatrix *matrix, hypre_ParVector *x,
                                     hypre_ParVector *b );
HYPRE_Int hypre_IJMatrixDestroyPETSc ( hypre_IJMatrix *matrix );
HYPRE_Int hypre_IJMatrixSetTotalSizePETSc ( hypre_IJMatrix *matrix, HYPRE_Int size );

/* IJVector.c */
HYPRE_Int hypre_IJVectorDistribute ( HYPRE_IJVector vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorZeroValues ( HYPRE_IJVector vector );
HYPRE_Int hypre_IJVectorReadBinary ( MPI_Comm comm, const char *filename, HYPRE_Int type,
                                     HYPRE_IJVector *vector_ptr );

/* IJVector_parcsr.c */
HYPRE_Int hypre_IJVectorCreatePar ( hypre_IJVector *vector, HYPRE_BigInt *IJpartitioning );
HYPRE_Int hypre_IJVectorDestroyPar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializeParShell (hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorSetParData( hypre_IJVector *vector, HYPRE_Complex *data );
HYPRE_Int hypre_IJVectorSetTagsPar( hypre_IJVector *vector, HYPRE_Int owns_tags,
                                    HYPRE_Int num_tags, HYPRE_Int *tags );
HYPRE_Int hypre_IJVectorInitializePar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorInitializePar_v2(hypre_IJVector *vector,
                                         HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_IJVectorSetMaxOffProcElmtsPar ( hypre_IJVector *vector,
                                                HYPRE_Int max_off_proc_elmts );
HYPRE_Int hypre_IJVectorDistributePar ( hypre_IJVector *vector, const HYPRE_Int *vec_starts );
HYPRE_Int hypre_IJVectorZeroValuesPar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorSetComponentPar ( hypre_IJVector *vector, HYPRE_Int component);
HYPRE_Int hypre_IJVectorSetValuesPar ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, const HYPRE_Complex *values );
HYPRE_Int hypre_IJVectorSetConstantValuesPar ( hypre_IJVector *vector, HYPRE_Complex value );
HYPRE_Int hypre_IJVectorAddToValuesPar ( hypre_IJVector *vector, HYPRE_Int num_values,
                                         const HYPRE_BigInt *indices, const HYPRE_Complex *values );
HYPRE_Int hypre_IJVectorAssemblePar ( hypre_IJVector *vector );
HYPRE_Int hypre_IJVectorGetValuesPar ( hypre_IJVector *vector, HYPRE_Int num_values,
                                       const HYPRE_BigInt *indices, HYPRE_Complex *values );
HYPRE_Int hypre_IJVectorAssembleOffProcValsPar ( hypre_IJVector *vector,
                                                 HYPRE_Int max_off_proc_elmts, HYPRE_Int current_num_elmts, HYPRE_MemoryLocation memory_location,
                                                 HYPRE_BigInt *off_proc_i, HYPRE_Complex *off_proc_data );
HYPRE_Int hypre_IJVectorMigrateParCSR(hypre_IJVector *vector, HYPRE_MemoryLocation memory_location);

/* IJVector_parcsr_device.c */
HYPRE_Int hypre_IJVectorSetAddValuesParDevice(hypre_IJVector *vector, HYPRE_Int num_values,
                                              const HYPRE_BigInt *indices, const HYPRE_Complex *values, const char *action);
HYPRE_Int hypre_IJVectorAssembleParDevice(hypre_IJVector *vector);
HYPRE_Int hypre_IJVectorUpdateValuesDevice( hypre_IJVector *vector, HYPRE_Int num_values,
                                            const HYPRE_BigInt *indices, const HYPRE_Complex *values, HYPRE_Int action);


