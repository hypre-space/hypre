/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* communicationT.c */
void hypre_RowsWithColumn_original ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                                     hypre_ParCSRMatrix *A );
void hypre_RowsWithColumn ( HYPRE_Int *rowmin, HYPRE_Int *rowmax, HYPRE_BigInt column,
                            HYPRE_Int num_rows_diag, HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag,
                            HYPRE_Int *mat_j_diag, HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd );
void hypre_MatTCommPkgCreate_core ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_rows_diag,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd, HYPRE_BigInt *row_starts,
                                    HYPRE_BigInt firstColDiag, HYPRE_BigInt *colMapOffd, HYPRE_Int *mat_i_diag, HYPRE_Int *mat_j_diag,
                                    HYPRE_Int *mat_i_offd, HYPRE_Int *mat_j_offd, HYPRE_Int data, HYPRE_Int *p_num_recvs,
                                    HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends,
                                    HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
HYPRE_Int hypre_MatTCommPkgCreate ( hypre_ParCSRMatrix *A );

/* driver_aat.c */

/* driver_boolaat.c */

/* driver_boolmatmul.c */

/* driver.c */

/* driver_matmul.c */

/* driver_mat_multivec.c */

/* driver_matvec.c */

/* driver_multivec.c */

/* HYPRE_parcsr_matrix.c */
HYPRE_Int HYPRE_ParCSRMatrixCreate ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                     HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts,
                                     HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag, HYPRE_Int num_nonzeros_offd,
                                     HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixInitialize ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixBigInitialize ( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixRead ( MPI_Comm comm, const char *file_name,
                                   HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixPrint ( HYPRE_ParCSRMatrix matrix, const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixGetComm ( HYPRE_ParCSRMatrix matrix, MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetDims ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *M, HYPRE_BigInt *N );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetGlobalRowPartitioning ( HYPRE_ParCSRMatrix matrix,
                                                       HYPRE_Int all_procs, HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning ( HYPRE_ParCSRMatrix matrix,
                                                 HYPRE_BigInt **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetRow ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, HYPRE_Complex **values );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow ( HYPRE_ParCSRMatrix matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, HYPRE_Complex **values );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                          HYPRE_BigInt *row_partitioning, HYPRE_BigInt *col_partitioning, HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning ( MPI_Comm comm, HYPRE_CSRMatrix A_CSR,
                                                              HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixMatvec ( HYPRE_Complex alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                     HYPRE_Complex beta, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT ( HYPRE_Complex alpha, HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                                      HYPRE_Complex beta, HYPRE_ParVector y );

/* HYPRE_parcsr_vector.c */
HYPRE_Int HYPRE_ParVectorCreate ( MPI_Comm comm, HYPRE_BigInt global_size,
                                  HYPRE_BigInt *partitioning, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParMultiVectorCreate ( MPI_Comm comm, HYPRE_BigInt global_size,
                                       HYPRE_BigInt *partitioning, HYPRE_Int number_vectors, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorDestroy ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInitialize ( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorRead ( MPI_Comm comm, const char *file_name, HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorPrint ( HYPRE_ParVector vector, const char *file_name );
HYPRE_Int HYPRE_ParVectorPrintBinaryIJ ( HYPRE_ParVector vector, const char *file_name );
HYPRE_Int HYPRE_ParVectorSetConstantValues ( HYPRE_ParVector vector, HYPRE_Complex value );
HYPRE_Int HYPRE_ParVectorSetRandomValues ( HYPRE_ParVector vector, HYPRE_Int seed );
HYPRE_Int HYPRE_ParVectorCopy ( HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_Int hypre_ParVectorStridedCopy( hypre_ParVector *x, HYPRE_Int istride, HYPRE_Int ostride,
                                      HYPRE_Int size, HYPRE_Complex *data );
HYPRE_ParVector HYPRE_ParVectorCloneShallow ( HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorScale ( HYPRE_Complex value, HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorAxpy ( HYPRE_Complex alpha, HYPRE_ParVector x, HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorInnerProd ( HYPRE_ParVector x, HYPRE_ParVector y, HYPRE_Real *prod );
HYPRE_Int HYPRE_VectorToParVector ( MPI_Comm comm, HYPRE_Vector b, HYPRE_BigInt *partitioning,
                                    HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorGetValues ( HYPRE_ParVector vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, HYPRE_Complex *values);

/* gen_fffc.c */
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCHost( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                              hypre_ParCSRMatrix **A_FC_ptr,
                                              hypre_ParCSRMatrix **A_FF_ptr ) ;
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                          hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr ) ;
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC3(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                          hypre_ParCSRMatrix **A_FF_ptr ) ;
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCD3(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                           hypre_ParCSRMatrix **A_FF_ptr, HYPRE_Real **D_lambda_ptr ) ;
HYPRE_Int hypre_ParCSRMatrixGenerateFFFC3Device(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **A_FC_ptr,
                                                hypre_ParCSRMatrix **A_FF_ptr ) ;
HYPRE_Int hypre_ParCSRMatrixGenerateCFDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **ACF_ptr) ;
HYPRE_Int hypre_ParCSRMatrixGenerateCCDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **ACC_ptr) ;
HYPRE_Int hypre_ParCSRMatrixGenerate1DCFDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S, hypre_ParCSRMatrix **ACX_ptr,
                                                hypre_ParCSRMatrix **AXC_ptr ) ;

/* new_commpkg.c */
HYPRE_Int hypre_PrintCommpkg ( hypre_ParCSRMatrix *A, const char *file_name );
HYPRE_Int hypre_ParCSRCommPkgCreateApart_core ( MPI_Comm comm, HYPRE_BigInt *col_map_off_d,
                                                HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_off_d, HYPRE_BigInt global_num_cols,
                                                HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs, HYPRE_Int **p_recv_vec_starts,
                                                HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts,
                                                HYPRE_Int **p_send_map_elements, hypre_IJAssumedPart *apart );
HYPRE_Int hypre_ParCSRCommPkgCreateApart ( MPI_Comm  comm, HYPRE_BigInt *col_map_off_d,
                                           HYPRE_BigInt  first_col_diag, HYPRE_Int  num_cols_off_d, HYPRE_BigInt  global_num_cols,
                                           hypre_IJAssumedPart *apart, hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_NewCommPkgDestroy ( hypre_ParCSRMatrix *parcsr_A );
HYPRE_Int hypre_RangeFillResponseIJDetermineRecvProcs ( void *p_recv_contact_buf,
                                                        HYPRE_Int contact_size, HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                        HYPRE_Int *response_message_size );
HYPRE_Int hypre_FillResponseIJDetermineSendProcs ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                                   HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                                   HYPRE_Int *response_message_size );

/* numbers.c */
hypre_NumbersNode *hypre_NumbersNewNode ( void );
void hypre_NumbersDeleteNode ( hypre_NumbersNode *node );
HYPRE_Int hypre_NumbersEnter ( hypre_NumbersNode *node, const HYPRE_Int n );
HYPRE_Int hypre_NumbersNEntered ( hypre_NumbersNode *node );
HYPRE_Int hypre_NumbersQuery ( hypre_NumbersNode *node, const HYPRE_Int n );
HYPRE_Int *hypre_NumbersArray ( hypre_NumbersNode *node );

/* parchord_to_parcsr.c */
void hypre_ParChordMatrix_RowStarts ( hypre_ParChordMatrix *Ac, MPI_Comm comm,
                                      HYPRE_BigInt **row_starts, HYPRE_BigInt *global_num_cols );
HYPRE_Int hypre_ParChordMatrixToParCSRMatrix ( hypre_ParChordMatrix *Ac, MPI_Comm comm,
                                               hypre_ParCSRMatrix **pAp );
HYPRE_Int hypre_ParCSRMatrixToParChordMatrix ( hypre_ParCSRMatrix *Ap, MPI_Comm comm,
                                               hypre_ParChordMatrix **pAc );

/* par_csr_aat.c */
void hypre_ParAat_RowSizes ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i, HYPRE_Int *B_marker,
                             HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                             HYPRE_BigInt *A_col_map_offd, HYPRE_Int *A_ext_i, HYPRE_BigInt *A_ext_j,
                             HYPRE_BigInt *A_ext_row_map, HYPRE_Int *C_diag_size, HYPRE_Int *C_offd_size,
                             HYPRE_Int num_rows_diag_A, HYPRE_Int num_cols_offd_A, HYPRE_Int num_rows_A_ext,
                             HYPRE_BigInt first_col_diag_A, HYPRE_BigInt first_row_index_A );
hypre_ParCSRMatrix *hypre_ParCSRAAt ( hypre_ParCSRMatrix *A );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractAExt ( hypre_ParCSRMatrix *A, HYPRE_Int data,
                                                 HYPRE_BigInt **pA_ext_row_map );

/* par_csr_assumed_part.c */
HYPRE_Int hypre_LocateAssumedPartition ( MPI_Comm comm, HYPRE_BigInt row_start,
                                         HYPRE_BigInt row_end, HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows,
                                         hypre_IJAssumedPart *part, HYPRE_Int myid );
hypre_IJAssumedPart *hypre_AssumedPartitionCreate ( MPI_Comm comm, HYPRE_BigInt global_num,
                                                    HYPRE_BigInt start, HYPRE_BigInt end );
HYPRE_Int hypre_ParCSRMatrixCreateAssumedPartition ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_AssumedPartitionDestroy ( hypre_IJAssumedPart *apart );
HYPRE_Int hypre_GetAssumedPartitionProcFromRow ( MPI_Comm comm, HYPRE_BigInt row,
                                                 HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_Int *proc_id );
HYPRE_Int hypre_GetAssumedPartitionRowRange ( MPI_Comm comm, HYPRE_Int proc_id,
                                              HYPRE_BigInt global_first_row, HYPRE_BigInt global_num_rows, HYPRE_BigInt *row_start,
                                              HYPRE_BigInt *row_end );
HYPRE_Int hypre_ParVectorCreateAssumedPartition ( hypre_ParVector *vector );

/* par_csr_bool_matop.c */
hypre_ParCSRBooleanMatrix *hypre_ParBooleanMatmul ( hypre_ParCSRBooleanMatrix *A,
                                                    hypre_ParCSRBooleanMatrix *B );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractBExt ( hypre_ParCSRBooleanMatrix *B,
                                                               hypre_ParCSRBooleanMatrix *A );
hypre_CSRBooleanMatrix *hypre_ParCSRBooleanMatrixExtractAExt ( hypre_ParCSRBooleanMatrix *A,
                                                               HYPRE_BigInt **pA_ext_row_map );
hypre_ParCSRBooleanMatrix *hypre_ParBooleanAAt ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatTCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );
HYPRE_Int hypre_BooleanMatvecCommPkgCreate ( hypre_ParCSRBooleanMatrix *A );

/* par_csr_bool_matrix.c */
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixCreate ( HYPRE_Int num_rows, HYPRE_Int num_cols,
                                                       HYPRE_Int num_nonzeros );
HYPRE_Int hypre_CSRBooleanMatrixDestroy ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixInitialize ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixBigInitialize ( hypre_CSRBooleanMatrix *matrix );
HYPRE_Int hypre_CSRBooleanMatrixSetDataOwner ( hypre_CSRBooleanMatrix *matrix,
                                               HYPRE_Int owns_data );
HYPRE_Int hypre_CSRBooleanMatrixSetBigDataOwner ( hypre_CSRBooleanMatrix *matrix,
                                                  HYPRE_Int owns_data );
hypre_CSRBooleanMatrix *hypre_CSRBooleanMatrixRead ( const char *file_name );
HYPRE_Int hypre_CSRBooleanMatrixPrint ( hypre_CSRBooleanMatrix *matrix, const char *file_name );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixCreate ( MPI_Comm comm,
                                                             HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols, HYPRE_BigInt *row_starts,
                                                             HYPRE_BigInt *col_starts, HYPRE_Int num_cols_offd, HYPRE_Int num_nonzeros_diag,
                                                             HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRBooleanMatrixDestroy ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixInitialize ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetNNZ ( hypre_ParCSRBooleanMatrix *matrix );
HYPRE_Int hypre_ParCSRBooleanMatrixSetDataOwner ( hypre_ParCSRBooleanMatrix *matrix,
                                                  HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRBooleanMatrixSetRowStartsOwner ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_row_starts );
HYPRE_Int hypre_ParCSRBooleanMatrixSetColStartsOwner ( hypre_ParCSRBooleanMatrix *matrix,
                                                       HYPRE_Int owns_col_starts );
hypre_ParCSRBooleanMatrix *hypre_ParCSRBooleanMatrixRead ( MPI_Comm comm, const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrint ( hypre_ParCSRBooleanMatrix *matrix,
                                           const char *file_name );
HYPRE_Int hypre_ParCSRBooleanMatrixPrintIJ ( hypre_ParCSRBooleanMatrix *matrix,
                                             const char *filename );
HYPRE_Int hypre_ParCSRBooleanMatrixGetLocalRange ( hypre_ParCSRBooleanMatrix *matrix,
                                                   HYPRE_BigInt *row_start, HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRBooleanMatrixGetRow ( hypre_ParCSRBooleanMatrix *mat, HYPRE_BigInt row,
                                            HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_ParCSRBooleanMatrixRestoreRow ( hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt row,
                                                HYPRE_Int *size, HYPRE_BigInt **col_ind );
HYPRE_Int hypre_BuildCSRBooleanMatrixMPIDataType ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                                   HYPRE_Int *a_i, HYPRE_Int *a_j, hypre_MPI_Datatype *csr_matrix_datatype );
hypre_ParCSRBooleanMatrix *hypre_CSRBooleanMatrixToParCSRBooleanMatrix ( MPI_Comm comm,
                                                                         hypre_CSRBooleanMatrix *A, HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
HYPRE_Int hypre_BooleanGenerateDiagAndOffd ( hypre_CSRBooleanMatrix *A,
                                             hypre_ParCSRBooleanMatrix *matrix, HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );

/* par_csr_communication.c */
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate ( HYPRE_Int job, hypre_ParCSRCommPkg *comm_pkg,
                                                       void *send_data, void *recv_data );
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate_v2 ( HYPRE_Int job,
                                                          hypre_ParCSRCommPkg *comm_pkg,
                                                          HYPRE_MemoryLocation send_memory_location,
                                                          void *send_data_in,
                                                          HYPRE_MemoryLocation recv_memory_location,
                                                          void *recv_data_in );
HYPRE_Int hypre_ParCSRCommHandleDestroy ( hypre_ParCSRCommHandle *comm_handle );
void hypre_ParCSRCommPkgCreate_core ( MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                      HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts, HYPRE_Int num_cols_diag,
                                      HYPRE_Int num_cols_offd, HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs,
                                      HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends, HYPRE_Int **p_send_procs,
                                      HYPRE_Int **p_send_map_starts, HYPRE_Int **p_send_map_elmts );
HYPRE_Int hypre_ParCSRCommPkgCreate(MPI_Comm comm, HYPRE_BigInt *col_map_offd,
                                    HYPRE_BigInt first_col_diag, HYPRE_BigInt *col_starts,
                                    HYPRE_Int num_cols_diag, HYPRE_Int num_cols_offd,
                                    hypre_ParCSRCommPkg *comm_pkg);
HYPRE_Int hypre_ParCSRCommPkgCreateAndFill ( MPI_Comm comm, HYPRE_Int num_recvs,
                                             HYPRE_Int *recv_procs, HYPRE_Int *recv_vec_starts,
                                             HYPRE_Int num_sends, HYPRE_Int *send_procs,
                                             HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                             hypre_ParCSRCommPkg **comm_pkg_ptr );
HYPRE_Int hypre_ParCSRCommPkgUpdateVecStarts ( hypre_ParCSRCommPkg *comm_pkg,
                                               HYPRE_Int num_components_in,
                                               HYPRE_Int vecstride, HYPRE_Int idxstride );
HYPRE_Int hypre_MatvecCommPkgCreate ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MatvecCommPkgDestroy ( hypre_ParCSRCommPkg *comm_pkg );
HYPRE_Int hypre_BuildCSRMatrixMPIDataType ( HYPRE_Int num_nonzeros, HYPRE_Int num_rows,
                                            HYPRE_Complex *a_data, HYPRE_Int *a_i, HYPRE_Int *a_j,
                                            hypre_MPI_Datatype *csr_matrix_datatype );
HYPRE_Int hypre_BuildCSRJDataType ( HYPRE_Int num_nonzeros, HYPRE_Complex *a_data, HYPRE_Int *a_j,
                                    hypre_MPI_Datatype *csr_jdata_datatype );
HYPRE_Int hypre_ParCSRFindExtendCommPkg(MPI_Comm comm, HYPRE_BigInt global_num_cols,
                                        HYPRE_BigInt first_col_diag, HYPRE_Int num_cols_diag, HYPRE_BigInt *col_starts,
                                        hypre_IJAssumedPart *apart, HYPRE_Int indices_len, HYPRE_BigInt *indices,
                                        hypre_ParCSRCommPkg **extend_comm_pkg);

/* par_csr_matop.c */
HYPRE_Int hypre_ParCSRMatrixScale(hypre_ParCSRMatrix *A, HYPRE_Complex scalar);
void hypre_ParMatmul_RowSizes ( HYPRE_MemoryLocation memory_location, HYPRE_Int **C_diag_i,
                                HYPRE_Int **C_offd_i, HYPRE_Int *rownnz_A, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j, HYPRE_Int *B_diag_i, HYPRE_Int *B_diag_j,
                                HYPRE_Int *B_offd_i, HYPRE_Int *B_offd_j, HYPRE_Int *B_ext_diag_i, HYPRE_Int *B_ext_diag_j,
                                HYPRE_Int *B_ext_offd_i, HYPRE_Int *B_ext_offd_j, HYPRE_Int *map_B_to_C, HYPRE_Int *C_diag_size,
                                HYPRE_Int *C_offd_size, HYPRE_Int num_rownnz_A, HYPRE_Int num_rows_diag_A,
                                HYPRE_Int num_cols_offd_A, HYPRE_Int  allsquare, HYPRE_Int num_cols_diag_B,
                                HYPRE_Int num_cols_offd_B, HYPRE_Int num_cols_offd_C );
hypre_ParCSRMatrix *hypre_ParMatmul ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
void hypre_ParCSRMatrixExtractBExt_Arrays ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                            HYPRE_Complex **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                            HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                            HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                            HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                            HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                            HYPRE_BigInt *col_map_offd, HYPRE_Real *diag_data, HYPRE_Real *offd_data );
void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap ( HYPRE_Int **pB_ext_i, HYPRE_BigInt **pB_ext_j,
                                                    HYPRE_Complex **pB_ext_data, HYPRE_BigInt **pB_ext_row_map, HYPRE_Int *num_nonzeros, HYPRE_Int data,
                                                    HYPRE_Int find_row_map, MPI_Comm comm, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols_B,
                                                    HYPRE_Int num_recvs, HYPRE_Int num_sends, HYPRE_BigInt first_col_diag, HYPRE_BigInt *row_starts,
                                                    HYPRE_Int *recv_vec_starts, HYPRE_Int *send_map_starts, HYPRE_Int *send_map_elmts,
                                                    HYPRE_Int *diag_i, HYPRE_Int *diag_j, HYPRE_Int *offd_i, HYPRE_Int *offd_j,
                                                    HYPRE_BigInt *col_map_offd, HYPRE_Real *diag_data, HYPRE_Real *offd_data,
                                                    hypre_ParCSRCommHandle **comm_handle_idx, hypre_ParCSRCommHandle **comm_handle_data,
                                                    HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                 HYPRE_Int data );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractBExt_Overlap ( hypre_ParCSRMatrix *B,
                                                         hypre_ParCSRMatrix *A, HYPRE_Int data, hypre_ParCSRCommHandle **comm_handle_idx,
                                                         hypre_ParCSRCommHandle **comm_handle_data, HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
                                                         HYPRE_Int skip_fine, HYPRE_Int skip_same_sign );
HYPRE_Int hypre_ParCSRMatrixExtractBExtDeviceInit( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                   HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParCSRMatrixExtractBExtDeviceWait(void *request);
hypre_CSRMatrix* hypre_ParCSRMatrixExtractBExtDevice( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                                      HYPRE_Int want_data );
HYPRE_Int hypre_ParCSRMatrixLocalTranspose( hypre_ParCSRMatrix  *A );
HYPRE_Int hypre_ParCSRMatrixTranspose ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                        HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTransposeHost ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                            HYPRE_Int data );
HYPRE_Int hypre_ParCSRMatrixTransposeDevice ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr,
                                              HYPRE_Int data );
void hypre_ParCSRMatrixGenSpanningTree ( hypre_ParCSRMatrix *G_csr, HYPRE_Int **indices,
                                         HYPRE_Int G_type );
void hypre_ParCSRMatrixExtractSubmatrices ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                            hypre_ParCSRMatrix ***submatrices );
void hypre_ParCSRMatrixExtractRowSubmatrices ( hypre_ParCSRMatrix *A_csr, HYPRE_Int *indices2,
                                               hypre_ParCSRMatrix ***submatrices );
HYPRE_Complex hypre_ParCSRMatrixLocalSumElts ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixAminvDB ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                      HYPRE_Complex *d, hypre_ParCSRMatrix **C_ptr );
hypre_ParCSRMatrix *hypre_ParTMatmul ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
HYPRE_Real hypre_ParCSRMatrixFnorm( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixInfNorm ( hypre_ParCSRMatrix *A, HYPRE_Real *norm );
HYPRE_Int hypre_ExchangeExternalRowsInit( hypre_CSRMatrix *B_ext, hypre_ParCSRCommPkg *comm_pkg_A,
                                          void **request_ptr);
hypre_CSRMatrix* hypre_ExchangeExternalRowsWait(void *vequest);
HYPRE_Int hypre_ExchangeExternalRowsDeviceInit( hypre_CSRMatrix *B_ext,
                                                hypre_ParCSRCommPkg *comm_pkg_A, HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ExchangeExternalRowsDeviceWait(void *vrequest);
HYPRE_Int hypre_ParCSRMatrixGenerateFFFCDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                                hypre_ParCSRMatrix **A_FC_ptr,
                                                hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateFFCFDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                                hypre_ParCSRMatrix **A_CF_ptr,
                                                hypre_ParCSRMatrix **A_FF_ptr );
HYPRE_Int hypre_ParCSRMatrixGenerateCCCFDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, hypre_ParCSRMatrix *S,
                                                hypre_ParCSRMatrix **A_CF_ptr,
                                                hypre_ParCSRMatrix **A_CC_ptr );
hypre_CSRMatrix* hypre_ConcatDiagAndOffdDevice(hypre_ParCSRMatrix *A);
HYPRE_Int hypre_ConcatDiagOffdAndExtDevice(hypre_ParCSRMatrix *A, hypre_CSRMatrix *E,
                                           hypre_CSRMatrix **B_ptr, HYPRE_Int *num_cols_offd_ptr, HYPRE_BigInt **cols_map_offd_ptr);
HYPRE_Int hypre_ParCSRMatrixGetRowDevice( hypre_ParCSRMatrix *mat, HYPRE_BigInt row,
                                          HYPRE_Int *size, HYPRE_BigInt **col_ind, HYPRE_Complex **values );
HYPRE_Int hypre_ParCSRDiagScaleVector( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                       hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRDiagScaleVectorHost( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                           hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRDiagScaleVectorDevice( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_y,
                                             hypre_ParVector *par_x );
HYPRE_Int hypre_ParCSRMatrixDropSmallEntries( hypre_ParCSRMatrix *A, HYPRE_Real tol,
                                              HYPRE_Int type);
HYPRE_Int hypre_ParCSRMatrixDropSmallEntriesHost( hypre_ParCSRMatrix *A, HYPRE_Real tol,
                                                  HYPRE_Int type);
HYPRE_Int hypre_ParCSRMatrixDropSmallEntriesDevice( hypre_ParCSRMatrix *A, HYPRE_Complex tol,
                                                    HYPRE_Int type);

HYPRE_Int hypre_ParCSRCommPkgCreateMatrixE( hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int local_ncols );

#ifdef HYPRE_USING_PERSISTENT_COMM
hypre_ParCSRPersistentCommHandle* hypre_ParCSRPersistentCommHandleCreate(HYPRE_Int job,
                                                                         hypre_ParCSRCommPkg *comm_pkg);
hypre_ParCSRPersistentCommHandle* hypre_ParCSRCommPkgGetPersistentCommHandle(HYPRE_Int job,
                                                                             hypre_ParCSRCommPkg *comm_pkg);
void hypre_ParCSRPersistentCommHandleDestroy(hypre_ParCSRPersistentCommHandle *comm_handle);
void hypre_ParCSRPersistentCommHandleStart(hypre_ParCSRPersistentCommHandle *comm_handle,
                                           HYPRE_MemoryLocation send_memory_location, void *send_data);
void hypre_ParCSRPersistentCommHandleWait(hypre_ParCSRPersistentCommHandle *comm_handle,
                                          HYPRE_MemoryLocation recv_memory_location, void *recv_data);
#endif

HYPRE_Int hypre_ParcsrGetExternalRowsInit( hypre_ParCSRMatrix *A, HYPRE_Int indices_len,
                                           HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsWait(void *vrequest);
HYPRE_Int hypre_ParcsrGetExternalRowsDeviceInit( hypre_ParCSRMatrix *A, HYPRE_Int indices_len,
                                                 HYPRE_BigInt *indices, hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int want_data, void **request_ptr);
hypre_CSRMatrix* hypre_ParcsrGetExternalRowsDeviceWait(void *vrequest);

HYPRE_Int hypre_ParvecBdiagInvScal( hypre_ParVector *b, HYPRE_Int blockSize, hypre_ParVector **bs,
                                    hypre_ParCSRMatrix *A);

HYPRE_Int hypre_ParcsrBdiagInvScal( hypre_ParCSRMatrix *A, HYPRE_Int blockSize,
                                    hypre_ParCSRMatrix **As);

HYPRE_Int hypre_ParCSRMatrixExtractSubmatrixFC( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                HYPRE_BigInt *cpts_starts, const char *job,
                                                hypre_ParCSRMatrix **B_ptr,
                                                HYPRE_Real strength_thresh);
HYPRE_Int hypre_ParCSRMatrixDiagScale( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_ld,
                                       hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixReorder ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixAdd( HYPRE_Complex alpha, hypre_ParCSRMatrix *A, HYPRE_Complex beta,
                                 hypre_ParCSRMatrix *B, hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAddHost( HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                     HYPRE_Complex beta, hypre_ParCSRMatrix *B,
                                     hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixAddDevice( HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                       HYPRE_Complex beta, hypre_ParCSRMatrix *B,
                                       hypre_ParCSRMatrix **Cout);
HYPRE_Int hypre_ParCSRMatrixBlockColSum( hypre_ParCSRMatrix *A, HYPRE_Int row_major,
                                         HYPRE_Int num_rows_block, HYPRE_Int num_cols_block,
                                         hypre_DenseBlockMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixColSum( hypre_ParCSRMatrix *A, hypre_ParVector **B_ptr );

/* par_csr_matop_device.c */
HYPRE_Int hypre_ParCSRMatrixDiagScaleDevice ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_ld,
                                              hypre_ParVector *par_rd );
HYPRE_Int hypre_ParCSRMatrixCompressOffdMapDevice(hypre_ParCSRMatrix *A);
HYPRE_Int hypre_ParCSRMatrixCompressOffdMap(hypre_ParCSRMatrix *A);

/* par_csr_matop_marked.c */
void hypre_ParMatmul_RowSizes_Marked ( HYPRE_Int **C_diag_i, HYPRE_Int **C_offd_i,
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
hypre_ParCSRMatrix *hypre_ParMatmul_FC ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                         HYPRE_Int *CF_marker, HYPRE_Int *dof_func,
                                         HYPRE_Int *dof_func_offd );
void hypre_ParMatScaleDiagInv_F ( hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *A,
                                  HYPRE_Complex weight, HYPRE_Int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                          HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixZero_F ( hypre_ParCSRMatrix *P, HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixCopy_C ( hypre_ParCSRMatrix *P, hypre_ParCSRMatrix *C,
                                HYPRE_Int *CF_marker );
void hypre_ParCSRMatrixDropEntries ( hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *P,
                                     HYPRE_Int *CF_marker );

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate ( MPI_Comm comm, HYPRE_BigInt global_num_rows,
                                               HYPRE_BigInt global_num_cols,
                                               HYPRE_BigInt *row_starts_in,
                                               HYPRE_BigInt *col_starts_in,
                                               HYPRE_Int num_cols_offd,
                                               HYPRE_Int num_nonzeros_diag,
                                               HYPRE_Int num_nonzeros_offd );
HYPRE_Int hypre_ParCSRMatrixDestroy ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixInitialize_v2( hypre_ParCSRMatrix *matrix,
                                           HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParCSRMatrixInitialize ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumNonzeros ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetDNumNonzeros ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetNumRownnz ( hypre_ParCSRMatrix *matrix );
HYPRE_Int hypre_ParCSRMatrixSetDataOwner ( hypre_ParCSRMatrix *matrix, HYPRE_Int owns_data );
HYPRE_Int hypre_ParCSRMatrixSetPatternOnly( hypre_ParCSRMatrix *matrix, HYPRE_Int pattern_only);
hypre_ParCSRMatrix* hypre_ParCSRMatrixCreateFromDenseBlockMatrix(MPI_Comm comm,
                                                                 HYPRE_BigInt global_num_rows,
                                                                 HYPRE_BigInt global_num_cols,
                                                                 HYPRE_BigInt *row_starts,
                                                                 HYPRE_BigInt *col_starts,
                                                                 hypre_DenseBlockMatrix *B);
hypre_ParCSRMatrix* hypre_ParCSRMatrixCreateFromParVector(hypre_ParVector *b,
                                                          HYPRE_BigInt global_num_rows,
                                                          HYPRE_BigInt global_num_cols,
                                                          HYPRE_BigInt *row_starts,
                                                          HYPRE_BigInt *col_starts);
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead ( MPI_Comm comm, const char *file_name );
HYPRE_Int hypre_ParCSRMatrixPrint ( hypre_ParCSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_ParCSRMatrixPrintIJ ( const hypre_ParCSRMatrix *matrix, const HYPRE_Int base_i,
                                      const HYPRE_Int base_j, const char *filename );
HYPRE_Int hypre_ParCSRMatrixPrintBinaryIJ ( hypre_ParCSRMatrix *matrix, HYPRE_Int base_i,
                                            HYPRE_Int base_j, const char *filename );
HYPRE_Int hypre_ParCSRMatrixReadIJ ( MPI_Comm comm, const char *filename, HYPRE_Int *base_i_ptr,
                                     HYPRE_Int *base_j_ptr, hypre_ParCSRMatrix **matrix_ptr );
HYPRE_Int hypre_ParCSRMatrixGetLocalRange ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt *row_start,
                                            HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_ParCSRMatrixGetRow ( hypre_ParCSRMatrix *mat, HYPRE_BigInt row, HYPRE_Int *size,
                                     HYPRE_BigInt **col_ind, HYPRE_Complex **values );
HYPRE_Int hypre_ParCSRMatrixRestoreRow ( hypre_ParCSRMatrix *matrix, HYPRE_BigInt row,
                                         HYPRE_Int *size, HYPRE_BigInt **col_ind, HYPRE_Complex **values );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix ( MPI_Comm comm, hypre_CSRMatrix *A,
                                                    HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts );
HYPRE_Int GenerateDiagAndOffd ( hypre_CSRMatrix *A, hypre_ParCSRMatrix *matrix,
                                HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag );
hypre_CSRMatrix *hypre_MergeDiagAndOffd ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_MergeDiagAndOffdDevice ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll_v2 ( hypre_ParCSRMatrix *par_matrix,
                                                       HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParCSRMatrixCopy ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                                   HYPRE_Int copy_data );
HYPRE_Int hypre_FillResponseParToCSRMatrix ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data );
#define hypre_ParCSRMatrixCompleteClone(A) hypre_ParCSRMatrixClone(A,0)
hypre_ParCSRMatrix* hypre_ParCSRMatrixClone_v2 ( hypre_ParCSRMatrix *A, HYPRE_Int copy_data,
                                                 HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParCSRMatrixTruncate(hypre_ParCSRMatrix *A, HYPRE_Real tol,
                                     HYPRE_Int max_row_elmts, HYPRE_Int rescale,
                                     HYPRE_Int nrm_type);
HYPRE_Int hypre_ParCSRMatrixMigrate(hypre_ParCSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParCSRMatrixSetConstantValues( hypre_ParCSRMatrix *A, HYPRE_Complex value );
void hypre_ParCSRMatrixCopyColMapOffdToDevice(hypre_ParCSRMatrix *A);
void hypre_ParCSRMatrixCopyColMapOffdToHost(hypre_ParCSRMatrix *A);

/* par_csr_matrix_stats.c */
HYPRE_Int hypre_ParCSRMatrixStatsArrayCompute( HYPRE_Int num_matrices,
                                               hypre_ParCSRMatrix **matrices,
                                               hypre_MatrixStatsArray *stats_array );

/* par_csr_matvec.c */
// y = alpha*A*x + beta*b
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlace ( HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                               hypre_ParVector *x, HYPRE_Complex beta,
                                               hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecOutOfPlaceDevice ( HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                                     hypre_ParVector *x, HYPRE_Complex beta,
                                                     hypre_ParVector *b, hypre_ParVector *y );
// y = alpha*A*x + beta*y
HYPRE_Int hypre_ParCSRMatrixMatvec ( HYPRE_Complex alpha, hypre_ParCSRMatrix *A, hypre_ParVector *x,
                                     HYPRE_Complex beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecT ( HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                      hypre_ParVector *x, HYPRE_Complex beta, hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecTDevice ( HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                            hypre_ParVector *x, HYPRE_Complex beta,
                                            hypre_ParVector *y );
HYPRE_Int hypre_ParCSRMatrixMatvecT_unpack( hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int num_cols,
                                            HYPRE_Complex *recv_data, HYPRE_Complex *local_data );
HYPRE_Int hypre_ParCSRMatrixMatvec_FF ( HYPRE_Complex alpha, hypre_ParCSRMatrix *A,
                                        hypre_ParVector *x, HYPRE_Complex beta, hypre_ParVector *y,
                                        HYPRE_Int *CF_marker, HYPRE_Int fpt );

/* par_csr_triplemat.c */
HYPRE_Int hypre_ParCSRTMatMatPartialAddDevice( hypre_ParCSRCommPkg *comm_pkg_A,
                                               HYPRE_Int num_cols_A, HYPRE_Int num_cols_B, HYPRE_BigInt first_col_diag_B,
                                               HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B,
                                               HYPRE_Int local_nnz_Cbar, hypre_CSRMatrix *Cbar, hypre_CSRMatrix *Cext,
                                               hypre_CSRMatrix **C_diag_ptr, hypre_CSRMatrix **C_offd_ptr, HYPRE_Int *num_cols_offd_C_ptr,
                                               HYPRE_BigInt **col_map_offd_C_ptr );
hypre_ParCSRMatrix *hypre_ParCSRMatMat( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatHost( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRMatMatDevice( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B );
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTHost( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                               HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKTDevice( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                                 HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMatKT( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B,
                                           HYPRE_Int keep_transpose);
hypre_ParCSRMatrix *hypre_ParCSRTMatMat( hypre_ParCSRMatrix  *A, hypre_ParCSRMatrix  *B);
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAPKT( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                             hypre_ParCSRMatrix  *P, HYPRE_Int keepTranspose );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRAP( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix  *A,
                                           hypre_ParCSRMatrix  *P );
hypre_ParCSRMatrix* hypre_ParCSRMatrixRAPKTDevice( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix *A,
                                                   hypre_ParCSRMatrix *P, HYPRE_Int keep_transpose );
hypre_ParCSRMatrix* hypre_ParCSRMatrixRAPKTHost( hypre_ParCSRMatrix *R, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keep_transpose );

/* par_make_system.c */
HYPRE_ParCSR_System_Problem *HYPRE_Generate2DSystem ( HYPRE_ParCSRMatrix H_L1,
                                                      HYPRE_ParCSRMatrix H_L2, HYPRE_ParVector H_b1, HYPRE_ParVector H_b2, HYPRE_ParVector H_x1,
                                                      HYPRE_ParVector H_x2, HYPRE_Complex *M_vals );
HYPRE_Int HYPRE_Destroy2DSystem ( HYPRE_ParCSR_System_Problem *sys_prob );

/* par_vector.c */
hypre_ParVector *hypre_ParVectorCreate ( MPI_Comm comm, HYPRE_BigInt global_size,
                                         HYPRE_BigInt *partitioning_in );
hypre_ParVector *hypre_ParMultiVectorCreate ( MPI_Comm comm, HYPRE_BigInt global_size,
                                              HYPRE_BigInt *partitioning, HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorDestroy ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorInitialize ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorInitialize_v2( hypre_ParVector *vector,
                                        HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParVectorSetDataOwner ( hypre_ParVector *vector, HYPRE_Int owns_data );
HYPRE_Int hypre_ParVectorSetLocalSize ( hypre_ParVector *vector, HYPRE_Int local_size );
HYPRE_Int hypre_ParVectorSetNumVectors ( hypre_ParVector *vector, HYPRE_Int num_vectors );
HYPRE_Int hypre_ParVectorSetComponent ( hypre_ParVector *vector, HYPRE_Int component );
HYPRE_Int hypre_ParVectorResize ( hypre_ParVector *vector, HYPRE_Int num_vectors );
hypre_ParVector *hypre_ParVectorRead ( MPI_Comm comm, const char *file_name );
HYPRE_Int hypre_ParVectorSetConstantValues ( hypre_ParVector *v, HYPRE_Complex value );
HYPRE_Int hypre_ParVectorSetZeros( hypre_ParVector *v );
HYPRE_Int hypre_ParVectorSetRandomValues ( hypre_ParVector *v, HYPRE_Int seed );
HYPRE_Int hypre_ParVectorCopy ( hypre_ParVector *x, hypre_ParVector *y );
hypre_ParVector *hypre_ParVectorCloneShallow ( hypre_ParVector *x );
hypre_ParVector *hypre_ParVectorCloneDeep_v2( hypre_ParVector *x,
                                              HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParVectorMigrate(hypre_ParVector *x, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_ParVectorScale ( HYPRE_Complex alpha, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorAxpy ( HYPRE_Complex alpha, hypre_ParVector *x, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorAxpyz ( HYPRE_Complex alpha, hypre_ParVector *x,
                                 HYPRE_Complex beta, hypre_ParVector *y,
                                 hypre_ParVector *z );
HYPRE_Int hypre_ParVectorMassAxpy ( HYPRE_Complex *alpha, hypre_ParVector **x, hypre_ParVector *y,
                                    HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Real hypre_ParVectorInnerProd ( hypre_ParVector *x, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorMassInnerProd ( hypre_ParVector *x, hypre_ParVector **y, HYPRE_Int k,
                                         HYPRE_Int unroll, HYPRE_Real *prod );
HYPRE_Int hypre_ParVectorMassDotpTwo ( hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector **z,
                                       HYPRE_Int k, HYPRE_Int unroll, HYPRE_Real *prod_x, HYPRE_Real *prod_y );
hypre_ParVector *hypre_VectorToParVector ( MPI_Comm comm, hypre_Vector *v,
                                           HYPRE_BigInt *vec_starts );
hypre_Vector *hypre_ParVectorToVectorAll ( hypre_ParVector *par_v );
hypre_Vector *hypre_ParVectorToVectorAll_v2 ( hypre_ParVector *par_v,
                                              HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_ParVectorPrint ( hypre_ParVector *vector, const char *file_name );
HYPRE_Int hypre_ParVectorPrintIJ ( hypre_ParVector *vector, HYPRE_Int base_j,
                                   const char *filename );
HYPRE_Int hypre_ParVectorPrintBinaryIJ ( hypre_ParVector *par_vector, const char *filename );
HYPRE_Int hypre_ParVectorReadIJ ( MPI_Comm comm, const char *filename, HYPRE_Int *base_j_ptr,
                                  hypre_ParVector **vector_ptr );
HYPRE_Int hypre_FillResponseParToVectorAll ( void *p_recv_contact_buf, HYPRE_Int contact_size,
                                             HYPRE_Int contact_proc, void *ro, MPI_Comm comm, void **p_send_response_buf,
                                             HYPRE_Int *response_message_size );
HYPRE_Complex hypre_ParVectorLocalSumElts ( hypre_ParVector *vector );
HYPRE_Int hypre_ParVectorGetValues ( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, HYPRE_Complex *values);
HYPRE_Int hypre_ParVectorGetValues2( hypre_ParVector *vector, HYPRE_Int num_values,
                                     HYPRE_BigInt *indices, HYPRE_BigInt base, HYPRE_Complex *values );
HYPRE_Int hypre_ParVectorGetValuesHost(hypre_ParVector *vector, HYPRE_Int num_values,
                                       HYPRE_BigInt *indices, HYPRE_BigInt base, HYPRE_Complex *values);
HYPRE_Int hypre_ParVectorElmdivpy( hypre_ParVector *x, hypre_ParVector *b, hypre_ParVector *y );
HYPRE_Int hypre_ParVectorElmdivpyMarked( hypre_ParVector *x, hypre_ParVector *b,
                                         hypre_ParVector *y, HYPRE_Int *marker,
                                         HYPRE_Int marker_val );
/* par_vector_device.c */
HYPRE_Int hypre_ParVectorGetValuesDevice(hypre_ParVector *vector, HYPRE_Int num_values,
                                         HYPRE_BigInt *indices, HYPRE_BigInt base,
                                         HYPRE_Complex *values);
