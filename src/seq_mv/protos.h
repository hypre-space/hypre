/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* csr_matop.c */
hypre_CSRMatrix *hypre_CSRMatrixAddHost ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixAdd ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixBigAdd ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyHost ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiply ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros ( hypre_CSRMatrix *A , HYPRE_Real tol );
HYPRE_Int hypre_CSRMatrixTransposeHost ( hypre_CSRMatrix *A , hypre_CSRMatrix **AT , HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixTranspose ( hypre_CSRMatrix *A , hypre_CSRMatrix **AT , HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixReorder ( hypre_CSRMatrix *A );
HYPRE_Complex hypre_CSRMatrixSumElts ( hypre_CSRMatrix *A );
HYPRE_Real hypre_CSRMatrixFnorm( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixSplit(hypre_CSRMatrix *Bs_ext, HYPRE_BigInt first_col_diag_B, HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B, HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr, hypre_CSRMatrix **Bext_diag_ptr, hypre_CSRMatrix **Bext_offd_ptr);
hypre_CSRMatrix * hypre_CSRMatrixAddPartial( hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int *row_nums);
void hypre_CSRMatrixComputeRowSum( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j, HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixExtractDiagonal( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
void hypre_CSRMatrixExtractDiagonalHost( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);

/* csr_matop_device.c */
#if defined(HYPRE_USING_CUDA)
hypre_CSRMatrix *hypre_CSRMatrixAddDevice ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiplyDevice ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixTripleMultiplyDevice ( hypre_CSRMatrix *A, hypre_CSRMatrix *B, hypre_CSRMatrix *C );
HYPRE_Int hypre_CSRMatrixSplitDevice_core( HYPRE_Int job, HYPRE_Int num_rows, HYPRE_Int B_ext_nnz, HYPRE_Int *B_ext_ii, HYPRE_BigInt *B_ext_bigj, HYPRE_Complex *B_ext_data, char *B_ext_xata, HYPRE_BigInt first_col_diag_B, HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B, HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr, HYPRE_Int *B_ext_diag_nnz_ptr, HYPRE_Int *B_ext_diag_ii, HYPRE_Int *B_ext_diag_j, HYPRE_Complex *B_ext_diag_data, char *B_ext_diag_xata, HYPRE_Int *B_ext_offd_nnz_ptr, HYPRE_Int *B_ext_offd_ii, HYPRE_Int *B_ext_offd_j, HYPRE_Complex *B_ext_offd_data, char *B_ext_offd_xata );
HYPRE_Int hypre_CSRMatrixSplitDevice(hypre_CSRMatrix *B_ext, HYPRE_BigInt first_col_diag_B, HYPRE_BigInt last_col_diag_B, HYPRE_Int num_cols_offd_B, HYPRE_BigInt *col_map_offd_B, HYPRE_Int **map_B_to_C_ptr, HYPRE_Int *num_cols_offd_C_ptr, HYPRE_BigInt **col_map_offd_C_ptr, hypre_CSRMatrix **B_ext_diag_ptr, hypre_CSRMatrix **B_ext_offd_ptr);
HYPRE_Int hypre_CSRMatrixTransposeDevice ( hypre_CSRMatrix *A , hypre_CSRMatrix **AT , HYPRE_Int data );
hypre_CSRMatrix* hypre_CSRMatrixAddPartialDevice( hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int *row_nums);
HYPRE_Int hypre_CSRMatrixColNNzRealDevice( hypre_CSRMatrix *A, HYPRE_Real *colnnz);
HYPRE_Int hypre_CSRMatrixMoveDiagFirstDevice( hypre_CSRMatrix  *A );
HYPRE_Int hypre_CSRMatrixCheckDiagFirstDevice( hypre_CSRMatrix  *A );
void hypre_CSRMatrixComputeRowSumDevice( hypre_CSRMatrix *A, HYPRE_Int *CF_i, HYPRE_Int *CF_j, HYPRE_Complex *row_sum, HYPRE_Int type, HYPRE_Complex scal, const char *set_or_add);
void hypre_CSRMatrixExtractDiagonalDevice( hypre_CSRMatrix *A, HYPRE_Complex *d, HYPRE_Int type);
hypre_CSRMatrix* hypre_CSRMatrixStack2Device(hypre_CSRMatrix *A, hypre_CSRMatrix *B);
#endif

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CSRMatrixCreate ( HYPRE_Int num_rows , HYPRE_Int num_cols , HYPRE_Int num_nonzeros );
HYPRE_Int hypre_CSRMatrixDestroy ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixInitialize_v2( hypre_CSRMatrix *matrix, HYPRE_Int bigInit, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_CSRMatrixInitialize ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigInitialize ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixBigJtoJ ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixJtoBigJ ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixSetDataOwner ( hypre_CSRMatrix *matrix , HYPRE_Int owns_data );
HYPRE_Int hypre_CSRMatrixSetRownnz ( hypre_CSRMatrix *matrix );
hypre_CSRMatrix *hypre_CSRMatrixRead ( char *file_name );
HYPRE_Int hypre_CSRMatrixPrint ( hypre_CSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_CSRMatrixPrint2( hypre_CSRMatrix *matrix, const char *file_name );
HYPRE_Int hypre_CSRMatrixPrintHB ( hypre_CSRMatrix *matrix_input , char *file_name );
HYPRE_Int hypre_CSRMatrixPrintMM( hypre_CSRMatrix *matrix, HYPRE_Int basei, HYPRE_Int basej, HYPRE_Int trans, const char *file_name );
HYPRE_Int hypre_CSRMatrixCopy ( hypre_CSRMatrix *A , hypre_CSRMatrix *B , HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone ( hypre_CSRMatrix *A, HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone_v2( hypre_CSRMatrix *A, HYPRE_Int copy_data, HYPRE_MemoryLocation memory_location );
hypre_CSRMatrix *hypre_CSRMatrixUnion ( hypre_CSRMatrix *A , hypre_CSRMatrix *B , HYPRE_BigInt *col_map_offd_A , HYPRE_BigInt *col_map_offd_B , HYPRE_BigInt **col_map_offd_C );
HYPRE_Int hypre_CSRMatrixPrefetch( hypre_CSRMatrix *A, HYPRE_MemoryLocation memory_location);
HYPRE_Int hypre_CSRMatrixCheckSetNumNonzeros( hypre_CSRMatrix *matrix );

/* csr_matvec.c */
// y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end]
HYPRE_Int hypre_CSRMatrixMatvecOutOfPlace ( HYPRE_Complex alpha , hypre_CSRMatrix *A , hypre_Vector *x , HYPRE_Complex beta , hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecOutOfPlaceOOMP (HYPRE_Int trans, HYPRE_Complex alpha , hypre_CSRMatrix *A , hypre_Vector *x , HYPRE_Complex beta , hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
// y = alpha*A + beta*y
HYPRE_Int hypre_CSRMatrixMatvec ( HYPRE_Complex alpha , hypre_CSRMatrix *A , hypre_Vector *x , HYPRE_Complex beta , hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvecT ( HYPRE_Complex alpha , hypre_CSRMatrix *A , hypre_Vector *x , HYPRE_Complex beta , hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvec_FF ( HYPRE_Complex alpha , hypre_CSRMatrix *A , hypre_Vector *x , HYPRE_Complex beta , hypre_Vector *y , HYPRE_Int *CF_marker_x , HYPRE_Int *CF_marker_y , HYPRE_Int fpt );
#ifdef HYPRE_USING_CUDA
HYPRE_Int hypre_CSRMatrixMatvecDevice(HYPRE_Int trans, HYPRE_Complex alpha , hypre_CSRMatrix *A , hypre_Vector *x , HYPRE_Complex beta , hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
HYPRE_Int hypre_CSRMatrixMatvecDeviceBIGINT( HYPRE_Complex alpha , hypre_CSRMatrix *A , hypre_Vector *x , HYPRE_Complex beta , hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset );
#endif
/* genpart.c */
HYPRE_Int hypre_GeneratePartitioning ( HYPRE_BigInt length , HYPRE_Int num_procs , HYPRE_BigInt **part_ptr );
HYPRE_Int hypre_GenerateLocalPartitioning ( HYPRE_BigInt length , HYPRE_Int num_procs , HYPRE_Int myid , HYPRE_BigInt **part_ptr );

/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate ( HYPRE_Int num_rows , HYPRE_Int num_cols , HYPRE_Int *row_sizes );
HYPRE_Int HYPRE_CSRMatrixDestroy ( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixInitialize ( HYPRE_CSRMatrix matrix );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead ( char *file_name );
void HYPRE_CSRMatrixPrint ( HYPRE_CSRMatrix matrix , char *file_name );
HYPRE_Int HYPRE_CSRMatrixGetNumRows ( HYPRE_CSRMatrix matrix , HYPRE_Int *num_rows );

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate ( void );
HYPRE_Int HYPRE_MappedMatrixDestroy ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixLimitedDestroy ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixInitialize ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixAssemble ( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixGetColIndex ( HYPRE_MappedMatrix matrix , HYPRE_Int j );
void *HYPRE_MappedMatrixGetMatrix ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixSetMatrix ( HYPRE_MappedMatrix matrix , void *matrix_data );
HYPRE_Int HYPRE_MappedMatrixSetColMap ( HYPRE_MappedMatrix matrix , HYPRE_Int (*ColMap )(HYPRE_Int ,void *));
HYPRE_Int HYPRE_MappedMatrixSetMapData ( HYPRE_MappedMatrix matrix , void *MapData );

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate ( void );
HYPRE_Int HYPRE_MultiblockMatrixDestroy ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixLimitedDestroy ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixInitialize ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixAssemble ( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixSetNumSubmatrices ( HYPRE_MultiblockMatrix matrix , HYPRE_Int n );
HYPRE_Int HYPRE_MultiblockMatrixSetSubmatrixType ( HYPRE_MultiblockMatrix matrix , HYPRE_Int j , HYPRE_Int type );

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate ( HYPRE_Int size );
HYPRE_Int HYPRE_VectorDestroy ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorInitialize ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorPrint ( HYPRE_Vector vector , char *file_name );
HYPRE_Vector HYPRE_VectorRead ( char *file_name );

/* mapped_matrix.c */
hypre_MappedMatrix *hypre_MappedMatrixCreate ( void );
HYPRE_Int hypre_MappedMatrixDestroy ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixLimitedDestroy ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixInitialize ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixAssemble ( hypre_MappedMatrix *matrix );
void hypre_MappedMatrixPrint ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixGetColIndex ( hypre_MappedMatrix *matrix , HYPRE_Int j );
void *hypre_MappedMatrixGetMatrix ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixSetMatrix ( hypre_MappedMatrix *matrix , void *matrix_data );
HYPRE_Int hypre_MappedMatrixSetColMap ( hypre_MappedMatrix *matrix , HYPRE_Int (*ColMap )(HYPRE_Int ,void *));
HYPRE_Int hypre_MappedMatrixSetMapData ( hypre_MappedMatrix *matrix , void *map_data );

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate ( void );
HYPRE_Int hypre_MultiblockMatrixDestroy ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixLimitedDestroy ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixInitialize ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixAssemble ( hypre_MultiblockMatrix *matrix );
void hypre_MultiblockMatrixPrint ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixSetNumSubmatrices ( hypre_MultiblockMatrix *matrix , HYPRE_Int n );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrixType ( hypre_MultiblockMatrix *matrix , HYPRE_Int j , HYPRE_Int type );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrix ( hypre_MultiblockMatrix *matrix , HYPRE_Int j , void *submatrix );

/* vector.c */
hypre_Vector *hypre_SeqVectorCreate ( HYPRE_Int size );
hypre_Vector *hypre_SeqMultiVectorCreate ( HYPRE_Int size , HYPRE_Int num_vectors );
HYPRE_Int hypre_SeqVectorDestroy ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorInitialize_v2( hypre_Vector *vector, HYPRE_MemoryLocation memory_location );
HYPRE_Int hypre_SeqVectorInitialize ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorSetDataOwner ( hypre_Vector *vector , HYPRE_Int owns_data );
hypre_Vector *hypre_SeqVectorRead ( char *file_name );
HYPRE_Int hypre_SeqVectorPrint ( hypre_Vector *vector , char *file_name );
HYPRE_Int hypre_SeqVectorSetConstantValues ( hypre_Vector *v , HYPRE_Complex value );
HYPRE_Int hypre_SeqVectorSetRandomValues ( hypre_Vector *v , HYPRE_Int seed );
HYPRE_Int hypre_SeqVectorCopy ( hypre_Vector *x , hypre_Vector *y );
hypre_Vector *hypre_SeqVectorCloneDeep ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneDeep_v2( hypre_Vector *x, HYPRE_MemoryLocation memory_location );
hypre_Vector *hypre_SeqVectorCloneShallow ( hypre_Vector *x );
HYPRE_Int hypre_SeqVectorScale( HYPRE_Complex alpha, hypre_Vector *y );

HYPRE_Int hypre_SeqVectorAxpy ( HYPRE_Complex alpha , hypre_Vector *x , hypre_Vector *y );
HYPRE_Real hypre_SeqVectorInnerProd ( hypre_Vector *x , hypre_Vector *y );
HYPRE_Int hypre_SeqVectorMassInnerProd(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k, HYPRE_Int unroll, HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd4(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,  HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassInnerProd8(hypre_Vector *x, hypre_Vector **y, HYPRE_Int k,  HYPRE_Real *result);
HYPRE_Int hypre_SeqVectorMassDotpTwo(hypre_Vector *x, hypre_Vector *y , hypre_Vector **z, HYPRE_Int k, HYPRE_Int unroll,  HYPRE_Real *result_x , HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo4(hypre_Vector *x, hypre_Vector *y , hypre_Vector **z, HYPRE_Int k, HYPRE_Real *result_x , HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassDotpTwo8(hypre_Vector *x, hypre_Vector *y , hypre_Vector **z, HYPRE_Int k,  HYPRE_Real *result_x , HYPRE_Real *result_y);
HYPRE_Int hypre_SeqVectorMassAxpy(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y, HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_SeqVectorMassAxpy4(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y, HYPRE_Int k);
HYPRE_Int hypre_SeqVectorMassAxpy8(HYPRE_Complex *alpha, hypre_Vector **x, hypre_Vector *y, HYPRE_Int k);
HYPRE_Complex hypre_SeqVectorSumElts ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorPrefetch(hypre_Vector *x, HYPRE_MemoryLocation memory_location);
//HYPRE_Int hypre_SeqVectorMax( HYPRE_Complex alpha, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y );

//HYPRE_Int hypre_CSRMatrixMatvecOutOfPlaceOOMP3( HYPRE_Complex alpha, hypre_CSRMatrix *A, hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *b, hypre_Vector *y, HYPRE_Int offset);

#if defined(HYPRE_USING_CUDA)

HYPRE_Int hypreDevice_CSRSpAdd(HYPRE_Int ma, HYPRE_Int mb, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int nnzB, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_ab, HYPRE_Int *d_num_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out);

HYPRE_Int hypreDevice_CSRSpTrans(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out, HYPRE_Int want_data);

HYPRE_Int hypreDevice_CSRSpGemm(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int nnza, HYPRE_Int nnzb, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

#endif
