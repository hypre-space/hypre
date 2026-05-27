/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* dense_block_matrix.c */
hypre_DenseBlockMatrix *
hypre_DenseBlockMatrixCreate( HYPRE_Int  row_major,
                              HYPRE_Int  num_rows,
                              HYPRE_Int  num_cols,
                              HYPRE_Int  num_rows_block,
                              HYPRE_Int  num_cols_block );
hypre_DenseBlockMatrix *
hypre_DenseBlockMatrixCreateByBlock( HYPRE_Int  row_major,
                                     HYPRE_Int  num_blocks,
                                     HYPRE_Int  num_rows_block,
                                     HYPRE_Int  num_cols_block );
hypre_DenseBlockMatrix*
hypre_DenseBlockMatrixClone( hypre_DenseBlockMatrix *A,
                             HYPRE_Int               copy_data );
HYPRE_Int
hypre_DenseBlockMatrixDestroy( hypre_DenseBlockMatrix *A );
HYPRE_Int
hypre_DenseBlockMatrixInitializeOn( hypre_DenseBlockMatrix  *A,
                                    HYPRE_MemoryLocation     memory_location );
HYPRE_Int
hypre_DenseBlockMatrixInitialize( hypre_DenseBlockMatrix *A );
HYPRE_Int
hypre_DenseBlockMatrixBuildAOP( hypre_DenseBlockMatrix *A );
HYPRE_Int
hypre_DenseBlockMatrixCopy( hypre_DenseBlockMatrix *A,
                            hypre_DenseBlockMatrix *B );
HYPRE_Int
hypre_DenseBlockMatrixMigrate( hypre_DenseBlockMatrix *A,
                               HYPRE_MemoryLocation    memory_location );
HYPRE_Int
hypre_DenseBlockMatrixTranspose( hypre_DenseBlockMatrix  *A,
                                 hypre_DenseBlockMatrix **B_ptr );
HYPRE_Int
hypre_DenseBlockMatrixPrint( MPI_Comm                comm,
                             hypre_DenseBlockMatrix *A,
                             const char*             filename );

/* dense_block_matmult.c */
HYPRE_Int
hypre_DenseBlockMatrixMultiplyHost( hypre_DenseBlockMatrix  *A,
                                    hypre_DenseBlockMatrix  *B,
                                    hypre_DenseBlockMatrix  *C);
HYPRE_Int
hypre_DenseBlockMatrixMultiply( hypre_DenseBlockMatrix   *A,
                                hypre_DenseBlockMatrix   *B,
                                hypre_DenseBlockMatrix  **C_ptr);
