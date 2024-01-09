/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* dense_block_matrix.c */
hypre_DenseBlockMatrix* hypre_DenseBlockMatrixCreate(HYPRE_Int, HYPRE_Int, HYPRE_Int,
                                                     HYPRE_Int, HYPRE_Int);
hypre_DenseBlockMatrix* hypre_DenseBlockMatrixCreateByBlock(HYPRE_Int, HYPRE_Int,
                                                            HYPRE_Int, HYPRE_Int);
hypre_DenseBlockMatrix* hypre_DenseBlockMatrixClone(hypre_DenseBlockMatrix*, HYPRE_Int);
HYPRE_Int hypre_DenseBlockMatrixDestroy(hypre_DenseBlockMatrix*);
HYPRE_Int hypre_DenseBlockMatrixInitializeOn(hypre_DenseBlockMatrix*, HYPRE_MemoryLocation);
HYPRE_Int hypre_DenseBlockMatrixInitialize(hypre_DenseBlockMatrix*);
HYPRE_Int hypre_DenseBlockMatrixBuildAOP(hypre_DenseBlockMatrix*);
HYPRE_Int hypre_DenseBlockMatrixCopy(hypre_DenseBlockMatrix*, hypre_DenseBlockMatrix*);
HYPRE_Int hypre_DenseBlockMatrixMigrate(hypre_DenseBlockMatrix*, HYPRE_MemoryLocation);
HYPRE_Int hypre_DenseBlockMatrixPrint(MPI_Comm, hypre_DenseBlockMatrix*, const char*);

/* dense_block_matmult.c */
HYPRE_Int hypre_DenseBlockMatrixMultiply(hypre_DenseBlockMatrix*, hypre_DenseBlockMatrix*,
                                         hypre_DenseBlockMatrix**);
