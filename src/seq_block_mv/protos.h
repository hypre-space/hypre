/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* csr_block_matrix.c */
hypre_CSRBlockMatrix* hypre_CSRBlockMatrixCreate(HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixDestroy(hypre_CSRBlockMatrix*);
HYPRE_Int hypre_CSRBlockMatrixInitialize(hypre_CSRBlockMatrix*);
HYPRE_Int hypre_CSRBlockMatrixBigInitialize(hypre_CSRBlockMatrix*);
HYPRE_Int hypre_CSRBlockMatrixSetDataOwner(hypre_CSRBlockMatrix*, HYPRE_Int);
hypre_CSRMatrix* hypre_CSRBlockMatrixCompress(hypre_CSRBlockMatrix*);
hypre_CSRMatrix* hypre_CSRBlockMatrixConvertToCSRMatrix(hypre_CSRBlockMatrix*);
hypre_CSRBlockMatrix* hypre_CSRBlockMatrixConvertFromCSRMatrix(hypre_CSRMatrix*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAdd(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulate(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulateDiag(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(HYPRE_Complex*, HYPRE_Complex*,
                                                              HYPRE_Int, HYPRE_Real*);
HYPRE_Int hypre_CSRBlockMatrixComputeSign(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockSetScalar(HYPRE_Complex*, HYPRE_Complex, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockCopyData(HYPRE_Complex*, HYPRE_Complex*,
                                            HYPRE_Complex, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockCopyDataDiag(HYPRE_Complex*, HYPRE_Complex*,
                                                HYPRE_Complex, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixTranspose(hypre_CSRBlockMatrix*, hypre_CSRBlockMatrix**, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockNorm(HYPRE_Int, HYPRE_Complex*, HYPRE_Real*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAdd(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex,
                                           HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiag(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex,
                                               HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(HYPRE_Complex*, HYPRE_Complex*,
                                                        HYPRE_Complex, HYPRE_Complex*,
                                                        HYPRE_Int, HYPRE_Real*);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiag2(HYPRE_Complex*, HYPRE_Complex*,
                                                HYPRE_Complex, HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMultAddDiag3(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex,
                                                HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockInvMatvec(HYPRE_Complex*, HYPRE_Complex*,
                                             HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockInvMult(HYPRE_Complex*, HYPRE_Complex*,
                                           HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMultInv(HYPRE_Complex*, HYPRE_Complex*,
                                           HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockInvMultDiag(HYPRE_Complex*, HYPRE_Complex*,
                                               HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockInvMultDiag2(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex*,
                                                HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockInvMultDiag3(HYPRE_Complex*, HYPRE_Complex*, HYPRE_Complex*,
                                                HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockMatvec(HYPRE_Complex, HYPRE_Complex*, HYPRE_Complex*,
                                          HYPRE_Complex, HYPRE_Complex*, HYPRE_Int);
HYPRE_Int hypre_CSRBlockMatrixBlockTranspose(HYPRE_Complex *, HYPRE_Complex *, HYPRE_Int);

/* csr_block_matvec.c */
HYPRE_Int hypre_CSRBlockMatrixMatvec(HYPRE_Complex, hypre_CSRBlockMatrix*,
                                     hypre_Vector*, HYPRE_Complex, hypre_Vector*);
HYPRE_Int hypre_CSRBlockMatrixMatvecT(HYPRE_Complex, hypre_CSRBlockMatrix*, hypre_Vector*,
                                      HYPRE_Complex, hypre_Vector*);

/* csr_block_matop.c */
hypre_CSRBlockMatrix* hypre_CSRBlockMatrixAdd(hypre_CSRBlockMatrix*, hypre_CSRBlockMatrix*);
hypre_CSRBlockMatrix* hypre_CSRBlockMatrixMultiply(hypre_CSRBlockMatrix*, hypre_CSRBlockMatrix*);

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
