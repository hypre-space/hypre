/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for HYPRE_parcsr_mv library
 *
 *****************************************************************************/

#ifndef HYPRE_PARCSR_MV_HEADER
#define HYPRE_PARCSR_MV_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_seq_mv.h"

#ifdef HYPRE_MIXED_PRECISION
#include "_hypre_parcsr_mv_mup_def.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup ParCSRInterface ParCSR Object Interface
 *
 * Interface for ParCSR matrices and vectors. This is an interface for matrices
 * and vectors with object type HYPRE_PARCSR.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Matrices
 *
 * @{
 **/

struct hypre_ParCSRMatrix_struct;
/**
 * The matrix object. ParCSR matrices use a parallel compressed-sparse-row (CSR)
 * storage format that consists of a diagonal CSR matrix for intra-processor
 * couplings and an off-diagonal CSR matrix for inter-processor couplings.
 **/
typedef struct hypre_ParCSRMatrix_struct *HYPRE_ParCSRMatrix;

/**
 * Create a matrix object. More info here about arguments...
 **/
HYPRE_Int
HYPRE_ParCSRMatrixCreate(MPI_Comm            comm,
                         HYPRE_BigInt        global_num_rows,
                         HYPRE_BigInt        global_num_cols,
                         HYPRE_BigInt       *row_starts,
                         HYPRE_BigInt       *col_starts,
                         HYPRE_Int           num_cols_offd,
                         HYPRE_Int           num_nonzeros_diag,
                         HYPRE_Int           num_nonzeros_offd,
                         HYPRE_ParCSRMatrix *matrix);

/**
 * Destroy a matrix object.
 **/
HYPRE_Int
HYPRE_ParCSRMatrixDestroy(HYPRE_ParCSRMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.
 **/
HYPRE_Int
HYPRE_ParCSRMatrixInitialize(HYPRE_ParCSRMatrix matrix);

/**
 * Read a matrix from file.
 **/
HYPRE_Int
HYPRE_ParCSRMatrixRead(MPI_Comm            comm,
                       const char         *file_name,
                       HYPRE_ParCSRMatrix *matrix);

/**
 * Print a matrix to file.
 **/
HYPRE_Int
HYPRE_ParCSRMatrixPrint(HYPRE_ParCSRMatrix  matrix,
                        const char         *file_name);

/*===== BEGIN 1 - IGNORE CODE IN DOCS =====*/  /*! \cond */

HYPRE_Int
HYPRE_ParCSRMatrixGetComm(HYPRE_ParCSRMatrix  matrix,
                          MPI_Comm           *comm);

HYPRE_Int
HYPRE_ParCSRMatrixGetDims(HYPRE_ParCSRMatrix  matrix,
                          HYPRE_BigInt       *M,
                          HYPRE_BigInt       *N);

HYPRE_Int
HYPRE_ParCSRMatrixGetRowPartitioning(HYPRE_ParCSRMatrix   matrix,
                                     HYPRE_BigInt       **row_partitioning_ptr);

HYPRE_Int
HYPRE_ParCSRMatrixGetGlobalRowPartitioning (HYPRE_ParCSRMatrix   matrix,
                                            HYPRE_Int            all_procs,
                                            HYPRE_BigInt       **row_partitioning_ptr);

HYPRE_Int
HYPRE_ParCSRMatrixGetColPartitioning(HYPRE_ParCSRMatrix   matrix,
                                     HYPRE_BigInt       **col_partitioning_ptr);

HYPRE_Int
HYPRE_ParCSRMatrixGetLocalRange(HYPRE_ParCSRMatrix  matrix,
                                HYPRE_BigInt       *row_start,
                                HYPRE_BigInt       *row_end,
                                HYPRE_BigInt       *col_start,
                                HYPRE_BigInt       *col_end);

HYPRE_Int
HYPRE_ParCSRMatrixGetRow(HYPRE_ParCSRMatrix   matrix,
                         HYPRE_BigInt         row,
                         HYPRE_Int           *size,
                         HYPRE_BigInt       **col_ind,
                         HYPRE_Complex      **values);

HYPRE_Int
HYPRE_ParCSRMatrixRestoreRow(HYPRE_ParCSRMatrix  matrix,
                             HYPRE_BigInt        row,
                             HYPRE_Int          *size,
                             HYPRE_BigInt      **col_ind,
                             HYPRE_Complex     **values);

HYPRE_Int
HYPRE_CSRMatrixToParCSRMatrix(MPI_Comm            comm,
                              HYPRE_CSRMatrix     A_CSR,
                              HYPRE_BigInt       *row_partitioning,
                              HYPRE_BigInt       *col_partitioning,
                              HYPRE_ParCSRMatrix *matrix);

HYPRE_Int
HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning (MPI_Comm            comm,
                                                   HYPRE_CSRMatrix     A_CSR,
                                                   HYPRE_ParCSRMatrix *matrix);

/*===== END 1 - IGNORE CODE IN DOCS =====*/  /*! \endcond */

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Vectors
 *
 * @{
 **/

struct hypre_ParVector_struct;
/**
 * The vector object. A Par vector is an array storage format compatible with
 * ParCSR matrices.
 **/
typedef struct hypre_ParVector_struct *HYPRE_ParVector;

/**
 * Create a vector object.
 **/
HYPRE_Int
HYPRE_ParVectorCreate(MPI_Comm         comm,
                      HYPRE_BigInt     global_size,
                      HYPRE_BigInt    *partitioning,
                      HYPRE_ParVector *vector);

/**
 * Destroy a vector object.
 **/
HYPRE_Int
HYPRE_ParVectorDestroy(HYPRE_ParVector vector);


/**
 * Prepare a vector object for setting coefficient values.
 **/
HYPRE_Int
HYPRE_ParVectorInitialize(HYPRE_ParVector vector);

/**
 * Read a vector from file.
 **/
HYPRE_Int
HYPRE_ParVectorRead(MPI_Comm         comm,
                    const char      *file_name,
                    HYPRE_ParVector *vector);

/**
 * Print a vector to file.
 **/
HYPRE_Int
HYPRE_ParVectorPrint(HYPRE_ParVector  vector,
                     const char      *file_name);

/*===== BEGIN 2 - IGNORE CODE IN DOCS =====*/  /*! \cond */

HYPRE_Int
HYPRE_ParMultiVectorCreate (MPI_Comm         comm,
                            HYPRE_BigInt     global_size,
                            HYPRE_BigInt    *partitioning,
                            HYPRE_Int        number_vectors,
                            HYPRE_ParVector *vector);

HYPRE_Int
HYPRE_ParVectorPrintBinaryIJ(HYPRE_ParVector  vector,
                             const char      *file_name);

HYPRE_Int
HYPRE_ParVectorSetConstantValues(HYPRE_ParVector vector,
                                 HYPRE_Complex   value);

HYPRE_Int
HYPRE_ParVectorSetRandomValues(HYPRE_ParVector vector,
                               HYPRE_Int       seed);


HYPRE_ParVector
HYPRE_ParVectorCloneShallow(HYPRE_ParVector x);


/*===== END 2 - IGNORE CODE IN DOCS =====*/  /*! \endcond */

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Basic Matrix/vector routines
 *
 * @{
 **/

/**
 * Copy vector x into y (\f$y \leftarrow x\f$).
 **/
HYPRE_Int
HYPRE_ParVectorCopy(HYPRE_ParVector x,
                    HYPRE_ParVector y);

/**
 * Scale a vector by \e alpha (\f$y \leftarrow \alpha y\f$).
 **/
HYPRE_Int
HYPRE_ParVectorScale(HYPRE_Complex   value,
                     HYPRE_ParVector x);

/**
 * Compute \f$y = y + \alpha x\f$.
 **/
HYPRE_Int
HYPRE_ParVectorAxpy(HYPRE_Complex   alpha,
                    HYPRE_ParVector x,
                    HYPRE_ParVector y);

/**
 * Compute \e result, the inner product of vectors \e x and \e y.
 **/
HYPRE_Int
HYPRE_ParVectorInnerProd(HYPRE_ParVector  x,
                         HYPRE_ParVector  y,
                         HYPRE_Real      *result);

/**
 * Compute a matrix-vector product \f$y = \alpha A x + \beta y\f$.
 **/
HYPRE_Int
HYPRE_ParCSRMatrixMatvec(HYPRE_Complex      alpha,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    x,
                         HYPRE_Complex      beta,
                         HYPRE_ParVector    y);

/**
 * Compute a transpose matrix-vector product \f$y = \alpha A^T x + \beta y\f$.
 **/
HYPRE_Int
HYPRE_ParCSRMatrixMatvecT(HYPRE_Complex      alpha,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    x,
                          HYPRE_Complex      beta,
                          HYPRE_ParVector    y);

/**
 * Matrix-matrix multiply.
 **/
HYPRE_Int
HYPRE_ParCSRMatrixMatmat(HYPRE_ParCSRMatrix  A,
                         HYPRE_ParCSRMatrix  B,
                         HYPRE_ParCSRMatrix *C);

/*===== BEGIN 3 - IGNORE CODE IN DOCS =====*/  /*! \cond */

HYPRE_Int
HYPRE_ParCSRMatrixDiagScale(HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector    left,
                            HYPRE_ParVector    right);

HYPRE_Int
HYPRE_ParCSRMatrixComputeScalingTagged(HYPRE_ParCSRMatrix    A,
                                       HYPRE_Int             type,
                                       HYPRE_MemoryLocation  memloc_tags,
                                       HYPRE_Int             num_tags,
                                       HYPRE_Int            *tags,
                                       HYPRE_ParVector      *scaling_ptr);

HYPRE_Int
HYPRE_VectorToParVector(MPI_Comm         comm,
                        HYPRE_Vector     b,
                        HYPRE_BigInt    *partitioning,
                        HYPRE_ParVector *vector);

HYPRE_Int
HYPRE_ParVectorGetValues(HYPRE_ParVector vector,
                         HYPRE_Int       num_values,
                         HYPRE_BigInt   *indices,
                         HYPRE_Complex  *values);

/*===== END 3 - IGNORE CODE IN DOCS =====*/  /*! \endcond */

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#ifdef HYPRE_MIXED_PRECISION
/* The following is for user compiles and the order is important.  The first
 * header ensures that we do not change prototype names in user files or in the
 * second header file.  The second header contains all the prototypes needed by
 * users for mixed precision. */
#ifndef hypre_MP_BUILD
#include "_hypre_parcsr_mv_mup_undef.h"
#include "HYPRE_parcsr_mv_mup.h"
#include "HYPRE_parcsr_mv_mp.h"
#endif
#endif

#endif
