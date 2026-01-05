/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for overlap matrix extraction
 *
 * This test file contains:
 * 1. Unit tests for overlap extraction with small hard-coded matrices
 * 2. TODO - Benchmarking with generated laplacian matrices
 *--------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Macro to print test result (PASSED/FAILED) on rank 0
 *--------------------------------------------------------------------------*/
#define PRINT_TEST_RESULT(my_id, error) \
   do { \
      if ((my_id) == 0) \
      { \
         if ((error) == 0) \
         { \
            hypre_printf("PASSED\n"); \
         } \
         else \
         { \
            hypre_printf("FAILED\n"); \
         } \
      } \
   } while (0)

/*--------------------------------------------------------------------------
 * Helper function to compare two CSR matrices
 * Returns 0 if matrices are equal (within tolerance), 1 otherwise
 * Uses Frobenius norm of the difference matrix
 *--------------------------------------------------------------------------*/
static HYPRE_Int
CompareCSRMatrices(hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Real tol)
{
   HYPRE_Int num_rows_A = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols_A = hypre_CSRMatrixNumCols(A);
   HYPRE_Int num_rows_B = hypre_CSRMatrixNumRows(B);
   HYPRE_Int num_cols_B = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix *diff;
   HYPRE_Real fnorm_diff, fnorm_A;
   HYPRE_Real rel_error;

   /* Check dimensions */
   if (num_rows_A != num_rows_B || num_cols_A != num_cols_B)
   {
      hypre_printf("Matrix dimensions mismatch: (%d x %d) vs (%d x %d)\n",
                   num_rows_A, num_cols_A, num_rows_B, num_cols_B);
      return 1;
   }

   /* Compute difference matrix: diff = A - B */
   diff = hypre_CSRMatrixAdd(1.0, A, -1.0, B);
   if (!diff)
   {
      hypre_printf("Failed to compute difference matrix\n");
      return 1;
   }

   /* Compute Frobenius norm of difference */
   fnorm_diff = hypre_CSRMatrixFnorm(diff);
   fnorm_A = hypre_CSRMatrixFnorm(A);

   /* Compute relative error */
   rel_error = (fnorm_A > 0.0) ? (fnorm_diff / fnorm_A) : fnorm_diff;

   /* Cleanup */
   hypre_CSRMatrixDestroy(diff);

   /* Check if relative error is within tolerance */
   if (rel_error > tol)
   {
      hypre_printf("Matrix comparison failed: ||A - B||_F / ||A||_F = %e (tolerance = %e)\n",
                   rel_error, tol);
      return 1;
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Create a CSR matrix from hard-coded values
 * num_rows: number of rows
 * num_cols: number of columns
 * I: row pointer array (length num_rows+1)
 * J: column index array (length nnz)
 * data: value array (length nnz)
 *--------------------------------------------------------------------------*/
static hypre_CSRMatrix*
CreateCSRMatrixFromData(HYPRE_Int num_rows, HYPRE_Int num_cols, HYPRE_Int nnz,
                        HYPRE_Int *I, HYPRE_Int *J, HYPRE_Real *data)
{
   hypre_CSRMatrix *matrix;
   HYPRE_Int i;

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, nnz);
   hypre_CSRMatrixInitialize(matrix);

   /* Copy I array */
   for (i = 0; i <= num_rows; i++)
   {
      hypre_CSRMatrixI(matrix)[i] = I[i];
   }

   /* Copy J and data arrays */
   for (i = 0; i < nnz; i++)
   {
      hypre_CSRMatrixJ(matrix)[i] = J[i];
      hypre_CSRMatrixData(matrix)[i] = data[i];
   }

   hypre_CSRMatrixNumNonzeros(matrix) = nnz;

   return matrix;
}

/*--------------------------------------------------------------------------
 * Create a simple 1D Laplacian matrix using GenerateLaplacian
 * Global size: n, partitioned across num_procs processors (1D partitioning)
 *--------------------------------------------------------------------------*/
static hypre_ParCSRMatrix*
Create1DLaplacian(MPI_Comm comm, HYPRE_Int n, HYPRE_Int num_procs, HYPRE_Int my_id)
{
   HYPRE_Real *values;
   HYPRE_Int p, q, r;
   HYPRE_Int P = num_procs, Q = 1, R = 1;

   /* Compute processor coordinates for 1D partitioning */
   p = my_id % P;
   q = ((my_id - p) / P) % Q;
   r = (my_id - p - P * q) / (P * Q);

   /* Set up 1D stencil values (tridiagonal) */
   values = hypre_CTAlloc(HYPRE_Real, 4, HYPRE_MEMORY_HOST);
   values[0] = 2.0;  /* diagonal */
   values[1] = -1.0; /* x-direction */
   values[2] = 0.0;  /* y-direction (not used) */
   values[3] = 0.0;  /* z-direction (not used) */

   return GenerateLaplacian(comm, n, 1, 1, P, Q, R, p, q, r, values);
}

/*--------------------------------------------------------------------------
 * Unit test: 1D grid with 1D partitioning, overlap=1
 * Test case: 4x4 matrix on 2 processors
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test1_Grid1D_Part1D_Overlap1(MPI_Comm comm, HYPRE_Int print_matrices)
{
   hypre_ParCSRMatrix *A;
   hypre_OverlapData *overlap_data;
   hypre_CSRMatrix *A_local;
   HYPRE_BigInt *col_map;
   HYPRE_Int num_cols_local;
   HYPRE_Int error = 0;
   HYPRE_Int overlap_order = 1;
   HYPRE_Int my_id, num_procs;
   MPI_Comm test_comm = MPI_COMM_NULL;
   HYPRE_Int participate = 0;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   /* Create subcommunicator with first 2 processors */
   if (num_procs >= 2)
   {
      participate = (my_id < 2) ? 1 : hypre_MPI_UNDEFINED;
      hypre_MPI_Comm_split(comm, participate, my_id, &test_comm);
   }
   else
   {
      if (my_id == 0)
      {
         hypre_printf("Test1_Grid1D_Part1D_Overlap1: Skipping (requires at least 2 processors)\n");
      }
      hypre_MPI_Barrier(comm);
      return 0;
   }

   /* Only participating processes run the test */
   if (test_comm == MPI_COMM_NULL)
   {
      /* Non-participating processes must wait for test to complete */
      hypre_MPI_Barrier(comm);
      return 0;
   }

   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      if (test_my_id == 0)
      {
         hypre_printf("Test1_Grid1D_Part1D_Overlap1 (2 procs): ");
      }
   }

   /* Create 4x4 1D Laplacian */
   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      HYPRE_Real *values = hypre_CTAlloc(HYPRE_Real, 4, HYPRE_MEMORY_HOST);
      values[0] = 2.0;
      values[1] = -1.0;
      values[2] = 0.0;
      values[3] = 0.0;
      A = Create1DLaplacian(test_comm, 4, 2, test_my_id);
      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }

   /* Compute overlap */
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }
   hypre_ParCSRMatrixComputeOverlap(A, overlap_order, &overlap_data);

   /* Get overlap rows */
   hypre_ParCSRMatrixGetOverlapRows(A, overlap_data);

   /* Extract local overlap matrix */
   hypre_ParCSRMatrixExtractLocalOverlap(A, overlap_data, &A_local, &col_map, &num_cols_local);

   /* Create expected matrix and compare */
   {
      hypre_CSRMatrix *A_expected = NULL;
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);

      if (test_my_id == 0)
      {
         /* Proc 0: Extended domain is rows 0,1,2 with columns 0,1,2
          * Global matrix row 0: [2, -1] at columns [0,1] -> [2, -1] at local cols [0,1]
          * Global matrix row 1: [-1, 2, -1] at columns [0,1,2] -> [-1, 2, -1] at local cols [0,1,2]
          * Global matrix row 2: [-1, 2] at columns [1,2] -> [-1, 2] at local cols [1,2]
          * Expected 3x3 matrix:
          */
         HYPRE_Int I_expected[4] = {0, 2, 5, 7};
         HYPRE_Int J_expected[7] = {0, 1,       /* row 0 (global 0): columns 0,1 (global 0,1) */
                                    0, 1, 2,    /* row 1 (global 1): columns 0,1,2 (global 0,1,2) */
                                    1, 2        /* row 2 (global 2): columns 1,2 (global 1,2) */
                                   };
         HYPRE_Real data_expected[7] = {2.0, -1.0,        /* row 0 */
                                        -1.0, 2.0, -1.0,  /* row 1 */
                                        -1.0, 2.0         /* row 2 */
                                       };

         A_expected = CreateCSRMatrixFromData(3, 3, 7, I_expected, J_expected, data_expected);
      }
      else if (test_my_id == 1)
      {
         /* Proc 1: Extended domain is rows 1,2,3 with columns 1,2,3
          * Expected 3x3 matrix:
          */
         HYPRE_Int I_expected[4] = {0, 2, 5, 7};
         HYPRE_Int J_expected[7] = {0, 1,       /* row 0 (global 1): columns 0,1 (global 1,2) */
                                    0, 1, 2,    /* row 1 (global 2): columns 0,1,2 (global 1,2,3) */
                                    1, 2        /* row 2 (global 3): columns 1,2 (global 2,3) */
                                   };
         HYPRE_Real data_expected[7] = {2.0, -1.0,        /* row 0 */
                                        -1.0, 2.0, -1.0,  /* row 1 */
                                        -1.0, 2.0         /* row 2 */
                                       };

         A_expected = CreateCSRMatrixFromData(3, 3, 7, I_expected, J_expected, data_expected);
      }

      if (A_expected)
      {
         HYPRE_Real tol = 1e-10;
         if (print_matrices)
         {
            HYPRE_Int test_my_id;
            hypre_MPI_Comm_rank(test_comm, &test_my_id);
            char filename_expected[256];
            char filename_computed[256];
            hypre_sprintf(filename_expected, "test1_expected_ij.out.%05d", test_my_id);
            hypre_sprintf(filename_computed, "test1_computed_ij.out.%05d", test_my_id);
            hypre_CSRMatrixPrintIJ(A_expected, 0, 0, filename_expected);
            hypre_CSRMatrixPrintIJ(A_local, 0, 0, filename_computed);
         }
         if (CompareCSRMatrices(A_expected, A_local, tol) != 0)
         {
            HYPRE_Int test_my_id;
            hypre_MPI_Comm_rank(test_comm, &test_my_id);
            hypre_printf("Proc %d: Extracted matrix does not match expected matrix\n", test_my_id);
            error = 1;
         }
         hypre_CSRMatrixDestroy(A_expected);
      }
   }

   /* Cleanup */
   if (A_local)
   {
      hypre_CSRMatrixDestroy(A_local);
   }
   if (col_map)
   {
      hypre_TFree(col_map, HYPRE_MEMORY_HOST);
   }
   if (overlap_data)
   {
      hypre_OverlapDataDestroy(overlap_data);
   }
   if (A)
   {
      hypre_ParCSRMatrixDestroy(A);
   }

   if (test_comm != MPI_COMM_NULL)
   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      PRINT_TEST_RESULT(test_my_id, error);
      hypre_MPI_Comm_free(&test_comm);
   }

   /* Synchronize all processes before returning */
   hypre_MPI_Barrier(comm);

   return error;
}

/*--------------------------------------------------------------------------
 * Unit test: 1D grid with 1D partitioning, overlap=2
 * Test case: 6x6 matrix on 3 processors
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test2_Grid1D_Part1D_Overlap2(MPI_Comm comm, HYPRE_Int print_matrices)
{
   hypre_ParCSRMatrix *A;
   hypre_OverlapData *overlap_data;
   hypre_CSRMatrix *A_local;
   HYPRE_BigInt *col_map;
   HYPRE_Int num_cols_local;
   HYPRE_Int error = 0;
   HYPRE_Int overlap_order = 2;
   HYPRE_Int my_id, num_procs;
   MPI_Comm test_comm = MPI_COMM_NULL;
   HYPRE_Int participate = 0;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   /* Create subcommunicator with first 3 processors */
   if (num_procs >= 3)
   {
      participate = (my_id < 3) ? 1 : hypre_MPI_UNDEFINED;
      hypre_MPI_Comm_split(comm, participate, my_id, &test_comm);
   }
   else
   {
      if (my_id == 0)
      {
         hypre_printf("Test2_Grid1D_Part1D_Overlap2: Skipping (requires at least 3 processors)\n");
      }
      return 0;
   }

   /* Only participating processes run the test */
   if (test_comm == MPI_COMM_NULL)
   {
      /* Non-participating processes must still synchronize */
      hypre_MPI_Barrier(comm);
      return 0;
   }

   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      if (test_my_id == 0)
      {
         hypre_printf("Test2_Grid1D_Part1D_Overlap2 (3 procs): ");
      }
   }

   /* Create 6x6 1D Laplacian */
   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      HYPRE_Real *values = hypre_CTAlloc(HYPRE_Real, 4, HYPRE_MEMORY_HOST);
      values[0] = 2.0;
      values[1] = -1.0;
      values[2] = 0.0;
      values[3] = 0.0;
      A = Create1DLaplacian(test_comm, 6, 3, test_my_id);
      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }

   /* Compute overlap */
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }
   hypre_ParCSRMatrixComputeOverlap(A, overlap_order, &overlap_data);

   /* Get overlap rows */
   hypre_ParCSRMatrixGetOverlapRows(A, overlap_data);

   /* Extract local overlap matrix */
   hypre_ParCSRMatrixExtractLocalOverlap(A, overlap_data, &A_local, &col_map, &num_cols_local);

   /* Create expected matrix and compare */
   {
      hypre_CSRMatrix *A_expected = NULL;
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);

      if (test_my_id == 0)
      {
         /* Proc 0: Owns rows 0,1. With overlap=2:
          * - 1st hop: gets row 2 (from proc 1)
          * - 2nd hop: gets row 3 (via row 2 from proc 1)
          * Extended domain: rows 0,1,2,3 with columns 0,1,2,3.
          * Expected 4x4 matrix:
          */
         HYPRE_Int I_expected[5] = {0, 2, 5, 8, 10};
         HYPRE_Int J_expected[10] = {0, 1,       /* row 0 (global 0): columns 0,1 */
                                     0, 1, 2,    /* row 1 (global 1): columns 0,1,2 */
                                     1, 2, 3,    /* row 2 (global 2): columns 1,2,3 */
                                     2, 3        /* row 3 (global 3): columns 2,3 */
                                    };
         HYPRE_Real data_expected[10] = {2.0, -1.0,        /* row 0 */
                                         -1.0, 2.0, -1.0,  /* row 1 */
                                         -1.0, 2.0, -1.0,  /* row 2 */
                                         -1.0, 2.0         /* row 3 */
                                        };

         A_expected = CreateCSRMatrixFromData(4, 4, 10, I_expected, J_expected, data_expected);
      }
      else if (test_my_id == 1)
      {
         /* Proc 1: Owns rows 2,3. With overlap=2, should get rows 0,1 (from proc 0) and rows 4,5 (from proc 2)
          * Extended domain (full): rows 0,1,2,3,4,5 with columns 0,1,2,3,4,5.
          * Expected 6x6 matrix:
          */
         HYPRE_Int I_expected[7] = {0, 2, 5, 8, 11, 14, 16};
         HYPRE_Int J_expected[16] = {0, 1,          /* row 0 (global 0): columns 0,1 */
                                     0, 1, 2,       /* row 1 (global 1): columns 0,1,2 */
                                     1, 2, 3,       /* row 2 (global 2): columns 1,2,3 */
                                     2, 3, 4,       /* row 3 (global 3): columns 2,3,4 */
                                     3, 4, 5,       /* row 4 (global 4): columns 3,4,5 */
                                     4, 5           /* row 5 (global 5): columns 4,5 */
                                    };
         HYPRE_Real data_expected[16] = {2.0, -1.0,           /* row 0 */
                                         -1.0, 2.0, -1.0,     /* row 1 */
                                         -1.0, 2.0, -1.0,     /* row 2 */
                                         -1.0, 2.0, -1.0,     /* row 3 */
                                         -1.0, 2.0, -1.0,     /* row 4 */
                                         -1.0, 2.0            /* row 5 */
                                        };

         A_expected = CreateCSRMatrixFromData(6, 6, 16, I_expected, J_expected, data_expected);
      }
      else if (test_my_id == 2)
      {
         /* Proc 2: Owns rows 4,5. With overlap=2:
          * - 1st hop: gets row 3 (from proc 1)
          * - 2nd hop: gets row 2 (via row 3 from proc 1)
          * Extended domain: rows 2,3,4,5 with columns 2,3,4,5.
          * Expected 4x4 matrix:
          */
         HYPRE_Int I_expected[5] = {0, 2, 5, 8, 10};
         HYPRE_Int J_expected[10] = {0, 1,       /* row 0 (global 2): columns 0,1 (global 2,3) */
                                     0, 1, 2,    /* row 1 (global 3): columns 0,1,2 (global 2,3,4) */
                                     1, 2, 3,    /* row 2 (global 4): columns 1,2,3 (global 3,4,5) */
                                     2, 3        /* row 3 (global 5): columns 2,3 (global 4,5) */
                                    };
         HYPRE_Real data_expected[10] = {2.0, -1.0,        /* row 0 */
                                         -1.0, 2.0, -1.0,  /* row 1 */
                                         -1.0, 2.0, -1.0,  /* row 2 */
                                         -1.0, 2.0         /* row 3 */
                                        };

         A_expected = CreateCSRMatrixFromData(4, 4, 10, I_expected, J_expected, data_expected);
      }

      if (A_expected)
      {
         HYPRE_Real tol = 1e-10;
         if (CompareCSRMatrices(A_expected, A_local, tol) != 0)
         {
            hypre_printf("Proc %d: Extracted matrix does not match expected matrix\n", test_my_id);
            error = 1;
         }
         hypre_CSRMatrixDestroy(A_expected);
      }
      if (print_matrices)
      {
         char filename_expected[256];
         char filename_computed[256];
         hypre_sprintf(filename_expected, "test2_expected_ij.out.%05d", test_my_id);
         hypre_sprintf(filename_computed, "test2_computed_ij.out.%05d", test_my_id);
         hypre_CSRMatrixPrintIJ(A_expected, 0, 0, filename_expected);
         hypre_CSRMatrixPrintIJ(A_local, 0, 0, filename_computed);
      }
   }

   /* Cleanup */
   if (A_local)
   {
      hypre_CSRMatrixDestroy(A_local);
   }
   if (col_map)
   {
      hypre_TFree(col_map, HYPRE_MEMORY_HOST);
   }
   if (overlap_data)
   {
      hypre_OverlapDataDestroy(overlap_data);
   }
   if (A)
   {
      hypre_ParCSRMatrixDestroy(A);
   }

   if (test_comm != MPI_COMM_NULL)
   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      PRINT_TEST_RESULT(test_my_id, error);
      hypre_MPI_Comm_free(&test_comm);
   }

   /* Synchronize all processes before returning */
   hypre_MPI_Barrier(comm);

   return error;
}

/*--------------------------------------------------------------------------
 * Unit test: 1D grid with 1D partitioning, overlap=8
 * Test case: 8x8 matrix on 4 processors
 * With overlap=8, each processor should gather the entire global matrix
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test3_Grid1D_Part1D_Overlap8(MPI_Comm comm, HYPRE_Int print_matrices)
{
   hypre_ParCSRMatrix *A;
   hypre_OverlapData *overlap_data;
   hypre_CSRMatrix *A_local;
   HYPRE_BigInt *col_map;
   HYPRE_Int num_cols_local;
   HYPRE_Int error = 0;
   HYPRE_Int overlap_order = 8;
   HYPRE_Int my_id, num_procs;
   MPI_Comm test_comm = MPI_COMM_NULL;
   HYPRE_Int participate = 0;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   /* Create subcommunicator with first 4 processors */
   if (num_procs >= 4)
   {
      participate = (my_id < 4) ? 1 : hypre_MPI_UNDEFINED;
      hypre_MPI_Comm_split(comm, participate, my_id, &test_comm);
   }
   else
   {
      if (my_id == 0)
      {
         hypre_printf("Test3_Grid1D_Part1D_Overlap8: Skipping (requires at least 4 processors)\n");
      }
      return 0;
   }

   /* Only participating processes run the test */
   if (test_comm == MPI_COMM_NULL)
   {
      /* Non-participating processes must still synchronize */
      hypre_MPI_Barrier(comm);
      return 0;
   }

   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      if (test_my_id == 0)
      {
         hypre_printf("Test3_Grid1D_Part1D_Overlap8 (4 procs): ");
      }
   }

   /* Create 8x8 1D Laplacian */
   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      HYPRE_Real *values = hypre_CTAlloc(HYPRE_Real, 4, HYPRE_MEMORY_HOST);
      values[0] = 2.0;
      values[1] = -1.0;
      values[2] = 0.0;
      values[3] = 0.0;
      A = Create1DLaplacian(test_comm, 8, 4, test_my_id);
      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }

   /* Compute overlap */
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }
   hypre_ParCSRMatrixComputeOverlap(A, overlap_order, &overlap_data);

   /* Get overlap rows */
   hypre_ParCSRMatrixGetOverlapRows(A, overlap_data);

   /* Extract local overlap matrix */
   hypre_ParCSRMatrixExtractLocalOverlap(A, overlap_data, &A_local, &col_map, &num_cols_local);

   /* Create expected matrix - with overlap=8, all processors should have the full 8x8 matrix */
   {
      hypre_CSRMatrix *A_expected = NULL;
      HYPRE_Int num_extended = hypre_OverlapDataNumExtendedRows(overlap_data);
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);

      /* Expected: full 8x8 matrix for all processors */
      HYPRE_Int I_expected[9] = {0, 2, 5, 8, 11, 14, 17, 20, 22};
      HYPRE_Int J_expected[22] = {0, 1,          /* row 0: columns 0,1 */
                                  0, 1, 2,       /* row 1: columns 0,1,2 */
                                  1, 2, 3,       /* row 2: columns 1,2,3 */
                                  2, 3, 4,       /* row 3: columns 2,3,4 */
                                  3, 4, 5,       /* row 4: columns 3,4,5 */
                                  4, 5, 6,       /* row 5: columns 4,5,6 */
                                  5, 6, 7,       /* row 6: columns 5,6,7 */
                                  6, 7           /* row 7: columns 6,7 */
                                 };
      HYPRE_Real data_expected[22] = {2.0, -1.0,           /* row 0 */
                                      -1.0, 2.0, -1.0,     /* row 1 */
                                      -1.0, 2.0, -1.0,     /* row 2 */
                                      -1.0, 2.0, -1.0,     /* row 3 */
                                      -1.0, 2.0, -1.0,     /* row 4 */
                                      -1.0, 2.0, -1.0,     /* row 5 */
                                      -1.0, 2.0, -1.0,     /* row 6 */
                                      -1.0, 2.0            /* row 7 */
                                     };

      A_expected = CreateCSRMatrixFromData(8, 8, 22, I_expected, J_expected, data_expected);

      if (A_expected)
      {
         HYPRE_Real tol = 1e-10;
         if (print_matrices)
         {
            char filename_expected[256];
            char filename_computed[256];
            hypre_sprintf(filename_expected, "test3_expected_ij.out.%05d", test_my_id);
            hypre_sprintf(filename_computed, "test3_computed_ij.out.%05d", test_my_id);
            hypre_CSRMatrixPrintIJ(A_expected, 0, 0, filename_expected);
            hypre_CSRMatrixPrintIJ(A_local, 0, 0, filename_computed);
         }

         /* Verify that we got all 8 rows */
         if (num_extended != 8)
         {
            hypre_printf("Proc %d: Expected 8 extended rows, got %d\n", test_my_id, num_extended);
            error = 1;
         }

         if (CompareCSRMatrices(A_expected, A_local, tol) != 0)
         {
            hypre_printf("Proc %d: Extracted matrix does not match expected matrix\n", test_my_id);
            error = 1;
         }
         hypre_CSRMatrixDestroy(A_expected);
      }
   }

   /* Cleanup */
   if (A_local)
   {
      hypre_CSRMatrixDestroy(A_local);
   }
   if (col_map)
   {
      hypre_TFree(col_map, HYPRE_MEMORY_HOST);
   }
   if (overlap_data)
   {
      hypre_OverlapDataDestroy(overlap_data);
   }
   if (A)
   {
      hypre_ParCSRMatrixDestroy(A);
   }

   if (test_comm != MPI_COMM_NULL)
   {
      HYPRE_Int test_my_id;
      hypre_MPI_Comm_rank(test_comm, &test_my_id);
      PRINT_TEST_RESULT(test_my_id, error);
      hypre_MPI_Comm_free(&test_comm);
   }

   /* Synchronize all processes before returning */
   hypre_MPI_Barrier(comm);

   return error;
}

/*--------------------------------------------------------------------------
 * Main function
 *--------------------------------------------------------------------------*/
int
main(int argc, char *argv[])
{
   MPI_Comm comm;
   HYPRE_Int my_id, num_procs;
   HYPRE_Int error = 0;
   HYPRE_Int test_mode = 1;  /* 1=unit tests, 0=benchmark */
   HYPRE_Int nx = 20, ny = 20, nz = 20;
   HYPRE_Int Px = 2, Py = 2, Pz = 2;
   HYPRE_Int overlap_order = 1;
   HYPRE_Int print_matrices = 0;
   HYPRE_Int i;

   /* Initialize MPI and HYPRE */
   hypre_MPI_Init(&argc, &argv);
   comm = hypre_MPI_COMM_WORLD;
   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   HYPRE_Init();

   /* Parse command line */
   i = 1;
   while (i < argc)
   {
      if (strcmp(argv[i], "-benchmark") == 0)
      {
         test_mode = 0;
         i++;
      }
      else if (strcmp(argv[i], "-n") == 0)
      {
         nx = atoi(argv[++i]);
         ny = atoi(argv[++i]);
         nz = atoi(argv[++i]);
         i++;
      }
      else if (strcmp(argv[i], "-P") == 0)
      {
         Px = atoi(argv[++i]);
         Py = atoi(argv[++i]);
         Pz = atoi(argv[++i]);
         i++;
      }
      else if (strcmp(argv[i], "-ov") == 0)
      {
         overlap_order = atoi(argv[++i]);
         i++;
      }
      else if (strcmp(argv[i], "-print") == 0)
      {
         print_matrices = 1;
         i++;
      }
      else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0)
      {
         if (my_id == 0)
         {
            hypre_printf("Usage: %s [options]\n", argv[0]);
            hypre_printf("Options:\n");
            hypre_printf("  -benchmark        : Run benchmark instead of unit tests\n");
            hypre_printf("  -n <nx> <ny> <nz> : Problem size (default: 20 20 20)\n");
            hypre_printf("  -P <Px> <Py> <Pz> : Processor grid (default: 2 2 2)\n");
            hypre_printf("  -ov <order>       : Overlap order (default: 1)\n");
            hypre_printf("  -print            : Print expected and computed matrices to .ij files\n");
            hypre_printf("  -h, -help         : Print this help\n");
         }
         HYPRE_Finalize();
         hypre_MPI_Finalize();
         return 0;
      }
      else
      {
         if (my_id == 0)
         {
            hypre_printf("Unknown option: %s\n", argv[i]);
         }
         i++;
      }
   }

   if (test_mode)
   {
      /* Run unit tests */
      if (my_id == 0)
      {
         hypre_printf("\n========================================\n");
         hypre_printf("Unit Tests for Overlap Extraction\n");
         hypre_printf("========================================\n\n");
      }

      error += Test1_Grid1D_Part1D_Overlap1(comm, print_matrices);
      error += Test2_Grid1D_Part1D_Overlap2(comm, print_matrices);
      error += Test3_Grid1D_Part1D_Overlap8(comm, print_matrices);

      if (my_id == 0)
      {
         hypre_printf("\n");
         if (error == 0)
         {
            hypre_printf("All unit tests PASSED\n");
         }
         else
         {
            hypre_printf("Some unit tests FAILED (errors: %d)\n", error);
         }
      }
   }
   else
   {
      /* TODO: benchmark mode */
   }

   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return error;
}
