/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for overlap matrix extraction
 *
 * This test file works according to the following modes:
 *  0 - Unit tests for overlap extraction with small hard-coded matrices
 *  1 - Overlap benchmarking with laplacian matrices
 *  2 - Matrix/Matrix product benchmarking with laplacian matrices
 *--------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"

#define CHECK_TOLERANCE 1.0e-10

/*--------------------------------------------------------------------------
 * Macros for test infrastructure
 *--------------------------------------------------------------------------*/
#define PRINT_TEST_RESULT(my_id, error, comm) \
   do { \
      HYPRE_Int global_error = 0; \
      hypre_MPI_Allreduce(&(error), &global_error, 1, HYPRE_MPI_INT, hypre_MPI_MAX, comm); \
      if ((my_id) == 0) hypre_printf("%s\n", global_error ? "FAILED" : "PASSED"); \
      (error) = global_error; \
   } while (0)

#define TEST_VARS() \
   hypre_ParCSRMatrix *A; hypre_OverlapData *overlap_data; hypre_CSRMatrix *A_local, *A_expected = NULL; \
   HYPRE_BigInt *col_map; HYPRE_Int num_cols_local, error = 0, test_my_id, my_id, num_procs; \
   MPI_Comm test_comm = MPI_COMM_NULL;

#define TEST_SETUP(min_procs) \
   hypre_MPI_Comm_rank(comm, &my_id); hypre_MPI_Comm_size(comm, &num_procs); \
   if (num_procs >= min_procs) { \
      HYPRE_Int part = (my_id < min_procs) ? 1 : hypre_MPI_UNDEFINED; \
      hypre_MPI_Comm_split(comm, part, my_id, &test_comm); \
   } else { \
      if (my_id == 0) hypre_printf("%s: Skipping (requires at least %d processors)\n", __func__, min_procs); \
      hypre_MPI_Barrier(comm); return 0; \
   } \
   if (test_comm == hypre_MPI_COMM_NULL) { hypre_MPI_Barrier(comm); return 0; } \
   hypre_MPI_Comm_rank(test_comm, &test_my_id); \
   if (test_my_id == 0) hypre_printf("%s (%d procs): ", __func__, min_procs);

#define TEST_OVERLAP(overlap_order) \
   if (!hypre_ParCSRMatrixCommPkg(A)) hypre_MatvecCommPkgCreate(A); \
   hypre_ParCSRMatrixComputeOverlap(A, overlap_order, &overlap_data); \
   hypre_ParCSRMatrixGetExternalMatrix(A, overlap_data); \
   hypre_ParCSRMatrixCreateExtendedMatrix(A, overlap_data, &A_local, &col_map, &num_cols_local);

#define TEST_CLEANUP() \
   if (A_expected) hypre_CSRMatrixDestroy(A_expected); \
   if (A_local) hypre_CSRMatrixDestroy(A_local); \
   if (col_map) hypre_TFree(col_map, HYPRE_MEMORY_HOST); \
   if (overlap_data) hypre_OverlapDataDestroy(overlap_data); \
   if (A) hypre_ParCSRMatrixDestroy(A); \
   if (test_comm != hypre_MPI_COMM_NULL) { PRINT_TEST_RESULT(test_my_id, error, test_comm); hypre_MPI_Comm_free(&test_comm); } \
   hypre_MPI_Barrier(comm); return error;

#define TEST_PRINT_MATRICES(tag) \
   if (print_matrices && A_expected) { \
      char filename_expected[256]; \
      char filename_computed[256]; \
      hypre_sprintf(filename_expected, tag "_expected_ij.out.%05d", test_my_id); \
      hypre_sprintf(filename_computed, tag "_computed_ij.out.%05d", test_my_id); \
      hypre_CSRMatrixPrintIJ(A_expected, 0, 0, filename_expected); \
      hypre_CSRMatrixPrintIJ(A_local, 0, 0, filename_computed); \
   }

#define TEST_COMPARE_MATRICES(tag) \
   if (A_expected) { \
      TEST_PRINT_MATRICES(tag) \
      if (CompareCSRMatrices(A_expected, A_local, CHECK_TOLERANCE) != 0) { \
         hypre_printf("Proc %d: Extracted matrix does not match expected matrix\n", test_my_id); error = 1; \
      } }

/*--------------------------------------------------------------------------
 * Helper: Compare two CSR matrices using Frobenius norm
 *--------------------------------------------------------------------------*/
static HYPRE_Int
CompareCSRMatrices(hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Real tol)
{
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumRows(B) ||
       hypre_CSRMatrixNumCols(A) != hypre_CSRMatrixNumCols(B))
   {
      hypre_printf("Matrix dimensions mismatch: (%d x %d) vs (%d x %d)\n",
                   hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A),
                   hypre_CSRMatrixNumRows(B), hypre_CSRMatrixNumCols(B));
      return 1;
   }
   hypre_CSRMatrix *diff = hypre_CSRMatrixAdd(1.0, A, -1.0, B);
   if (!diff) { return 1; }
   HYPRE_Real fnorm_diff = hypre_CSRMatrixFnorm(diff);
   HYPRE_Real fnorm_A = hypre_CSRMatrixFnorm(A);
   HYPRE_Real rel_error = (fnorm_A > 0.0) ? (fnorm_diff / fnorm_A) : fnorm_diff;
   hypre_CSRMatrixDestroy(diff);
   if (rel_error > tol)
   {
      hypre_printf("Matrix comparison failed: ||A - B||_F / ||A||_F = %e (tolerance = %e)\n", rel_error,
                   tol);
      return 1;
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * Helper: Create CSR matrix from arrays
 *--------------------------------------------------------------------------*/
static hypre_CSRMatrix*
CreateCSRMatrixFromData(HYPRE_Int nr, HYPRE_Int nc, HYPRE_Int nnz,
                        HYPRE_Int *I, HYPRE_Int *J, HYPRE_Real *data)
{
   hypre_CSRMatrix *m = hypre_CSRMatrixCreate(nr, nc, nnz);
   hypre_CSRMatrixInitialize(m);
   for (HYPRE_Int i = 0; i <= nr; i++) { hypre_CSRMatrixI(m)[i] = I[i]; }
   for (HYPRE_Int i = 0; i < nnz; i++) { hypre_CSRMatrixJ(m)[i] = J[i]; hypre_CSRMatrixData(m)[i] = data[i]; }
   hypre_CSRMatrixNumNonzeros(m) = nnz;
   return m;
}

/*--------------------------------------------------------------------------
 * Laplacian matrix creators
 *--------------------------------------------------------------------------*/
static hypre_ParCSRMatrix*
Create1DLaplacian(MPI_Comm comm, HYPRE_Int n, HYPRE_Int np, HYPRE_Int id)
{
   HYPRE_Real v[4] = {2.0, -1.0, 0.0, 0.0};
   return GenerateLaplacian(comm, n, 1, 1, np, 1, 1, id % np, 0, 0, v);
}

static hypre_ParCSRMatrix*
Create2DLaplacian(MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int np, HYPRE_Int id)
{
   HYPRE_Real v[4] = {4.0, -1.0, -1.0, 0.0};
   return GenerateLaplacian(comm, nx, ny, 1, np, 1, 1, id % np, 0, 0, v);
}

static hypre_ParCSRMatrix*
Create2DLaplacian2DPart(MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int Px, HYPRE_Int Py,
                        HYPRE_Int id)
{
   HYPRE_Real v[4] = {4.0, -1.0, -1.0, 0.0};
   HYPRE_Int p = id % Px, q = (id / Px) % Py;
   return GenerateLaplacian(comm, nx, ny, 1, Px, Py, 1, p, q, 0, v);
}

static hypre_ParCSRMatrix*
Create3DLaplacian3DPart(MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                        HYPRE_Int Px, HYPRE_Int Py, HYPRE_Int Pz, HYPRE_Int id)
{
   HYPRE_Real v[4] = {6.0, -1.0, -1.0, -1.0};
   HYPRE_Int p = id % Px, q = (id / Px) % Py, r = id / (Px * Py);
   return GenerateLaplacian(comm, nx, ny, nz, Px, Py, Pz, p, q, r, v);
}

/*--------------------------------------------------------------------------
 * Unit test: 1D grid with 1D partitioning, overlap=1 (4x4 matrix on 2 procs)
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test1_Grid1D_Part1D_Overlap1(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(2)
   A = Create1DLaplacian(test_comm, 4, 2, test_my_id);
   TEST_OVERLAP(1)

   /* 3x3 expected matrices - same for both procs */
   HYPRE_Int I[] = {0, 2, 5, 7};
   HYPRE_Int J[] = {0, 1, 0, 1, 2, 1, 2};
   HYPRE_Real D[] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
   A_expected = CreateCSRMatrixFromData(3, 3, 7, I, J, D);
   TEST_COMPARE_MATRICES("test1")
   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 1D grid with 1D partitioning, overlap=2 (6x6 matrix on 3 procs)
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test2_Grid1D_Part1D_Overlap2(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(3)
   A = Create1DLaplacian(test_comm, 6, 3, test_my_id);
   TEST_OVERLAP(2)

   if (test_my_id == 0)
   {
      HYPRE_Int I[] = {0, 2, 5, 8, 10};
      HYPRE_Int J[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
      HYPRE_Real D[] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
      A_expected = CreateCSRMatrixFromData(4, 4, 10, I, J, D);
   }
   else if (test_my_id == 1)
   {
      HYPRE_Int I[] = {0, 2, 5, 8, 11, 14, 16};
      HYPRE_Int J[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
      HYPRE_Real D[] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
      A_expected = CreateCSRMatrixFromData(6, 6, 16, I, J, D);
   }
   else
   {
      HYPRE_Int I[] = {0, 2, 5, 8, 10};
      HYPRE_Int J[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
      HYPRE_Real D[] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
      A_expected = CreateCSRMatrixFromData(4, 4, 10, I, J, D);
   }
   TEST_COMPARE_MATRICES("test2")
   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 1D grid with 1D partitioning, overlap=8 (full gather, 8x8 on 4 procs)
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test3_Grid1D_Part1D_Overlap8(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(4)
   A = Create1DLaplacian(test_comm, 8, 4, test_my_id);
   TEST_OVERLAP(8)

   HYPRE_Int I[] = {0, 2, 5, 8, 11, 14, 17, 20, 22};
   HYPRE_Int J[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7};
   HYPRE_Real D[] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
   A_expected = CreateCSRMatrixFromData(8, 8, 22, I, J, D);
   if (hypre_OverlapDataNumExtendedRows(overlap_data) != 8) { hypre_printf("Proc %d: Expected 8 extended rows, got %d\n", test_my_id, hypre_OverlapDataNumExtendedRows(overlap_data)); error = 1; }
   TEST_COMPARE_MATRICES("test3")
   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 2D grid with 1D partitioning, overlap=2 (4x4 grid on 2 procs)
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test4_Grid2D_Part1D_Overlap2(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(2)
   A = Create2DLaplacian(test_comm, 4, 4, 2, test_my_id);
   TEST_OVERLAP(2)

   HYPRE_Int I[] = {0, 3, 7, 11, 16, 20, 25, 28, 32, 36, 39, 44, 48, 53, 57, 61, 64};
   HYPRE_Int J[] = {0, 1, 2, 1, 0, 3, 8, 2, 0, 3, 4, 3, 1, 2, 5, 10, 4, 2, 5, 6, 5, 3, 4, 7, 12, 6, 4, 7, 7, 5, 6, 14, 8, 1, 9, 10, 9, 8, 11, 10, 3, 8, 11, 12, 11, 9, 10, 13, 12, 5, 10, 13, 14, 13, 11, 12, 15, 14, 7, 12, 15, 15, 13, 14};
   HYPRE_Real D[] = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0};
   A_expected = CreateCSRMatrixFromData(16, 16, 64, I, J, D);
   TEST_COMPARE_MATRICES("test4")
   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 2D grid with 2D partitioning, overlap=1 (4x4 grid on 2x2 procs)
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test5_Grid2D_Part2D_Overlap1(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(4)
   A = Create2DLaplacian2DPart(test_comm, 4, 4, 2, 2, test_my_id);
   TEST_OVERLAP(1)

   if (test_my_id == 0)
   {
      HYPRE_Int I[] = {0, 3, 7, 11, 16, 19, 22, 25, 28};
      HYPRE_Int J[] = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 6, 1, 2, 3, 5, 7, 1, 4, 5, 3, 4, 5, 2, 6, 7, 3, 6, 7};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(8, 8, 28, I, J, D);
   }
   else if (test_my_id == 1)
   {
      HYPRE_Int I[] = {0, 3, 6, 10, 13, 18, 22, 25, 28};
      HYPRE_Int J[] = {0, 1, 2, 0, 1, 4, 0, 2, 3, 4, 2, 3, 5, 1, 2, 4, 5, 6, 3, 4, 5, 7, 4, 6, 7, 5, 6, 7};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(8, 8, 28, I, J, D);
   }
   else if (test_my_id == 2)
   {
      HYPRE_Int I[] = {0, 3, 6, 10, 15, 18, 22, 25, 28};
      HYPRE_Int J[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 4, 1, 2, 3, 5, 6, 2, 4, 5, 3, 4, 5, 7, 3, 6, 7, 5, 6, 7};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(8, 8, 28, I, J, D);
   }
   else
   {
      HYPRE_Int I[] = {0, 3, 6, 9, 12, 17, 21, 25, 28};
      HYPRE_Int J[] = {0, 1, 4, 0, 1, 5, 2, 3, 4, 2, 3, 6, 0, 2, 4, 5, 6, 1, 4, 5, 7, 3, 4, 6, 7, 5, 6, 7};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(8, 8, 28, I, J, D);
   }
   TEST_COMPARE_MATRICES("test5")
   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 2D grid with 2D partitioning, overlap=2 (4x4 grid on 2x2 procs)
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test6_Grid2D_Part2D_Overlap2(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(4)
   A = Create2DLaplacian2DPart(test_comm, 4, 4, 2, 2, test_my_id);
   TEST_OVERLAP(2)

   if (test_my_id == 0)
   {
      HYPRE_Int I[] = {0, 3, 7, 11, 16, 20, 23, 28, 31, 35, 40, 43, 46, 49};
      HYPRE_Int J[] = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 8, 1, 2, 3, 6, 9, 1, 4, 5, 6, 4, 5, 7, 3, 4, 6, 7, 12, 5, 6, 7, 2, 8, 9, 10, 3, 8, 9, 11, 12, 8, 10, 11, 9, 10, 11, 6, 9, 12};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(13, 13, 49, I, J, D);
   }
   else if (test_my_id == 1)
   {
      HYPRE_Int I[] = {0, 3, 7, 10, 15, 19, 22, 27, 31, 34, 39, 43, 46, 49};
      HYPRE_Int J[] = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 1, 2, 3, 6, 8, 1, 4, 5, 6, 4, 5, 7, 3, 4, 6, 7, 9, 5, 6, 7, 10, 3, 8, 9, 6, 8, 9, 10, 11, 7, 9, 10, 12, 9, 11, 12, 10, 11, 12};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(13, 13, 49, I, J, D);
   }
   else if (test_my_id == 2)
   {
      HYPRE_Int I[] = {0, 3, 6, 10, 15, 18, 22, 27, 30, 34, 39, 42, 46, 49};
      HYPRE_Int J[] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 5, 1, 2, 3, 4, 6, 3, 4, 9, 2, 5, 6, 7, 3, 5, 6, 8, 9, 5, 7, 8, 6, 7, 8, 11, 4, 6, 9, 10, 11, 9, 10, 12, 8, 9, 11, 12, 10, 11, 12};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(13, 13, 49, I, J, D);
   }
   else
   {
      HYPRE_Int I[] = {0, 3, 6, 9, 14, 18, 21, 26, 29, 33, 38, 42, 46, 49};
      HYPRE_Int J[] = {0, 3, 6, 1, 2, 3, 1, 2, 4, 0, 1, 3, 4, 9, 2, 3, 4, 10, 5, 6, 7, 0, 5, 6, 8, 9, 5, 7, 8, 6, 7, 8, 11, 3, 6, 9, 10, 11, 4, 9, 10, 12, 8, 9, 11, 12, 10, 11, 12};
      HYPRE_Real D[] = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
      A_expected = CreateCSRMatrixFromData(13, 13, 49, I, J, D);
   }
   TEST_COMPARE_MATRICES("test6")
   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 2D grid with 2D partitioning, overlap=3
 * Test case: 4x4 2D grid on 2x2 processor grid (4 processors)
 * Tests overlap extraction for 2D problem with 2D processor layout and overlap=3
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test7_Grid2D_Part2D_Overlap3(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(4)
   A = Create2DLaplacian2DPart(test_comm, 4, 4, 2, 2, test_my_id);
   TEST_OVERLAP(3)

   if (!A_local) { hypre_printf("Proc %d: Failed to extract local overlap matrix\n", test_my_id); error = 1; }
   else
   {
      HYPRE_Int num_rows_local = hypre_CSRMatrixNumRows(A_local);
      HYPRE_Int num_cols_local_actual = hypre_CSRMatrixNumCols(A_local);
      if (num_rows_local == 15 && num_cols_local_actual == 15)
      {
         if (test_my_id == 0)
         {
            HYPRE_Int I[16] = {0, 3, 7, 11, 16, 20, 23, 28, 32, 36, 41, 44, 48, 53, 56, 59};
            HYPRE_Int J[59] = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 8, 1, 2, 3, 6, 9, 1, 4, 5, 6, 4, 5, 7, 3, 4, 6, 7, 12, 5, 6, 7, 13, 2, 8, 9, 10, 3, 8, 9, 11, 12, 8, 10, 11, 9, 10, 11, 14, 6, 9, 12, 13, 14, 7, 12, 13, 11, 12, 14};
            HYPRE_Real D[59] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};
            A_expected = CreateCSRMatrixFromData(15, 15, 59, I, J, D);
         }
         else if (test_my_id == 1)
         {
            HYPRE_Int I[16] = {0, 3, 7, 11, 16, 20, 23, 28, 32, 35, 40, 43, 48, 52, 56, 59};
            HYPRE_Int J[59] = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 8, 1, 2, 3, 6, 9, 1, 4, 5, 6, 4, 5, 7, 3, 4, 6, 7, 11, 5, 6, 7, 12, 2, 8, 9, 3, 8, 9, 10, 11, 9, 10, 13, 6, 9, 11, 12, 13, 7, 11, 12, 14, 10, 11, 13, 14, 12, 13, 14};
            HYPRE_Real D[59] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
            A_expected = CreateCSRMatrixFromData(15, 15, 59, I, J, D);
         }
         else if (test_my_id == 2)
         {
            HYPRE_Int I[16] = {0, 3, 7, 11, 16, 19, 24, 27, 31, 36, 39, 43, 48, 52, 56, 59};
            HYPRE_Int J[59] = {0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 7, 1, 2, 3, 5, 8, 1, 4, 5, 3, 4, 5, 6, 11, 5, 6, 12, 2, 7, 8, 9, 3, 7, 8, 10, 11, 7, 9, 10, 8, 9, 10, 13, 5, 8, 11, 12, 13, 6, 11, 12, 14, 10, 11, 13, 14, 12, 13, 14};
            HYPRE_Real D[59] = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
            A_expected = CreateCSRMatrixFromData(15, 15, 59, I, J, D);
         }
         else if (test_my_id == 3)
         {
            HYPRE_Int I[16] = {0, 3, 6, 11, 15, 18, 23, 27, 31, 36, 39, 43, 48, 52, 56, 59};
            HYPRE_Int J[59] = {0, 2, 3, 1, 2, 7, 0, 1, 2, 5, 8, 0, 3, 4, 5, 3, 4, 6, 2, 3, 5, 6, 11, 4, 5, 6, 12, 1, 7, 8, 9, 2, 7, 8, 10, 11, 7, 9, 10, 8, 9, 10, 13, 5, 8, 11, 12, 13, 6, 11, 12, 14, 10, 11, 13, 14, 12, 13, 14};
            HYPRE_Real D[59] = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
            A_expected = CreateCSRMatrixFromData(15, 15, 59, I, J, D);
         }
      }
      else { hypre_printf("Proc %d: Unexpected matrix dimensions: %d x %d (expected 15 x 15)\n", test_my_id, num_rows_local, num_cols_local_actual); error = 1; }
      TEST_COMPARE_MATRICES("test7")
   }

   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 3D grid with 3D partitioning, overlap=1
 * Test case: 3x3x3 3D grid on 2x2x2 processor grid (8 processors)
 * Tests overlap extraction for 3D problem with 3D processor layout and overlap=1
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test8_Grid3D_Part3D_Overlap1(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(8)
   A = Create3DLaplacian3DPart(test_comm, 3, 3, 3, 2, 2, 2, test_my_id);
   TEST_OVERLAP(1)
   if (!A_local) { hypre_printf("Proc %d: Failed to extract local overlap matrix\n", test_my_id); error = 1; }
   else
   {
      HYPRE_Int num_rows_local = hypre_CSRMatrixNumRows(A_local);
      HYPRE_Int num_cols_local_actual = hypre_CSRMatrixNumCols(A_local);
      if (test_my_id == 0 && num_rows_local == 20 && num_cols_local_actual == 20)
      {
         HYPRE_Int I[21] = {0, 4, 9, 14, 20, 25, 31, 37, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92};
         HYPRE_Int J[92] = {0, 1, 2, 4, 0, 1, 3, 5, 8, 0, 2, 3, 6, 12, 1, 2, 3, 7, 9, 13, 0, 4, 5, 6, 16, 1, 4, 5, 7, 10, 17, 2, 4, 6, 7, 14, 18, 3, 5, 6, 7, 11, 15, 19, 1, 8, 9, 10, 3, 8, 9, 11, 5, 8, 10, 11, 7, 9, 10, 11, 2, 12, 13, 14, 3, 12, 13, 15, 6, 12, 14, 15, 7, 13, 14, 15, 4, 16, 17, 18, 5, 16, 17, 19, 6, 16, 18, 19, 7, 17, 18, 19};
         HYPRE_Real D[92] = {6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(20, 20, 92, I, J, D);
      }
      else if (test_my_id == 1 && num_rows_local == 12 && num_cols_local_actual == 12)
      {
         HYPRE_Int I[13] = {0, 4, 8, 12, 16, 20, 25, 30, 36, 39, 42, 45, 48};
         HYPRE_Int J[48] = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 8, 2, 4, 6, 7, 10, 3, 5, 6, 7, 9, 11, 5, 8, 9, 7, 8, 9, 6, 10, 11, 7, 10, 11};
         HYPRE_Real D[48] = {6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(12, 12, 48, I, J, D);
      }
      else if (test_my_id == 2 && num_rows_local == 12 && num_cols_local_actual == 12)
      {
         HYPRE_Int I[13] = {0, 4, 8, 12, 16, 20, 25, 30, 36, 39, 42, 45, 48};
         HYPRE_Int J[48] = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 8, 2, 4, 6, 7, 10, 3, 5, 6, 7, 9, 11, 5, 8, 9, 7, 8, 9, 6, 10, 11, 7, 10, 11};
         HYPRE_Real D[48] = {6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(12, 12, 48, I, J, D);
      }
      else if (test_my_id == 3 && num_rows_local == 7 && num_cols_local_actual == 7)
      {
         HYPRE_Int I[8] = {0, 3, 6, 9, 12, 16, 21, 23};
         HYPRE_Int J[23] = {0, 1, 4, 0, 1, 5, 2, 3, 4, 2, 3, 5, 0, 2, 4, 5, 1, 3, 4, 5, 6, 5, 6};
         HYPRE_Real D[23] = {6.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(7, 7, 23, I, J, D);
      }
      else if (test_my_id == 4 && num_rows_local == 12 && num_cols_local_actual == 12)
      {
         HYPRE_Int I[13] = {0, 4, 8, 12, 16, 20, 25, 30, 36, 39, 42, 45, 48};
         HYPRE_Int J[48] = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 8, 2, 4, 6, 7, 10, 3, 5, 6, 7, 9, 11, 5, 8, 9, 7, 8, 9, 6, 10, 11, 7, 10, 11};
         HYPRE_Real D[48] = {6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(12, 12, 48, I, J, D);
      }
      else if (test_my_id == 5 && num_rows_local == 7 && num_cols_local_actual == 7)
      {
         HYPRE_Int I[8] = {0, 3, 6, 9, 12, 16, 21, 23};
         HYPRE_Int J[23] = {0, 1, 4, 0, 1, 5, 2, 3, 4, 2, 3, 5, 0, 2, 4, 5, 1, 3, 4, 5, 6, 5, 6};
         HYPRE_Real D[23] = {6.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(7, 7, 23, I, J, D);
      }
      else if (test_my_id == 6 && num_rows_local == 7 && num_cols_local_actual == 7)
      {
         HYPRE_Int I[8] = {0, 3, 6, 9, 12, 16, 21, 23};
         HYPRE_Int J[23] = {0, 1, 4, 0, 1, 5, 2, 3, 4, 2, 3, 5, 0, 2, 4, 5, 1, 3, 4, 5, 6, 5, 6};
         HYPRE_Real D[23] = {6.0, -1.0, -1.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(7, 7, 23, I, J, D);
      }
      else if (test_my_id == 7 && num_rows_local == 4 && num_cols_local_actual == 4)
      {
         HYPRE_Int I[5] = {0, 2, 4, 6, 10};
         HYPRE_Int J[10] = {0, 3, 1, 3, 2, 3, 0, 1, 2, 3};
         HYPRE_Real D[10] = {6.0, -1.0, 6.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(4, 4, 10, I, J, D);
      }
      else if (test_my_id >= 0 && test_my_id <= 7)
      {
         hypre_printf("Proc %d: Unexpected matrix dimensions: %d x %d\n", test_my_id, num_rows_local,
                      num_cols_local_actual);
         error = 1;
      }
      else { hypre_printf("Proc %d: Unexpected processor ID (expected 0-7)\n", test_my_id); error = 1; }
      TEST_COMPARE_MATRICES("test8")
   }

   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Unit test: 3D grid with 3D partitioning, overlap=6 (full domain coverage)
 * Test case: 3x3x3 3D grid on 2x2x2 processor grid (8 processors)
 * Tests overlap extraction for 3D problem with 3D processor layout and overlap=6
 * With overlap=6, all processors should have the full global matrix (27x27)
 *--------------------------------------------------------------------------*/
static HYPRE_Int
Test9_Grid3D_Part3D_Overlap6(MPI_Comm comm, HYPRE_Int print_matrices)
{
   TEST_VARS()
   TEST_SETUP(8)
   A = Create3DLaplacian3DPart(test_comm, 3, 3, 3, 2, 2, 2, test_my_id);
   TEST_OVERLAP(6)
   if (!A_local) { hypre_printf("Proc %d: Failed to extract local overlap matrix\n", test_my_id); error = 1; }
   else
   {
      HYPRE_Int num_rows_local = hypre_CSRMatrixNumRows(A_local);
      HYPRE_Int num_cols_local_actual = hypre_CSRMatrixNumCols(A_local);
      if (num_rows_local == 27 && num_cols_local_actual == 27)
      {
         HYPRE_Int I[28] = {0, 4, 9, 14, 20, 25, 31, 37, 44, 48, 53, 58, 64, 68, 73, 78, 84, 88, 93, 97, 102, 107, 113, 117, 122, 126, 131, 135};
         HYPRE_Int J[135] = {0, 1, 2, 4, 0, 1, 3, 5, 8, 0, 2, 3, 6, 12, 1, 2, 3, 7, 9, 13, 0, 4, 5, 6, 18, 1, 4, 5, 7, 10, 19, 2, 4, 6, 7, 14, 20, 3, 5, 6, 7, 11, 15, 21, 1, 8, 9, 10, 3, 8, 9, 11, 16, 5, 8, 10, 11, 22, 7, 9, 10, 11, 17, 23, 2, 12, 13, 14, 3, 12, 13, 15, 16, 6, 12, 14, 15, 24, 7, 13, 14, 15, 17, 25, 9, 13, 16, 17, 11, 15, 16, 17, 26, 4, 18, 19, 20, 5, 18, 19, 21, 22, 6, 18, 20, 21, 24, 7, 19, 20, 21, 23, 25, 10, 19, 22, 23, 11, 21, 22, 23, 26, 14, 20, 24, 25, 15, 21, 24, 25, 26, 17, 23, 25, 26};
         HYPRE_Real D[135] = {6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0, -1.0, 6.0};
         A_expected = CreateCSRMatrixFromData(27, 27, 135, I, J, D);
      }
      else { hypre_printf("Proc %d: Unexpected matrix dimensions: %d x %d (expected 27 x 27)\n", test_my_id, num_rows_local, num_cols_local_actual); error = 1; }
      TEST_COMPARE_MATRICES("test9")
   }

   TEST_CLEANUP()
}

/*--------------------------------------------------------------------------
 * Benchmark: Generate laplacian and test overlap extraction
 *--------------------------------------------------------------------------*/
static HYPRE_Int
BenchmarkOverlap(MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                 HYPRE_Int Px, HYPRE_Int Py, HYPRE_Int Pz,
                 HYPRE_Int overlap_order, HYPRE_Int print_matrices)
{
   HYPRE_Int my_id, num_procs;
   HYPRE_ParCSRMatrix A;
   hypre_OverlapData *overlap_data;
   hypre_CSRMatrix *A_local;
   HYPRE_BigInt *col_map;
   HYPRE_Int num_cols_local;
   HYPRE_Real time_start, time_end, time_overlap, time_extract;
   HYPRE_Int num_extended, num_overlap;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   if (my_id == 0)
   {
      hypre_printf("\nBenchmark: Overlap extraction\n");
      hypre_printf("  Problem size: %d x %d x %d\n", nx, ny, nz);
      hypre_printf("  Processor grid: %d x %d x %d\n", Px, Py, Pz);
      hypre_printf("  Overlap order: %d\n", overlap_order);
   }

   /* Generate laplacian using GenerateLaplacian */
   {
      HYPRE_Real *values;
      HYPRE_Int p, q, r;

      /* Compute processor coordinates */
      p = my_id % Px;
      q = ((my_id - p) / Px) % Py;
      r = (my_id - p - Px * q) / (Px * Py);

      /* Set up 7-point stencil values */
      values = hypre_CTAlloc(HYPRE_Real, 4, HYPRE_MEMORY_HOST);
      if (nx > 1)
      {
         values[0] += 2.0;
         values[1] = -1.0;
      }
      if (ny > 1)
      {
         values[0] += 2.0;
         values[2] = -1.0;
      }
      if (nz > 1)
      {
         values[0] += 2.0;
         values[3] = -1.0;
      }

      A = GenerateLaplacian(comm, nx, ny, nz, Px, Py, Pz, p, q, r, values);

      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }

   /* Print global matrix if requested */
   if (print_matrices)
   {
      char filename[256];
      hypre_sprintf(filename, "benchmark_global_matrix_ij.out");
      hypre_ParCSRMatrixPrintIJ(A, 0, 0, filename);
      if (my_id == 0)
      {
         hypre_printf("  Printed global matrix to %s.<proc_id>\n", filename);
      }
   }

   /* Benchmark overlap computation */
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   time_start = hypre_MPI_Wtime();
   hypre_ParCSRMatrixComputeOverlap(A, overlap_order, &overlap_data);
   hypre_MPI_Barrier(comm);
   time_end = hypre_MPI_Wtime();
   time_overlap = time_end - time_start;

   /* Benchmark overlap row fetching */
   time_start = hypre_MPI_Wtime();
   hypre_ParCSRMatrixGetExternalMatrix(A, overlap_data);
   hypre_MPI_Barrier(comm);
   time_end = hypre_MPI_Wtime();
   time_overlap += (time_end - time_start);

   /* Benchmark local matrix extraction */
   time_start = hypre_MPI_Wtime();
   hypre_ParCSRMatrixCreateExtendedMatrix(A, overlap_data, &A_local, &col_map, &num_cols_local);
   hypre_MPI_Barrier(comm);
   time_end = hypre_MPI_Wtime();
   time_extract = time_end - time_start;

   /* Print overlapped matrices if requested */
   if (print_matrices)
   {
      char filename[256];
      hypre_sprintf(filename, "benchmark_overlap_matrix_ij.out.%05d", my_id);
      hypre_CSRMatrixPrintIJ(A_local, 0, 0, filename);
   }

   /* Gather statistics */
   num_extended = hypre_OverlapDataNumExtendedRows(overlap_data);
   num_overlap  = hypre_OverlapDataNumOverlapRows(overlap_data);

   if (my_id == 0)
   {
      hypre_printf("  Overlap computation time: %e seconds\n", time_overlap);
      hypre_printf("  Extraction time: %e seconds\n", time_extract);
      hypre_printf("  Total time: %e seconds\n", time_overlap + time_extract);
   }

   /* Print per-processor stats */
   {
      HYPRE_Int i;
      HYPRE_Int *local_nnz = hypre_TAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int *local_extended = hypre_TAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int *local_overlap = hypre_TAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int nnz_local = hypre_CSRMatrixNumNonzeros(A_local);

      hypre_MPI_Gather(&nnz_local, 1, HYPRE_MPI_INT, local_nnz, 1, HYPRE_MPI_INT, 0, comm);
      hypre_MPI_Gather(&num_extended, 1, HYPRE_MPI_INT, local_extended, 1, HYPRE_MPI_INT, 0, comm);
      hypre_MPI_Gather(&num_overlap, 1, HYPRE_MPI_INT, local_overlap, 1, HYPRE_MPI_INT, 0, comm);

      if (my_id == 0)
      {
         hypre_printf("\n  Per-processor statistics:\n");
         hypre_printf("  Proc  Extended Rows  Overlap Rows  Local NNZ\n");
         for (i = 0; i < num_procs; i++)
         {
            hypre_printf("  %3d      %8d      %8d    %10d\n",
                         i, local_extended[i], local_overlap[i], local_nnz[i]);
         }
      }

      hypre_TFree(local_nnz, HYPRE_MEMORY_HOST);
      hypre_TFree(local_extended, HYPRE_MEMORY_HOST);
      hypre_TFree(local_overlap, HYPRE_MEMORY_HOST);
   }

   /* Cleanup */
   hypre_CSRMatrixDestroy(A_local);
   hypre_TFree(col_map, HYPRE_MEMORY_HOST);
   hypre_OverlapDataDestroy(overlap_data);
   HYPRE_ParCSRMatrixDestroy(A);

   return 0;
}

/*--------------------------------------------------------------------------
 * Benchmark: Generate laplacian and compute matrix powers via matrix-matrix multiplication
 * Computes A^overlap_order using repeated matrix-matrix products
 * This serves as a reference for the new overlap capabilities
 *--------------------------------------------------------------------------*/
static HYPRE_Int
BenchmarkMatMat(MPI_Comm comm, HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                HYPRE_Int Px, HYPRE_Int Py, HYPRE_Int Pz,
                HYPRE_Int overlap_order, HYPRE_Int print_matrices)
{
   HYPRE_Int my_id, num_procs;
   hypre_ParCSRMatrix *A;
   hypre_ParCSRMatrix *A_power;
   hypre_ParCSRMatrix **buffer;
   HYPRE_Real time_start, time_end, time_total;
   HYPRE_Int i;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   if (my_id == 0)
   {
      hypre_printf("\nBenchmark: Matrix-Matrix Multiplication (A^%d)\n", overlap_order + 1);
      hypre_printf("  Problem size: %d x %d x %d\n", nx, ny, nz);
      hypre_printf("  Processor grid: %d x %d x %d\n", Px, Py, Pz);
      hypre_printf("  Overlap order: %d (computes A^%d)\n", overlap_order, overlap_order + 1);
   }

   /* Generate laplacian using GenerateLaplacian */
   {
      HYPRE_Real *values;
      HYPRE_Int p, q, r;

      /* Compute processor coordinates */
      p = my_id % Px;
      q = ((my_id - p) / Px) % Py;
      r = (my_id - p - Px * q) / (Px * Py);

      /* Set up 7-point stencil values */
      values = hypre_CTAlloc(HYPRE_Real, 4, HYPRE_MEMORY_HOST);
      if (nx > 1)
      {
         values[0] += 2.0;
         values[1] = -1.0;
      }
      if (ny > 1)
      {
         values[0] += 2.0;
         values[2] = -1.0;
      }
      if (nz > 1)
      {
         values[0] += 2.0;
         values[3] = -1.0;
      }

      A = GenerateLaplacian(comm, nx, ny, nz, Px, Py, Pz, p, q, r, values);

      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }

   /* Print original matrix if requested */
   if (print_matrices)
   {
      char filename[256];
      hypre_sprintf(filename, "benchmark_matmat_original_ij.out");
      hypre_ParCSRMatrixPrintIJ(A, 0, 0, filename);
      if (my_id == 0)
      {
         hypre_printf("  Printed original matrix to %s.<proc_id>\n", filename);
      }
   }

   /* Ensure communication package exists */
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   /* Allocate array to track intermediate matrices for cleanup */
   buffer = hypre_TAlloc(hypre_ParCSRMatrix*, overlap_order, HYPRE_MEMORY_HOST);
   for (i = 0; i < overlap_order; i++)
   {
      buffer[i] = NULL;
   }

   /* Compute A^(overlap_order+1) */
   time_start = hypre_MPI_Wtime();

   HYPRE_Int exponent = overlap_order + 1;
   HYPRE_Int buffer_idx = 0;
   hypre_ParCSRMatrix *temp, *base = A;

   /* Binary exponentiation: process exponent bit by bit */
   A_power = NULL;
   while (exponent > 0)
   {
      if (exponent % 2 == 1)
      {
         /* Current bit is 1: multiply result by current base */
         if (A_power == NULL)
         {
            /* First multiplication: result = base (which is A initially) */
            A_power = base;
         }
         else
         {
            /* Multiply result by base */
            temp = hypre_ParCSRMatMat(A_power, base);
            /* Store old A_power in buffer for cleanup (if not original A) */
            if (A_power != A)
            {
               if (buffer_idx < overlap_order)
               {
                  buffer[buffer_idx++] = A_power;
               }
               else
               {
                  /* Buffer full: destroy old A_power immediately */
                  hypre_ParCSRMatrixDestroy(A_power);
               }
            }
            A_power = temp;
         }
      }

      /* Square the base for next bit: base = base^2 */
      if (exponent > 1)
      {
         temp = hypre_ParCSRMatMat(base, base);
         /* Store old base in buffer for cleanup (if not original A) */
         if (base != A)
         {
            if (buffer_idx < overlap_order)
            {
               buffer[buffer_idx++] = base;
            }
            else
            {
               /* Buffer full: destroy old base immediately */
               hypre_ParCSRMatrixDestroy(base);
            }
         }
         base = temp;
      }

      exponent /= 2;
   }

   hypre_MPI_Barrier(comm);
   time_end = hypre_MPI_Wtime();
   time_total = time_end - time_start;

   /* Print result matrix if requested */
   if (print_matrices)
   {
      char filename[256];
      hypre_sprintf(filename, "benchmark_matmat_power_ij.out.%05d", my_id);
      hypre_ParCSRMatrixPrintIJ(A_power, 0, 0, filename);
      if (my_id == 0)
      {
         hypre_printf("  Printed A^%d matrix to %s.<proc_id>\n", overlap_order + 1, filename);
      }
   }

   /* Gather statistics */
   {
      HYPRE_Int i;
      HYPRE_Int *local_nnz = hypre_TAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int *local_rows = hypre_TAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
      HYPRE_Int nnz_local = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A_power)) +
                            hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A_power));
      HYPRE_Int num_rows_local = hypre_ParCSRMatrixNumRows(A_power);

      hypre_MPI_Gather(&nnz_local, 1, HYPRE_MPI_INT, local_nnz, 1, HYPRE_MPI_INT, 0, comm);
      hypre_MPI_Gather(&num_rows_local, 1, HYPRE_MPI_INT, local_rows, 1, HYPRE_MPI_INT, 0, comm);

      if (my_id == 0)
      {
         hypre_printf("  Total matrix-matrix multiplication time: %e seconds\n", time_total);
         hypre_printf("\n  Per-processor statistics:\n");
         hypre_printf("  Proc  Local Rows  Local NNZ\n");
         for (i = 0; i < num_procs; i++)
         {
            hypre_printf("  %3d      %8d    %10d\n", i, local_rows[i], local_nnz[i]);
         }
      }

      hypre_TFree(local_nnz, HYPRE_MEMORY_HOST);
      hypre_TFree(local_rows, HYPRE_MEMORY_HOST);
   }

   /* Cleanup: destroy all intermediate matrices and final result */
   /* Destroy all intermediate matrices stored in buffer */
   for (i = 0; i < overlap_order; i++)
   {
      if (buffer[i] != NULL)
      {
         hypre_ParCSRMatrixDestroy(buffer[i]);
      }
   }

   /* Destroy final base if it's not the original A and not the same as A_power */
   if (base != A && base != A_power)
   {
      hypre_ParCSRMatrixDestroy(base);
   }

   /* Destroy final result A_power if it's not the original A */
   if (A_power != A)
   {
      hypre_ParCSRMatrixDestroy(A_power);
   }

   /* Destroy original matrix A */
   hypre_ParCSRMatrixDestroy(A);

   /* Free the buffer array */
   hypre_TFree(buffer, HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * Main function
 *--------------------------------------------------------------------------*/
hypre_int
main(hypre_int argc, char *argv[])
{
   MPI_Comm comm;
   HYPRE_Int my_id, num_procs;
   HYPRE_Int error = 0;
   HYPRE_Int test_mode = 0;  /* 0=unit tests, 1=benchmark, 2=benchmark-matmat */
   HYPRE_Int nx, ny, nz;
   HYPRE_Int Px, Py, Pz;
   HYPRE_Int overlap_order = 1;
   HYPRE_Int print_matrices = 0;
   HYPRE_Int i;

   /* Initialize MPI and HYPRE */
   hypre_MPI_Init(&argc, &argv);
   comm = hypre_MPI_COMM_WORLD;
   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   HYPRE_Initialize();

   /* Default to CPU execution */
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);

   /* Parse command line */
   i = 1;
   nx = num_procs * 64; ny = 64; nz = 64;
   Px = num_procs; Py = 1; Pz = 1;
   while (i < argc)
   {
      if (strcmp(argv[i], "-benchmark") == 0)
      {
         test_mode = 1;
         i++;
      }
      else if (strcmp(argv[i], "-benchmark-matmat") == 0)
      {
         test_mode = 2;
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
            hypre_printf("  -benchmark        : Run overlap extraction benchmark\n");
            hypre_printf("  -benchmark-matmat : Run matrix-matrix multiplication benchmark (A^overlap_order)\n");
            hypre_printf("  -n <nx> <ny> <nz> : Problem size (default: 20 20 20)\n");
            hypre_printf("  -P <Px> <Py> <Pz> : Processor grid (default: 2 2 2)\n");
            hypre_printf("  -ov <order>       : Overlap order / matrix power (default: 1)\n");
            hypre_printf("  -print            : Print expected and computed matrices to .ij files\n");
            hypre_printf("  -h, -help         : Print this help\n");
         }
         goto end;
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

   if (!test_mode)
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
      error += Test4_Grid2D_Part1D_Overlap2(comm, print_matrices);
      error += Test5_Grid2D_Part2D_Overlap1(comm, print_matrices);
      error += Test6_Grid2D_Part2D_Overlap2(comm, print_matrices);
      error += Test7_Grid2D_Part2D_Overlap3(comm, print_matrices);
      error += Test8_Grid3D_Part3D_Overlap1(comm, print_matrices);
      error += Test9_Grid3D_Part3D_Overlap6(comm, print_matrices);

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
   else if (test_mode == 1)
   {
      /* Run Warmup */
      if (my_id == 0)
      {
         hypre_printf("\nWarmup phase...");
      }
      BenchmarkOverlap(comm, 10 * num_procs, 10, 10, num_procs, 1, 1, 0, 0);

      /* Run Overlap Benchmark */
      BenchmarkOverlap(comm, nx, ny, nz, Px, Py, Pz, overlap_order, print_matrices);
   }
   else if (test_mode == 2)
   {
      /* Run Warmup */
      if (my_id == 0)
      {
         hypre_printf("\nWarmup phase...");
      }
      BenchmarkMatMat(comm, 10 * num_procs, 10, 10, num_procs, 1, 1, 1, 0);

      /* Run Matrix-Matrix Multiplication Benchmark */
      BenchmarkMatMat(comm, nx, ny, nz, Px, Py, Pz, overlap_order, print_matrices);
   }

end:
   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return error;
}
