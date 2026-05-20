/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Unit tests for ParCSR matrix statistics.
 *--------------------------------------------------------------------------*/

#include <math.h>
#include <stdio.h>

#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.h"
#include "hypre_unit.h"

#define MATRIX_STATS_TOL 1.0e-12

static hypre_ParCSRMatrix *
CreateHostParCSRMatrix(HYPRE_Int        num_rows,
                       HYPRE_Int        num_cols,
                       HYPRE_Int        num_nonzeros,
                       const HYPRE_Int *row_ptr,
                       const HYPRE_Int *col_ind,
                       const HYPRE_Real *data)
{
   HYPRE_BigInt row_starts[2] = {0, num_rows};
   HYPRE_BigInt col_starts[2] = {0, num_cols};
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag;

   A = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_SELF, num_rows, num_cols,
                                row_starts, col_starts, 0, num_nonzeros, 0);
   hypre_ParCSRMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   diag = hypre_ParCSRMatrixDiag(A);

   for (HYPRE_Int i = 0; i <= num_rows; i++)
   {
      hypre_CSRMatrixI(diag)[i] = row_ptr[i];
   }

   for (HYPRE_Int i = 0; i < num_nonzeros; i++)
   {
      hypre_CSRMatrixJ(diag)[i]    = col_ind[i];
      hypre_CSRMatrixData(diag)[i] = data[i];
   }

   hypre_ParCSRMatrixNumNonzeros(A)  = num_nonzeros;
   hypre_ParCSRMatrixDNumNonzeros(A) = (hypre_double) num_nonzeros;

   return A;
}

static HYPRE_Int
CheckStats(const char        *tag,
           hypre_MatrixStats *stats,
           HYPRE_BigInt       num_rows,
           HYPRE_BigInt       num_cols,
           hypre_ulonglongint num_nonzeros,
           hypre_ulonglongint actual_nonzeros,
           HYPRE_Real         sparsity,
           HYPRE_Int          nnz_min,
           HYPRE_Int          nnz_max,
           HYPRE_Real         nnz_avg,
           HYPRE_Real         nnz_sqsum,
           HYPRE_Real         rowsum_min,
           HYPRE_Real         rowsum_max,
           HYPRE_Real         rowsum_avg,
           HYPRE_Real         rowsum_sqsum,
           HYPRE_Real         absrowsum_min,
           HYPRE_Real         absrowsum_max,
           HYPRE_Real         absrowsum_avg,
           HYPRE_Real         absrowsum_sqsum,
           HYPRE_Int          print_result)
{
   HYPRE_Int error = 0;

   HYPRE_UNIT_CHECK_BIGINT(error, "num_rows", hypre_MatrixStatsNumRows(stats), num_rows);
   HYPRE_UNIT_CHECK_BIGINT(error, "num_cols", hypre_MatrixStatsNumCols(stats), num_cols);
   HYPRE_UNIT_CHECK_BIGINT(error, "num_nonzeros", hypre_MatrixStatsNumNonzeros(stats), num_nonzeros);
   HYPRE_UNIT_CHECK_BIGINT(error, "actual_nonzeros", hypre_MatrixStatsActualNonzeros(stats),
                           actual_nonzeros);
   HYPRE_UNIT_CHECK_REAL(error, "sparsity", hypre_MatrixStatsSparsity(stats),
                         sparsity, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_INT(error, "nnzrow_min", hypre_MatrixStatsNnzrowMin(stats), nnz_min);
   HYPRE_UNIT_CHECK_INT(error, "nnzrow_max", hypre_MatrixStatsNnzrowMax(stats), nnz_max);
   HYPRE_UNIT_CHECK_REAL(error, "nnzrow_avg", hypre_MatrixStatsNnzrowAvg(stats),
                         nnz_avg, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "nnzrow_sqsum", hypre_MatrixStatsNnzrowSqsum(stats),
                         nnz_sqsum, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "nnzrow_stdev", hypre_MatrixStatsNnzrowStDev(stats),
                         hypre_sqrt(nnz_sqsum / (HYPRE_Real) num_rows), MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "rowsum_min", hypre_MatrixStatsRowsumMin(stats),
                         rowsum_min, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "rowsum_max", hypre_MatrixStatsRowsumMax(stats),
                         rowsum_max, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "rowsum_avg", hypre_MatrixStatsRowsumAvg(stats),
                         rowsum_avg, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "rowsum_sqsum", hypre_MatrixStatsRowsumSqsum(stats),
                         rowsum_sqsum, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "rowsum_stdev", hypre_MatrixStatsRowsumStDev(stats),
                         hypre_sqrt(rowsum_sqsum / (HYPRE_Real) num_rows), MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "absrowsum_min", hypre_MatrixStatsAbsrowsumMin(stats),
                         absrowsum_min, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "absrowsum_max", hypre_MatrixStatsAbsrowsumMax(stats),
                         absrowsum_max, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "absrowsum_avg", hypre_MatrixStatsAbsrowsumAvg(stats),
                         absrowsum_avg, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "absrowsum_sqsum", hypre_MatrixStatsAbsrowsumSqsum(stats),
                         absrowsum_sqsum, MATRIX_STATS_TOL);
   HYPRE_UNIT_CHECK_REAL(error, "absrowsum_stdev", hypre_MatrixStatsAbsrowsumStDev(stats),
                         hypre_sqrt(absrowsum_sqsum / (HYPRE_Real) num_rows), MATRIX_STATS_TOL);

   if (!print_result)
   {
      return error;
   }

   if (error)
   {
      hypre_printf("%s: FAILED\n", tag);
   }
   else
   {
      hypre_printf("%s: PASSED\n", tag);
   }

   return error;
}

static HYPRE_Int
Test1_ParCSRMatrixStatsComputeLocal(MPI_Comm comm)
{
   MPI_Comm          test_comm = hypre_MPI_COMM_NULL;
   HYPRE_Int         test_my_id;
   HYPRE_Int         error = 0;
   HYPRE_Int         row_ptr[5] = {0, 2, 3, 3, 7};
   HYPRE_Int         col_ind[7] = {0, 1, 2, 0, 2, 3, 4};
   HYPRE_Real        data[7]    = {1.0, -3.0, 5.0, -1.0, -2.0, 3.0, -4.0};
   hypre_ParCSRMatrix *A;
   hypre_MatrixStats *stats;

   HYPRE_UNIT_TEST_SETUP(comm, 1, test_comm, test_my_id);

   A     = CreateHostParCSRMatrix(4, 5, 7, row_ptr, col_ind, data);
   stats = hypre_MatrixStatsCreate();

   hypre_ParCSRMatrixStatsComputeLocal(A, stats);
   error += CheckStats("Test1_ParCSRMatrixStatsComputeLocal", stats,
                       4, 5, 7, 7, 65.0, 0, 4, 1.75, 8.75,
                       -4.0, 5.0, -0.25, 44.75,
                       0.0, 10.0, 4.75, 50.75, 0);

#if defined(HYPRE_USING_GPU)
   /* Re-run on device data to cover the native device reduction path. */
   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRMatrixStatsComputeLocal(A, stats);
   error += CheckStats("Test1_ParCSRMatrixStatsComputeLocal", stats,
                       4, 5, 7, 7, 65.0, 0, 4, 1.75, 8.75,
                       -4.0, 5.0, -0.25, 44.75,
                       0.0, 10.0, 4.75, 50.75, 0);
   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_HOST);
#endif

   hypre_MatrixStatsDestroy(stats);
   hypre_ParCSRMatrixDestroy(A);

   HYPRE_UNIT_PRINT_TEST_RESULT(test_my_id, error, test_comm);
   hypre_MPI_Comm_free(&test_comm);
   hypre_MPI_Barrier(comm);

   return error;
}

static HYPRE_Int
Test2_MatrixStatsReduce(MPI_Comm comm)
{
   MPI_Comm           test_comm = hypre_MPI_COMM_NULL;
   HYPRE_Int          test_my_id;
   HYPRE_Int          error = 0;
   hypre_ParCSRMatrix *A;
   hypre_MatrixStats *local_stats;
   hypre_MatrixStats *global_stats;

   HYPRE_UNIT_TEST_SETUP(comm, 2, test_comm, test_my_id);

   if (test_my_id == 0)
   {
      HYPRE_Int  row_ptr[3] = {0, 2, 3};
      HYPRE_Int  col_ind[3] = {0, 1, 2};
      HYPRE_Real data[3]    = {1.0, -2.0, 3.0};
      A = CreateHostParCSRMatrix(2, 5, 3, row_ptr, col_ind, data);
   }
   else
   {
      HYPRE_Int  row_ptr[4] = {0, 0, 1, 4};
      HYPRE_Int  col_ind[4] = {0, 1, 2, 3};
      HYPRE_Real data[4]    = {-4.0, 2.0, -3.0, 5.0};
      A = CreateHostParCSRMatrix(3, 5, 4, row_ptr, col_ind, data);
   }

   local_stats  = hypre_MatrixStatsCreate();
   global_stats = hypre_MatrixStatsCreate();

   hypre_ParCSRMatrixStatsComputeLocal(A, local_stats);
   hypre_MatrixStatsReduce(local_stats, global_stats, test_comm);

   error += CheckStats("Test2_MatrixStatsReduce", global_stats,
                       5, 5, 7, 7, 72.0, 0, 3, 1.4, 5.2,
                       -4.0, 4.0, 0.4, 41.2,
                       0.0, 10.0, 4.0, 54.0, 0);

   hypre_MatrixStatsDestroy(global_stats);
   hypre_MatrixStatsDestroy(local_stats);
   hypre_ParCSRMatrixDestroy(A);

   HYPRE_UNIT_PRINT_TEST_RESULT(test_my_id, error, test_comm);
   hypre_MPI_Comm_free(&test_comm);
   hypre_MPI_Barrier(comm);

   return error;
}

static HYPRE_Int
Test3_ParCSRMatrixStatsArrayCompute(MPI_Comm comm)
{
   MPI_Comm          test_comm = hypre_MPI_COMM_NULL;
   HYPRE_Int         test_my_id;
   HYPRE_Int         error = 0;
   HYPRE_Int         row_ptr[5] = {0, 2, 3, 3, 7};
   HYPRE_Int         col_ind[7] = {0, 1, 2, 0, 2, 3, 3};
   HYPRE_Real        data[7]    = {1.0, -3.0, 5.0, -1.0, -2.0, 3.0, -4.0};
   hypre_ParCSRMatrix *A;
   hypre_MatrixStatsArray *stats_array;

   HYPRE_UNIT_TEST_SETUP(comm, 1, test_comm, test_my_id);

   A = CreateHostParCSRMatrix(4, 4, 7, row_ptr, col_ind, data);
   stats_array = hypre_MatrixStatsArrayCreate(1);

   hypre_ParCSRMatrixStatsArrayCompute(1, &A, stats_array);
   error += CheckStats("Test3_ParCSRMatrixStatsArrayCompute",
                       hypre_MatrixStatsArrayEntry(stats_array, 0),
                       4, 4, 7, 7, 56.25, 0, 4, 1.75, 8.75,
                       -4.0, 5.0, -0.25, 44.75,
                       0.0, 10.0, 4.75, 50.75, 0);

#if defined(HYPRE_USING_GPU)
   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRMatrixStatsArrayCompute(1, &A, stats_array);
   error += CheckStats("Test3_ParCSRMatrixStatsArrayCompute",
                       hypre_MatrixStatsArrayEntry(stats_array, 0),
                       4, 4, 7, 7, 56.25, 0, 4, 1.75, 8.75,
                       -4.0, 5.0, -0.25, 44.75,
                       0.0, 10.0, 4.75, 50.75, 0);
   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_HOST);
#endif

   hypre_MatrixStatsArrayDestroy(stats_array);
   hypre_ParCSRMatrixDestroy(A);

   HYPRE_UNIT_PRINT_TEST_RESULT(test_my_id, error, test_comm);
   hypre_MPI_Comm_free(&test_comm);
   hypre_MPI_Barrier(comm);

   return error;
}

static HYPRE_Int
Test4_MatrixStatsReduceEmptyRank(MPI_Comm comm)
{
   MPI_Comm           test_comm = hypre_MPI_COMM_NULL;
   HYPRE_Int          test_my_id;
   HYPRE_Int          error = 0;
   hypre_ParCSRMatrix *A;
   hypre_MatrixStats *local_stats;
   hypre_MatrixStats *global_stats;

   HYPRE_UNIT_TEST_SETUP(comm, 3, test_comm, test_my_id);

   if (test_my_id == 0)
   {
      HYPRE_Int  row_ptr[3] = {0, 2, 3};
      HYPRE_Int  col_ind[3] = {0, 1, 2};
      HYPRE_Real data[3]    = {1.0, -2.0, 3.0};
      A = CreateHostParCSRMatrix(2, 5, 3, row_ptr, col_ind, data);
   }
   else if (test_my_id == 1)
   {
      HYPRE_Int  row_ptr[4] = {0, 0, 1, 4};
      HYPRE_Int  col_ind[4] = {0, 1, 2, 3};
      HYPRE_Real data[4]    = {-4.0, 2.0, -3.0, 5.0};
      A = CreateHostParCSRMatrix(3, 5, 4, row_ptr, col_ind, data);
   }
   else
   {
      HYPRE_Int row_ptr[1] = {0};
      A = CreateHostParCSRMatrix(0, 5, 0, row_ptr, NULL, NULL);
   }

   local_stats  = hypre_MatrixStatsCreate();
   global_stats = hypre_MatrixStatsCreate();

   hypre_ParCSRMatrixStatsComputeLocal(A, local_stats);
   hypre_MatrixStatsReduce(local_stats, global_stats, test_comm);

   error += CheckStats("Test4_MatrixStatsReduceEmptyRank", global_stats,
                       5, 5, 7, 7, 72.0, 0, 3, 1.4, 5.2,
                       -4.0, 4.0, 0.4, 41.2,
                       0.0, 10.0, 4.0, 54.0, 0);

   hypre_MatrixStatsDestroy(global_stats);
   hypre_MatrixStatsDestroy(local_stats);
   hypre_ParCSRMatrixDestroy(A);

   HYPRE_UNIT_PRINT_TEST_RESULT(test_my_id, error, test_comm);
   hypre_MPI_Comm_free(&test_comm);
   hypre_MPI_Barrier(comm);

   return error;
}

hypre_int
main(hypre_int argc, char *argv[])
{
   MPI_Comm  comm;
   HYPRE_Int my_id, num_procs;
   HYPRE_Int error = 0;

   hypre_MPI_Init(&argc, &argv);
   comm = hypre_MPI_COMM_WORLD;
   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

#if defined(HYPRE_USING_GPU)
   hypre_bind_device_id(-1, my_id, num_procs, comm);
#endif

   HYPRE_Initialize();

#if defined(HYPRE_USING_GPU)
   HYPRE_DeviceInitialize();
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#else
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
#endif

   if (my_id == 0)
   {
      hypre_printf("\n========================================\n");
      hypre_printf("Unit Tests for ParCSR Matrix Statistics\n");
      hypre_printf("========================================\n\n");
   }

   error += Test1_ParCSRMatrixStatsComputeLocal(comm);
   error += Test2_MatrixStatsReduce(comm);
   error += Test3_ParCSRMatrixStatsArrayCompute(comm);
   error += Test4_MatrixStatsReduceEmptyRank(comm);

   if (my_id == 0)
   {
      hypre_printf("\n");
      if (error == 0)
      {
         hypre_printf("All matrix statistics unit tests PASSED\n");
      }
      else
      {
         hypre_printf("Some matrix statistics unit tests FAILED (errors: %d)\n", error);
      }
   }

   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return error;
}
