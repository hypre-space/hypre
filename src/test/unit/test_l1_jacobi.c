/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Unit tests for l1-Jacobi relaxation (relax type 18).
 *--------------------------------------------------------------------------*/

#include "HYPRE.h"
#include "_hypre_parcsr_ls.h"
#include "hypre_unit.h"

static hypre_ParCSRMatrix *
CreateHostParCSRMatrix(HYPRE_Int         num_rows,
                       HYPRE_Int         num_nonzeros,
                       const HYPRE_Int  *row_ptr,
                       const HYPRE_Int  *col_ind,
                       const HYPRE_Real *data)
{
   HYPRE_BigInt        row_starts[2] = {0, num_rows};
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag;

   A = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_SELF, num_rows, num_rows,
                                row_starts, row_starts, 0, num_nonzeros, 0);
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

   return A;
}

static hypre_ParVector *
CreateHostParVector(HYPRE_Int size, const HYPRE_Real *data)
{
   HYPRE_BigInt     partitioning[2] = {0, size};
   hypre_ParVector *x;

   x = hypre_ParVectorCreate(hypre_MPI_COMM_SELF, size, partitioning);
   hypre_ParVectorInitialize_v2(x, HYPRE_MEMORY_HOST);

   for (HYPRE_Int i = 0; i < size; i++)
   {
      hypre_VectorData(hypre_ParVectorLocalVector(x))[i] = data[i];
   }

   return x;
}

/*--------------------------------------------------------------------------
 * One sweep of l1-Jacobi for all points on A = [2 -1 0; -1 2 0; 0 0 0], whose
 * last row is zero, with f = [3 3 0] and u = [0 0 1]. The point of the zero row
 * must be left unchanged: u = [1 1 1].
 *--------------------------------------------------------------------------*/

static HYPRE_Int
L1JacobiZeroRow(HYPRE_MemoryLocation memory_location)
{
   HYPRE_Int           error = 0;
   HYPRE_Int           row_ptr[4] = {0, 2, 4, 5};
   HYPRE_Int           col_ind[5] = {0, 1, 1, 0, 2};
   HYPRE_Real          data[5]    = {2.0, -1.0, 2.0, -1.0, 0.0};
   HYPRE_Real          f_data[3]  = {3.0, 3.0, 0.0};
   HYPRE_Real          u_data[3]  = {0.0, 0.0, 1.0};
   HYPRE_Real         *l1_norms   = NULL;
   HYPRE_Real         *u_local;
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *f, *u, *Vtemp;

   A     = CreateHostParCSRMatrix(3, 5, row_ptr, col_ind, data);
   f     = CreateHostParVector(3, f_data);
   u     = CreateHostParVector(3, u_data);
   Vtemp = hypre_ParVectorCreate(hypre_MPI_COMM_SELF, 3, hypre_ParVectorPartitioning(f));
   hypre_ParVectorInitialize_v2(Vtemp, memory_location);
   hypre_ParCSRMatrixMigrate(A, memory_location);
   hypre_ParVectorMigrate(f, memory_location);
   hypre_ParVectorMigrate(u, memory_location);

   hypre_ParCSRComputeL1Norms(A, 1, NULL, &l1_norms);
   hypre_BoomerAMGRelax(A, f, NULL, 18, 0, 1.0, 1.0, l1_norms, u, Vtemp, NULL);

   hypre_ParVectorMigrate(u, HYPRE_MEMORY_HOST);
   u_local = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_UNIT_CHECK_REAL(error, "u[0]", u_local[0], 1.0, HYPRE_REAL_EPSILON);
   HYPRE_UNIT_CHECK_REAL(error, "u[1]", u_local[1], 1.0, HYPRE_REAL_EPSILON);
   HYPRE_UNIT_CHECK_REAL(error, "u[2]", u_local[2], 1.0, HYPRE_REAL_EPSILON);

   hypre_TFree(l1_norms, memory_location);
   hypre_ParVectorDestroy(Vtemp);
   hypre_ParVectorDestroy(u);
   hypre_ParVectorDestroy(f);
   hypre_ParCSRMatrixDestroy(A);

   return error;
}

static HYPRE_Int
Test1_L1JacobiZeroRow(MPI_Comm comm)
{
   MPI_Comm  test_comm = hypre_MPI_COMM_NULL;
   HYPRE_Int test_my_id;
   HYPRE_Int error = 0;

   HYPRE_UNIT_TEST_SETUP(comm, 1, test_comm, test_my_id);

   error += L1JacobiZeroRow(HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
   error += L1JacobiZeroRow(HYPRE_MEMORY_DEVICE);
#endif

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
      hypre_printf("Unit Tests for l1-Jacobi Relaxation\n");
      hypre_printf("========================================\n\n");
   }

   error += Test1_L1JacobiZeroRow(comm);

   if (my_id == 0)
   {
      hypre_printf("\n");
      if (error == 0)
      {
         hypre_printf("All l1-Jacobi unit tests PASSED\n");
      }
      else
      {
         hypre_printf("Some l1-Jacobi unit tests FAILED (errors: %d)\n", error);
      }
   }

   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return error;
}
