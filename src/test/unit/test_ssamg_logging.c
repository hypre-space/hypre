/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Unit tests for SSAMG final residual logging.
 *--------------------------------------------------------------------------*/

#include <math.h>
#include <stdio.h>

#include "HYPRE.h"
#include "HYPRE_sstruct_ls.h"
#include "_hypre_utilities.h"
#include "hypre_unit.h"

#define TEST_NDIM 3
#define TEST_NX 8
#define TEST_NY 4
#define TEST_NZ 4
#define TEST_NSTENCIL 7
#define TEST_RESIDUAL_TOL hypre_max(1.0e-12, 100.0 * HYPRE_REAL_EPSILON)

typedef struct
{
   HYPRE_SStructGrid    grid;
   HYPRE_SStructStencil stencil;
   HYPRE_SStructGraph   graph;
   HYPRE_SStructMatrix  A;
   HYPRE_SStructVector  b;
   HYPRE_SStructVector  x;
   HYPRE_SStructVector  r;
   HYPRE_Int            ilower[TEST_NDIM];
   HYPRE_Int            iupper[TEST_NDIM];
   HYPRE_Int            global_upper[TEST_NDIM];
} SSAMGLoggingTestSystem;

static HYPRE_Int
SetMatrixBoundaryValue(SSAMGLoggingTestSystem *system,
                       HYPRE_Int               dim,
                       HYPRE_Int               entry,
                       HYPRE_Int               upper_boundary)
{
   HYPRE_Int boundary_lower[TEST_NDIM];
   HYPRE_Int boundary_upper[TEST_NDIM];
   HYPRE_Int boundary = upper_boundary ? system->global_upper[dim] : 0;
   HYPRE_Int local_boundary = upper_boundary ? system->iupper[dim] : system->ilower[dim];
   HYPRE_Int boundary_size = 1;
   HYPRE_Complex *zero_values;
   HYPRE_Int d;

   if (local_boundary != boundary)
   {
      return hypre_error_flag;
   }

   for (d = 0; d < TEST_NDIM; d++)
   {
      boundary_lower[d] = (d == dim) ? boundary : system->ilower[d];
      boundary_upper[d] = (d == dim) ? boundary : system->iupper[d];
      boundary_size *= boundary_upper[d] - boundary_lower[d] + 1;
   }
   zero_values = hypre_CTAlloc(HYPRE_Complex, boundary_size, HYPRE_MEMORY_HOST);
   HYPRE_SStructMatrixSetBoxValues(system->A, 0, boundary_lower, boundary_upper,
                                   0, 1, &entry, zero_values);
   hypre_TFree(zero_values, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

static HYPRE_Int
CreateLoggingTestSystem(MPI_Comm comm, SSAMGLoggingTestSystem *system)
{
   HYPRE_SStructVariable vartype = HYPRE_SSTRUCT_VARIABLE_CELL;
   HYPRE_Int offsets[TEST_NSTENCIL][TEST_NDIM] =
   {
      { 0,  0,  0}, {-1,  0,  0}, { 1,  0,  0},
      { 0, -1,  0}, { 0,  1,  0}, { 0,  0, -1}, { 0,  0,  1}
   };
   HYPRE_Int stencil_entries[TEST_NSTENCIL] = {0, 1, 2, 3, 4, 5, 6};
   HYPRE_Complex matrix_values[TEST_NSTENCIL] = {7.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
   HYPRE_Int boundary_entries[6] = {1, 2, 3, 4, 5, 6};
   HYPRE_Real one_value = 1.0;
   HYPRE_Real zero_value = 0.0;
   HYPRE_Int local_size;
   HYPRE_Int values_size;
   HYPRE_Int i, d, rank, num_procs;
   HYPRE_Int base, remainder;

   hypre_MPI_Comm_rank(comm, &rank);
   hypre_MPI_Comm_size(comm, &num_procs);

   system->global_upper[0] = TEST_NX - 1;
   system->global_upper[1] = TEST_NY - 1;
   system->global_upper[2] = TEST_NZ - 1;

   base = TEST_NX / num_procs;
   remainder = TEST_NX % num_procs;
   system->ilower[0] = rank * base + ((rank < remainder) ? rank : remainder);
   system->iupper[0] = system->ilower[0] + base - 1 + ((rank < remainder) ? 1 : 0);
   system->ilower[1] = 0;
   system->iupper[1] = TEST_NY - 1;
   system->ilower[2] = 0;
   system->iupper[2] = TEST_NZ - 1;

   HYPRE_SStructGridCreate(comm, TEST_NDIM, 1, &system->grid);
   HYPRE_SStructGridSetExtents(system->grid, 0, system->ilower, system->iupper);
   HYPRE_SStructGridSetVariables(system->grid, 0, 1, &vartype);
   HYPRE_SStructGridAssemble(system->grid);

   HYPRE_SStructStencilCreate(TEST_NDIM, TEST_NSTENCIL, &system->stencil);
   for (i = 0; i < TEST_NSTENCIL; i++)
   {
      HYPRE_SStructStencilSetEntry(system->stencil, i, offsets[i], 0);
   }

   HYPRE_SStructGraphCreate(comm, system->grid, &system->graph);
   HYPRE_SStructGraphSetObjectType(system->graph, HYPRE_SSTRUCT);
   HYPRE_SStructGraphSetStencil(system->graph, 0, 0, system->stencil);
   HYPRE_SStructGraphAssemble(system->graph);

   HYPRE_SStructMatrixCreate(comm, system->graph, &system->A);
   HYPRE_SStructMatrixSetObjectType(system->A, HYPRE_SSTRUCT);
   HYPRE_SStructMatrixInitialize(system->A);

   local_size = (system->iupper[0] - system->ilower[0] + 1) *
                (system->iupper[1] - system->ilower[1] + 1) *
                (system->iupper[2] - system->ilower[2] + 1);
   values_size = TEST_NSTENCIL * local_size;
   {
      HYPRE_Complex *values = hypre_TAlloc(HYPRE_Complex, values_size, HYPRE_MEMORY_HOST);
      HYPRE_Int j;

      for (i = 0; i < values_size; i += TEST_NSTENCIL)
      {
         for (j = 0; j < TEST_NSTENCIL; j++)
         {
            values[i + j] = matrix_values[j];
         }
      }
      HYPRE_SStructMatrixSetBoxValues(system->A, 0, system->ilower, system->iupper,
                                      0, TEST_NSTENCIL, stencil_entries, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }

   for (d = 0; d < TEST_NDIM; d++)
   {
      SetMatrixBoundaryValue(system, d, boundary_entries[2 * d], 0);
      SetMatrixBoundaryValue(system, d, boundary_entries[2 * d + 1], 1);
   }
   HYPRE_SStructMatrixAssemble(system->A);

   HYPRE_SStructVectorCreate(comm, system->grid, &system->b);
   HYPRE_SStructVectorSetObjectType(system->b, HYPRE_SSTRUCT);
   HYPRE_SStructVectorInitialize(system->b);
   {
      HYPRE_Int nvalues = local_size;
      HYPRE_Complex *values = hypre_TAlloc(HYPRE_Complex, nvalues, HYPRE_MEMORY_HOST);

      for (i = 0; i < nvalues; i++)
      {
         values[i] = one_value;
      }
      HYPRE_SStructVectorSetBoxValues(system->b, 0, system->ilower, system->iupper,
                                      0, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }
   HYPRE_SStructVectorAssemble(system->b);

   HYPRE_SStructVectorCreate(comm, system->grid, &system->x);
   HYPRE_SStructVectorSetObjectType(system->x, HYPRE_SSTRUCT);
   HYPRE_SStructVectorInitialize(system->x);
   HYPRE_SStructVectorSetConstantValues(system->x, zero_value);
   HYPRE_SStructVectorAssemble(system->x);

   HYPRE_SStructVectorCreate(comm, system->grid, &system->r);
   HYPRE_SStructVectorSetObjectType(system->r, HYPRE_SSTRUCT);
   HYPRE_SStructVectorInitialize(system->r);
   HYPRE_SStructVectorSetConstantValues(system->r, zero_value);
   HYPRE_SStructVectorAssemble(system->r);

   return hypre_error_flag;
}

static HYPRE_Int
DestroyLoggingTestSystem(SSAMGLoggingTestSystem *system)
{
   HYPRE_SStructVectorDestroy(system->r);
   HYPRE_SStructVectorDestroy(system->x);
   HYPRE_SStructVectorDestroy(system->b);
   HYPRE_SStructMatrixDestroy(system->A);
   HYPRE_SStructGraphDestroy(system->graph);
   HYPRE_SStructStencilDestroy(system->stencil);
   HYPRE_SStructGridDestroy(system->grid);

   return hypre_error_flag;
}

static HYPRE_Int
ComputeRelativeResidual(SSAMGLoggingTestSystem *system, HYPRE_Real *relative_residual)
{
   HYPRE_Real b_dot_b, r_dot_r;

   HYPRE_SStructVectorCopy(system->b, system->r);
   HYPRE_SStructMatrixMatvec(-1.0, system->A, system->x, 1.0, system->r);
   HYPRE_SStructVectorInnerProd(system->b, system->b, &b_dot_b);
   HYPRE_SStructVectorInnerProd(system->r, system->r, &r_dot_r);
   *relative_residual = hypre_sqrt(r_dot_r / b_dot_b);

   return hypre_error_flag;
}

static HYPRE_Int
RunLoggingCase(MPI_Comm comm,
               SSAMGLoggingTestSystem *system,
               HYPRE_Real              tol,
               HYPRE_Int               max_iter,
               HYPRE_Int               logging,
               HYPRE_Int               expected_iterations,
               HYPRE_Int               require_early_exit,
               HYPRE_Int               check_residual,
               HYPRE_Int               check_argument_error)
{
   HYPRE_SStructSolver solver;
   HYPRE_Real reported = -1.0;
   HYPRE_Real expected = -1.0;
   HYPRE_Int iterations = -1;
   HYPRE_Int error = 0;

   HYPRE_ClearAllErrors();
   HYPRE_SStructVectorSetConstantValues(system->x, 0.0);
   HYPRE_SStructVectorAssemble(system->x);

   HYPRE_SStructSSAMGCreate(comm, &solver);
   HYPRE_SStructSSAMGSetMaxLevels(solver, 2);
   HYPRE_SStructSSAMGSetMaxIter(solver, max_iter);
   HYPRE_SStructSSAMGSetTol(solver, tol);
   HYPRE_SStructSSAMGSetRelChange(solver, 0);
   HYPRE_SStructSSAMGSetNumPreRelax(solver, 1);
   HYPRE_SStructSSAMGSetNumPostRelax(solver, 1);
   HYPRE_SStructSSAMGSetNumCoarseRelax(solver, 1);
   HYPRE_SStructSSAMGSetLogging(solver, logging);
   HYPRE_SStructSSAMGSetPrintLevel(solver, 0);
   HYPRE_SStructSSAMGSetPrintFreq(solver, 1);
   HYPRE_SStructSSAMGSetZeroGuess(solver);
   HYPRE_SStructSSAMGSetup(solver, system->A, system->b, system->x);
   HYPRE_SStructSSAMGSolve(solver, system->A, system->b, system->x);
   HYPRE_SStructSSAMGGetNumIterations(solver, &iterations);
   HYPRE_UNIT_CHECK_INT(error, "solve error", HYPRE_GetError(), 0);

   if (expected_iterations >= 0)
   {
      HYPRE_UNIT_CHECK_INT(error, "num_iterations", iterations, expected_iterations);
   }
   if (require_early_exit && (iterations >= max_iter))
   {
      hypre_printf("  early exit mismatch: got %d iterations with max_iter %d\n",
                   iterations, max_iter);
      error = 1;
   }

   if (check_residual)
   {
      HYPRE_SStructSSAMGGetFinalRelativeResidualNorm(solver, &reported);
      ComputeRelativeResidual(system, &expected);
      if (!isfinite(reported) || !isfinite(expected))
      {
         error = 1;
      }
      HYPRE_UNIT_CHECK_REAL(error, "reported residual", reported, expected, TEST_RESIDUAL_TOL);
   }
   else if ((logging > 0) && !check_argument_error)
   {
      HYPRE_SStructSSAMGGetFinalRelativeResidualNorm(solver, &reported);
      HYPRE_UNIT_CHECK_REAL(error, "disabled residual", reported, 0.0, TEST_RESIDUAL_TOL);
   }

   if (check_argument_error)
   {
      HYPRE_ClearAllErrors();
      HYPRE_SStructSSAMGGetFinalRelativeResidualNorm(solver, &reported);
      HYPRE_UNIT_CHECK_INT(error, "argument error", (HYPRE_GetError() & HYPRE_ERROR_ARG) != 0, 1);
      HYPRE_ClearAllErrors();
   }

   if (logging == 0)
   {
      HYPRE_SStructSSAMGGetFinalRelativeResidualNorm(solver, &reported);
      HYPRE_UNIT_CHECK_REAL(error, "logging disabled", reported, -1.0, TEST_RESIDUAL_TOL);
   }

   HYPRE_SStructSSAMGDestroy(solver);
   HYPRE_UNIT_CHECK_INT(error, "HYPRE error", HYPRE_GetError(), 0);
   return error;
}

static HYPRE_Int
TestSSAMGLogging(MPI_Comm comm)
{
   SSAMGLoggingTestSystem system;
   HYPRE_Int error = 0;
   HYPRE_Int test_my_id, num_procs, test_num_procs;
   MPI_Comm test_comm = hypre_MPI_COMM_NULL;

   hypre_MPI_Comm_size(comm, &num_procs);
   test_num_procs = hypre_min(num_procs, TEST_NX);
#ifdef HYPRE_SEQUENTIAL
   test_comm = comm;
   test_my_id = 0;
   hypre_printf("%s (%d procs): ", __func__, test_num_procs);
#else
   HYPRE_UNIT_TEST_SETUP(comm, test_num_procs, test_comm, test_my_id);
#endif
   CreateLoggingTestSystem(test_comm, &system);
   HYPRE_UNIT_CHECK_INT(error, "system setup error", HYPRE_GetError(), 0);

   error += RunLoggingCase(test_comm, &system, 1.0e-12, 1, 1, 1, 0, 1, 0);
   error += RunLoggingCase(test_comm, &system, 1.0e-12, 2, 1, 2, 0, 1, 0);
   error += RunLoggingCase(test_comm, &system, 1.0e-3, 100, 1, -1, 1, 1, 0);
   error += RunLoggingCase(test_comm, &system, 1.0e-12, 1, 0, 1, 0, 0, 0);
   error += RunLoggingCase(test_comm, &system, 0.0, 1, 1, 1, 0, 0, 0);
   error += RunLoggingCase(test_comm, &system, 1.0e-6, 0, 1, 0, 0, 0, 1);

   DestroyLoggingTestSystem(&system);
   HYPRE_UNIT_PRINT_TEST_RESULT(test_my_id, error, test_comm);
#ifndef HYPRE_SEQUENTIAL
   hypre_MPI_Comm_free(&test_comm);
#endif
   hypre_MPI_Barrier(comm);

   return error;
}

hypre_int
main(hypre_int argc, char *argv[])
{
   MPI_Comm comm;
   HYPRE_Int my_id, global_error;
   HYPRE_Int error = 0;

   hypre_MPI_Init(&argc, &argv);
   comm = hypre_MPI_COMM_WORLD;
   hypre_MPI_Comm_rank(comm, &my_id);
   HYPRE_Initialize();
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);

   if (my_id == 0)
   {
      hypre_printf("\n========================================\n");
      hypre_printf("Unit Tests for SSAMG Logging\n");
      hypre_printf("========================================\n\n");
   }

   error += TestSSAMGLogging(comm);
   hypre_MPI_Allreduce(&error, &global_error, 1, HYPRE_MPI_INT, hypre_MPI_MAX, comm);
   error = global_error;

   if (my_id == 0)
   {
      hypre_printf("\n");
      if (error == 0)
      {
         hypre_printf("All SSAMG logging unit tests PASSED\n");
      }
      else
      {
         hypre_printf("Some SSAMG logging unit tests FAILED (errors: %d)\n", error);
      }
   }

   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return error;
}
