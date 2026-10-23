/* Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. SPDX-License-Identifier: (Apache-2.0 OR MIT) */

/* Independent scalar sparse-pattern ILU0 oracle. No GPU permutation algorithm
 * is used by the oracle: its coordinates are the initial factor ordering. */
#include "_hypre_parcsr_ls.h"
#include <math.h>
#include <fenv.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if defined(HYPRE_USING_HIP)
#include <hip/hip_runtime_api.h>
#include "_hypre_utilities.hpp"
#elif defined(HYPRE_USING_CUDA)
#include <cuda_runtime_api.h>
#include "_hypre_utilities.hpp"
#endif

#define ORACLE_TOL (sizeof(HYPRE_Real) == sizeof(float) ? 2e-5 : 1e-11)

static HYPRE_Int failures;
static int reverse_input_columns;
static void check(int ok, const char *what, int n, unsigned seed)
{
   if (!ok)
   {
      printf("FAILED: %s n=%d seed=%u\n", what, n, seed);
      failures++;
   }
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
/* Context for numerical assertions; ordinary oracle cases keep one stream. */
static int case_ordering, case_path = -1, case_rhs = -1;
static void numerical_check(double actual, double expected, double tolerance,
                            const char *what, int n, unsigned seed, int row)
{
   double error = fabs(actual - expected);
   if (!isfinite(actual) || error > tolerance)
   {
      check(0, what, n, seed);
      printf("  ordering=%d path=%s rhs=%d stream=%d row=%d "
             "actual=%.17g expected=%.17g error=%.17g limit=%.17g\n",
             case_ordering, case_path < 0 ? "setup" :
             (case_path ? "single-block" : "level-set"), case_rhs,
             hypre_HandleComputeStreamNum(hypre_handle()),
             row, actual, expected, error, tolerance);
   }
}

static unsigned next(unsigned *s)
{
   *s = 1664525u * *s + 1013904223u;
   return *s;
}

#endif

static hypre_ParCSRMatrix *matrix(int n, const double *a, const int *mask)
{
   int i, j, k = 0, nnz = 0;
   HYPRE_BigInt starts[2] = {0, n};
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *D;
   for (i = 0; i < n * n; i++) { nnz += mask[i]; }
   A = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_SELF, n, n, starts, starts, 0, nnz, 0);
   hypre_ParCSRMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   D = hypre_ParCSRMatrixDiag(A);
   for (i = 0; i < n; i++)
   {
      hypre_CSRMatrixI(D)[i] = k;
      /* Exercise both numeric ordering and valid unsorted input. */
      for (j = 0; j < n; j++)
      {
         int column = reverse_input_columns ? n - 1 - j : j;
         if (mask[i * n + column])
         {
            hypre_CSRMatrixJ(D)[k] = column;
            hypre_CSRMatrixData(D)[k++] = a[i * n + column];
         }
      }
   }
   hypre_CSRMatrixI(D)[n] = k;
   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   return A;
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
static hypre_ParVector *vector(int n)
{
   HYPRE_BigInt starts[2] = {0, n};
   hypre_ParVector *v = hypre_ParVectorCreate(hypre_MPI_COMM_SELF, n, starts);
   hypre_ParVectorInitialize_v2(v, HYPRE_MEMORY_DEVICE);
   return v;
}

static void schedule(hypre_CSRMatrix *F, int levels, HYPRE_Int *offsets, HYPRE_Int *drows,
                     int lower, unsigned seed)
{
   int n = hypre_CSRMatrixNumRows(F), i, l, k;
   HYPRE_Int *rows = (HYPRE_Int *) calloc(n ? n : 1, sizeof(HYPRE_Int));
   int *seen = (int *) calloc(n ? n : 1, sizeof(int));
   int *level = (int *) calloc(n ? n : 1, sizeof(int));
   hypre_TMemcpy(rows, drows, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   check(offsets[0] == 0 && offsets[levels] == n, "schedule extent", n, seed);
   for (l = 0; l < levels; l++)
   {
      for (k = offsets[l]; k < offsets[l + 1]; k++)
      {
         check(rows[k] >= 0 && rows[k] < n, "schedule row bounds", n, seed);
         if (rows[k] >= 0 && rows[k] < n) { seen[rows[k]]++; level[rows[k]] = l; }
      }
   }
   for (i = 0; i < n; i++)
   {
      check(seen[i] == 1, "schedule covers each row once", n, seed);
      for (k = hypre_CSRMatrixI(F)[i]; k < hypre_CSRMatrixI(F)[i + 1]; k++)
      {
         int j = hypre_CSRMatrixJ(F)[k];
         if ((lower && j < i) || (!lower && j > i))
         { check(level[j] < level[i], "dependency precedes row", n, seed); }
      }
   }
   free(rows); free(seen); free(level);
}

static void ordered_diagonal(void)
{
   /* First/middle/last diagonals and irregular rows, with distinct values. */
   double a[16] = {4, 1, 2, 3, 0, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14};
   int mask[16] = {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1};
   int variant, i, p;
   for (variant = 0; variant < 2; variant++)
   {
      hypre_ParCSRMatrix *A = matrix(4, a, mask);
      hypre_CSRMatrix *D = hypre_ParCSRMatrixDiag(A), *h;
      if (variant) { hypre_CSRMatrixMoveDiagFirstOrderedDevice(D); }
      else { hypre_CSRMatrixMoveDiagFirstDevice(D); }
      h = hypre_CSRMatrixClone_v2(D, 1, HYPRE_MEMORY_HOST);
      for (i = 0; i < 4; i++)
      {
         check(hypre_CSRMatrixJ(h)[hypre_CSRMatrixI(h)[i]] == i, "diagonal moved first", 4, 0);
         for (p = hypre_CSRMatrixI(h)[i]; p < hypre_CSRMatrixI(h)[i + 1]; p++)
         {
            int j = hypre_CSRMatrixJ(h)[p];
            check(hypre_CSRMatrixData(h)[p] == a[i * 4 + j], "diagonal helper preserves values", 4, 0);
            if (variant && p > hypre_CSRMatrixI(h)[i] + 1)
            {
               check(hypre_CSRMatrixJ(h)[p - 1] < j, "ordered off-diagonals", 4, 0);
            }
         }
      }
      hypre_CSRMatrixDestroy(h); hypre_ParCSRMatrixDestroy(A);
      A = matrix(0, a, mask);
      hypre_CSRMatrixMoveDiagFirstOrderedDevice(hypre_ParCSRMatrixDiag(A));
      hypre_ParCSRMatrixDestroy(A);
   }
}

static void run(HYPRE_Solver solver, int n, double *a, int *mask,
                int ordering, int invalid, unsigned seed);

/* Account live allocations, not reserved device pool capacity. All test-owned
 * host buffers use calloc/free and are excluded. Requires memory tracker build. */
static size_t live_allocations(void)
{
#if defined(HYPRE_USING_MEMORY_TRACKER)
   hypre_MemoryTracker *t = hypre_memory_tracker();
   return t->queue[HYPRE_MEMORY_EVENT_ALLOC].actual_size -
          t->queue[HYPRE_MEMORY_EVENT_FREE].actual_size;
#else
   return 0;
#endif
}

/* Sparse levels of widths 2,1,1,1 form one scalar factor batch. A failed
 * middle-level row must not strand inactive threads at a block barrier, leak
 * the early-owned offsets, or prevent a later setup on the same solver. */
static void factor_pivot_ownership(void)
{
   HYPRE_Solver solver;
   double a[25] = {1, 0, .5, 0, 0,
                   0, 4, 0, 0, 0,
                   2, 0, 4, .02, 0,
                   0, 0, .01, 4, .02,
                   0, 0, 0, .01, 4
                  };
   int mask[25] = {1, 0, 1, 0, 0,
                   0, 1, 0, 0, 0,
                   1, 0, 1, 1, 0,
                   0, 0, 1, 1, 1,
                   0, 0, 0, 1, 1
                  };
   size_t before, stable;
   int rep;
   HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
   HYPRE_ILUSetTol(solver, 0); HYPRE_ILUSetMaxIter(solver, 1);
   /* Strict diagonal dominance permits the float oracle bound, while
    * 2 * .5 / 1 == 1 preserves the exact elimination-created zero below. */
   run(solver, 5, a, mask, 0, 0, 74126);
   a[12] = 1; /* Exactly zero after eliminating the row-0 unit pivot. */
   run(solver, 5, a, mask, 0, 1, 74127);
   HYPRE_ILUDestroy(solver);
   before = live_allocations();
   HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
   HYPRE_ILUSetTol(solver, 0); HYPRE_ILUSetMaxIter(solver, 1);
   run(solver, 5, a, mask, 0, 1, 74128);
   stable = live_allocations();
   for (rep = 0; rep < 16; rep++)
   {
      run(solver, 5, a, mask, 0, 1, 74129 + rep);
      check(live_allocations() == stable, "failed factorization releases schedules", 5, rep);
   }
   a[12] = NAN;
   run(solver, 5, a, mask, 0, 1, 74146);
   check(live_allocations() == stable, "nonfinite factorization releases schedules", 5, 0);
   a[12] = 4;
   run(solver, 5, a, mask, 0, 0, 74147);
   HYPRE_ILUDestroy(solver);
   check(live_allocations() == before, "factorization recovery destruction returns allocations", 5, 0);
}

/* Stored zeros are structural dependencies and must remain finite under
 * elimination, including when a zero multiplier updates its diagonal. */
static void stored_zero_pattern(HYPRE_Solver solver)
{
   double a[49] = {0}; int mask[49] = {0};
   const struct { int row, col; double value; } entries[] = {
      {0,0,2.1469999999999998}, {0,2,-.041999999999999996}, {0,3,-.105}, {1,1,2},
      {2,0,-.059999999999999998}, {2,2,2.1319999999999997}, {2,4,0},
      {2,5,.029999999999999999}, {2,6,-.041999999999999996},
      {3,0,-.14999999999999999}, {3,3,2.2969999999999997}, {3,5,-.14699999999999999},
      {4,2,0}, {4,4,2.0840000000000001}, {4,6,-.083999999999999991},
      {5,2,.020999999999999998}, {5,3,-.20999999999999999}, {5,5,2.399}, {5,6,-.16799999999999998},
      {6,2,-.059999999999999998}, {6,4,-.12}, {6,5,-.23999999999999999}, {6,6,2.4199999999999999}
   };
   for (size_t i = 0; i < sizeof(entries)/sizeof(entries[0]); i++)
   {
      int p = entries[i].row * 7 + entries[i].col;
      a[p] = entries[i].value; mask[p] = 1;
   }
   run(solver, 7, a, mask, 0, 0, 62031);
   run(solver, 7, a, mask, 1, 0, 62031);
}

/* Failed RCM setup followed by another type and size must not reuse stale
 * permutation coordinates. Check ownership before attempting an unsafe gather
 * on old implementations, then exercise the complete transition numerically. */
static void failed_ordering_transition(void)
{
   HYPRE_Solver solver;
   double a[9] = {1, 1, 0, 1, 1, 0, 0, 0, 4};
   int mask[9] = {1, 1, 0, 1, 1, 0, 0, 0, 1};
   HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
   HYPRE_ILUSetTol(solver, 0); HYPRE_ILUSetMaxIter(solver, 1);
   run(solver, 3, a, mask, 1, 1, 75026);
   hypre_ParILUData *data = (hypre_ParILUData *) solver;
   int retired = !hypre_ParILUDataPerm(data) && !hypre_ParILUDataQPerm(data);
   check(retired, "failed RCM setup retires permutation before a changed-type setup", 3, 75026);
   if (!retired)
   {
      HYPRE_ILUDestroy(solver);
      return; /* Do not attempt an out-of-bounds old-coordinate gather. */
   }
   if (retired)
   {
      double diagonal[25] = {0}; int pattern[25] = {0};
      for (int i = 0; i < 5; i++) { diagonal[i * 5 + i] = i + 2; pattern[i * 5 + i] = 1; }
      hypre_ParCSRMatrix *A = matrix(5, diagonal, pattern);
      hypre_ParVector *b = vector(5), *u = vector(5);
      HYPRE_Complex x[5];
      hypre_ParVectorSetConstantValues(b, 1); hypre_ParVectorSetConstantValues(u, 0);
      HYPRE_ILUSetType(solver, 0);
      check(!HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                            (HYPRE_ParVector) u), "type 0 setup after failed RCM with changed size", 5, 75027);
      check(!HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                            (HYPRE_ParVector) u), "type 0 solve after failed RCM with changed size", 5, 75027);
      hypre_TMemcpy(x, hypre_VectorData(hypre_ParVectorLocalVector(u)), HYPRE_Complex, 5,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (int i = 0; i < 5; i++)
      { numerical_check(x[i], 1.0 / (i + 2), ORACLE_TOL * 2, "changed-type diagonal correction", 5, 75027, i); }
      hypre_ParCSRMatrixDestroy(A); hypre_ParVectorDestroy(b); hypre_ParVectorDestroy(u);
      HYPRE_ILUSetType(solver, 60);
      a[0] = a[4] = 4;
      run(solver, 3, a, mask, 1, 0, 75028);
   }
   /* Default allocation policy may change independently of factor ownership. */
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   check(!HYPRE_ILUDestroy(solver), "device permutation cleanup after changing default memory", 3,
         75029);
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
}

static void ownership(void)
{
   HYPRE_Solver solver;
   int rep;
   double a[9] = {4, 0, 0, 0, 4, 0, 0, 0, 4}; int mask[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
   size_t before, stable = 0;
   /* Warm handle/library caches before measuring solver ownership. */
   HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
   HYPRE_ILUSetTol(solver, 0); HYPRE_ILUSetMaxIter(solver, 1);
   run(solver, 3, a, mask, 0, 0, 0); HYPRE_ILUDestroy(solver);
   before = live_allocations();
   HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
   HYPRE_ILUSetTol(solver, 0); HYPRE_ILUSetMaxIter(solver, 1);
   for (rep = 0; rep < 105; rep++)
   {
      run(solver, 3, a, mask, 0, 0, 0);
      if (rep == 4)
      {
         stable = live_allocations();
      }
      if (rep > 4) { check(live_allocations() == stable, "stable live allocations after setup", 3, rep); }
   }
   HYPRE_ILUDestroy(solver);
   check(live_allocations() == before, "destruction returns allocation count to baseline", 3, 0);
#if !defined(HYPRE_USING_MEMORY_TRACKER)
   printf("PENDING: allocation accounting requires HYPRE_ENABLE_MEMORY_TRACKER=ON\n");
#endif
}

static void run(HYPRE_Solver solver, int n, double *a, int *mask,
                int ordering, int invalid, unsigned seed)
{
   hypre_ParILUData *data = (hypre_ParILUData *) solver;
   hypre_ParCSRMatrix *A = matrix(n, a, mask);
   hypre_ParVector *b = vector(n), *u = vector(n);
   int i, j, k, rhs, path, error, before = failures;
   HYPRE_Int *perm = (HYPRE_Int *) calloc(n ? n : 1, sizeof(HYPRE_Int));
   HYPRE_Int *storage = (HYPRE_Int *) calloc(n ? n : 1, sizeof(HYPRE_Int));
   double *f = (double *) calloc(n * n ? n*n : 1, sizeof(double));
   int *pattern = (int *) calloc(n * n ? n*n : 1, sizeof(int));
   HYPRE_Complex *x = (HYPRE_Complex *) calloc(n ? n : 1, sizeof(HYPRE_Complex));
   HYPRE_Complex *initial = (HYPRE_Complex *) calloc(n ? n : 1, sizeof(HYPRE_Complex));
   HYPRE_Complex *bv = (HYPRE_Complex *) calloc(n ? n : 1, sizeof(HYPRE_Complex));
   double *expected = (double *) calloc(n ? n : 1, sizeof(double));
   hypre_CSRMatrix *F = NULL;
   double tolerance = 1e-11;
   case_ordering = ordering; case_path = case_rhs = -1;
   HYPRE_ILUSetLocalReordering(solver, ordering);
   error = HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix) A,
                          (HYPRE_ParVector) b, (HYPRE_ParVector) u);
   if (invalid)
   {
      check(error != 0, "invalid input rejected", n, seed);
      HYPRE_ClearAllErrors();
      check(HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix) A,
                           (HYPRE_ParVector) b, (HYPRE_ParVector) u) != 0,
            "solve after failed setup rejected", n, seed);
      HYPRE_ClearAllErrors();
      goto done;
   }
   check(error == 0, "setup", n, seed);
   if (error) { HYPRE_ClearAllErrors(); goto done; }
   if (!n)
   {
      check(!HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix) A,
                            (HYPRE_ParVector) b, (HYPRE_ParVector) u), "empty solve", n, seed);
      goto done;
   }
   {
      int symmetric = 1;
      for (i = 0; i < n; i++) for (j = 0; j < n; j++)
         {
            symmetric &= mask[i * n + j] == mask[j * n + i];
         }
      if (!symmetric && !ordering)
         for (i = 0; i < n; i++)
         {
            HYPRE_Int entry;
            hypre_TMemcpy(&entry, hypre_ParILUDataCombinedPermD(data) + i, HYPRE_Int, 1,
                          HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
            check(entry == i, "asymmetric factor retains initial order", n, seed);
         }
   }
   if (sizeof(HYPRE_Real) == sizeof(float))
   {
      /* Strict row diagonal dominance bounds kappa_inf(A) by max row sum
       * divided by the smallest dominance margin. gamma_n bounds accumulated
       * floating-point rounding in these bounded scalar oracle cases. */
      double max_sum = 0, min_margin = HUGE_VAL;
      double gamma = n * (double) HYPRE_REAL_EPSILON;
      for (i = 0; i < n; i++)
      {
         double off = 0;
         for (j = 0; j < n; j++) { if (j != i) { off += fabs(a[i * n + j]); } }
         if (fabs(a[i * n + i]) + off > max_sum) { max_sum = fabs(a[i * n + i]) + off; }
         if (fabs(a[i * n + i]) - off < min_margin) { min_margin = fabs(a[i * n + i]) - off; }
      }
      int bounded = isfinite(max_sum) && isfinite(min_margin) &&
                    min_margin > 0 && gamma < 1;
      check(bounded, "oracle conditioning bound", n, seed);
      /* A failed bound is a test failure; do not manufacture a negative or
       * infinite tolerance and then report misleading numerical failures. */
      if (!bounded) { goto done; }
      tolerance = 8 * gamma / (1 - gamma) * max_sum / min_margin;
   }
   if (hypre_ParILUDataPerm(data))
   {
      hypre_TMemcpy(perm, hypre_ParILUDataPerm(data), HYPRE_Int, n,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }
   else { for (i = 0; i < n; i++) { perm[i] = i; } }
   hypre_TMemcpy(storage, hypre_ParILUDataCombinedPermD(data), HYPRE_Int, n,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   for (i = 0; i < n; i++) for (j = 0; j < n; j++)
      {
         f[i * n + j] = a[perm[i] * n + perm[j]];
         pattern[i * n + j] = mask[perm[i] * n + perm[j]];
      }
   /* Scalar textbook ILU0 in initial ordering, fixed original sparsity. */
   for (i = 0; i < n; i++) for (k = 0; k < i; k++)
      {
         if (!pattern[i * n + k]) { continue; }
         f[i * n + k] /= f[k * n + k];
         for (j = k + 1; j < n; j++)
            if (pattern[i * n + j] && pattern[k * n + j]) { f[i * n + j] -= f[i * n + k] * f[k * n + j]; }
      }
   F = hypre_CSRMatrixClone_v2(hypre_ParILUDataMatBILUDevice(data), 1, HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      int oi = 0;
      while (perm[oi] != storage[i]) { oi++; }
      for (k = hypre_CSRMatrixI(F)[i]; k < hypre_CSRMatrixI(F)[i + 1]; k++)
      {
         int oj = 0;
         j = hypre_CSRMatrixJ(F)[k];
         while (perm[oj] != storage[j]) { oj++; }
         numerical_check(hypre_CSRMatrixData(F)[k], f[oi * n + oj],
                         tolerance * (1 + fabs(f[oi * n + oj])), "factor entry", n, seed, i);
      }
   }
   schedule(F, hypre_ParILUDataNumLowLevels(data), hypre_ParILUDataLowLevelSetOffsets(data),
            hypre_ParILUDataDLowLevelSetRows(data), 1, seed);
   schedule(F, hypre_ParILUDataNumUppLevels(data), hypre_ParILUDataUppLevelSetOffsets(data),
            hypre_ParILUDataDUppLevelSetRows(data), 0, seed);
   for (path = 0; path < 1; path++)
   {
      for (rhs = 0; rhs < 3; rhs++)
      {
         case_path = path; case_rhs = rhs;
         for (i = 0; i < n; i++) { initial[i] = rhs ? 0.1 * (i % 3) : 0; bv[i] = 1 + (i + rhs) % 5; }
         if (n == 3 && seed == 0) { bv[0] = 4; bv[1] = 6; bv[2] = 4; }
         for (i = 0; i < n; i++)
         {
            double r = bv[perm[i]];
            for (j = 0; j < n; j++) { r -= a[perm[i] * n + j] * initial[j]; }
            expected[i] = r;
         }
         for (i = 0; i < n; i++) for (j = 0; j < i; j++)
               if (pattern[i * n + j]) { expected[i] -= f[i * n + j] * expected[j]; }
         for (i = n - 1; i >= 0; i--)
         {
            for (j = i + 1; j < n; j++)
               if (pattern[i * n + j]) { expected[i] -= f[i * n + j] * expected[j]; }
            expected[i] /= f[i * n + i];
         }
         hypre_TMemcpy(hypre_VectorData(hypre_ParVectorLocalVector(b)), bv, HYPRE_Complex, n,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(hypre_VectorData(hypre_ParVectorLocalVector(u)), initial, HYPRE_Complex, n,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         error = HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix) A,
                                (HYPRE_ParVector) b, (HYPRE_ParVector) u);
         check(!error, "solve", n, seed);
         hypre_TMemcpy(x, hypre_VectorData(hypre_ParVectorLocalVector(u)), HYPRE_Complex, n,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
         for (i = 0; i < n; i++)
            numerical_check(x[perm[i]] - initial[perm[i]], expected[i],
                            tolerance * (1 + fabs(expected[i])), "correction", n, seed, perm[i]);
         if (n == 3 && seed == 0)
            for (i = 0; i < n; i++)
            {
               double r = bv[i];
               for (j = 0; j < n; j++) { r -= a[i * n + j] * x[j]; }
               numerical_check(r, 0, tolerance, "minimal independently recomputed residual", n, seed, i);
            }
      }
   }
done:
   if (failures != before)
   {
      printf("case ordering=%d invalid=%d seed=%u\n", ordering, invalid, seed);
      for (i = 0; i < n; i++) for (j = 0; j < n; j++)
            if (mask[i * n + j]) { printf("(%d,%d)=%.17g\n", i, j, a[i * n + j]); }
   }
   hypre_CSRMatrixDestroy(F);
   hypre_ParCSRMatrixDestroy(A); hypre_ParVectorDestroy(b); hypre_ParVectorDestroy(u);
   free(perm); free(storage); free(f); free(pattern); free(x); free(initial); free(bv); free(expected);
}
/* Exercise changed streams and RHS pointers with explicit lifetime boundaries. */
static void stream_scalar(void)
{
   double a[9] = {4, 0, 0, 1, 4, 1, 0, 0, 4};
   int mask[9] = {1, 0, 0, 1, 1, 1, 0, 0, 1};
   HYPRE_Solver solver;
   hypre_ParILUData *data;
   hypre_ParCSRMatrix *A = matrix(3, a, mask);
   hypre_ParVector *b = vector(3), *u = vector(3);
   HYPRE_Complex *scratch[2], host[3];
   HYPRE_Int original_stream = hypre_HandleComputeStreamNum(hypre_handle());
   int rep, i, error;
   HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
   HYPRE_ILUSetLocalReordering(solver, 0);
   data = (hypre_ParILUData *) solver;
   error = HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix) A,
                          (HYPRE_ParVector) b, (HYPRE_ParVector) u);
   check(!error, "stream test setup", 3, 0);
   if (error) { HYPRE_ClearAllErrors(); goto done; }
   {
      auto first_stream = hypre_HandleComputeStream(hypre_handle());
      hypre_HandleComputeStreamNum(hypre_handle()) = (original_stream + 1) % HYPRE_MAX_NUM_STREAMS;
      check(first_stream != hypre_HandleComputeStream(hypre_handle()),
            "stream test requires distinct streams", 3, 0);
      hypre_HandleComputeStreamNum(hypre_handle()) = original_stream;
   }
   scratch[0] = hypre_TAlloc(HYPRE_Complex, 3, HYPRE_MEMORY_DEVICE);
   scratch[1] = hypre_TAlloc(HYPRE_Complex, 3, HYPRE_MEMORY_DEVICE);
   case_ordering = 0; case_path = 0;
   /* Reuse on each stream, then switch back; also change RHS pointer. */
   for (rep = 0; rep < 8; rep++)
   {
      int which = (rep / 4) % 2;
      double scale = rep + 1;
      hypre_SyncDevice();
      hypre_HandleComputeStreamNum(hypre_handle()) =
         (rep % 4 < 2) ? original_stream : (original_stream + 1) % HYPRE_MAX_NUM_STREAMS;
      host[0] = 4 * scale; host[1] = 6 * scale; host[2] = 4 * scale;
      hypre_TMemcpy(scratch[which], host, HYPRE_Complex, 3,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_SyncDevice();
      case_rhs = rep;
      error = hypre_CSRMatrixILU0LevelSetLSolve(
                 hypre_ParILUDataMatBILUDevice(data), hypre_ParILUDataNumLowLevels(data),
                 hypre_ParILUDataLowLevelSetOffsets(data),
                 hypre_ParILUDataDLowLevelSetRows(data),
                 scratch[which]);
      if (!error)
      {
         error = hypre_CSRMatrixILU0LevelSetUSolve(
                    hypre_ParILUDataMatBILUDevice(data), hypre_ParILUDataNumUppLevels(data),
                    hypre_ParILUDataUppLevelSetOffsets(data),
                    hypre_ParILUDataDUppLevelSetRows(data),
                    scratch[which]);
      }
      check(!error, "stream scalar solve", 3, rep);
      if (error) { HYPRE_ClearAllErrors(); break; }
      hypre_SyncDevice();
      hypre_TMemcpy(host, scratch[which], HYPRE_Complex, 3,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_SyncDevice();
      for (i = 0; i < 3; i++)
      {
         numerical_check(host[i], scale, ORACLE_TOL * (1 + scale),
                         "stream scalar correction", 3, rep, i);
      }
   }
   hypre_SyncDevice();
   HYPRE_ILUDestroy(solver); solver = NULL;
   hypre_TFree(scratch[0], HYPRE_MEMORY_DEVICE);
   hypre_TFree(scratch[1], HYPRE_MEMORY_DEVICE);
done:
   hypre_SyncDevice();
   hypre_HandleComputeStreamNum(hypre_handle()) = original_stream;
   if (solver) { HYPRE_ILUDestroy(solver); }
   hypre_ParCSRMatrixDestroy(A); hypre_ParVectorDestroy(b); hypre_ParVectorDestroy(u);
}

static void invalid_structure(HYPRE_Solver solver)
{
   double a[9] = {4, 1, 1, 0, 4, 0, 0, 0, 4}; int mask[9] = {1, 1, 1, 0, 1, 0, 0, 0, 1};
   int issue;
   for (issue = 0; issue < 4; issue++)
   {
      if (issue == 3)
      {
         mask[1] = mask[2] = 0; a[1] = a[2] = 0;
         mask[3] = 1; a[3] = .1;
      }
      hypre_ParCSRMatrix *A = matrix(3, a, mask);
      hypre_CSRMatrix *D = hypre_ParCSRMatrixDiag(A);
      hypre_ParVector *b = vector(3), *u = vector(3);
      HYPRE_Int column = issue == 0 ? 1 : 3;
      if (issue < 2)
         hypre_TMemcpy(hypre_CSRMatrixJ(D) + 2, &column, HYPRE_Int, 1,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      else if (issue == 2) { hypre_CSRMatrixNumCols(D) = 4; }
      else
         hypre_TMemcpy(hypre_CSRMatrixJ(D) + hypre_CSRMatrixNumNonzeros(D) - 1,
                       &column, HYPRE_Int, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      check(HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                           (HYPRE_ParVector)u) != 0, "invalid structure rejected before permutation", 3, issue);
      HYPRE_ClearAllErrors();
      check(HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                           (HYPRE_ParVector)u) != 0, "invalid structure leaves solve disabled", 3, issue);
      HYPRE_ClearAllErrors();
      hypre_ParCSRMatrixDestroy(A); hypre_ParVectorDestroy(b); hypre_ParVectorDestroy(u);
   }
   /* Malformed CSR offsets must fail before a device permutation can use
    * them as addresses. A successful setup/solve after each rejection also
    * verifies that invalid structure leaves no stale usable factor state. */
   for (issue = 0; issue < 4; issue++)
   {
      hypre_ParCSRMatrix *A = matrix(3, a, mask);
      hypre_CSRMatrix *D = hypre_ParCSRMatrixDiag(A);
      hypre_ParVector *b = vector(3), *u = vector(3);
      HYPRE_Int n = hypre_CSRMatrixNumRows(D);
      HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(D);
      HYPRE_Int index = issue == 0 ? 0 : (issue == 1 ? n : (issue == 2 ? 1 : 2));
      HYPRE_Int offset = issue == 0 ? 1 : (issue == 1 ? nnz - 1 :
                                           (issue == 2 ? nnz + 1 : HYPRE_INT_MIN));
      /* Last case retains a valid first row and positive ia[1], then makes
       * ia[2] negative: subtraction before bounds checking would overflow. */
      hypre_TMemcpy(hypre_CSRMatrixI(D) + index, &offset, HYPRE_Int, 1,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      check(HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                           (HYPRE_ParVector)u) != 0,
            "invalid CSR row offsets rejected before permutation", 3, 66026u + issue);
      HYPRE_ClearAllErrors();
      check(HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                           (HYPRE_ParVector)u) != 0,
            "invalid CSR row offsets leave solve disabled", 3, 66026u + issue);
      HYPRE_ClearAllErrors();
      hypre_ParCSRMatrixDestroy(A); hypre_ParVectorDestroy(b); hypre_ParVectorDestroy(u);
      run(solver, 3, a, mask, 0, 0, 66026u + issue);
   }
   run(solver, 3, a, mask, 0, 0, 18);
}

/* Failures from a long-row factor must propagate collectively,
 * including to an empty rank, and a valid retry must restore usable state. */
static void distributed_longrow_failures(int rank, int active)
{
   const int width = 33;
   int n = rank < active ? width : 0, i, j, issue;
   HYPRE_BigInt total = width * active;
   HYPRE_BigInt starts[2] = {width * (rank < active ? rank : active), 0};
   HYPRE_Complex values[33 * 33], result[33];
   HYPRE_Solver solver;
   hypre_ParCSRMatrix *A;
   hypre_ParVector *b, *u;
   hypre_CSRMatrix *D;
   starts[1] = starts[0] + n;
   A = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_WORLD, total, total, starts, starts, 0, n * n, 0);
   hypre_ParCSRMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   D = hypre_ParCSRMatrixDiag(A);
   for (i = 0; i < n; i++)
   {
      hypre_CSRMatrixI(D)[i] = i * n;
      for (j = 0; j < n; j++)
      {
         hypre_CSRMatrixJ(D)[i * n + j] = j;
         values[i * n + j] = hypre_CSRMatrixData(D)[i * n + j] = i == j ? 4 : .01;
      }
   }
   hypre_CSRMatrixI(D)[n] = n * n;
   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   D = hypre_ParCSRMatrixDiag(A);
   b = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, total, starts);
   u = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, total, starts);
   hypre_ParVectorInitialize_v2(b, HYPRE_MEMORY_DEVICE);
   hypre_ParVectorInitialize_v2(u, HYPRE_MEMORY_DEVICE);
   hypre_ParVectorSetConstantValues(b, 1);
   HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
   HYPRE_ILUSetMaxIter(solver, 1); HYPRE_ILUSetTol(solver, 0);
   HYPRE_ILUSetLocalReordering(solver, 0);
   for (issue = 0; issue < 2; issue++)
   {
      hypre_ParVectorSetConstantValues(u, 0);
      if (!rank)
      {
         if (!issue) { values[0] = values[1] = values[width] = values[width + 1] = 1; }
         else { values[width * width - 2] = NAN; }
         hypre_TMemcpy(hypre_CSRMatrixData(D), values, HYPRE_Complex, n * n,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      }
      check(HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                           (HYPRE_ParVector) u) != 0, "MPI long-row factor failure rejected", n, issue);
      HYPRE_ClearAllErrors();
      check(HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                           (HYPRE_ParVector) u) != 0, "MPI long-row failed setup disables solve", n, issue);
      HYPRE_ClearAllErrors();
      if (!rank)
      {
         for (i = 0; i < n; i++) for (j = 0; j < n; j++)
            {
               values[i * n + j] = i == j ? 4 : .01;
            }
         hypre_TMemcpy(hypre_CSRMatrixData(D), values, HYPRE_Complex, n * n,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      }
      check(!HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                            (HYPRE_ParVector) u), "MPI long-row setup recovery", n, issue);
      check(!HYPRE_ILUSolve(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                            (HYPRE_ParVector) u), "MPI long-row solve recovery", n, issue);
      hypre_TMemcpy(result, hypre_VectorData(hypre_ParVectorLocalVector(u)), HYPRE_Complex, n,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < n; i++)
         numerical_check(result[i], 1 / (4 + (width - 1) * .01), ORACLE_TOL,
                         "MPI long-row recovered correction", n, issue, i);
   }
   HYPRE_ILUDestroy(solver); hypre_ParCSRMatrixDestroy(A);
   hypre_ParVectorDestroy(b); hypre_ParVectorDestroy(u);
}

/* Distributed integration: at least two active ranks and one empty rank.
 * Off-rank entries remain outside the local ILU, preserving block Jacobi. */
static void distributed(void)
{
   HYPRE_Int rank, size;
   int active, n, i;
   HYPRE_BigInt starts[2], total;
   double a[9] = {4, 0, 0, 1, 4, 1, 0, 0, 4}; int mask[9] = {1, 0, 0, 1, 1, 1, 0, 0, 1};
   hypre_ParCSRMatrix *local, *A;
   hypre_ParVector *b, *u, *r;
   HYPRE_Solver ilu, gmres;
   HYPRE_Complex bv[3] = {4.125, 6, 4}, uv[3] = {.25, .25, .25};
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &rank);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &size);
   check(size >= 3, "MPI integration needs at least 3 ranks", size, 0);
   if (size < 3) { return; }
   active = size - 1;
   distributed_longrow_failures(rank, active);
   n = rank < active ? 3 : 0; total = 3 * active;
   starts[0] = 3 * (rank < active ? rank : active); starts[1] = starts[0] + n;
   local = matrix(n, a, mask);
   /* The assembled distributed matrix is host-resident until both blocks are
    * populated. Transferring a device diagonal into it violates that contract. */
   hypre_ParCSRMatrixMigrate(local, HYPRE_MEMORY_HOST);
   A = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_WORLD, total, total, starts, starts,
                                n ? 1 : 0, 0, n ? 1 : 0);
   hypre_ParCSRMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   if (n)
   {
      hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
      hypre_CSRMatrixI(offd)[0] = 0;
      for (i = 1; i <= n; i++) { hypre_CSRMatrixI(offd)[i] = 1; }
      hypre_CSRMatrixJ(offd)[0] = 0; hypre_CSRMatrixData(offd)[0] = .125;
      hypre_ParCSRMatrixColMapOffd(A)[0] = 3 * ((rank + 1) % active);
   }
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(A));
   hypre_ParCSRMatrixDiag(A) = hypre_ParCSRMatrixDiag(local);
   hypre_ParCSRMatrixDiag(local) = NULL;
   hypre_ParCSRMatrixDestroy(local);
   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
   b = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, total, starts);
   u = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, total, starts);
   r = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, total, starts);
   hypre_ParVectorInitialize_v2(b, HYPRE_MEMORY_DEVICE);
   hypre_ParVectorInitialize_v2(u, HYPRE_MEMORY_DEVICE);
   hypre_ParVectorInitialize_v2(r, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(hypre_VectorData(hypre_ParVectorLocalVector(b)), bv, HYPRE_Complex, n,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_VectorData(hypre_ParVectorLocalVector(u)), uv, HYPRE_Complex, n,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   HYPRE_ILUCreate(&ilu); HYPRE_ILUSetType(ilu, 60); HYPRE_ILUSetMaxIter(ilu, 1);
   HYPRE_ILUSetTol(ilu, 0); HYPRE_ILUSetLocalReordering(ilu, 0);
   /* Rank-local option and structure failures must agree before entering setup. */
   if (!rank) { HYPRE_ILUSetLevelOfFill(ilu, 1); }
   check(HYPRE_ILUSetup(ilu, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                        (HYPRE_ParVector) u) != 0, "MPI rank-local option rejection", n, 0);
   HYPRE_ClearAllErrors();
   HYPRE_ILUSetLevelOfFill(ilu, 0);
   if (!rank)
   {
      HYPRE_Int invalid = 1;
      hypre_TMemcpy(hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A)), &invalid,
                    HYPRE_Int, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
   check(HYPRE_ILUSetup(ilu, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                        (HYPRE_ParVector) u) != 0, "MPI rank-local structure rejection", n, 0);
   HYPRE_ClearAllErrors();
   if (!rank)
   {
      HYPRE_Int valid = 0;
      hypre_TMemcpy(hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A)), &valid,
                    HYPRE_Int, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
   /* A failure on one active rank must reject setup on every rank without
    * mismatching collectives, including the empty rank. Then retry successfully. */
   if (rank == 0)
   {
      HYPRE_Complex zero = 0;
      hypre_TMemcpy(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A)), &zero,
                    HYPRE_Complex, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
   check(HYPRE_ILUSetup(ilu, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                        (HYPRE_ParVector) u) != 0, "MPI rank-local pivot failure rejected", n, 0);
   HYPRE_ClearAllErrors();
   check(HYPRE_ILUSolve(ilu, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                        (HYPRE_ParVector) u) != 0, "MPI failed setup disables solve", n, 0);
   HYPRE_ClearAllErrors();
   if (rank == 0)
   {
      HYPRE_Complex diagonal = 4;
      hypre_TMemcpy(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A)), &diagonal,
                    HYPRE_Complex, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
   HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &gmres);
   HYPRE_ParCSRGMRESSetTol(gmres, ORACLE_TOL); HYPRE_ParCSRGMRESSetMaxIter(gmres, 100);
   HYPRE_ParCSRGMRESSetPrecond(gmres, HYPRE_ILUSolve, HYPRE_ILUSetup, ilu);
   check(!HYPRE_ParCSRGMRESSetup(gmres, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                                 (HYPRE_ParVector)u), "MPI setup", n, 0);
   check(!HYPRE_ParCSRGMRESSolve(gmres, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                                 (HYPRE_ParVector)u), "MPI converged solve", n, 0);
   hypre_ParCSRMatrixMatvecOutOfPlace(-1, A, u, 1, b, r);
   check(sqrt(hypre_ParVectorInnerProd(r, r) / hypre_ParVectorInnerProd(b, b)) < 10 * ORACLE_TOL,
         "MPI independently recomputed residual", n, 0);
   HYPRE_ParCSRGMRESDestroy(gmres); HYPRE_ILUDestroy(ilu);
   hypre_ParCSRMatrixDestroy(A); hypre_ParVectorDestroy(b);
   hypre_ParVectorDestroy(u); hypre_ParVectorDestroy(r);
}

#endif

static int test_main(int argc, char **argv)
{
   HYPRE_Solver solver;
   hypre_MPI_Init(&argc, &argv);
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   {
      hypre_int count = 0;
#if defined(HYPRE_USING_HIP)
      hipError_t status = hipGetDeviceCount(&count);
      if (status != hipSuccess && status != hipErrorNoDevice)
      { printf("FAILED: HIP device discovery error %d\n", (int) status); hypre_MPI_Finalize(); return 1; }
#else
      cudaError_t status = cudaGetDeviceCount(&count);
      if (status != cudaSuccess && status != cudaErrorNoDevice)
      { printf("FAILED: CUDA device discovery error %d\n", (int) status); hypre_MPI_Finalize(); return 1; }
#endif
      if (!count) { printf("SKIP: CUDA/HIP device unavailable\n"); hypre_MPI_Finalize(); return 77; }
   }
   HYPRE_Init();
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
   if (argc > 1 && !strcmp(argv[1], "--stream"))
   {
      stream_scalar();
      printf("ilu60 stream: %s (%lld failures)\n", failures ? "FAILED" : "PASSED",
             (long long) failures);
      HYPRE_Finalize(); hypre_MPI_Finalize(); return failures ? 1 : 0;
   }
   if (argc > 1 && !strcmp(argv[1], "--ownership"))
   {
      ownership();
      HYPRE_Finalize(); hypre_MPI_Finalize(); return failures ? 1 : 0;
   }
   if (argc > 1 && !strcmp(argv[1], "--mpi"))
   {
      HYPRE_Int global_failures;
      distributed();
      hypre_MPI_Allreduce(&failures, &global_failures, 1, HYPRE_MPI_INT, hypre_MPI_MAX,
                          hypre_MPI_COMM_WORLD);
      HYPRE_Finalize(); hypre_MPI_Finalize(); return global_failures ? 1 : 0;
   }
   HYPRE_ILUCreate(&solver);
   HYPRE_ILUSetType(solver, 60); HYPRE_ILUSetMaxIter(solver, 1); HYPRE_ILUSetTol(solver, 0);
   if (argc > 1 && !strcmp(argv[1], "--stored-zero"))
   {
      stored_zero_pattern(solver);
      HYPRE_ILUDestroy(solver); HYPRE_Finalize(); hypre_MPI_Finalize();
      return failures ? 1 : 0;
   }
   if (argc > 1 && !strcmp(argv[1], "--diagonal"))
   {
      double a[4] = {2, 0, 0, 2}; int mask[4] = {1, 0, 0, 1};
      run(solver, 2, a, mask, 0, 0, 62026);
      run(solver, 2, a, mask, 1, 0, 62026);
      HYPRE_ILUDestroy(solver); HYPRE_Finalize(); hypre_MPI_Finalize();
      return failures ? 1 : 0;
   }
   {
      double a[9] = {4, 0, 0, 1, 4, 1, 0, 0, 4};
      int mask[9] = {1, 0, 0, 1, 1, 1, 0, 0, 1};
      /* Exercise an asymmetric pattern with identity ordering. */
      run(solver, 3, a, mask, 0, 0, 0);
      run(solver, 3, a, mask, 1, 0, 0);
      if (argc > 1 && !strcmp(argv[1], "--minimal"))
      {
         HYPRE_ILUDestroy(solver); HYPRE_Finalize(); hypre_MPI_Finalize();
         return failures ? 1 : 0;
      }
   }
   {
      /* Equal lower/upper counts do not imply matching reverse edges. Also
       * test the unmatched-upper exhaustion and unmatched-lower paths. */
      int issue, ordering;
      for (issue = 0; issue < 3; issue++)
      {
         double a[16] = {4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4};
         int mask[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
         if (issue != 2) { mask[1] = 1; a[1] = .2; }
         if (issue != 1) { mask[8] = 1; a[8] = .1; }
         for (ordering = 0; ordering < 2; ordering++)
         {
            run(solver, 4, a, mask, ordering, 0, 65026 + issue);
         }
      }
   }
   {
      /* Unsorted unique rows must retain the scalar oracle result, including
       * after RCM. These rows take the validation sorting fallback. */
      double a[16] = {4, .2, .1, .3, -.1, 4, .2, 0,
                      .5, 0, 4, .1, 0, .3, .1, 4
                     };
      int mask[16] = {1, 1, 1, 1, 1, 1, 1, 0,
                      1, 0, 1, 1, 0, 1, 1, 1
                     };
      reverse_input_columns = 1;
      run(solver, 4, a, mask, 0, 0, 64026);
      run(solver, 4, a, mask, 1, 0, 64026);
      reverse_input_columns = 0;
   }
   {
      int repetition, ordering;
      for (repetition = 0; repetition < 120; repetition++)
      {
         int n = 2 + repetition % 35, i, j, symmetric = repetition % 2;
         unsigned seed = 62026u + repetition, state = seed;
         double *a = (double *) calloc(n * n, sizeof(double));
         int *mask = (int *) calloc(n * n, sizeof(int));
         for (i = 0; i < n; i++) for (j = 0; j < n; j++)
               if (i != j && next(&state) % 5 == 0)
               {
                  mask[i * n + j] = 1; a[i * n + j] = ((int)(next(&state) % 21) - 10) * 0.03;
                  if (symmetric) { mask[j * n + i] = 1; a[j * n + i] = a[i * n + j] * 0.7; }
               }
         for (i = 0; i < n; i++)
         {
            mask[i * n + i] = 1; a[i * n + i] = 2;
            for (j = 0; j < n; j++) if (j != i) { a[i * n + i] += fabs(a[i * n + j]); }
         }
         for (ordering = 0; ordering < 2; ordering++) { run(solver, n, a, mask, ordering, 0, seed); }
         free(a); free(mask);
      }
   }
   stored_zero_pattern(solver);
   {
      /* Long-row correctness: 65 independent rows
       * followed by 65 dependent rows exercise partial blocks and batches.
       * Both orderings are checked against the independent scalar oracle. */
      int n = 130, i, j;
      double *a = (double *) calloc(n * n, sizeof(double));
      int *mask = (int *) calloc(n * n, sizeof(int));
      for (i = 0; i < n; i++) for (j = 0; j < n; j++)
         {
            mask[i * n + j] = i == j || i >= 65 || j >= 65;
            a[i * n + j] = mask[i * n + j] ? (i == j ? 4 : ((i + j) % 3 - 1) * .01) : 0;
         }
      run(solver, n, a, mask, 0, 0, 83226);
      run(solver, n, a, mask, 1, 0, 83226);
      /* Keep the long rows while forcing the asymmetric
       * structure fallback: one stored edge has no reverse edge. */
      mask[3 * n + 100] = 0; a[3 * n + 100] = 0;
      a[100 * n + 3] = .01;
      run(solver, n, a, mask, 0, 0, 83326);
      run(solver, n, a, mask, 1, 0, 83326);
      mask[3 * n + 100] = 1;
      a[3 * n + 100] = a[100 * n + 3] = 0;
      {
         double small = sizeof(HYPRE_Real) == sizeof(float) ? 1e-10 : 1e-40;
         double large = 1 / small;
         for (i = 0; i < n * n; i++) { a[i] *= small; }
         run(solver, n, a, mask, 0, 0, 83227);
         run(solver, n, a, mask, 1, 0, 83227);
         for (i = 0; i < n * n; i++) { a[i] *= large / small; }
         run(solver, n, a, mask, 0, 0, 83228);
         run(solver, n, a, mask, 1, 0, 83228);
      }
      free(a); free(mask);
   }
   {
      int widths[] = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025}, w;
      for (w = 0; w < 13; w++)
      {
         int n = widths[w], i;
         double *a = (double *) calloc(n * n, sizeof(double));
         int *mask = (int *) calloc(n * n, sizeof(int));
         for (i = 0; i < n; i++) { a[i * n + i] = 4; mask[i * n + i] = 1; }
         run(solver, n, a, mask, 0, 0, 1);
         free(a); free(mask);
         /* Width w followed by a one-row level exercises inactive barriers and
          * long rows. Stored reverse edges preserve the optimized storage path. */
         n++;
         a = (double *) calloc(n * n, sizeof(double));
         mask = (int *) calloc(n * n, sizeof(int));
         for (i = 0; i < n; i++) { a[i * n + i] = 4; mask[i * n + i] = 1; }
         for (i = 0; i < n - 1; i++)
         {
            mask[(n - 1)*n + i] = mask[i * n + n - 1] = 1;
            a[(n - 1)*n + i] = .01;
            a[i * n + n - 1] = .02;
         }
         a[n * n - 1] += .01 * n;
         run(solver, n, a, mask, 0, 0, 17);
         free(a); free(mask);
      }
   }
   {
      /* Both schedules have widths 1,1,w,1,1. Scalar factorization and
       * level solves must cover widths above 128 and an oversized level
       * between narrower levels. */
      int widths[] = {128, 129, 256, 512, 1023, 1024, 1025}, w;
      for (w = 0; w < 7; w++)
      {
         int n = widths[w] + 4, i;
         double *a = (double *) calloc(n * n, sizeof(double));
         int *mask = (int *) calloc(n * n, sizeof(int));
         for (i = 0; i < n; i++) { a[i * n + i] = 8; mask[i * n + i] = 1; }
         mask[1] = mask[n] = mask[(n - 2) * n + n - 1] = mask[(n - 1) * n + n - 2] = 1;
         a[1] = a[(n - 2) * n + n - 1] = .02;
         a[n] = a[(n - 1) * n + n - 2] = .01;
         for (i = 2; i < n - 2; i++)
         {
            mask[n + i] = mask[i * n + 1] = 1;
            a[n + i] = .02; a[i * n + 1] = .01;
            mask[i * n + n - 2] = mask[(n - 2) * n + i] = 1;
            a[i * n + n - 2] = .02; a[(n - 2) * n + i] = .01;
         }
         /* Preserve nonzero arithmetic and every dependency edge as width
          * grows, while maintaining a positive row-dominance margin for
          * the independent float oracle's conditioning bound. */
         a[n + 1] += .01 + .02 * widths[w];
         a[(n - 2) * n + n - 2] += .02 + .01 * widths[w];
         run(solver, n, a, mask, 0, 0, 72826 + w);
         /* Invalid pivot inside a batch, including schedules interrupted by
          * the 1025-row wide level, must complete without barrier deadlock. */
         if (w == 0 || w == 6)
         {
            double second_diagonal = a[n + 1];
            a[0] = a[n + 1] = a[1] = a[n] = 1;
            run(solver, n, a, mask, 0, 1, 72926 + w);
            a[n + 1] = NAN;
            run(solver, n, a, mask, 0, 1, 73026 + w);
            a[0] = 8; a[n + 1] = second_diagonal; a[1] = .02; a[n] = .01;
            run(solver, n, a, mask, 0, 0, 73126 + w);
         }
         free(a); free(mask);
      }
   }
   {
      double a[4] = {1, 1, 1, 1}; int mask[4] = {1, 1, 1, 1};
      run(solver, 2, a, mask, 0, 1, 2); /* elimination-created terminal zero */
      a[0] = 0; run(solver, 2, a, mask, 0, 1, 3);
      a[0] = 1; mask[0] = 0; run(solver, 2, a, mask, 0, 1, 4);
      mask[0] = 1; a[0] = NAN; run(solver, 2, a, mask, 0, 1, 5);
      a[0] = INFINITY; run(solver, 2, a, mask, 0, 1, 6);
      a[0] = sizeof(HYPRE_Real) == sizeof(float) ? 1e-15 : 1e-100; a[1] = a[2] = 0; a[3] = 2 * a[0];
      mask[1] = mask[2] = 0; run(solver, 2, a, mask, 0, 0, 7);
      run(solver, 0, a, mask, 0, 0, 8);
      a[3] = 0; run(solver, 2, a, mask, 0, 1, 9); /* input terminal zero */
      mask[3] = 0; run(solver, 2, a, mask, 0, 1, 10); /* empty nonzero-size row */
      mask[3] = 1; a[0] = 2; a[3] = 3; run(solver, 2, a, mask, 0, 0, 11);
      HYPRE_ILUSetLevelOfFill(solver, 1); run(solver, 2, a, mask, 0, 1, 12);
      HYPRE_ILUSetLevelOfFill(solver, 0);
      HYPRE_ILUSetTriSolve(solver, 0); run(solver, 2, a, mask, 0, 1, 13);
      HYPRE_ILUSetTriSolve(solver, 1);
      HYPRE_ILUSetIterativeSetupType(solver, 1); run(solver, 2, a, mask, 0, 1, 14);
      HYPRE_ILUSetIterativeSetupType(solver, 0); run(solver, 2, a, mask, 0, 0, 15);
      {
         hypre_ParCSRMatrix *A = matrix(2, a, mask);
         hypre_ParVector *b = vector(2), *u = vector(2);
         HYPRE_ILUSetType(solver, 0);
         check(!HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) b,
                               (HYPRE_ParVector) u), "transition to type 0", 2, 0);
         HYPRE_ILUSetType(solver, 60);
         hypre_ParCSRMatrixDestroy(A); hypre_ParVectorDestroy(b); hypre_ParVectorDestroy(u);
         run(solver, 2, a, mask, 0, 0, 16);
      }
   }
   invalid_structure(solver);
   HYPRE_ILUDestroy(solver);
   ordered_diagonal();
   ownership();
   failed_ordering_transition();
   factor_pivot_ownership();
   HYPRE_Finalize();
#else
   HYPRE_Init();
   {
      double a[1] = {1}; int mask[1] = {1};
      hypre_ParCSRMatrix *A = matrix(1, a, mask);
      HYPRE_ILUCreate(&solver); HYPRE_ILUSetType(solver, 60);
      check(HYPRE_ILUSetup(solver, (HYPRE_ParCSRMatrix) A, NULL, NULL) != 0,
            "unsupported backend rejected", 1, 0);
      HYPRE_ClearAllErrors(); HYPRE_ILUDestroy(solver); hypre_ParCSRMatrixDestroy(A);
   }
   HYPRE_Finalize();
#endif
   printf("ilu60: %s (%lld failures)\n", failures ? "FAILED" : "PASSED",
          (long long) failures);
   hypre_MPI_Finalize();
   return failures ? 1 : 0;
}

/* HYPRE_Init enables Debug floating-point traps without restoring the caller's
 * environment. Keep those traps throughout tests and hypre/MPI finalization,
 * then restore the entry environment before ROCm static runtime teardown. */
int main(int argc, char **argv)
{
   fenv_t entry_environment;
   int status;
   if (fegetenv(&entry_environment))
   {
      printf("FAILED: cannot save floating-point environment\n");
      return 1;
   }
   status = test_main(argc, argv);
   if (fesetenv(&entry_environment))
   {
      printf("FAILED: cannot restore floating-point environment\n");
      return 1;
   }
   return status;
}
