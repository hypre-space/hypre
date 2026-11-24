/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

#define N 8

/* Independent residual-form reference, with old values across thread blocks. */
static void
reference_sweep(HYPRE_Complex *u, HYPRE_Complex *f, HYPRE_Real *l1,
                HYPRE_Int type, HYPRE_Int threads, HYPRE_Real weight)
{
   HYPRE_Complex old[N];
   HYPRE_Int t, sweep, i, begin, end, direction;
   for (i = 0; i < N; i++) { old[i] = u[i]; }
   for (t = 0; t < threads; t++)
   {
      begin = t * (N / threads);
      end = begin + N / threads;
      for (sweep = 0; sweep < (type == 8 ? 2 : 1); sweep++)
      {
         direction = type == 14 || sweep == 1 ? -1 : 1;
         for (i = direction > 0 ? begin : end - 1;
              i != (direction > 0 ? end : begin - 1); i += direction)
         {
            HYPRE_Complex residual = f[i] - (3.0 + i) * u[i];
            if (i > 0) { residual += i - 1 >= begin ? u[i - 1] : old[i - 1]; }
            if (i + 1 < N) { residual += i + 1 < end ? u[i + 1] : old[i + 1]; }
            u[i] += weight * residual / l1[i];
         }
      }
   }
}

int main(int argc, char *argv[])
{
   HYPRE_BigInt starts[2] = {0, N};
   HYPRE_Int types[3] = {8, 13, 14};
   HYPRE_Real weights[2] = {1.0, 2.0 / 3.0};
   HYPRE_Real omegas[2] = {1.0, 0.75};
   HYPRE_Real l1[N];
   HYPRE_Complex expected[N];
   HYPRE_Int i, j, ti, wi, oi, threads, iteration, size, failures = 0;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_ParVector *f, *u, *v, *z;
   HYPRE_Complex *rhs, *solution;
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &size);
   if (size != 1)
   {
      hypre_printf("l1_gs_damping requires one MPI rank\n");
      hypre_MPI_Finalize();
      return 1;
   }
   HYPRE_Initialize();
   A = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_WORLD, N, N, starts, starts, 0, 3 * N - 2, 0);
   hypre_ParCSRMatrixInitialize_v2(A, HYPRE_MEMORY_HOST);
   diag = hypre_ParCSRMatrixDiag(A);
   j = 0;
   for (i = 0; i < N; i++)
   {
      hypre_CSRMatrixI(diag)[i] = j;
      hypre_CSRMatrixJ(diag)[j] = i;
      hypre_CSRMatrixData(diag)[j++] = 3.0 + i;
      if (i > 0)
      {
         hypre_CSRMatrixJ(diag)[j] = i - 1;
         hypre_CSRMatrixData(diag)[j++] = -1.0;
      }
      if (i + 1 < N)
      {
         hypre_CSRMatrixJ(diag)[j] = i + 1;
         hypre_CSRMatrixData(diag)[j++] = -1.0;
      }
      l1[i] = 3.0 + i + (i > 0) + (i + 1 < N);
   }
   hypre_CSRMatrixI(diag)[N] = j;
   f = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, N, starts);
   u = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, N, starts);
   v = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, N, starts);
   z = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, N, starts);
   hypre_ParVectorInitialize_v2(f, HYPRE_MEMORY_HOST);
   hypre_ParVectorInitialize_v2(u, HYPRE_MEMORY_HOST);
   hypre_ParVectorInitialize_v2(v, HYPRE_MEMORY_HOST);
   hypre_ParVectorInitialize_v2(z, HYPRE_MEMORY_HOST);
   rhs = hypre_VectorData(hypre_ParVectorLocalVector(f));
   solution = hypre_VectorData(hypre_ParVectorLocalVector(u));
   for (i = 0; i < N; i++)
   {
      rhs[i] = (3.0 + i) * (i + 1) - (i > 0 ? i : 0) - (i + 1 < N ? i + 2 : 0);
   }
   for (threads = 1; threads <= 4; threads *= 4)
   {
#ifndef HYPRE_USING_OPENMP
      if (threads > 1) { break; }
#endif
      hypre_SetNumThreads(threads);
      for (ti = 0; ti < 3; ti++)
      {
         for (wi = 0; wi < 2; wi++)
         {
            for (oi = 0; oi < 2; oi++)
            {
               for (i = 0; i < N; i++) { solution[i] = i + 1; }
               hypre_BoomerAMGRelax(A, f, NULL, types[ti], 0, weights[wi], omegas[oi], l1, u, v, z);
               for (i = 0; i < N; i++)
               {
                  if (!(hypre_cabs(solution[i] - (i + 1)) <= 100 * HYPRE_REAL_EPSILON * (i + 1)))
                  {
                     hypre_printf("fixed point failed: type=%d threads=%d weight=%g omega=%g row=%d\n",
                                  types[ti], threads, (double) weights[wi], (double) omegas[oi], i);
                     failures++;
                     break;
                  }
               }
            }
            for (i = 0; i < N; i++) { solution[i] = expected[i] = 0.0; }
            for (iteration = 0; iteration < 3; iteration++)
            {
               reference_sweep(expected, rhs, l1, types[ti], threads, weights[wi]);
               hypre_BoomerAMGRelax(A, f, NULL, types[ti], 0, weights[wi], 1.0, l1, u, v, z);
               for (i = 0; i < N; i++)
               {
                  if (!(hypre_cabs(solution[i] - expected[i]) <= 100 * HYPRE_REAL_EPSILON * (i + 1)))
                  {
                     hypre_printf("reference failed: type=%d threads=%d weight=%g iteration=%d row=%d\n",
                                  types[ti], threads, (double) weights[wi], iteration, i);
                     failures++;
                     break;
                  }
               }
            }
         }
      }
   }
   hypre_printf("l1 GS damping: %d failed checks\n", failures);
   hypre_ParVectorDestroy(f);
   hypre_ParVectorDestroy(u);
   hypre_ParVectorDestroy(v);
   hypre_ParVectorDestroy(z);
   hypre_ParCSRMatrixDestroy(A);
   HYPRE_Finalize();
   hypre_MPI_Finalize();
   return failures ? 1 : 0;
}
