/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "HYPRE.h"
#include "_hypre_parcsr_ls.h"

#define CHECK(call) do { if ((call)) { hypre_printf("%s failed at line %d\n", #call, __LINE__); return 1; } } while (0)

/* An independent recursive traversal: one terminal coarse solve, complete
 * child cycles repeated twice below W nodes, and two root W passes for F+W.
 * Like BoomerAMG, do not repeat the terminal coarse solve within a child visit.
 * This reference does not use BoomerAMGCycle's traversal counters. */
static void
relax_phase(hypre_ParAMGData *d, HYPRE_Int level, HYPRE_Int phase, HYPRE_Int repeated)
{
   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(d)[level];
   hypre_ParVector *f = hypre_ParAMGDataFArray(d)[level];
   hypre_ParVector *u = hypre_ParAMGDataUArray(d)[level];
   hypre_ParVector *v = hypre_ParAMGDataVtemp(d);
   hypre_ParVector *z = hypre_ParAMGDataZtemp(d);
   hypre_ParVector *p = hypre_ParAMGDataPtemp(d);
   hypre_ParVector *r = hypre_ParAMGDataRtemp(d);
   HYPRE_Int type = hypre_ParAMGDataGridRelaxType(d)[phase];
   HYPRE_Int sweeps = hypre_ParAMGDataNumGridSweeps(d)[phase];
   HYPRE_Int outer_sweeps = level < hypre_ParAMGDataSmoothNumLevels(d) ?
                            hypre_ParAMGDataSmoothNumSweeps(d) : sweeps;
   HYPRE_Int n = hypre_VectorSize(hypre_ParVectorLocalVector(u));
   hypre_IntArray *markers = hypre_ParAMGDataCFMarkerArray(d)[level];
   HYPRE_Real *l1 = hypre_ParAMGDataL1Norms(d) && hypre_ParAMGDataL1Norms(d)[level] ?
                    hypre_VectorData(hypre_ParAMGDataL1Norms(d)[level]) : NULL;
   hypre_ParVectorSetLocalSize(v, n);
   if (z) { hypre_ParVectorSetLocalSize(z, n); }
   if (p) { hypre_ParVectorSetLocalSize(p, n); }
   if (r) { hypre_ParVectorSetLocalSize(r, n); }
   for (HYPRE_Int j = 0; j < outer_sweeps; j++)
   {
      if (level < hypre_ParAMGDataSmoothNumLevels(d))
      {
         HYPRE_Solver smoother = hypre_ParAMGDataSmoother(d)[level];
         HYPRE_Int zero = phase == 1 && !repeated;
         HYPRE_FSAISetZeroGuess(smoother, zero);
         HYPRE_FSAISetMaxIterations(smoother, sweeps);
         HYPRE_FSAISolve(smoother, (HYPRE_ParCSRMatrix) A,
                         (HYPRE_ParVector) f, (HYPRE_ParVector) u);
      }
      else if (type == 9)
      {
         hypre_GaussElimSolve(d, level, type);
      }
      else
      {
         hypre_BoomerAMGRelaxIF(A, f, markers ? hypre_IntArrayData(markers) : NULL,
                                type, hypre_ParAMGDataRelaxOrder(d), phase,
                                hypre_ParAMGDataRelaxWeight(d)[level],
                                hypre_ParAMGDataOmega(d)[level], l1, u, v, z);
      }
   }
}

static void
reference_cycle(hypre_ParAMGData *d, HYPRE_Int level, HYPRE_Int wcycle, HYPRE_Int repeated)
{
   HYPRE_Int last = hypre_ParAMGDataNumLevels(d) - 1;
   if (level == last) { relax_phase(d, level, 3, 0); return; }
   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(d)[level];
   hypre_ParCSRMatrix *P = hypre_ParAMGDataPArray(d)[level];
   hypre_ParCSRMatrix *R = hypre_ParAMGDataRArray(d)[level];
   hypre_ParVector **f = hypre_ParAMGDataFArray(d);
   hypre_ParVector **u = hypre_ParAMGDataUArray(d);
   hypre_ParVector *v = hypre_ParAMGDataVtemp(d);
   relax_phase(d, level, 1, repeated);
   hypre_ParVectorSetZeros(u[level + 1]);
   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u[level], 1.0, f[level], v);
   hypre_ParCSRMatrixMatvecT(1.0, R, v, 0.0, f[level + 1]);
   HYPRE_Int repeats = wcycle && level + 1 < last ? 2 : 1;
   for (HYPRE_Int j = 0; j < repeats; j++)
   {
      reference_cycle(d, level + 1, wcycle, j > 0);
   }
   hypre_ParCSRMatrixMatvec(1.0, P, u[level + 1], 1.0, u[level]);
   hypre_ParVectorAllZeros(u[level]) = 0;
   relax_phase(d, level, 2, 0);
}

int main(int argc, char **argv)
{
   const HYPRE_Int n = 97;
   HYPRE_Int size, failures = 0, cases = 0, symmetry_cases = 0;
   HYPRE_Int failed_cycles[3] = {0, 0, 0};
   HYPRE_IJMatrix ij;
   HYPRE_IJVector ib, ix;
   HYPRE_ParCSRMatrix A;
   HYPRE_ParVector b, x;
   HYPRE_Complex expected[97];
   HYPRE_Complex first_solution[97];
   /* The final pair uses internal FSAI smoothing instead of point relaxation. */
   HYPRE_Int pairs[][2] = {{3, 4}, {13, 14}, {6, 6}, {8, 8}, {0, 0}, {18, 18}, {6, 6}};
   HYPRE_Int sweep_pairs[][2] = {{1, 1}, {2, 2}, {1, 2}, {2, 1}, {0, 1}, {1, 0}};
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &size);
   if (size != 1) { hypre_MPI_Finalize(); return 1; }
   CHECK(HYPRE_Initialize());
   CHECK(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST));
   CHECK(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST));
   CHECK(HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, 0, n - 1, 0, n - 1, &ij));
   CHECK(HYPRE_IJMatrixSetObjectType(ij, HYPRE_PARCSR));
   CHECK(HYPRE_IJMatrixInitialize(ij));
   for (HYPRE_Int i = 0; i < n; i++)
   {
      HYPRE_BigInt row = i, cols[3] = {i, i - 1, i + 1};
      HYPRE_Complex vals[3] = {2.1, -1.0, -1.0};
      HYPRE_Int count = 1;
      if (i > 0) { cols[count] = i - 1; vals[count++] = -1.0; }
      if (i + 1 < n) { cols[count] = i + 1; vals[count++] = -1.0; }
      CHECK(HYPRE_IJMatrixSetValues(ij, 1, &count, &row, cols, vals));
   }
   CHECK(HYPRE_IJMatrixAssemble(ij));
   CHECK(HYPRE_IJMatrixGetObject(ij, (void **) &A));
   CHECK(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, 0, n - 1, &ib));
   CHECK(HYPRE_IJVectorSetObjectType(ib, HYPRE_PARCSR));
   CHECK(HYPRE_IJVectorInitialize(ib)); CHECK(HYPRE_IJVectorAssemble(ib));
   CHECK(HYPRE_IJVectorGetObject(ib, (void **) &b));
   CHECK(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, 0, n - 1, &ix));
   CHECK(HYPRE_IJVectorSetObjectType(ix, HYPRE_PARCSR));
   CHECK(HYPRE_IJVectorInitialize(ix)); CHECK(HYPRE_IJVectorAssemble(ix));
   CHECK(HYPRE_IJVectorGetObject(ix, (void **) &x));
   HYPRE_Complex *rhs = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   HYPRE_Complex *solution = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   for (HYPRE_Int depth = 2; depth <= 4; depth++)
      for (HYPRE_Int pair = 0; pair < 7; pair++)
         for (HYPRE_Int coarse = 0; coarse < 2; coarse++)
         {
            HYPRE_Solver s;
            CHECK(HYPRE_BoomerAMGCreate(&s));
            CHECK(HYPRE_BoomerAMGSetMaxIter(s, 1));
            CHECK(HYPRE_BoomerAMGSetTol(s, 0.0));
            CHECK(HYPRE_BoomerAMGSetCoarsenType(s, 10));
            CHECK(HYPRE_BoomerAMGSetInterpType(s, 6));
            CHECK(HYPRE_BoomerAMGSetMaxLevels(s, depth));
            CHECK(HYPRE_BoomerAMGSetMaxCoarseSize(s, depth == 2 ? 60 : depth == 3 ? 30 : 15));
            CHECK(HYPRE_BoomerAMGSetRelaxOrder(s, 0));
            CHECK(HYPRE_BoomerAMGSetCycleRelaxType(s, pairs[pair][0], 1));
            CHECK(HYPRE_BoomerAMGSetCycleRelaxType(s, pairs[pair][1], 2));
            CHECK(HYPRE_BoomerAMGSetCycleRelaxType(s, coarse ? 0 : 9, 3));
            if (pair == 6)
            {
               CHECK(HYPRE_BoomerAMGSetSmoothType(s, 4));
               CHECK(HYPRE_BoomerAMGSetSmoothNumLevels(s, depth - 1));
            }
            CHECK(HYPRE_BoomerAMGSetup(s, A, b, x));
            hypre_ParAMGData *d = (hypre_ParAMGData *) s;
            if (hypre_ParAMGDataNumLevels(d) != depth ||
                hypre_ParAMGDataGridRelaxType(d)[3] != (coarse ? 0 : 9)) { return 2; }

            for (HYPRE_Int weight = 0; weight < 2; weight++)
               for (HYPRE_Int sw = 0; sw < 6; sw++)
                  for (HYPRE_Int cycle = 0; cycle < 3; cycle++)
                  {
                     CHECK(HYPRE_BoomerAMGSetRelaxWt(s, weight ? 2.0 / 3.0 : 1.0));
                     CHECK(HYPRE_BoomerAMGSetCycleNumSweeps(s, sweep_pairs[sw][0], 1));
                     CHECK(HYPRE_BoomerAMGSetCycleNumSweeps(s, sweep_pairs[sw][1], 2));
                     CHECK(HYPRE_BoomerAMGSetCycleType(s, cycle ? 2 : 1));
                     CHECK(HYPRE_BoomerAMGSetFCycle(s, cycle == 2));
                     for (HYPRE_Int repetition = 0; repetition < 2; repetition++)
                     {
                        for (HYPRE_Int i = 0; i < n; i++) { rhs[i] = sin((i + 1) * (repetition + 1.0)); }
                        CHECK(HYPRE_ParVectorSetConstantValues(x, 0.0));
                        CHECK(HYPRE_BoomerAMGSolve(s, A, b, x));
                        for (HYPRE_Int i = 0; i < n; i++) { expected[i] = solution[i]; }
                        if (repetition == 0)
                        {
                           for (HYPRE_Int i = 0; i < n; i++) { first_solution[i] = solution[i]; }
                        }
                        else if (pair < 6 && coarse == 0 && weight == 0 && sw < 2)
                        {
                           /* With matched adjoint sweeps and an exact coarse solve, test
                            * <b1, M b2> = <b2, M b1> independently of the reference. */
                           HYPRE_Complex lhs = 0.0, rhs_dot = 0.0;
                           HYPRE_Real scale_dot = 0.0;
                           for (HYPRE_Int i = 0; i < n; i++)
                           {
                              HYPRE_Complex a = sin(i + 1.0) * solution[i];
                              HYPRE_Complex bterm = sin(2 * (i + 1.0)) * first_solution[i];
                              lhs += a;
                              rhs_dot += bterm;
                              scale_dot += hypre_cabs(a) + hypre_cabs(bterm);
                           }
                           symmetry_cases++;
                           if (!(hypre_cabs(lhs - rhs_dot) <=
                                 1000 * HYPRE_REAL_EPSILON * hypre_max(1.0, scale_dot)))
                           {
                              hypre_printf("Symmetry failed: depth=%d pair=%d cycle=%d sweeps=%d\n",
                                           depth, pair, cycle, sw + 1);
                              failures++;
                           }
                        }
                        CHECK(HYPRE_ParVectorSetConstantValues(x, 0.0));
                        reference_cycle(d, 0, cycle != 0, 0);
                        if (cycle == 2) { reference_cycle(d, 0, 1, 1); }
                        HYPRE_Real error = 0.0, scale = 0.0;
                        HYPRE_Int finite = 1;
                        for (HYPRE_Int i = 0; i < n; i++)
                        {
                           finite = finite && isfinite(hypre_cabs(solution[i])) && isfinite(hypre_cabs(expected[i]));
                           error = hypre_max(error, hypre_cabs(solution[i] - expected[i]));
                           scale = hypre_max(scale, hypre_cabs(expected[i]));
                        }
                        cases++;
                        if (!finite || !(error <= 1000 * HYPRE_REAL_EPSILON * hypre_max(1.0, scale)))
                        {
                           if (failures < 12)
                           {
                              hypre_printf("depth=%d pair=%d coarse=%d cycle=%d sweeps=%d/%d weight=%d "
                                           "error=%g\n",
                                           depth, pair, coarse, cycle, sweep_pairs[sw][0], sweep_pairs[sw][1], weight,
                                           (double) error);
                           }
                           failures++;
                           failed_cycles[cycle]++;
                        }
                     }
                  }
            CHECK(HYPRE_BoomerAMGDestroy(s));
         }
   hypre_printf("W cycle: %d reference cases, %d symmetry cases, %d failures\n",
                cases, symmetry_cases, failures);
   hypre_printf("Reference failures by cycle: V=%d W=%d F+W=%d\n",
                failed_cycles[0], failed_cycles[1], failed_cycles[2]);
   CHECK(HYPRE_IJMatrixDestroy(ij));
   CHECK(HYPRE_IJVectorDestroy(ib)); CHECK(HYPRE_IJVectorDestroy(ix));
   CHECK(HYPRE_Finalize()); hypre_MPI_Finalize();
   return failures ? 1 : 0;
}
