/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Unit tests for BoomerAMG solves with multicomponent vectors: hybrid
 * Gauss-Seidel relaxation (with and without damping) and Gaussian
 * elimination on the coarsest level treat each component like a single
 * vector.
 *--------------------------------------------------------------------------*/

#include <math.h>
#include <stdio.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "hypre_unit.h"

#define MULTICOMPONENT_TOL (100 * HYPRE_REAL_EPSILON)

/* max_i |x_i - y_i| / max_i |y_i| over all processes */
static HYPRE_Real
RelMaxDiff(MPI_Comm comm, HYPRE_Int n, const HYPRE_Complex *x, const HYPRE_Complex *y)
{
   HYPRE_Real loc[2] = {0.0, 0.0}, glob[2];
   for (HYPRE_Int i = 0; i < n; i++)
   {
      loc[0] = hypre_max(loc[0], hypre_cabs(x[i] - y[i]));
      loc[1] = hypre_max(loc[1], hypre_cabs(y[i]));
   }
   hypre_MPI_Allreduce(loc, glob, 2, HYPRE_MPI_REAL, hypre_MPI_MAX, comm);
   return glob[1] > 0.0 ? glob[0] / glob[1] : glob[0];
}

static HYPRE_Int
test_MulticomponentRelax(MPI_Comm comm)
{
   HYPRE_Int           my_id, num_procs, error = 0;
   HYPRE_Int           relax_types[] = {3, 6, 8, 13, 14};
   HYPRE_Int           num_relax_types = 5, num_vectors_list[] = {2, 3};
   HYPRE_Real          relax_wts[] = {1.0, 0.75}, outer_wts[] = {1.0, 0.8};
   HYPRE_Real          values[4] = {6.0, -1.0, -1.0, -1.0};
   HYPRE_ParCSRMatrix  A;
   hypre_ParCSRMatrix *parA;

   /* All processes take part */
   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);
   if (my_id == 0)
   {
      hypre_printf("%s (%d procs): ", __func__, num_procs);
   }

   /* 3D 7-point Laplacian, slabs of one process each */
   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(comm, 12, 11, 10 * num_procs, 1, 1,
                                              num_procs, 0, 0, my_id, values);
   parA = (hypre_ParCSRMatrix *) A;

   for (HYPRE_Int ri = 0; ri < num_relax_types; ri++)
   {
      for (HYPRE_Int ci = 0; ci < 4; ci++)
      {
         HYPRE_Int        nv = num_vectors_list[ci % 2];
         HYPRE_Int        wi = ci / 2;
         HYPRE_Int        n = hypre_ParCSRMatrixNumRows(parA);
         HYPRE_BigInt     N = hypre_ParCSRMatrixGlobalNumRows(parA);
         HYPRE_BigInt    *starts = hypre_ParCSRMatrixRowStarts(parA);
         HYPRE_Solver     amg;
         hypre_ParVector *b_mv, *x_mv, *b, *x, *x_ref[3];
         HYPRE_Complex   *b_mv_data, *x_mv_data;
         HYPRE_Int        ierr;

         HYPRE_BoomerAMGCreate(&amg);
         HYPRE_BoomerAMGSetPrintLevel(amg, 0);
         HYPRE_BoomerAMGSetMaxIter(amg, 2);
         HYPRE_BoomerAMGSetTol(amg, 0.0);
         HYPRE_BoomerAMGSetRelaxType(amg, relax_types[ri]);
         HYPRE_BoomerAMGSetMaxCoarseSize(amg, 50); /* keeps Gaussian elimination on the coarsest level */
         HYPRE_BoomerAMGSetCycleRelaxType(amg, 9, 3);
         HYPRE_BoomerAMGSetNumSweeps(amg, 1);
         HYPRE_BoomerAMGSetRelaxWt(amg, relax_wts[wi]);
         HYPRE_BoomerAMGSetOuterWt(amg, outer_wts[wi]);

         b_mv = hypre_ParMultiVectorCreate(comm, N, starts, nv);
         x_mv = hypre_ParMultiVectorCreate(comm, N, starts, nv);
         hypre_ParVectorInitialize_v2(b_mv, HYPRE_MEMORY_HOST);
         hypre_ParVectorInitialize_v2(x_mv, HYPRE_MEMORY_HOST);
         b = hypre_ParVectorCreate(comm, N, starts);
         x = hypre_ParVectorCreate(comm, N, starts);
         hypre_ParVectorInitialize_v2(b, HYPRE_MEMORY_HOST);
         hypre_ParVectorInitialize_v2(x, HYPRE_MEMORY_HOST);
         b_mv_data = hypre_VectorData(hypre_ParVectorLocalVector(b_mv));
         x_mv_data = hypre_VectorData(hypre_ParVectorLocalVector(x_mv));
         for (HYPRE_Int k = 0; k < nv; k++)
         {
            for (HYPRE_Int i = 0; i < n; i++)
            {
               HYPRE_BigInt gi = starts[0] + i;
               b_mv_data[i + k * n] = sin(0.1 * (HYPRE_Real) gi + k) + 0.5 * k;
            }
         }
         HYPRE_BoomerAMGSetup(amg, A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);

         /* Separate solves of the components */
         for (HYPRE_Int k = 0; k < nv; k++)
         {
            hypre_TMemcpy(hypre_VectorData(hypre_ParVectorLocalVector(b)), b_mv_data + k * n,
                          HYPRE_Complex, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
            hypre_ParVectorSetZeros(x);
            HYPRE_BoomerAMGSolve(amg, A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
            x_ref[k] = hypre_ParVectorCloneDeep_v2(x, HYPRE_MEMORY_HOST);
         }

         /* All components in one solve, then a single-component solve again */
         hypre_ParVectorSetZeros(x_mv);
         ierr = HYPRE_BoomerAMGSolve(amg, A, (HYPRE_ParVector) b_mv, (HYPRE_ParVector) x_mv);
         if (ierr)
         {
            if (my_id == 0)
            {
               hypre_printf("\n  relax_type %d, weight %g, %d components: solve returned %d",
                            relax_types[ri], relax_wts[wi], nv, ierr);
            }
            error = 1;
            HYPRE_ClearAllErrors();
         }
         else
         {
            for (HYPRE_Int k = 0; k < nv; k++)
            {
               HYPRE_Real d = RelMaxDiff(comm, n, x_mv_data + k * n,
                                         hypre_VectorData(hypre_ParVectorLocalVector(x_ref[k])));
               if (!(d <= MULTICOMPONENT_TOL))
               {
                  if (my_id == 0)
                  {
                     hypre_printf("\n  relax_type %d, weight %g, %d components: component %d differs by %e",
                                  relax_types[ri], relax_wts[wi], nv, k, d);
                  }
                  error = 1;
               }
            }
         }
         hypre_TMemcpy(hypre_VectorData(hypre_ParVectorLocalVector(b)), b_mv_data,
                       HYPRE_Complex, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_ParVectorSetZeros(x);
         HYPRE_BoomerAMGSolve(amg, A, (HYPRE_ParVector) b, (HYPRE_ParVector) x);
         if (RelMaxDiff(comm, n, hypre_VectorData(hypre_ParVectorLocalVector(x)),
                        hypre_VectorData(hypre_ParVectorLocalVector(x_ref[0]))) != 0.0)
         {
            if (my_id == 0)
            {
               hypre_printf("\n  relax_type %d, weight %g, %d components: single solve after "
                            "the multicomponent one differs", relax_types[ri], relax_wts[wi], nv);
            }
            error = 1;
         }

         for (HYPRE_Int k = 0; k < nv; k++)
         {
            hypre_ParVectorDestroy(x_ref[k]);
         }
         hypre_ParVectorDestroy(b_mv);
         hypre_ParVectorDestroy(x_mv);
         hypre_ParVectorDestroy(b);
         hypre_ParVectorDestroy(x);
         HYPRE_BoomerAMGDestroy(amg);
      }
   }
   if (my_id == 0 && error) { hypre_printf("\n"); }

   HYPRE_ParCSRMatrixDestroy(A);
   HYPRE_UNIT_PRINT_TEST_RESULT(my_id, error, comm);

   return error;
}

hypre_int
main(hypre_int argc, char *argv[])
{
   MPI_Comm  comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int error = 0;

   hypre_MPI_Init(&argc, &argv);
   HYPRE_Initialize();
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
   hypre_SetNumThreads(1); /* multicomponent relaxation runs on one thread */

   error |= test_MulticomponentRelax(comm);

   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return error;
}
