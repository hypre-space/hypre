/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured Boolean matrix interface , A * A^T
 *--------------------------------------------------------------------------*/

HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   hypre_ParCSRBooleanMatrix     *A;
   hypre_ParCSRBooleanMatrix     *C;
   hypre_CSRBooleanMatrix *As;
   HYPRE_BigInt *row_starts, *col_starts;
   HYPRE_Int num_procs, my_id;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id);
   row_starts = NULL;
   col_starts = NULL;

   if (my_id == 0)
   {
      As = hypre_CSRBooleanMatrixRead("inpr");
      hypre_printf(" read input A\n");
   }
   A = hypre_CSRBooleanMatrixToParCSRBooleanMatrix(hypre_MPI_COMM_WORLD, As, row_starts,
                                                   col_starts);
   row_starts = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(A);

   hypre_ParCSRBooleanMatrixPrint(A, "echo_A" );
   hypre_ParCSRBooleanMatrixPrintIJ(A, "echo_AIJ" );
   C = hypre_ParBooleanAAt( A );
   hypre_ParCSRBooleanMatrixPrint(C, "result");
   hypre_ParCSRBooleanMatrixPrintIJ(C, "resultIJ");

   if (my_id == 0)
   {
      hypre_CSRBooleanMatrixDestroy(As);
   }
   hypre_ParCSRBooleanMatrixDestroy(A);
   hypre_ParCSRBooleanMatrixDestroy(C);

   hypre_MPI_Finalize();

   return 0;
}

