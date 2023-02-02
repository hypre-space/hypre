/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for Boolean matrix multiplication, C=A*B .
 *--------------------------------------------------------------------------*/

HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   hypre_ParCSRBooleanMatrix     *A;
   hypre_ParCSRBooleanMatrix     *B;
   hypre_ParCSRBooleanMatrix     *C;
   hypre_CSRBooleanMatrix *As;
   hypre_CSRBooleanMatrix *Bs;
   HYPRE_BigInt *row_starts, *col_starts;
   HYPRE_Int num_procs, my_id;
   HYPRE_Int a_nrows, a_ncols, b_nrows, b_ncols;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id);
   row_starts = NULL;
   col_starts = NULL;

   if (my_id == 0)
   {
      As = hypre_CSRBooleanMatrixRead("inpr");
      a_nrows = hypre_CSRBooleanMatrix_Get_NRows( As );
      a_ncols = hypre_CSRBooleanMatrix_Get_NCols( As );
      hypre_printf(" read input A(%i,%i)\n", a_nrows, a_ncols);
      Bs = hypre_CSRBooleanMatrixRead("input");
      b_nrows = hypre_CSRBooleanMatrix_Get_NRows( Bs );
      b_ncols = hypre_CSRBooleanMatrix_Get_NCols( Bs );
      hypre_printf(" read input B(%i,%i)\n", b_nrows, b_ncols);
      if ( a_ncols != b_nrows )
      {
         hypre_printf( "incompatible matrix dimensions! (%i,%i)*(%i,%i)\n",
                       a_nrows, a_ncols, b_nrows, b_ncols );
         exit(1);
      }

   }
   A = hypre_CSRBooleanMatrixToParCSRBooleanMatrix
       (hypre_MPI_COMM_WORLD, As, row_starts, col_starts);
   row_starts = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(A);
   B = hypre_CSRBooleanMatrixToParCSRBooleanMatrix
       (hypre_MPI_COMM_WORLD, Bs, col_starts, row_starts);
   hypre_ParCSRBooleanMatrixSetRowStartsOwner(B, 0);
   hypre_ParCSRBooleanMatrixSetColStartsOwner(B, 0);
   C = hypre_ParBooleanMatmul(A, B);
   hypre_ParCSRBooleanMatrixPrint(A, "echo_A" );
   hypre_ParCSRBooleanMatrixPrint(B, "echo_B" );
   hypre_ParCSRBooleanMatrixPrint(C, "result");
   hypre_ParCSRBooleanMatrixPrintIJ(C, "result_Cij");

   if (my_id == 0)
   {
      hypre_CSRBooleanMatrixDestroy(As);
      hypre_CSRBooleanMatrixDestroy(Bs);
   }
   hypre_ParCSRBooleanMatrixDestroy(A);
   hypre_ParCSRBooleanMatrixDestroy(B);
   hypre_ParCSRBooleanMatrixDestroy(C);

   hypre_MPI_Finalize();

   return 0;
}

