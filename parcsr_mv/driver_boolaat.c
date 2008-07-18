/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "headers.h" 
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured Boolean matrix interface , A * A^T
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   hypre_ParCSRBooleanMatrix     *A;
   hypre_ParCSRBooleanMatrix     *C;
   hypre_CSRBooleanMatrix *As;
   int *row_starts, *col_starts;
   int num_procs, my_id;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_id);
   row_starts = NULL;
   col_starts = NULL;

   if (my_id == 0)
   {
   	As = hypre_CSRBooleanMatrixRead("inpr");
   	printf(" read input A\n");
   }
   A = hypre_CSRBooleanMatrixToParCSRBooleanMatrix(MPI_COMM_WORLD, As, row_starts,
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

   MPI_Finalize();

   return 0;
}

