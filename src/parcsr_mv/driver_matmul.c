/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface 
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
main( HYPRE_Int   argc,
      char *argv[] )
{
   hypre_ParCSRMatrix     *A;
   hypre_ParCSRMatrix     *B;
   hypre_ParCSRMatrix     *C;
   hypre_CSRMatrix *As;
   hypre_CSRMatrix *Bs;
   HYPRE_Int *row_starts, *col_starts;
   HYPRE_Int num_procs, my_id;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD,&num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD,&my_id);
   row_starts = NULL;
   col_starts = NULL;

   if (my_id == 0)
   {
   	As = hypre_CSRMatrixRead("inpr");
   	hypre_printf(" read input A\n");
   	Bs = hypre_CSRMatrixRead("input");
   	hypre_printf(" read input B\n");
   }
   A = hypre_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, As, row_starts,
	col_starts);
   row_starts = hypre_ParCSRMatrixRowStarts(A);
   col_starts = hypre_ParCSRMatrixColStarts(A);
   B = hypre_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, Bs, col_starts,
	row_starts);
   hypre_ParCSRMatrixSetRowStartsOwner(B,0);
   hypre_ParCSRMatrixSetColStartsOwner(B,0);
   C = hypre_ParMatmul(B,A);
   hypre_ParCSRMatrixPrint(B, "echo_B" );
   hypre_ParCSRMatrixPrint(A, "echo_A" );
   hypre_ParCSRMatrixPrint(C, "result");

   if (my_id == 0)
   {
	hypre_CSRMatrixDestroy(As);
   	hypre_CSRMatrixDestroy(Bs);
   }
   hypre_ParCSRMatrixDestroy(A);
   hypre_ParCSRMatrixDestroy(B);
   hypre_ParCSRMatrixDestroy(C);

   hypre_MPI_Finalize();

   return 0;
}

