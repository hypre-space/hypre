/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for Boolean matrix multiplication, C=A*B . 
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   hypre_ParCSRBooleanMatrix     *A;
   hypre_ParCSRBooleanMatrix     *B;
   hypre_ParCSRBooleanMatrix     *C;
   hypre_CSRBooleanMatrix *As;
   hypre_CSRBooleanMatrix *Bs;
   int *row_starts, *col_starts;
   int num_procs, my_id;
   int a_nrows, a_ncols, b_nrows, b_ncols;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_id);
   row_starts = NULL;
   col_starts = NULL;

   if (my_id == 0)
   {
   	As = hypre_CSRBooleanMatrixRead("inpr");
        a_nrows = hypre_CSRBooleanMatrix_Get_NRows( As );
        a_ncols = hypre_CSRBooleanMatrix_Get_NCols( As );
   	printf(" read input A(%i,%i)\n",a_nrows,a_ncols);
   	Bs = hypre_CSRBooleanMatrixRead("input");
        b_nrows = hypre_CSRBooleanMatrix_Get_NRows( Bs );
        b_ncols = hypre_CSRBooleanMatrix_Get_NCols( Bs );
   	printf(" read input B(%i,%i)\n",b_nrows,b_ncols);
        if ( a_ncols != b_nrows ) {
           printf( "incompatible matrix dimensions! (%i,%i)*(%i,%i)\n",
                   a_nrows,a_ncols,b_nrows,b_ncols );
           exit(1);
        }
        
   }
   A = hypre_CSRBooleanMatrixToParCSRBooleanMatrix
      (MPI_COMM_WORLD, As, row_starts, col_starts);
   row_starts = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(A);
   B = hypre_CSRBooleanMatrixToParCSRBooleanMatrix
      (MPI_COMM_WORLD, Bs, col_starts, row_starts);
   hypre_ParCSRBooleanMatrixSetRowStartsOwner(B,0);
   hypre_ParCSRBooleanMatrixSetColStartsOwner(B,0);
   C = hypre_ParBooleanMatmul(A,B);
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

   MPI_Finalize();

   return 0;
}

