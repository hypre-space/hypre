 
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
   C = hypre_ParBooleanMatmul(B,A);
   hypre_ParCSRBooleanMatrixPrint(B, "echo_B" );
   hypre_ParCSRBooleanMatrixPrint(A, "echo_A" );
   hypre_ParCSRBooleanMatrixPrint(C, "result");

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

