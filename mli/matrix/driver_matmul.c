 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for Boolean matrix multiplication, C=A*B . 
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   MLI_ParCSRBooleanMatrix     *A;
   MLI_ParCSRBooleanMatrix     *B;
   MLI_ParCSRBooleanMatrix     *C;
   MLI_CSRBooleanMatrix *As;
   MLI_CSRBooleanMatrix *Bs;
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
   	As = MLI_CSRBooleanMatrixRead("inpr");
        a_nrows = MLI_CSRBooleanMatrix_Get_NRows( As );
        a_ncols = MLI_CSRBooleanMatrix_Get_NCols( As );
   	printf(" read input A(%i,%i)\n",a_nrows,a_ncols);
   	Bs = MLI_CSRBooleanMatrixRead("input");
        b_nrows = MLI_CSRBooleanMatrix_Get_NRows( Bs );
        b_ncols = MLI_CSRBooleanMatrix_Get_NCols( Bs );
   	printf(" read input B(%i,%i)\n",b_nrows,b_ncols);
        if ( a_ncols != b_nrows ) {
           printf( "incompatible matrix dimensions! (%i,%i)*(%i,%i)\n",
                   a_nrows,a_ncols,b_nrows,b_ncols );
           exit(1);
        }
        
   }
   A = MLI_CSRBooleanMatrixToParCSRBooleanMatrix
      (MPI_COMM_WORLD, As, row_starts, col_starts);
   row_starts = MLI_ParCSRBooleanMatrix_Get_RowStarts(A);
   col_starts = MLI_ParCSRBooleanMatrix_Get_ColStarts(A);
   B = MLI_CSRBooleanMatrixToParCSRBooleanMatrix
      (MPI_COMM_WORLD, Bs, col_starts, row_starts);
   MLI_ParCSRBooleanMatrixSetRowStartsOwner(B,0);
   MLI_ParCSRBooleanMatrixSetColStartsOwner(B,0);
   C = MLI_ParBooleanMatmul(B,A);
   MLI_ParCSRBooleanMatrixPrint(B, "echo_B" );
   MLI_ParCSRBooleanMatrixPrint(A, "echo_A" );
   MLI_ParCSRBooleanMatrixPrint(C, "result");

   if (my_id == 0)
   {
	MLI_CSRBooleanMatrixDestroy(As);
   	MLI_CSRBooleanMatrixDestroy(Bs);
   }
   MLI_ParCSRBooleanMatrixDestroy(A);
   MLI_ParCSRBooleanMatrixDestroy(B);
   MLI_ParCSRBooleanMatrixDestroy(C);

   MPI_Finalize();

   return 0;
}

