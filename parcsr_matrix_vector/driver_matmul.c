 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface 
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   hypre_ParCSRMatrix     *A;
   hypre_ParCSRMatrix     *B;
   hypre_ParCSRMatrix     *C;
   hypre_CSRMatrix *As;
   hypre_CSRMatrix *Bs;
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
   	As = hypre_CSRMatrixRead("inpr");
   	printf(" read input A\n");
   	Bs = hypre_CSRMatrixRead("input");
   	printf(" read input B\n");
   }
   A = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, As, row_starts,
	col_starts);
   row_starts = hypre_ParCSRMatrixRowStarts(A);
   col_starts = hypre_ParCSRMatrixColStarts(A);
   B = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, Bs, col_starts,
	row_starts);
   hypre_ParCSRMatrixSetRowStartsOwner(B,0);
   hypre_ParCSRMatrixSetColStartsOwner(B,0);
   C = hypre_ParMatmul(B,A);
   hypre_ParCSRMatrixPrint(C, "result");

   if (my_id == 0)
   {
	hypre_CSRMatrixDestroy(As);
   	hypre_CSRMatrixDestroy(Bs);
   }
   hypre_ParCSRMatrixDestroy(A);
   hypre_ParCSRMatrixDestroy(B);
   hypre_ParCSRMatrixDestroy(C);

   MPI_Finalize();

   return 0;
}

