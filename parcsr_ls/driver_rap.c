 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface 
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   hypre_ParCSRMatrix  *P;
   hypre_ParCSRMatrix  *A;
   hypre_ParCSRMatrix  *RAP;
   hypre_CSRMatrix     *P_in;
   hypre_CSRMatrix     *A_in;
   int 		       *fine_partitioning;
   int 		       *coarse_partitioning;
   int          num_procs, my_id;
   int 		ierr=0;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   printf(" my_id: %d num_procs: %d\n", my_id, num_procs);
 
   if (my_id == 0) 
   {
	P_in = hypre_ReadCSRMatrix("prolongation");
   	printf(" read prolongation\n");
	A_in = hypre_ReadCSRMatrix("fine_op");
   	printf(" read fine_op\n");
   }
   P = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, P_in);
   printf(" P converted\n");

   fine_partitioning = NULL;
   coarse_partitioning = NULL;

   hypre_GenerateMatvecCommunicationInfo(P,coarse_partitioning,
		fine_partitioning);
   printf(" generated P_CommPkg \n");

   A = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_in);
   printf(" A converted\n");

   hypre_GenerateMatvecCommunicationInfo(A,fine_partitioning,fine_partitioning);
   printf(" generated A_CommPkg \n");
 
   hypre_ParAMGBuildCoarseOperator(P,A,P,&RAP,coarse_partitioning);
   printf(" did rap\n");

   hypre_PrintParCSRMatrix(RAP, "rap"); 

   hypre_DestroyParCSRMatrix(A);
   hypre_DestroyParCSRMatrix(P);
   hypre_DestroyParCSRMatrix(RAP);
   if (my_id == 0)
   {
   	hypre_DestroyCSRMatrix(A_in);
   	hypre_DestroyCSRMatrix(P_in);
   }
   hypre_TFree(fine_partitioning);
   hypre_TFree(coarse_partitioning);
   /* Finalize MPI */
   MPI_Finalize();

   return ierr;
}

