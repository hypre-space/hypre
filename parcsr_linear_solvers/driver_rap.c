 
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
	P_in = hypre_ReadCSRMatrix("../../Parrap/prolongation");
   	printf(" read prolongation\n");
	A_in = hypre_ReadCSRMatrix("../../Parrap/fine_op");
   	printf(" read fine_op\n");
   }

   fine_partitioning = NULL;
   coarse_partitioning = NULL;
/*   coarse_partitioning = hypre_CTAlloc(int,5);
   coarse_partitioning[0] = 0;
   coarse_partitioning[1] = 2;
   coarse_partitioning[2] = 4;
   coarse_partitioning[3] = 7;
   coarse_partitioning[4] = 10;
*/

   P = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, P_in, fine_partitioning,
		coarse_partitioning);

   printf(" P converted\n");

   fine_partitioning = hypre_ParCSRMatrixRowStarts(P);
   A = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_in, fine_partitioning,
	fine_partitioning);
   hypre_SetParCSRMatrixPartitioningOwner(A,0);
   printf(" A converted\n");

   hypre_GenerateMatvecCommunicationInfo(A);
   printf(" generated A_CommPkg \n");
 
   hypre_GetCommPkgRTFromCommPkgA(P,A);
   printf(" generated P_CommPkg \n");

   hypre_ParAMGBuildCoarseOperator(P,A,P,&RAP);
   hypre_SetParCSRMatrixPartitioningOwner(RAP,0);
   printf(" did rap\n");

   hypre_PrintParCSRMatrix(RAP, "rap"); 
   hypre_DestroyParCSRMatrix(RAP);

   hypre_DestroyParCSRMatrix(A);
   hypre_DestroyParCSRMatrix(P);
   if (my_id == 0)
   {
   	hypre_DestroyCSRMatrix(A_in);
   	hypre_DestroyCSRMatrix(P_in);
   }
   /* Finalize MPI */
   MPI_Finalize();

   return ierr;
}

