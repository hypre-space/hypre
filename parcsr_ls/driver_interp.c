 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface 
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   hypre_ParCSRMatrix  *S;
   hypre_ParCSRMatrix  *A;
   hypre_ParCSRMatrix  *P;
   hypre_ParVector     *CF_marker; 
   int                 *CF_marker_int;

   int 		       *A_row_starts, *A_col_starts;
   int 		       *S_row_starts, *S_col_starts;
   int                 *CF_marker_starts;

   hypre_CSRMatrix     *S_in;
   hypre_CSRMatrix     *A_in; 
   hypre_Vector        *CF_marker_in;


   hypre_Vector        *CF_local;
   double              *CF_local_data;

   int                  num_procs, my_id;
   int 	         	ierr=0;
   int                  fine_size;

   int		        j;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   
   printf(" my_id: %d num_procs: %d\n", my_id, num_procs);
 
   if (my_id == 0) 
   {
	S_in = hypre_ReadCSRMatrix("Smatrix_0.ysmp");
   	printf(" read influence matrix\n");
	A_in = hypre_ReadCSRMatrix("Amatrix_0.ysmp");
   	printf(" read fine_op\n");

        fine_size =  hypre_CSRMatrixNumRows(A_in);

/*        CF_marker_in = hypre_ReadVector("CF_marker_0"); 
        printf(" read CF_marker\n");  */
   }
   else
   {
        S_in = NULL;
        A_in = NULL;
        CF_marker_in = NULL;
   }

   /* Why does it run with the next two lines outside the if (my_id==0) 
      test, but not with them inside it???  Or, why does it run with 
      A_in and S_in read inside that block, but not CF_marker_in??? */
 
   CF_marker_in = hypre_ReadVector("CF_marker_0"); 
   printf(" read CF_marker\n"); 

   A_row_starts = NULL;
   A_col_starts = NULL;
   A = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_in, 
                                     A_row_starts, A_col_starts);
   printf(" A converted\n");


   hypre_GenerateMatvecCommunicationInfo(A);
   printf(" generated A_CommPkg \n");

   S_row_starts = NULL;
   S_col_starts = NULL;
   S = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, S_in, 
                                     S_row_starts, S_col_starts);
   printf(" S converted\n");

   hypre_GenerateMatvecCommunicationInfo(S);
   printf(" generated S_CommPkg \n");
 
   CF_marker_starts = NULL;
   CF_marker = hypre_VectorToParVector(MPI_COMM_WORLD, CF_marker_in,
                                       &CF_marker_starts);

   CF_local = hypre_ParVectorLocalVector(CF_marker);
   CF_local_data = hypre_VectorData(CF_local);


   CF_marker_int = hypre_CTAlloc(int,hypre_VectorSize(CF_local));
   for (j = 0; j < hypre_VectorSize(CF_local); j++)
   {
     CF_marker_int[j] = (int) CF_local_data[j];
   }
   
 
   printf(" CF_marker converted\n");

   hypre_ParAMGBuildInterp(A,CF_marker_int,S,&P); 
   printf(" built interp\n");

/*   hypre_PrintParCSRMatrix(P, "Pmatrix_0.ysmp"); */

   hypre_DestroyParCSRMatrix(A);
   hypre_DestroyParCSRMatrix(S);
   hypre_DestroyParCSRMatrix(P); 
   if (my_id == 0)
   {
   	hypre_DestroyCSRMatrix(A_in);
   	hypre_DestroyCSRMatrix(S_in);
   }

   /* Finalize MPI */
   MPI_Finalize();

   return ierr;
}

