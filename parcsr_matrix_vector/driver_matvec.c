 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface 
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   hypre_CSRMatrix     *matrix;
   hypre_CSRMatrix     *matrix1;
   hypre_ParCSRMatrix  *par_matrix;
   hypre_Vector        *x_local;
   hypre_Vector        *y_local;
   hypre_Vector        *y2_local;
   hypre_ParVector     *x;
   hypre_ParVector     *x2;
   hypre_ParVector     *y;
   hypre_ParVector     *y2;

   int          num_procs, my_id;
   int		local_size;
   int		global_num_rows;
   int		global_num_cols;
   int		first_index;
   int 		i, ierr=0;
   double 	*data, *data2;
   int 		*row_starts, *col_starts;
   char		file_name[80];
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   printf(" my_id: %d num_procs: %d\n", my_id, num_procs);
 
   if (my_id == 0) 
   {
	matrix = hypre_ReadCSRMatrix("input");
   	printf(" read input\n");
   }
/*   row_starts = hypre_CTAlloc(int,4);
   col_starts = hypre_CTAlloc(int,4);
   row_starts[0] = 0;
   row_starts[1] = 3;
   row_starts[2] = 3;
   row_starts[3] = 7;
   col_starts[0] = 0;
   col_starts[1] = 3;
   col_starts[2] = 3;
   col_starts[3] = 9;
*/
   row_starts = NULL;
   col_starts = NULL; 
   par_matrix = hypre_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, matrix, 
		row_starts, col_starts);
   printf(" converted\n");

   matrix1 = hypre_ParCSRMatrixToCSRMatrixAll(par_matrix);

   sprintf(file_name,"matrix1.%d",my_id);

   if (matrix1) hypre_PrintCSRMatrix(matrix1, file_name);

   hypre_PrintParCSRMatrix(par_matrix,"matrix");

   par_matrix = hypre_ReadParCSRMatrix(MPI_COMM_WORLD,"matrix");

   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   printf(" global_num_cols %d\n", global_num_cols);
   global_num_rows = hypre_ParCSRMatrixGlobalNumRows(par_matrix);
 
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   first_index = col_starts[my_id];
   local_size = col_starts[my_id+1] - first_index;

   x = hypre_CreateParVector(MPI_COMM_WORLD,global_num_cols,col_starts);
   hypre_SetParVectorPartitioningOwner(x,0);
   hypre_InitializeParVector(x);
   x_local = hypre_ParVectorLocalVector(x);
   data = hypre_VectorData(x_local);
 
   for (i=0; i < local_size; i++)
        data[i] = first_index+i+1;
   x2 = hypre_CreateParVector(MPI_COMM_WORLD,global_num_cols,col_starts);
   hypre_SetParVectorPartitioningOwner(x2,0);
   hypre_InitializeParVector(x2);
   hypre_SetParVectorConstantValues(x2,2.0);

   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   first_index = row_starts[my_id];
   local_size = row_starts[my_id+1] - first_index;
   y = hypre_CreateParVector(MPI_COMM_WORLD,global_num_rows,row_starts);
   hypre_SetParVectorPartitioningOwner(y,0);
   hypre_InitializeParVector(y);
   y_local = hypre_ParVectorLocalVector(y);

   y2 = hypre_CreateParVector(MPI_COMM_WORLD,global_num_rows,row_starts);
   hypre_SetParVectorPartitioningOwner(y2,0);
   hypre_InitializeParVector(y2);
   y2_local = hypre_ParVectorLocalVector(y2);
   data2 = hypre_VectorData(y2_local);
 
   for (i=0; i < local_size; i++)
        data2[i] = first_index+i+1;

   hypre_SetParVectorConstantValues(y,1.0);
   printf(" initialized vectors\n");

   hypre_GenerateMatvecCommunicationInfo(par_matrix);

   hypre_ParMatvec ( 1.0, par_matrix, x, 1.0, y);
   printf(" did matvec\n");

   hypre_PrintParVector(y, "result");

   ierr = hypre_ParMatvecT ( 1.0, par_matrix, y2, 1.0, x2);
   printf(" did matvecT %d\n", ierr);

   hypre_PrintParVector(x2, "transp"); 

   hypre_DestroyParCSRMatrix(par_matrix);
   hypre_DestroyParVector(x);
   hypre_DestroyParVector(x2);
   hypre_DestroyParVector(y);
   hypre_DestroyParVector(y2);
   if (my_id == 0) hypre_DestroyCSRMatrix(matrix);
   if (matrix1) hypre_DestroyCSRMatrix(matrix1);

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

