 
#include "headers.h"
 
/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface 
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   hypre_ParVector   *vector1;
   hypre_ParVector   *vector2;
   hypre_ParVector   *tmp_vector;

   int          num_procs, my_id;
   int	 	global_size = 20;
   int		local_size;
   int		first_index, last_index;	
   int 		i;
   double	prod;
   double 	*data, *data2;
   hypre_Vector *local_vector; 
   hypre_Vector *local_vector2;
   char		*filename;
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id );

   printf(" my_id: %d num_procs: %d\n", my_id, num_procs);
 
   MPE_Decomp1d(global_size, num_procs, my_id, &first_index, &last_index);
   
   local_size = last_index - first_index + 1;
   vector1 = hypre_CreateParVector(MPI_COMM_WORLD,global_size,
				first_index-1,local_size);
   hypre_InitializeParVector(vector1);
   local_vector = hypre_ParVectorLocalVector(vector1);
   data = hypre_VectorData(local_vector);

   for (i=0; i < local_size; i++)
   	data[i] = first_index+i;

   hypre_PrintParVector(vector1, "Vector");

   local_vector2 = hypre_CreateVector(global_size);
   hypre_InitializeVector(local_vector2);
   data2 = hypre_VectorData(local_vector2);
   for (i=0; i < global_size; i++)
	data2[i] = i+1;

   vector2 = hypre_VectorToParVector(MPI_COMM_WORLD,local_vector2,NULL);

   hypre_PrintParVector(vector2, "Convert");

   /*-----------------------------------------------------------
    * Copy the vector into tmp_vector
    *-----------------------------------------------------------*/

   tmp_vector = hypre_CreateParVector(MPI_COMM_WORLD,global_size,
					first_index-1,local_size);
   hypre_InitializeParVector(tmp_vector);
   hypre_CopyParVector(vector1, tmp_vector);

   printf (" \n"); 
   filename = "Copy";
   hypre_PrintParVector(tmp_vector,filename);

   /*-----------------------------------------------------------
    * Scale tmp_vector
    *-----------------------------------------------------------*/

   hypre_ScaleParVector(2.0, tmp_vector);

   hypre_PrintParVector(tmp_vector,"Scale");

   /*-----------------------------------------------------------
    * Do an Axpy (2*vector - vector) = vector
    *-----------------------------------------------------------*/

   hypre_ParAxpy(-1.0, vector1, tmp_vector);

   hypre_PrintParVector(tmp_vector,"Axpy");

   /*-----------------------------------------------------------
    * Do an inner product vector* tmp_vector
    *-----------------------------------------------------------*/

   prod = hypre_ParInnerProd(MPI_COMM_WORLD, vector1, tmp_vector);

   printf (" prod: %8.2f \n", prod);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_DestroyParVector(vector1);
   hypre_DestroyParVector(vector2); 
   hypre_DestroyParVector(tmp_vector);
   hypre_DestroyVector(local_vector2); 

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

