 
#include "headers.h"
 
#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Example 3:
 *
 *    Read matrix and vector from disk.
 *----------------------------------------------------------------------*/
int
main( int   argc,
      char *argv[] )
{
   int                   matrix_num_ghost[6] = { 0, 0, 0, 0, 0, 0};
   int                   vector_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   hypre_StructMatrix   *matrix;
   hypre_StructVector   *vector;
   hypre_StructVector   *tmp_vector;

   hypre_StructGrid     *matrix_grid;
   hypre_StructGrid     *vector_grid;

   int                   num_procs, myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);
#endif

   hypre_InitMemoryDebug(myid);

   /*-----------------------------------------------------------
    * Read in the matrix
    *-----------------------------------------------------------*/

   matrix = hypre_ReadStructMatrix(MPI_COMM_WORLD,
                                   "zin_matrix", matrix_num_ghost);
   matrix_grid = hypre_StructMatrixGrid(matrix);
 
   hypre_PrintStructMatrix("zout_matrix", matrix, 0);

   /*-----------------------------------------------------------
    * Read in the vector
    *-----------------------------------------------------------*/

   vector = hypre_ReadStructVector(MPI_COMM_WORLD,
                                   "zin_vector", vector_num_ghost);
   vector_grid = hypre_StructVectorGrid(vector);
 
   hypre_PrintStructVector("zout_vector", vector, 0);

   /*-----------------------------------------------------------
    * Do a matvec
    *-----------------------------------------------------------*/

   tmp_vector = hypre_NewStructVector(MPI_COMM_WORLD, vector_grid);
   hypre_InitializeStructVector(tmp_vector);
   hypre_AssembleStructVector(tmp_vector);

   hypre_StructMatvec(1.0, matrix, vector, 0.0, tmp_vector);

   hypre_PrintStructVector("zout_matvec", tmp_vector, 0);

   /*-----------------------------------------------------------
    * Copy the vector into tmp_vector
    *-----------------------------------------------------------*/

   hypre_StructCopy(vector, tmp_vector);

   hypre_PrintStructVector("zout_copy", tmp_vector, 0);

   /*-----------------------------------------------------------
    * Scale tmp_vector
    *-----------------------------------------------------------*/

   hypre_StructScale(2.0, tmp_vector);

   hypre_PrintStructVector("zout_scale", tmp_vector, 0);

   /*-----------------------------------------------------------
    * Do an Axpy (2*vector - vector) = vector
    *-----------------------------------------------------------*/

   hypre_StructAxpy(-1.0, vector, tmp_vector);

   hypre_PrintStructVector("zout_axpy", tmp_vector, 0);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_FreeStructMatrix(matrix);
   hypre_FreeStructVector(vector);
   hypre_FreeStructVector(tmp_vector);

   hypre_FreeStructGrid(matrix_grid);
   hypre_FreeStructGrid(vector_grid);

   hypre_FinalizeMemoryDebug();

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

