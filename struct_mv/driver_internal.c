 
#include "headers.h"
 
#ifdef HYPRE_DEBUG
#include <cegdb.h>
#endif

#ifdef HYPRE_DEBUG
char malloc_logpath_memory[256];
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
   MPI_Comm             *comm;
   int                   matrix_num_ghost[6] = { 0, 0, 0, 0, 0, 0};
   int                   vector_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   hypre_StructMatrix   *matrix;
   hypre_StructVector   *vector;
   hypre_StructVector   *tmp_vector;

   int                   num_procs, myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   comm = hypre_TAlloc(MPI_Comm, 1);
   MPI_Comm_dup(MPI_COMM_WORLD, comm);
   MPI_Comm_size(*comm, &num_procs );
   MPI_Comm_rank(*comm, &myid );

#ifdef HYPRE_DEBUG
   cegdb(&argc, &argv, myid);
#endif

#ifdef HYPRE_DEBUG
   malloc_logpath = malloc_logpath_memory;
   sprintf(malloc_logpath, "malloc.log.%04d", myid);
#endif

   /*-----------------------------------------------------------
    * Read in the matrix
    *-----------------------------------------------------------*/

   matrix = hypre_ReadStructMatrix(comm, "zin_matrix", matrix_num_ghost);
 
   hypre_PrintStructMatrix("zout_matrix", matrix, 0);

   /*-----------------------------------------------------------
    * Read in the vector
    *-----------------------------------------------------------*/

   vector = hypre_ReadStructVector(comm, "zin_vector", vector_num_ghost);
 
   hypre_PrintStructVector("zout_vector", vector, 0);

   /*-----------------------------------------------------------
    * Do a matvec
    *-----------------------------------------------------------*/

   tmp_vector =
      hypre_NewStructVector(comm, hypre_StructVectorGrid(vector));
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

   hypre_FreeStructGrid(hypre_StructMatrixGrid(matrix));
   hypre_FreeStructMatrix(matrix);
   hypre_FreeStructGrid(hypre_StructVectorGrid(vector));
   hypre_FreeStructVector(vector);
   hypre_FreeStructVector(tmp_vector);
   hypre_TFree(comm);

#ifdef HYPRE_DEBUG
   malloc_verify(0);
   malloc_shutdown();
#endif

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

