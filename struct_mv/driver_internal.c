 
#include "headers.h"
 
/* debugging header */
#include <cegdb.h>

/* malloc debug stuff */
char malloc_logpath_memory[256];
 
/*--------------------------------------------------------------------------
 * Test driver for structured matrix interface (structured storage)
 *--------------------------------------------------------------------------*/
 
/*----------------------------------------------------------------------
 * Example 2: (symmetric storage)
 *
 *    Standard 5-point laplacian in 2D on a 10 x 7 grid,
 *    ignoring boundary conditions for simplicity.
 *----------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   int                 matrix_num_ghost[6] = { 1, 1, 0, 0, 0, 0};
   int                 vector_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   zzz_StructGrid     *grid;
   zzz_StructMatrix   *matrix;
   zzz_StructVector   *vector;

   int                 num_procs, myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   cegdb(&argc, &argv, myid);

   /* malloc debug stuff */
   malloc_logpath = malloc_logpath_memory;
   sprintf(malloc_logpath, "malloc.log.%04d", myid);

   /*-----------------------------------------------------------
    * Read in the matrix
    *-----------------------------------------------------------*/

   matrix = zzz_ReadStructMatrix("zin_matrix", matrix_num_ghost);
 
   zzz_PrintStructMatrix("zout_matrix", matrix, 0);

   /*-----------------------------------------------------------
    * Read in the vector
    *-----------------------------------------------------------*/

   vector = zzz_ReadStructVector("zin_vector", vector_num_ghost);
 
   zzz_PrintStructVector("zout_vector", vector, 0);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   zzz_FreeStructGrid(zzz_StructMatrixGrid(matrix));
   zzz_FreeStructStencil(zzz_StructMatrixUserStencil(matrix));
   zzz_FreeStructMatrix(matrix);
   zzz_FreeStructGrid(zzz_StructVectorGrid(vector));
   zzz_FreeStructVector(vector);

   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();

   /* Finalize MPI */
   MPI_Finalize();
}

