 
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
   int                 A_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
   int                 b_num_ghost[6] = { 0, 0, 0, 0, 0, 0};
   int                 x_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   zzz_StructMatrix   *A;
   zzz_StructVector   *b;
   zzz_StructVector   *x;

   void               *smg_data;

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
    * Set up the linear system
    *-----------------------------------------------------------*/

   A = zzz_ReadStructMatrix("zin_A", A_num_ghost);

   b = zzz_NewStructVector(&MPI_COMM_WORLD, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(b, b_num_ghost);
   zzz_InitializeStructVector(b);
   zzz_AssembleStructVector(b);
   zzz_SetStructVectorConstantValues(b, 1.0);

   x = zzz_NewStructVector(&MPI_COMM_WORLD, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(x, x_num_ghost);
   zzz_InitializeStructVector(x);
   zzz_AssembleStructVector(x);
   zzz_SetStructVectorConstantValues(x, 0.0);
 
   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   smg_data = zzz_SMGInitialize(&MPI_COMM_WORLD);
   zzz_SMGSetup(smg_data, A, b, x);
   zzz_SMGSolve(smg_data, b, x);

   /*-----------------------------------------------------------
    * Print the solution
    *-----------------------------------------------------------*/

   zzz_PrintStructVector("zout_x", x, 0);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   zzz_SMGFinalize(smg_data);
   zzz_FreeStructGrid(zzz_StructMatrixGrid(A));
   zzz_FreeStructMatrix(A);
   zzz_FreeStructVector(b);
   zzz_FreeStructVector(x);

   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

