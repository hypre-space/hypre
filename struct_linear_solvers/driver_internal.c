 
#include "headers.h"
 
#ifdef ZZZ_DEBUG
#include <cegdb.h>
#endif

#ifdef ZZZ_DEBUG
char malloc_logpath_memory[256];
#endif
 
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
   MPI_Comm           *comm;

   int                 A_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
   int                 b_num_ghost[6] = { 0, 0, 0, 0, 0, 0};
   int                 x_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   zzz_StructMatrix   *A;
   zzz_StructVector   *b;
   zzz_StructVector   *x;

   void               *smg_data;
   int                 num_iterations;

   int                 num_procs, myid;

   char                filename[255];

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
 
   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   comm = zzz_TAlloc(MPI_Comm, 1);
   MPI_Comm_dup(MPI_COMM_WORLD, comm);
   MPI_Comm_size(*comm, &num_procs );
   MPI_Comm_rank(*comm, &myid );

#ifdef ZZZ_DEBUG
   cegdb(&argc, &argv, myid);
#endif

#ifdef ZZZ_DEBUG
   malloc_logpath = malloc_logpath_memory;
   sprintf(malloc_logpath, "malloc.log.%04d", myid);
#endif

   if (argc > 1)
   {
      sprintf(filename, argv[1]);
   }
   else
   {
      printf("Usage: mpirun -np %d %s <input_problem>\n\n",
             num_procs, argv[0]);
      exit(1);
   }

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   A = zzz_ReadStructMatrix(comm, filename, A_num_ghost);

   b = zzz_NewStructVector(comm, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(b, b_num_ghost);
   zzz_InitializeStructVector(b);
   zzz_AssembleStructVector(b);
   zzz_SetStructVectorConstantValues(b, 1.0);

   x = zzz_NewStructVector(comm, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(x, x_num_ghost);
   zzz_InitializeStructVector(x);
   zzz_AssembleStructVector(x);
   zzz_SetStructVectorConstantValues(x, 0.0);
 
   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   smg_data = zzz_SMGInitialize(comm);
   zzz_SMGSetMaxIter(smg_data, 10);
   zzz_SMGSetup(smg_data, A, b, x);
   zzz_SMGSolve(smg_data, b, x);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   zzz_PrintStructVector("zout_x", x, 0);

   zzz_SMGGetNumIterations(smg_data, &num_iterations);
   if (myid == 0)
   {
      printf("Iterations = %d\n", num_iterations);
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   zzz_SMGFinalize(smg_data);
   zzz_FreeStructGrid(zzz_StructMatrixGrid(A));
   zzz_FreeStructMatrix(A);
   zzz_FreeStructVector(b);
   zzz_FreeStructVector(x);
   zzz_TFree(comm);

#ifdef ZZZ_DEBUG
   malloc_verify(0);
   malloc_shutdown();
#endif

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

