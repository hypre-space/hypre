
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
                     
   hypre_StructMatrix   *A;
   hypre_StructVector   *b;
   hypre_StructVector   *x;

   void               *smg_data;
   int                 num_iterations;
   int                 time_index;

   int                 num_procs, myid;

   char                filename[255];

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

   A = hypre_ReadStructMatrix(comm, filename, A_num_ghost);

   b = hypre_NewStructVector(comm, hypre_StructMatrixGrid(A));
   hypre_SetStructVectorNumGhost(b, b_num_ghost);
   hypre_InitializeStructVector(b);
   hypre_AssembleStructVector(b);
   hypre_SetStructVectorConstantValues(b, 1.0);

   x = hypre_NewStructVector(comm, hypre_StructMatrixGrid(A));
   hypre_SetStructVectorNumGhost(x, x_num_ghost);
   hypre_InitializeStructVector(x);
   hypre_AssembleStructVector(x);
   hypre_SetStructVectorConstantValues(x, 0.0);
 
   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   smg_data = hypre_SMGInitialize(comm);
   hypre_SMGSetMemoryUse(smg_data, 0);
   hypre_SMGSetMaxIter(smg_data, 50);
   hypre_SMGSetTol(smg_data, 1.0e-06);
   hypre_SMGSetNumPreRelax(smg_data, 1);
   hypre_SMGSetNumPostRelax(smg_data, 1);
   hypre_SMGSetLogging(smg_data, 0);
   hypre_SMGSetup(smg_data, A, b, x);

   time_index = hypre_InitializeTiming("Driver");
   hypre_BeginTiming(time_index);
   hypre_SMGSolve(smg_data, A, b, x);
   hypre_EndTiming(time_index);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   hypre_PrintStructVector("zout_x", x, 0);

   hypre_SMGGetNumIterations(smg_data, &num_iterations);
   if (myid == 0)
   {
      printf("Iterations = %d\n", num_iterations);
   }

   hypre_PrintTiming(comm);
   
   hypre_SMGPrintLogging(smg_data, myid);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_FinalizeTiming(time_index);
   hypre_SMGFinalize(smg_data);
   hypre_FreeStructGrid(hypre_StructMatrixGrid(A));
   hypre_FreeStructMatrix(A);
   hypre_FreeStructVector(b);
   hypre_FreeStructVector(x);
   hypre_TFree(comm);

#ifdef HYPRE_DEBUG
   malloc_verify(0);
   malloc_shutdown();
#endif

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


