
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
   /* int                 b_num_ghost[6] = { 0, 0, 0, 0, 0, 0}; */
   int                 b_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
   int                 x_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   hypre_StructMatrix   *A;
   hypre_StructVector   *b;
   hypre_StructVector   *b_l;
   hypre_StructVector   *x;
   hypre_StructVector   *x_l;


   HYPRE_PCGData        *pcg_data;
   HYPRE_PCGPrecondData *precond_data;
   int                 num_iterations;
   int                 num_procs, myid;

   char                filename[255];

   double              norm, rel_norm;

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

   /* A = hypre_ReadStructMatrix(comm, "Ares_Matrix", A_num_ghost); */

   /* b = hypre_ReadStructVector(comm, "Ares_Rhs", b_num_ghost); */

   /* x = hypre_ReadStructVector(comm, "Ares_InitialGuess", x_num_ghost); */
 
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
    * Allocate work vectors for preconditioner
    *-----------------------------------------------------------*/

   b_l = hypre_NewStructVector(comm, hypre_StructMatrixGrid(A));
   hypre_SetStructVectorNumGhost(b_l, b_num_ghost);
   hypre_InitializeStructVector(b_l);
   hypre_AssembleStructVector(b_l);
   hypre_SetStructVectorConstantValues(b_l, 1.0);

   x_l = hypre_NewStructVector(comm, hypre_StructMatrixGrid(A));
   hypre_SetStructVectorNumGhost(x_l, x_num_ghost);
   hypre_InitializeStructVector(x_l);
   hypre_AssembleStructVector(x_l);
   hypre_SetStructVectorConstantValues(x_l, 0.0);
 
   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   pcg_data = hypre_TAlloc(HYPRE_PCGData, 1);
   precond_data = hypre_TAlloc(HYPRE_PCGPrecondData, 1);

   HYPRE_PCGDataMaxIter(pcg_data) = 20;
   HYPRE_PCGDataTwoNorm(pcg_data) = 1;

#if 0
   HYPRE_PCGSMGPrecondSetup( A, b_l, x_l, precond_data );
   HYPRE_PCGSetup( A, HYPRE_PCGSMGPrecond, precond_data, pcg_data );
#endif
#if 1
   HYPRE_PCGDiagScalePrecondSetup( A, b_l, x_l, precond_data );
   HYPRE_PCGSetup( A, HYPRE_PCGDiagScalePrecond, precond_data, pcg_data );
#endif

   HYPRE_PCG( x, b, 1.e-6, pcg_data );
   num_iterations = HYPRE_PCGDataNumIterations(pcg_data);
   norm = HYPRE_PCGDataNorm(pcg_data);
   rel_norm = HYPRE_PCGDataRelNorm(pcg_data);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   hypre_PrintStructVector("zout_x", x, 0);

   if (myid == 0)
   {
      printf("Iterations = %d ", num_iterations);
      printf("Final Norm = %e Final Relative Norm = %e\n", norm, rel_norm);
   }

   
   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   HYPRE_FreePCGSMGData(pcg_data);
   hypre_FreeStructGrid(hypre_StructMatrixGrid(A));
   hypre_FreeStructMatrix(A);
   hypre_FreeStructVector(b);
   hypre_FreeStructVector(x);
   /* Already freed in HYPRE_FreePCGSMGData call */
   /* hypre_FreeStructVector(b_l); */
   /* hypre_FreeStructVector(x_l); */
   hypre_TFree(comm);

#ifdef HYPRE_DEBUG
   malloc_verify(0);
   malloc_shutdown();
#endif

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


