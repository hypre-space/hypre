
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
   /* int                 b_num_ghost[6] = { 0, 0, 0, 0, 0, 0}; */
   int                 b_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
   int                 x_num_ghost[6] = { 1, 1, 1, 1, 1, 1};
                     
   zzz_StructMatrix   *A;
   zzz_StructVector   *b;
   zzz_StructVector   *b_l;
   zzz_StructVector   *x;
   zzz_StructVector   *x_l;


   ZZZ_PCGData        *pcg_data;
   ZZZ_PCGPrecondData *precond_data;
   int                 num_iterations;
   int                 num_procs, myid;

   char                filename[255];

   double              norm, rel_norm;

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

   /* A = zzz_ReadStructMatrix(comm, "Ares_Matrix", A_num_ghost); */

   /* b = zzz_ReadStructVector(comm, "Ares_Rhs", b_num_ghost); */

   /* x = zzz_ReadStructVector(comm, "Ares_InitialGuess", x_num_ghost); */
 
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
    * Allocate work vectors for preconditioner
    *-----------------------------------------------------------*/

   b_l = zzz_NewStructVector(comm, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(b_l, b_num_ghost);
   zzz_InitializeStructVector(b_l);
   zzz_AssembleStructVector(b_l);
   zzz_SetStructVectorConstantValues(b_l, 1.0);

   x_l = zzz_NewStructVector(comm, zzz_StructMatrixGrid(A));
   zzz_SetStructVectorNumGhost(x_l, x_num_ghost);
   zzz_InitializeStructVector(x_l);
   zzz_AssembleStructVector(x_l);
   zzz_SetStructVectorConstantValues(x_l, 0.0);
 
   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   pcg_data = zzz_TAlloc(ZZZ_PCGData, 1);
   precond_data = zzz_TAlloc(ZZZ_PCGPrecondData, 1);

   ZZZ_PCGDataMaxIter(pcg_data) = 20;
   ZZZ_PCGDataTwoNorm(pcg_data) = 1;

#if 0
   ZZZ_PCGSMGPrecondSetup( A, b_l, x_l, precond_data );
   ZZZ_PCGSetup( A, ZZZ_PCGSMGPrecond, precond_data, pcg_data );
#endif
#if 1
   ZZZ_PCGDiagScalePrecondSetup( A, b_l, x_l, precond_data );
   ZZZ_PCGSetup( A, ZZZ_PCGDiagScalePrecond, precond_data, pcg_data );
#endif

   ZZZ_PCG( x, b, 1.e-6, pcg_data );
   num_iterations = ZZZ_PCGDataNumIterations(pcg_data);
   norm = ZZZ_PCGDataNorm(pcg_data);
   rel_norm = ZZZ_PCGDataRelNorm(pcg_data);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   zzz_PrintStructVector("zout_x", x, 0);

   if (myid == 0)
   {
      printf("Iterations = %d ", num_iterations);
      printf("Final Norm = %e Final Relative Norm = %e\n", norm, rel_norm);
   }

   
   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   ZZZ_FreePCGSMGData(pcg_data);
   zzz_FreeStructGrid(zzz_StructMatrixGrid(A));
   zzz_FreeStructMatrix(A);
   zzz_FreeStructVector(b);
   zzz_FreeStructVector(x);
   /* Already freed in ZZZ_FreePCGSMGData call */
   /* zzz_FreeStructVector(b_l); */
   /* zzz_FreeStructVector(x_l); */
   zzz_TFree(comm);

#ifdef ZZZ_DEBUG
   malloc_verify(0);
   malloc_shutdown();
#endif

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}


