/*
   Example 9

   Interface:      Structured interface (Struct)

   Compile with:   make ex9

   Sample run:     mpirun -np 16 ex9 -n 33 -solver 0 -v 1 1

   To see options: ex9 -help

   Description:    This code solves a system corresponding to a discretization
                   of the biharmonic problem treated as a system of equations
                   on the unit square.  In other words, we solve 
                   Laplacian(Laplacian(u))x = f via

                   A = [ Laplacian   -I;   0  Laplacian]  
                   x = [ u ; v]; 
                   b = [ 0 ; f] 

                   For boundary conditions, we use u=0 and Laplacian(u) = 0.

                   The domain is split into an N x N processor grid.
                   Thus, the given number of processors should be a
                   perfect square.  Each processor's piece of the grid
                   has n x n cells with n x n nodes. We use
                   cell-centered variables, and, therefore,the nodes
                   are not shared. Note that we have two variables, u and v, and 
                   need only one part to describe the domain. We use the standard 5-point 
                   stencil to discretize the laplace operators. 
                     
                   We incorporate boundary conditions as in Example 3.

                   We recommend viewing Examples 6 and 7 before this example.

*/

#include <math.h>
#include "utilities.h"
#include "HYPRE_sstruct_ls.h"

int main (int argc, char *argv[])
{
   int i, j;

   int myid, num_procs;

   int n, N, pi, pj;
   double h, h2;
   int ilower[2], iupper[2];

   int solver_id;
   int n_pre, n_post;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructStencil  stencil_v;
   HYPRE_SStructStencil  stencil_u;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;


   HYPRE_SStructSolver   solver;
   HYPRE_SStructSolver   precond;

   int print_solution;


   int object_type;


   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Set defaults */
   n = 33;
   solver_id = 0;
   n_pre  = 1;
   n_post = 1;
   print_solution  = 0;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-n") == 0 )
         {
            arg_index++;
            n = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-v") == 0 )
         {
            arg_index++;
            n_pre = atoi(argv[arg_index++]);
            n_post = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-print_solution") == 0 )
         {
            arg_index++;
            print_solution = 1;
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -n <n>              : problem size per procesor (default: 8)\n");
         printf("  -solver <ID>        : solver ID\n");
         printf("                        0  - GMRES with sysPFMG precond (default)\n");
         printf("                        1  - sysPFMG\n");
         printf("                        2  - GMRES with AMG precond (default)\n");
         printf("                        3  - AMG\n");
         printf("                        4  - GMRES with AMG precond with the 'unknown' approach for systems \n");
         printf("                        5  - AMG with the 'unknown' approach for systems \n");
         printf("  -v <n_pre> <n_post> : number of pre and post relaxations (default: 1 1)\n");
         printf("  -print_solution     : print the solution vector\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Figure out the processor grid (N x N).  The local problem
      size for the interior nodes is indicated by n (n x n).
      pi and pj indicate position in the processor grid. */
   N  = sqrt(num_procs);
   h  = 1.0 / (N*n+1); /* note that when calculating h we must
                          remember to count the bounday nodes */
   h2 = h*h;
   pj = myid / N;
   pi = myid - pj*N;

  /* Figure out the extents of each processor's piece of the grid. */
   ilower[0] = pi*n;
   ilower[1] = pj*n;

   iupper[0] = ilower[0] + n-1;
   iupper[1] = ilower[1] + n-1;

   /* 1. Set up a grid - we have one part and two variables */
   {

      int nparts = 1;
      int part = 0;

      /* Create an empty 2D grid object */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, 2, nparts, &grid);
      
      /* Add a new box to the grid */
      HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);

      /* Set the variable type and number of variables on each part.  These need
         to be set in each part which is neighboring or contains boxes owned by the
         Processor.  In this example, we only have one part. */
      {  
         int i;
         int nvars = 2;
         HYPRE_SStructVariable vartypes[2] = {HYPRE_SSTRUCT_VARIABLE_CELL, 
                                              HYPRE_SSTRUCT_VARIABLE_CELL };
         
         for (i = 0; i< nparts; i++)
            HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
      }


      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_SStructGridAssemble(grid);
   }


   /* 2. Define the discretization stencils */
   {

      int entry;
      int stencil_size;
      int var;
      int ndim = 2;
       

      /* Stencil object for variable u (variable 0)*/  
      {

         int offsets[6][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}, {0,0}};
         stencil_size = 6;

         HYPRE_SStructStencilCreate(ndim, stencil_size, &stencil_u);

         /* the first 5 entries are for the u-u connections */
         var = 0; /* connect to variable 0 */
         for (entry = 0; entry < stencil_size-1 ; entry++)
            HYPRE_SStructStencilSetEntry(stencil_u, entry, offsets[entry], var);

         /* the last entry is for the u-v connection */       
         var = 1;  /* connect to variable 1 */
         entry = 5;
         HYPRE_SStructStencilSetEntry(stencil_u, entry, offsets[entry], var);

      }

      /* Stencil object for variable v  (variable 1) */
      {

         int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
         stencil_size = 5;
       
         HYPRE_SStructStencilCreate(ndim, stencil_size, &stencil_v);

         /* these are all v-v connections */
         var = 1; /* connect to variable 1 */
         for (entry = 0; entry < stencil_size; entry++)
            HYPRE_SStructStencilSetEntry(stencil_v, entry, offsets[entry], var);
      }
      
   }


 /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix and allows non-stencil relationships between the parts. */
   {
      int var;
      int part = 0;

      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* Assign the u-stencil we created to variable u (variable 0)*/
      var = 0;
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil_u);

      /* Assign the v-stencil we created to variable v (variable 1)*/
      var = 1;
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil_v);

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }



   /* 4. Set up the SStruct Matrix */
   {
      int nentries;
      int nvalues;
      int var;
      int part = 0;
      

      /* Create an empty matrix object */
      HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);


 /* Set the object type (by default HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use unstructured
         solvers, e.g. BoomerAMG, the object type should be HYPRE_PARCSR.
         If the problem is purely structured (with one part), you may want to use
         HYPRE_STRUCT to access the structured solvers.  */

      object_type = HYPRE_SSTRUCT;
      HYPRE_SStructMatrixSetObjectType(A, object_type);


      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_SStructMatrixInitialize(A);

      /* Each processor must set the stencil values for their boxes on each part.
         In this example, we only set stencil entries and therefore use
         HYPRE_SStructMatrixSetBoxValues.  If we need to set non-stencil entries,
         we have to use HYPRE_SStructMatrixSetValues. */


      /* First set the u-stencil entries.  Note that we must set the entries for each 
         variable within a stencil with separate function calls. */

      {
         int     i, j;
         double *u_values;
         int     u_v_indices[1] = {5};
         int     u_u_indices[5] = {0, 1, 2, 3, 4};
         

         var = 0; /* u connections */
      
         /* the u-u connections */
         nentries = 5;
         nvalues = nentries*n*n;
         u_values = calloc(nvalues, sizeof(double));

         for (i = 0; i < nvalues; i += nentries)
         {
            u_values[i] = 4.0;
            for (j = 1; j < nentries; j++)
               u_values[i+j] = -1.0;
         }

         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, nentries,
                                         u_u_indices, u_values);
         free(u_values);

         /* the u-v connections */
         nentries = 1;
         nvalues = nentries*n*n;
         u_values = calloc(nvalues, sizeof(double));

         for (i = 0; i < nvalues; i++)
         {
            u_values[i] = -h2;
         }

         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, nentries,
                                         u_v_indices, u_values);

         free(u_values);

      }
      
    /*  Now set the v-stencil entries. */

      {
         int     i, j;
         double *v_values;
         int     v_v_indices[5] = {0, 1, 2, 3, 4};
         

         var = 1; /* the v connections */
         
         /* the v-v connections */
         nentries = 5;
         nvalues = nentries*n*n;
         v_values = calloc(nvalues, sizeof(double));

         for (i = 0; i < nvalues; i += nentries)
         {
            v_values[i] = 4.0;
            for (j = 1; j < nentries; j++)
               v_values[i+j] = -1.0;
         }

         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, nentries,
                                         v_v_indices, v_values);
        
         free(v_values);

      }

   }

   /* 5. Incorporate the zero boundary conditions: go along each edge of
         the domain and set the stencil entry that reaches to the boundary to
         zero.*/
   {
      int bc_ilower[2];
      int bc_iupper[2];
      int nentries = 1;
      int nvalues  = nentries*n; /*  number of stencil entries times the length
                                     of one side of my grid box */
      int var;
      double *values;
      int stencil_indices[1];

      int part = 0;
      

      values = calloc(nvalues, sizeof(double));
      for (j = 0; j < nvalues; j++)
            values[j] = 0.0;

      /* Recall: pi and pj describe position in the processor grid */
      if (pj == 0)
      {
         /* bottom row of grid points */
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 3;

         /* need to do this for u and for v */ 
         var = 0;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);


      }

      if (pj == N-1)
      {
         /* upper row of grid points */
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n + n-1;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 4;

         /* need to do this for u and for v */ 
         var = 0;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);

      }

      if (pi == 0)
      {
         /* left row of grid points */
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         stencil_indices[0] = 1;

         /* need to do this for u and for v */ 
         var = 0;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);


      }

      if (pi == N-1)
      {
         /* right row of grid points */
         bc_ilower[0] = pi*n + n-1;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         stencil_indices[0] = 2;

         /* need to do this for u and for v */ 
         var = 0;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);

         var = 1;
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, 
                                         var, nentries,
                                         stencil_indices, values);
         
      }

      free(values);
   }


   /* This is a collective call finalizing the matrix assembly.
      The matrix is now ``ready to be used'' */
   HYPRE_SStructMatrixAssemble(A);



   /* 5. Set up SStruct Vectors for b and x */
   {
      int    nvalues = n*n;
      double *values;
      int part = 0;
      int var;

      values = calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

    /* As with the matrix,  set the object type for the vectors
         to be the sstruct type */
      object_type = HYPRE_SSTRUCT;
      HYPRE_SStructVectorSetObjectType(b, object_type);
      HYPRE_SStructVectorSetObjectType(x, object_type);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(b);
      HYPRE_SStructVectorInitialize(x);

     /* Set the values for b*/
      for (i = 0; i < nvalues; i ++)
         values[i] = h2;
      var = 1;
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
      
      for (i = 0; i < nvalues; i ++)
         values[i] = 0.0;
      var = 0;
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
      

      /* Set the values for the initial guess */
      var = 0;
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

      var = 1;
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);


      free(values);

      /* This is a collective call finalizing the vector assembly.
         The vector is now ``ready to be used'' */
      HYPRE_SStructVectorAssemble(b);
      HYPRE_SStructVectorAssemble(x);
   }

   /* 6. Set up and use a solver
      (Solver options can be found in the Reference Manual.) */
   
   {

      double final_res_norm;
      int its;
         

      if (solver_id == 1) /* SysPFMG */
      {
         
        
         HYPRE_SStructSysPFMGCreate(MPI_COMM_WORLD, &solver);
         
         /* Set sysPFMG parameters */
         HYPRE_SStructSysPFMGSetTol(solver, 1.0e-6);
         HYPRE_SStructSysPFMGSetMaxIter(solver, 50);
         HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
         HYPRE_SStructSysPFMGSetPrintLevel(solver, 0);
         HYPRE_SStructSysPFMGSetLogging(solver, 1);
         HYPRE_SStructSysPFMGSetup(solver, A, b, x);
         

         /* do the solve */
         HYPRE_SStructSysPFMGSolve(solver, A, b, x);
         
        
         
         /* get some info */
         HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(solver,
                                                          &final_res_norm);
         HYPRE_SStructSysPFMGGetNumIterations(solver, &its);


         /* clean up */
         HYPRE_SStructSysPFMGDestroy(solver);

      }
      else if (solver_id == 2) /* GMRES with AMG */
      {
         if (myid ==0) printf("Sorry - solver not available yet!\n");
         
      }
      else if (solver_id == 3) /* AMG */
      {
         
         if (myid ==0) printf("Sorry - solver not available yet!\n");
      }
      else if (solver_id == 4) /* GMRES with AMG and unknown approach*/
      {
         
         if (myid ==0) printf("Sorry - solver not available yet!\n");
      }
      else if (solver_id == 3) /* AMG with unknown approach*/
      {
         if (myid ==0) printf("Sorry - solver not available yet!\n");
         
      }
      else /* GMRES with SysPFMG */
      {
         
         HYPRE_SStructGMRESCreate(MPI_COMM_WORLD, &solver);
         
         /* GMRES parameters */
         HYPRE_SStructGMRESSetMaxIter(solver, 50 );
         HYPRE_SStructGMRESSetTol(solver, 1.0e-06 );
         HYPRE_SStructGMRESSetPrintLevel(solver, 2 ); /* print each GMRES iteration */
         HYPRE_SStructGMRESSetLogging(solver, 1);
         
         /* use SysPFMG as precondititioner */
         HYPRE_SStructSysPFMGCreate(MPI_COMM_WORLD, &precond);
         
         /* Set sysPFMG parameters */
         HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         HYPRE_SStructSysPFMGSetPrintLevel(precond, 0);
         HYPRE_SStructSysPFMGSetZeroGuess(precond);
    
         /* Set the preconditioner and solve */
         HYPRE_SStructGMRESSetPrecond(solver, HYPRE_SStructSysPFMGSolve,
                                      HYPRE_SStructSysPFMGSetup, precond);
         HYPRE_SStructGMRESSetup(solver, A, b, x);
         HYPRE_SStructGMRESSolve(solver, A, b, x);
         
         
         /* get some info */
         HYPRE_SStructGMRESGetFinalRelativeResidualNorm(solver,
                                                        &final_res_norm);
         HYPRE_SStructGMRESGetNumIterations(solver, &its);
         
         /* clean up */
         HYPRE_SStructGMRESDestroy(solver);
         
      }
     

      /* Print the solution and other info */
      
      if (print_solution)
      {
         HYPRE_SStructVectorPrint("sstruct.ex9.out.x", x, 0);
      }
      
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", its);
         printf("Final Relative Residual Norm = %g\n", final_res_norm);
         printf("\n");
      }
      
   }
   

   /* Free memory */
   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructStencilDestroy(stencil_v);
   HYPRE_SStructStencilDestroy(stencil_u);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}
