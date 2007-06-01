/*
   Example 5

   Interface:    Linear-Algebraic (IJ), Babel-based version

   Compile with: make ex5b

   Sample run:   mpirun -np 4 ex5b

   Description:  This example solves the 2-D
                 Laplacian problem with zero boundary conditions
                 on an nxn grid.  The number of unknowns is N=n^2.
                 The standard 5-point stencil is used, and we solve
                 for the interior nodes only.

                 This example solves the same problem as Example 3.
                 Available solvers are AMG, PCG, and PCG with AMG or
                 Parasails preconditioners.
*/

#include <math.h>
#include <assert.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

/* Babel interface headers */
#include "bHYPRE.h"
#include "bHYPRE_Vector.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRVector.h"
#include "bHYPRE_ParCSRDiagScale.h"
#include "bHYPRE_BoomerAMG.h"
#include "sidl_Exception.h"

int main (int argc, char *argv[])
{
   int i;
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;

   int solver_id;
   int print_solution;

   double h, h2;
   MPI_Comm mpicommworld = MPI_COMM_WORLD;

   int ierr = 0;
   /* If this gets set to anything else, it's an error.
    Most functions return error flags, 0 unless there's an error.
    For clarity, they aren't checked much in this file. */

   bHYPRE_MPICommunicator mpi_comm;
   bHYPRE_IJParCSRMatrix parcsr_A;
   bHYPRE_Operator           op_A;
   bHYPRE_IJParCSRVector par_b;
   bHYPRE_IJParCSRVector par_x;
   bHYPRE_Vector vec_b;
   bHYPRE_Vector vec_x;

   bHYPRE_Solver precond;
   bHYPRE_BoomerAMG amg_solver;
   bHYPRE_ParaSails ps_solver;
   bHYPRE_PCG pcg_solver;
   bHYPRE_IdentitySolver identity;
   sidl_BaseInterface _ex = NULL;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   mpi_comm = bHYPRE_MPICommunicator_CreateC( &mpicommworld, &_ex );

   /* Default problem parameters */
   n = 33;
   solver_id = 0;
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
         printf("  -n <n>              : problem size in each direction (default: 33)\n");
         printf("  -solver <ID>        : solver ID\n");
         printf("                        0  - AMG (default) \n");
         printf("                        1  - AMG-PCG\n");
         printf("                        8  - ParaSails-PCG\n");
         printf("                        50 - PCG\n");
         printf("  -print_solution     : print the solution vector\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   parcsr_A = bHYPRE_IJParCSRMatrix_Create( mpi_comm,
                                            ilower, iupper, ilower, iupper, &_ex );

   op_A = bHYPRE_Operator__cast( parcsr_A, &_ex ); /* needed later as a function argument */

   /* Choose a parallel csr format storage (see the User's Manual) */
   /* Note: Here the HYPRE interface requires a SetObjectType call.
      I am using the bHYPRE interface in a way which does not because
      the object type is already specified through the class name. */

   /* Initialize before setting coefficients */
   bHYPRE_IJParCSRMatrix_Initialize( parcsr_A, &_ex );

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */
   {
      int nnz;
      double values[5];
      int cols[5];

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;

         /* The left identity block:position i-n */
         if ((i-n)>=0)
         {
	    cols[nnz] = i-n;
	    values[nnz] = -1.0;
	    nnz++;
         }

         /* The left -1: position i-1 */
         if (i%n)
         {
            cols[nnz] = i-1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the diagonal: position i */
         cols[nnz] = i;
         values[nnz] = 4.0;
         nnz++;

         /* The right -1: position i+1 */
         if ((i+1)%n)
         {
            cols[nnz] = i+1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i+n)< N)
         {
            cols[nnz] = i+n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the values for row i */
         bHYPRE_IJParCSRMatrix_SetValues( parcsr_A, 1, &nnz, &i, cols, values, 5, &_ex );
      }
   }

   /* Assemble after setting the coefficients */
   bHYPRE_IJParCSRMatrix_Assemble( parcsr_A, &_ex );

   /* Create the rhs and solution */
   par_b = bHYPRE_IJParCSRVector_Create( mpi_comm, ilower, iupper, &_ex );

   vec_b = bHYPRE_Vector__cast( par_b, &_ex ); /* needed later for function arguments */

   bHYPRE_IJParCSRVector_Initialize( par_b, &_ex );

   par_x = bHYPRE_IJParCSRVector_Create( mpi_comm, ilower, iupper, &_ex );

   vec_x = bHYPRE_Vector__cast( par_x, &_ex ); /* needed later for function arguments */

   bHYPRE_IJParCSRVector_Initialize( par_x, &_ex );

   /* Set the rhs values to h^2 and the solution to zero */
   {
      double *rhs_values, *x_values;
      int    *rows;

      rhs_values = calloc(local_size, sizeof(double));
      x_values = calloc(local_size, sizeof(double));
      rows = calloc(local_size, sizeof(int));

      for (i=0; i<local_size; i++)
      {
         rhs_values[i] = h2;
         x_values[i] = 0.0;
         rows[i] = ilower + i;
      }

      bHYPRE_IJParCSRVector_SetValues( par_b, local_size, rows, rhs_values, &_ex );
      bHYPRE_IJParCSRVector_SetValues( par_x, local_size, rows, x_values, &_ex );

      free(x_values);
      free(rhs_values);
      free(rows);
   }

   bHYPRE_IJParCSRVector_Assemble( par_b, &_ex );
   bHYPRE_IJParCSRVector_Assemble( par_x, &_ex );

   /* Choose a solver and solve the system */

   /* AMG */
   if (solver_id == 0)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      amg_solver = bHYPRE_BoomerAMG_Create( mpi_comm, parcsr_A, &_ex );

      /* Set some parameters (See Reference Manual for more parameters) */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "PrintLevel", 3, &_ex );  /* print solve info + parameters */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "CoarsenType", 6, &_ex ); /* Falgout coarsening */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "RelaxType", 3, &_ex );   /* G-S/Jacobi hybrid relaxation */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "NumSweeps", 1, &_ex );   /* Sweeeps on each level */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "MaxLevels", 20, &_ex );  /* maximum number of levels */
      bHYPRE_BoomerAMG_SetDoubleParameter( amg_solver, "Tolerance", 1e-7, &_ex );      /* conv. tolerance */

      /* Now setup and solve! */
      bHYPRE_BoomerAMG_Setup( amg_solver, vec_b, vec_x, &_ex );
      bHYPRE_BoomerAMG_Apply( amg_solver, vec_b, &vec_x, &_ex );

      /* Run info - needed logging turned on */
      ierr += bHYPRE_BoomerAMG_GetIntValue( amg_solver, "NumIterations", &num_iterations, &_ex );
      bHYPRE_BoomerAMG_GetDoubleValue( amg_solver, "RelResidualNorm", &final_res_norm, &_ex );

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      bHYPRE_BoomerAMG_deleteRef( amg_solver, &_ex );
   }

   /* PCG */
   else if (solver_id == 50)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      pcg_solver = bHYPRE_PCG_Create( mpi_comm, op_A, &_ex );

      /* Set some parameters (See Reference Manual for more parameters) */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "MaxIter", 1000, &_ex ); /* max iterations */
      bHYPRE_PCG_SetDoubleParameter( pcg_solver, "Tolerance", 1e-7, &_ex ); /* conv. tolerance */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "TwoNorm", 1, &_ex ); /* use the two norm as the stopping criteria */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "PrintLevel", 2, &_ex ); /* prints out the iteration info */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "Logging", 1, &_ex ); /* needed to get run info later */

      identity = bHYPRE_IdentitySolver_Create( mpi_comm, &_ex );
      precond = bHYPRE_Solver__cast( identity, &_ex );
      bHYPRE_PCG_SetPreconditioner( pcg_solver, precond, &_ex );

      /* Now setup and solve! */
      bHYPRE_PCG_Setup( pcg_solver, vec_b, vec_x, &_ex );
      bHYPRE_PCG_Apply( pcg_solver, vec_b, &vec_x, &_ex );

      /* Run info - needed logging turned on */
      bHYPRE_PCG_GetIntValue( pcg_solver, "NumIterations", &num_iterations, &_ex );
      bHYPRE_PCG_GetDoubleValue( pcg_solver, "RelResidualNorm", &final_res_norm, &_ex );
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solvers */
      bHYPRE_PCG_deleteRef( pcg_solver, &_ex );
      bHYPRE_Solver_deleteRef( precond, &_ex );
   }
   /* PCG with AMG preconditioner */
   else if (solver_id == 1)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      pcg_solver = bHYPRE_PCG_Create( mpi_comm, op_A, &_ex );

      /* Set some parameters (See Reference Manual for more parameters) */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "MaxIter", 1000, &_ex ); /* max iterations */
      bHYPRE_PCG_SetDoubleParameter( pcg_solver, "Tolerance", 1e-7, &_ex ); /* conv. tolerance */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "TwoNorm", 1, &_ex ); /* use the two norm as the stopping criteria */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "PrintLevel", 2, &_ex ); /* prints out the iteration info */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "Logging", 1, &_ex ); /* needed to get run info later */

      /* Now set up the AMG preconditioner and specify any parameters */
      amg_solver = bHYPRE_BoomerAMG_Create( mpi_comm, parcsr_A, &_ex );
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "PrintLevel", 1, &_ex ); /* print amg solution info*/
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "CoarsenType", 6, &_ex ); /* Falgout coarsening */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "RelaxType", 6, &_ex );   /* Sym G-S/Jacobi hybrid relaxation */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "NumSweeps", 1, &_ex );   /* Sweeeps on each level */
      bHYPRE_BoomerAMG_SetDoubleParameter( amg_solver, "Tolerance", 0, &_ex);      /* conv. tolerance */
      bHYPRE_BoomerAMG_SetIntParameter( amg_solver, "MaxIter", 1, &_ex ); /* do only one iteration! */

      /* Set the PCG preconditioner */
      precond = bHYPRE_Solver__cast( amg_solver, &_ex );
      bHYPRE_PCG_SetPreconditioner( pcg_solver, precond, &_ex );

      /* Now setup and solve! */
      bHYPRE_PCG_Setup( pcg_solver, vec_b, vec_x, &_ex );
      bHYPRE_PCG_Apply( pcg_solver, vec_b, &vec_x, &_ex );

      /* Run info - needed logging turned on */
      bHYPRE_PCG_GetIntValue( pcg_solver, "NumIterations", &num_iterations, &_ex );
      bHYPRE_PCG_GetDoubleValue( pcg_solver, "RelResidualNorm", &final_res_norm, &_ex );
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      bHYPRE_PCG_deleteRef( pcg_solver, &_ex );
      bHYPRE_Solver_deleteRef( precond, &_ex );
   }

   /* PCG with Parasails Preconditioner */
   else if (solver_id == 8)
   {
      int    num_iterations;
      double final_res_norm;

      int      sai_max_levels = 1;
      double   sai_threshold = 0.1;
      double   sai_filter = 0.05;
      int      sai_sym = 1;

      /* Create solver */
      pcg_solver = bHYPRE_PCG_Create( mpi_comm, op_A, &_ex );

      /* Set some parameters (See Reference Manual for more parameters) */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "MaxIter", 1000, &_ex ); /* max iterations */
      bHYPRE_PCG_SetDoubleParameter( pcg_solver, "Tolerance", 1e-7, &_ex ); /* conv. tolerance */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "TwoNorm", 1, &_ex ); /* use the two norm as the stopping criteria */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "PrintLevel", 2, &_ex ); /* prints out the iteration info */
      bHYPRE_PCG_SetIntParameter( pcg_solver, "Logging", 1, &_ex ); /* needed to get run info later */

      /* Now set up the ParaSails preconditioner and specify any parameters */
      ps_solver = bHYPRE_ParaSails_Create( mpi_comm, parcsr_A, &_ex );

      /* Set some parameters (See Reference Manual for more parameters) */
      bHYPRE_ParaSails_SetDoubleParameter( ps_solver, "Thresh", sai_threshold, &_ex );
      bHYPRE_ParaSails_SetIntParameter( ps_solver, "Nlevels", sai_max_levels, &_ex );
      bHYPRE_ParaSails_SetDoubleParameter( ps_solver, "Filter", sai_filter, &_ex );
      bHYPRE_ParaSails_SetIntParameter( ps_solver, "Sym", sai_sym, &_ex );
      bHYPRE_ParaSails_SetIntParameter( ps_solver, "Logging", 3, &_ex );

      /* Set the PCG preconditioner */
      precond = bHYPRE_Solver__cast( ps_solver, &_ex );
      bHYPRE_PCG_SetPreconditioner( pcg_solver, precond, &_ex );

      /* Now setup and solve! */
      bHYPRE_PCG_Setup( pcg_solver, vec_b, vec_x, &_ex );
      bHYPRE_PCG_Apply( pcg_solver, vec_b, &vec_x, &_ex );


      /* Run info - needed logging turned on */
      bHYPRE_PCG_GetIntValue( pcg_solver, "NumIterations", &num_iterations, &_ex );
      bHYPRE_PCG_GetDoubleValue( pcg_solver, "RelResidualNorm", &final_res_norm, &_ex );
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      bHYPRE_PCG_deleteRef( pcg_solver, &_ex );
      bHYPRE_Solver_deleteRef( precond, &_ex );
   }
   else
   {
      if (myid ==0) printf("Invalid solver id specified.\n");
   }

   /* Print the solution */
   if (print_solution)
      bHYPRE_IJParCSRVector_Print( par_x, "ij.out.x", &_ex );

   /* Clean up */
   bHYPRE_Operator_deleteRef( op_A, &_ex );
   bHYPRE_Vector_deleteRef( vec_b, &_ex );
   bHYPRE_Vector_deleteRef( vec_x, &_ex );
   bHYPRE_IJParCSRMatrix_deleteRef( parcsr_A, &_ex );
   bHYPRE_IJParCSRVector_deleteRef( par_b, &_ex );
   bHYPRE_IJParCSRVector_deleteRef( par_x, &_ex );
   bHYPRE_MPICommunicator_deleteRef( mpi_comm, &_ex );

   hypre_assert( ierr == 0 );

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
