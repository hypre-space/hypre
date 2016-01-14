/*
   Example 5

   Interface:    Linear-Algebraic (IJ), Babel-based version

   Compile with: make ex5bxx

   Sample run:   mpirun -np 4 ex5bxx

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
#include "bHYPRE.hxx"
#include "bHYPRE_Vector.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRVector.h"
#include "bHYPRE_ParCSRDiagScale.h"
#include "bHYPRE_BoomerAMG.h"

int main (int argc, char *argv[])
{
   using namespace ::bHYPRE;
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

   MPICommunicator mpi_comm;
   IJParCSRMatrix parcsr_A;
   Operator  op_A;
   IJParCSRVector par_b;
   IJParCSRVector par_x;
   Vector vec_x;

   BoomerAMG amg_solver;
   ParaSails ps_solver;
   PCG pcg_solver;
   IdentitySolver identity;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   mpi_comm = MPICommunicator::CreateC( &mpicommworld );

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
   if (n*n < num_procs) n = int(sqrt(num_procs)) + 1;
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
   parcsr_A = IJParCSRMatrix::Create( mpi_comm,
                                        ilower, iupper, ilower, iupper );

   op_A = parcsr_A; /* we can eliminate op_A later, it's really needed only in C */

   /* Choose a parallel csr format storage (see the User's Manual) */
   /* Note: Here the HYPRE interface requires a SetObjectType call.
      I am using the bHYPRE interface in a way which does not because
      the object type is already specified through the class name. */

   /* Initialize before setting coefficients */
   parcsr_A.Initialize();

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
         parcsr_A.SetValues( 1, &nnz, &i, cols, values, 5 );
      }
   }

   /* Assemble after setting the coefficients */
   parcsr_A.Assemble();

   /* Create the rhs and solution */
   par_b = IJParCSRVector::Create( mpi_comm, ilower, iupper );

   par_b.Initialize();

   par_x = IJParCSRVector::Create( mpi_comm, ilower, iupper );

   par_x.Initialize();

   /* Set the rhs values to h^2 and the solution to zero */
   {
      double *rhs_values, *x_values;
      int    *rows;

      rhs_values = new double[local_size];
      x_values = new double[local_size];
      rows = new int[local_size];

      for (i=0; i<local_size; i++)
      {
         rhs_values[i] = h2;
         x_values[i] = 0.0;
         rows[i] = ilower + i;
      }

      par_b.SetValues( local_size, rows, rhs_values );
      par_x.SetValues( local_size, rows, x_values );

      delete[] x_values;
      delete[] rhs_values;
      delete[] rows;
   }

   par_b.Assemble();
   par_x.Assemble();

   /* Choose a solver and solve the system */

   /* AMG */
   if (solver_id == 0)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      amg_solver = BoomerAMG::Create( mpi_comm, parcsr_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      amg_solver.SetIntParameter( "PrintLevel", 3 );  /* print solve info + parameters */
      amg_solver.SetIntParameter( "CoarsenType", 6); /* Falgout coarsening */
      amg_solver.SetIntParameter( "RelaxType", 3);   /* G-S/Jacobi hybrid relaxation */
      amg_solver.SetIntParameter( "NumSweeps", 1);   /* Sweeeps on each level */
      amg_solver.SetIntParameter( "MaxLevels", 20);  /* maximum number of levels */
      amg_solver.SetDoubleParameter( "Tolerance", 1e-7);      /* conv. tolerance */

      /* Now setup and solve! */
      // In C++, use of the Apply function is slightly more complicated than in other
      // languages.
      //   The second argument, the solution (and initial guess), is declared inout in
      // Interfaces.idl.  For C++, Babel implements an inout (or out) argument as a
      // reference argument, and obviously not a const one.
      //   So it is important not to implicitly ask the C++ environment for a type
      // conversion when supplying the second argument - results will be unpredictable!
      //   Our solution vector is par_x, declared type IJParCSRVector, derived from
      // Vector. In other languages we would use par_x directly for the second argument.
      // In C++ we must declare a Vector, vec_x, for use as the second argument.
      //   Fortunately, an assignment is all you have to do to set up vec_x.
      // Babel casts and code other inside the Apply function will recover the original
      // IJParCSRVector, par_x, to make sure it gets computed correctly.
      amg_solver.Setup( par_b, par_x );
      vec_x = par_x;
      amg_solver.Apply( par_b, vec_x );

      /* Run info - needed logging turned on */

      ierr += amg_solver.GetIntValue( "NumIterations", num_iterations );
      amg_solver.GetDoubleValue( "RelResidualNorm", final_res_norm );

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      /* In C++, unlike C, deleteRef of amg_solver is not needed here -
         it happens automatically. */
   }

   /* PCG */
   else if (solver_id == 50)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      pcg_solver = PCG::Create( mpi_comm, op_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      pcg_solver.SetIntParameter( "MaxIter", 1000 ); /* max iterations */
      pcg_solver.SetDoubleParameter( "Tolerance", 1e-7 ); /* conv. tolerance */
      pcg_solver.SetIntParameter( "TwoNorm", 1 ); /* use the two norm as the stopping criteria */
      pcg_solver.SetIntParameter( "PrintLevel", 2 ); /* prints out the iteration info */
      pcg_solver.SetIntParameter( "Logging", 1 ); /* needed to get run info later */

      identity = IdentitySolver::Create( mpi_comm );
      pcg_solver.SetPreconditioner( identity );

      /* Now setup and solve! */
      // See comments for solver 0 on Apply and vec_x.
      pcg_solver.Setup( par_b, par_x );
      vec_x = par_x;
      pcg_solver.Apply( par_b, vec_x );

      /* Run info - needed logging turned on */
      pcg_solver.GetIntValue( "NumIterations", num_iterations );
      pcg_solver.GetDoubleValue( "RelResidualNorm", final_res_norm );
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solvers */
      /* In C++, unlike C, deleteRef's of solvers are not needed here -
         it happens automatically. */
   }
   /* PCG with AMG preconditioner */
   else if (solver_id == 1)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      pcg_solver = PCG::Create( mpi_comm, op_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      pcg_solver.SetIntParameter( "MaxIter", 1000 ); /* max iterations */
      pcg_solver.SetDoubleParameter( "Tolerance", 1e-7 ); /* conv. tolerance */
      pcg_solver.SetIntParameter( "TwoNorm", 1 ); /* use the two norm as the stopping criteria */
      pcg_solver.SetIntParameter( "PrintLevel", 2 ); /* prints out the iteration info */
      pcg_solver.SetIntParameter( "Logging", 1 ); /* needed to get run info later */

      /* Now set up the AMG preconditioner and specify any parameters */
      amg_solver = BoomerAMG::Create( mpi_comm, parcsr_A );
      amg_solver.SetIntParameter( "PrintLevel", 1 ); /* print amg solution info*/
      amg_solver.SetIntParameter( "CoarsenType", 6); /* Falgout coarsening */
      amg_solver.SetIntParameter( "RelaxType", 6);   /* Sym G-S/Jacobi hybrid relaxation */
      amg_solver.SetIntParameter( "NumSweeps", 1);   /* Sweeeps on each level */
      amg_solver.SetDoubleParameter( "Tolerance", 0);      /* conv. tolerance */
      amg_solver.SetIntParameter( "MaxIter", 1 ); /* do only one iteration! */

      /* Set the PCG preconditioner */
      pcg_solver.SetPreconditioner( amg_solver );

      /* Now setup and solve! */
      // See comments for solver 0 on Apply and vec_x.
      pcg_solver.Setup( par_b, par_x );
      vec_x = par_x;
      pcg_solver.Apply( par_b, vec_x );

      /* Run info - needed logging turned on */
      pcg_solver.GetIntValue( "NumIterations", num_iterations );
      pcg_solver.GetDoubleValue( "RelResidualNorm", final_res_norm );
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      /* In C++, unlike C, deleteRef's of solvers are not needed here -
         it happens automatically. */
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
      pcg_solver = PCG::Create( mpi_comm, op_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      pcg_solver.SetIntParameter( "MaxIter", 1000 ); /* max iterations */
      pcg_solver.SetDoubleParameter( "Tolerance", 1e-7 ); /* conv. tolerance */
      pcg_solver.SetIntParameter( "TwoNorm", 1 ); /* use the two norm as the stopping criteria */
      pcg_solver.SetIntParameter( "PrintLevel", 2 ); /* prints out the iteration info */
      pcg_solver.SetIntParameter( "Logging", 1 ); /* needed to get run info later */

      /* Now set up the ParaSails preconditioner and specify any parameters */
      ps_solver = ParaSails::Create( mpi_comm, parcsr_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      ps_solver.SetDoubleParameter( "Thresh", sai_threshold );
      ps_solver.SetIntParameter( "Nlevels", sai_max_levels );
      ps_solver.SetDoubleParameter( "Filter", sai_filter );
      ps_solver.SetIntParameter( "Sym", sai_sym );
      ps_solver.SetIntParameter( "Logging", 3 );

      /* Set the PCG preconditioner */
      pcg_solver.SetPreconditioner( ps_solver );

      /* Now setup and solve! */
      // See comments for solver 0 on Apply and vec_x.
      pcg_solver.Setup( par_b, par_x);
      vec_x = par_x;
      pcg_solver.Apply( par_b, vec_x);


      /* Run info - needed logging turned on */
      pcg_solver.GetIntValue( "NumIterations", num_iterations );
      pcg_solver.GetDoubleValue( "RelResidualNorm", final_res_norm );
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      /* In C++, unlike C, deleteRef's of solvers are not needed here -
         it happens automatically. */
   }
   else
   {
      if (myid ==0) printf("Invalid solver id specified.\n");
   }

   /* Print the solution */
   if (print_solution)
      par_x.Print( "ij.out.x" );

   /* Clean up */
   /* In C++, unlike C, deleteRef gets called automatically, so we don't
      need any explicit cleanup. */

   hypre_assert( ierr == 0 );

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
