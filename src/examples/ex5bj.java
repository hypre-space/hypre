/**
   Example 5

   Interface:    Linear-Algebraic (IJ), Babel-based version, in Java.

   Note: The Java interface is not ready for distribution, because of:
     (1) MPI-Java conflicts (see below),
     (2) It requires more parts of Babel to be added to the Hypre distribution
   (see below), as well as minor build system changes, none of which are worth
   doing until (1) gets solved.

   Compile with: javac ex5bj.java

   Sample run:   java ex5b

   Description:  This example solves the 2-D
                 Laplacian problem with zero boundary conditions
                 on an nxn grid.  The number of unknowns is N=n^2.
                 The standard 5-point stencil is used, and we solve
                 for the interior nodes only.

                 This example solves the same problem as Example 3.
                 Available solvers are AMG, PCG, and PCG with AMG or
                 Parasails preconditioners.
*/
// --- MPI Notes:
// 
// With the mpich version on the CASC cluster, there are problems combining Java and
// MPI which involve threads and signals.
// This example code runs well on one processor with Java 1.6.0.  But, rather than
// use an MPI binding for Java, this example uses Hypre for all MPI calls, even
// MPI_Init and MPI_Finalize which are normally in user code.
//
// The only MPI binding for Java which I have experimented with is mpiJava.
// For information on the problems due to a signal handling conflict, see
// http://www.hpjava.org/mpijavareadme.html .
// The problem involving threads is that the MPI_Init call through mpiJava will not
// initialize MPI for the MPI calls in Hypre. My conjecture is that, for this example
// program, Java runs mpiJava and Hypre in different threads, and MPI_Init called in
// one thread will not necessarily set up MPI for another thread.  Neither mpiJava
// nor Hypre calls MPI_Init_Thread or other thread-oriented functions found in MPI-2.
//
// --- Building:
//
// Here are the environment variables which worked for me; change them as needed for your
// environment.  Note that CLASSPATH has a reference to the full Babel directory, which is
// not distributed in Hypre.  Also note that SIDL_DLL_PATH has a reference to my home
// library - this is for libraries which were copied there from a full Babel directory.
//
// setenv HYPRE_SRC_TOP_DIR /home/painter/linear_solvers
// ... just used to set environment variables below...
//
// setenv CLASSPATH .:${HYPRE_SRC_TOP_DIR}/babel-runtime/java:${HYPRE_SRC_TOP_DIR}/babel/bHYPREClient-J/:$HOME/babel-1.0.4/lib/sidlstubs/Java
// ... Moreover, if you want to try mpiJava, do:
// setenv CLASSPATH ${CLASSPATH}:$HOME/src/mpiJava/lib/classes
//
// setenv LD_LIBRARY_PATH /usr/apps/java/jdk1.6.0/lib:/usr/apps/lib:$HOME/lib:$HOME/linear_solvers/hypre/lib
// ... Moreover, if you want to try mpiJava, do:
// setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:$HOME/src/mpiJava/lib
//
// setenv SIDL_DLL_PATH $HOME/lib/libsidlstub_java.scl\;${HYPRE_SRC_TOP_DIR}/babel/bHYPREClient-P/libbHYPRE.scl\;${HYPRE_SRC_TOP_DIR}/hypre/lib/\;${HYPRE_SRC_TOP_DIR}/babel-runtime/sidl/libsidl.scl\;${HYPRE_SRC_TOP_DIR}/babel/bHYPREClient-J/libbHYPRE.scl
// ... you can "setenv SIDL_DEBUG_DLOPEN" to debug this, it helps a little but not much
// ... the Python directory bHYPREClient-P is not needed unless you may also run Python
//
// In hypre directory: ./configure --with-babel --enable-shared --enable-debug --enable-java
// (... --enable-debug is optional, but prudent for the first time through!)
// make
// In babel/bHYPREClient-J/bHYPRE do: javac -g *.java  (again, -g is optional but prudent)
//
// --- More notes:
// 1. there are no deleteRef calls as for other languages, as required in the Babel manual-
//    Java garbage collection does the job
// 2. The interface definitions (Interfaces.idl) often use R-arrays.  Babel only supports
//    sidl arrays in Java.

import java.lang.Math.*;
// use this if you have a working MPI for Java: import mpi.*;
import bHYPRE.*;

class ex5b {
// use this if you have a working MPI for Java:
//   public static void main(String[] args) throws MPIException {
   public static void main(String[] args) {

   int i;
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;

   int solver_id;
   boolean print_solution;
   sidl.Integer.Holder intout  = new sidl.Integer.Holder(0); // for output arguments
   sidl.Double.Holder doubout  = new sidl.Double.Holder(0); // for output arguments

   double h, h2;

   int ierr = 0;
   /* If this gets set to anything else, it's an error.
    Most functions return error flags, 0 unless there's an error.
    For clarity, they aren't checked much in this file. */

// use this if you have a working MPI for Java:   Comm mpi_comm_world = MPI.COMM_WORLD;
   MPICommunicator mpi_comm;
   IJParCSRMatrix parcsr_A;
   IJParCSRVector par_b;
   IJParCSRVector par_x;
   Vector vec_x;

   BoomerAMG amg_solver;
   ParaSails ps_solver;
   PCG pcg_solver;
   IdentitySolver identity;

   /* Initialize MPI */
   myid = 0; num_procs = 1;
   MPICommunicator.Init();
   /* use these instead if you have a working MPI library for Java...
   if ( args.length==0 )
   {  // it appears that the MPI.Init I'm using (mpiJava) won't accept a zero-length args...
      String[] noargs = new String[1];
      noargs[0] = "jfp";
      MPI.Init( noargs );
   }
   else
      MPI.Init(args);
   MPICommunicator.Init();
   myid = mpi_comm_world.Rank(); 
   num_procs = mpi_comm_world.Size();
   */

   mpi_comm = MPICommunicator.Create_MPICommWorld();

   /* Default problem parameters */
   n = 33;
   solver_id = 0;
   print_solution  = false;

   /* Parse command line */
   {
      int arg_index = 0;
      boolean print_usage = false;


      while (arg_index < args.length )
      {
         if ( (args[arg_index]).equals("-n") )
         {
            arg_index++;
            n = Integer.parseInt(args[arg_index++]);
         }
         else if ( (args[arg_index]).equals("-solver") )
         {
            arg_index++;
            solver_id = Integer.parseInt(args[arg_index++]);
         }
         else if ( (args[arg_index]).equals("-print_solution") )
         {
            arg_index++;
            print_solution = true;
         }
         else if ( (args[arg_index]).equals("-help") )
         {
            print_usage = true;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage) && (myid == 0))
      {
         System.out.println();
         System.out.print("Usage: java ex5b [<options>]\n");
         System.out.println();
         System.out.println("  -n <n>              : problem size in each direction (default: 33)");
         System.out.println("  -solver <ID>        : solver ID");
         System.out.println("                        0  - AMG (default) ");
         System.out.println("                        1  - AMG-PCG");
         System.out.println("                        8  - ParaSails-PCG");
         System.out.println("                        50 - PCG");
         System.out.println("  -print_solution     : print the solution vector");
         System.out.println();
      }

      if (print_usage)
      {
         MPICommunicator.Finalize();
         /* use this instead if you have a working MPI library for Java...
         MPI.Finalize();
         */
         System.exit(0);
      }
   }

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = (int)Math.sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += Math.min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += Math.min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   parcsr_A = IJParCSRMatrix.Create( mpi_comm,
                                            ilower, iupper, ilower, iupper );

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
      sidl.Double.Array1 values = new sidl.Double.Array1(5,true);
      sidl.Integer.Array1 cols = new sidl.Integer.Array1(5,true);
      sidl.Integer.Array1 nnza = new sidl.Integer.Array1( 1, true );
      sidl.Integer.Array1 ia = new sidl.Integer.Array1( 1, true );

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;

         /* The left identity block:position i-n */
         if ((i-n)>=0)
         {
	    cols.set( nnz, i-n );
	    values.set( nnz, -1.0 );
	    nnz++;
         }

         /* The left -1: position i-1 */
         if (i%n!=0)
         {
            cols.set( nnz, i-1 );
            values.set( nnz, -1.0 );
            nnz++;
         }

         /* Set the diagonal: position i */
         cols.set( nnz, i );
         values.set( nnz, 4.0 );
         nnz++;

         /* The right -1: position i+1 */
         if ((i+1)%n!=0)
         {
            cols.set( nnz, i+1 );
            values.set( nnz, -1.0 );
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i+n)< N)
         {
            cols.set( nnz, i+n );
            values.set( nnz, -1.0 );
            nnz++;
         }

         /* Set the values for row i */
         nnza.set(0,nnz);
         ia.set(0,i);
         parcsr_A.SetValues( nnza, ia, cols, values );
      }
   }

   /* Assemble after setting the coefficients */
   parcsr_A.Assemble();

   /* Create the rhs and solution */
   par_b = IJParCSRVector.Create( mpi_comm, ilower, iupper );

   par_b.Initialize();

   par_x = IJParCSRVector.Create( mpi_comm, ilower, iupper );

   // probably vec_x doesn't have to be initialized if par_x
   // is used as a solver arg; look into this later...
   vec_x = par_x;

   par_x.Initialize();

   /* Set the rhs values to h^2 and the solution to zero */
   {
      sidl.Double.Array1 rhs_values = new sidl.Double.Array1(local_size,true);
      sidl.Double.Array1 x_values = new sidl.Double.Array1(local_size,true);
      sidl.Integer.Array1 rows = new sidl.Integer.Array1(local_size,true);

      for (i=0; i<local_size; i++)
      {
         rhs_values.set( i, h2 );
         x_values.set( i, 0.0 );
         rows.set( i, ilower + i );
      }

      par_b.SetValues( rows, rhs_values );
      par_x.SetValues( rows, x_values );

      x_values = null;
      rhs_values = null;
      rows = null;
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
      amg_solver = BoomerAMG.Create( mpi_comm, parcsr_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      amg_solver.SetIntParameter( "PrintLevel", 3 );  /* print solve info + parameters */
      amg_solver.SetIntParameter( "CoarsenType", 6 ); /* Falgout coarsening */
      amg_solver.SetIntParameter( "RelaxType", 3 );   /* G-S/Jacobi hybrid relaxation */
      amg_solver.SetIntParameter( "NumSweeps", 1 );   /* Sweeeps on each level */
      amg_solver.SetIntParameter( "MaxLevels", 20 );  /* maximum number of levels */
      amg_solver.SetDoubleParameter( "Tolerance", 1e-7 );      /* conv. tolerance */

      /* Now setup and solve!
       Note the use of a Babel Holder class to deal with output arguments. */
      amg_solver.Setup( par_b, vec_x );
      bHYPRE.Vector.Holder vec_x_h = new bHYPRE.Vector.Holder(vec_x);
      amg_solver.Apply( par_b, vec_x_h );
      vec_x = vec_x_h.get();

      /* Run info - needed logging turned on.
       Note the use of Babel Holder objects to handle the output arguments. */
      ierr += amg_solver.GetIntValue( "NumIterations", intout );
      num_iterations = intout.get();
      amg_solver.GetDoubleValue( "RelResidualNorm", doubout );
      final_res_norm = doubout.get();

      if (myid == 0)
      {
         System.out.print("\n");
         System.out.print("Iterations = ");
         System.out.println(num_iterations);
         System.out.print("Final Relative Residual Norm = ");
         System.out.println(final_res_norm);
      }
   }

   /* PCG */
   else if (solver_id == 50)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      pcg_solver = PCG.Create( mpi_comm, parcsr_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      pcg_solver.SetIntParameter( "MaxIter", 1000 ); /* max iterations */
      pcg_solver.SetDoubleParameter( "Tolerance", 1e-7 ); /* conv. tolerance */
      pcg_solver.SetIntParameter( "TwoNorm", 1 ); /* use the two norm as the stopping criteria */
      pcg_solver.SetIntParameter( "PrintLevel", 2 ); /* prints out the iteration info */
      pcg_solver.SetIntParameter( "Logging", 1 ); /* needed to get run info later */

      identity = IdentitySolver.Create( mpi_comm );
      pcg_solver.SetPreconditioner( identity );

      /* Now setup and solve!
       Note the use of a Babel Holder class to deal with output arguments. */
      pcg_solver.Setup( par_b, vec_x );
      bHYPRE.Vector.Holder vec_x_h = new bHYPRE.Vector.Holder(vec_x);
      pcg_solver.Apply( par_b, vec_x_h );
      vec_x = vec_x_h.get();

      /* Run info - needed logging turned on 
       Note the use of Babel Holder objects to handle the output arguments. */
      pcg_solver.GetIntValue( "NumIterations", intout );
      num_iterations = intout.get();
      pcg_solver.GetDoubleValue( "RelResidualNorm", doubout );
      final_res_norm = doubout.get();

      if (myid == 0)
      {
         System.out.print("\n");
         System.out.print("Iterations = ");
         System.out.println(num_iterations);
         System.out.print("Final Relative Residual Norm = ");
         System.out.println(final_res_norm);
      }
   }
   /* PCG with AMG preconditioner */
   else if (solver_id == 1)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      pcg_solver = PCG.Create( mpi_comm, parcsr_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      pcg_solver.SetIntParameter( "MaxIter", 1000 ); /* max iterations */
      pcg_solver.SetDoubleParameter( "Tolerance", 1e-7 ); /* conv. tolerance */
      pcg_solver.SetIntParameter( "TwoNorm", 1 ); /* use the two norm as the stopping criteria */
      pcg_solver.SetIntParameter( "PrintLevel", 2 ); /* prints out the iteration info */
      pcg_solver.SetIntParameter( "Logging", 1 ); /* needed to get run info later */

      /* Now set up the AMG preconditioner and specify any parameters */
      amg_solver = BoomerAMG.Create( mpi_comm, parcsr_A );
      amg_solver.SetIntParameter( "PrintLevel", 1 ); /* print amg solution info*/
      amg_solver.SetIntParameter( "CoarsenType", 6 ); /* Falgout coarsening */
      amg_solver.SetIntParameter( "RelaxType", 6 );   /* Sym G-S/Jacobi hybrid relaxation */
      amg_solver.SetIntParameter( "NumSweeps", 1 );   /* Sweeeps on each level */
      amg_solver.SetDoubleParameter( "Tolerance", 0);      /* conv. tolerance */
      amg_solver.SetIntParameter( "MaxIter", 1 ); /* do only one iteration! */

      /* Set the PCG preconditioner */
      pcg_solver.SetPreconditioner( amg_solver );

      /* Now setup and solve!
       Note the use of a Babel Holder class to deal with output arguments. */
      pcg_solver.Setup( par_b, vec_x );
      bHYPRE.Vector.Holder vec_x_h = new bHYPRE.Vector.Holder(vec_x);
      pcg_solver.Apply( par_b, vec_x_h );
      vec_x = vec_x_h.get();

      /* Run info - needed logging turned on
       Note the use of Babel Holder objects to handle the output arguments.  */
      pcg_solver.GetIntValue( "NumIterations", intout );
      num_iterations = intout.get();
      pcg_solver.GetDoubleValue( "RelResidualNorm", doubout );
      final_res_norm = doubout.get();

      if (myid == 0)
      {
         System.out.print("\n");
         System.out.print("Iterations = ");
         System.out.println(num_iterations);
         System.out.print("Final Relative Residual Norm = ");
         System.out.println(final_res_norm);
      }
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
      pcg_solver = PCG.Create( mpi_comm, parcsr_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      pcg_solver.SetIntParameter( "MaxIter", 1000 ); /* max iterations */
      pcg_solver.SetDoubleParameter( "Tolerance", 1e-7 ); /* conv. tolerance */
      pcg_solver.SetIntParameter( "TwoNorm", 1 ); /* use the two norm as the stopping criteria */
      pcg_solver.SetIntParameter( "PrintLevel", 2 ); /* prints out the iteration info */
      pcg_solver.SetIntParameter( "Logging", 1 ); /* needed to get run info later */

      /* Now set up the ParaSails preconditioner and specify any parameters */
      ps_solver = ParaSails.Create( mpi_comm, parcsr_A );

      /* Set some parameters (See Reference Manual for more parameters) */
      ps_solver.SetDoubleParameter( "Thresh", sai_threshold );
      ps_solver.SetIntParameter( "Nlevels", sai_max_levels );
      ps_solver.SetDoubleParameter( "Filter", sai_filter );
      ps_solver.SetIntParameter( "Sym", sai_sym );
      ps_solver.SetIntParameter( "Logging", 3 );

      /* Set the PCG preconditioner */
      pcg_solver.SetPreconditioner( ps_solver );

      /* Now setup and solve!
       Note the use of a Babel Holder class to deal with output arguments. */
      pcg_solver.Setup( par_b, vec_x );
      bHYPRE.Vector.Holder vec_x_h = new bHYPRE.Vector.Holder(vec_x);
      pcg_solver.Apply( par_b, vec_x_h );
      vec_x = vec_x_h.get();


      /* Run info - needed logging turned on
       Note the use of Babel Holder objects to handle the output arguments.  */
      pcg_solver.GetIntValue( "NumIterations", intout );
      num_iterations = intout.get();
      pcg_solver.GetDoubleValue( "RelResidualNorm", doubout );
      final_res_norm = doubout.get();

      if (myid == 0)
      {
         System.out.print("\n");
         System.out.print("Iterations = ");
         System.out.println(num_iterations);
         System.out.print("Final Relative Residual Norm = ");
         System.out.println(final_res_norm);
      }
   }
   else
   {
      if (myid ==0) System.out.print("Invalid solver id specified.\n");
   }

   /* Print the solution */
   if (print_solution)
      par_x.Print( "ij.out.x" );

   // The following is commented out because assert is a relatively new Java feature
   // not correctly implemented in some old versions in use at LLNL:
   // assert ierr == 0 : ierr;

   /* Finalize MPI*/
   MPICommunicator.Finalize();
   /* use this instead if you have a working MPI library for Java...
   MPI.Finalize();
    */

   }}
