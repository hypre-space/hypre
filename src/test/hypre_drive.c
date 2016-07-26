/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.52 $
 ***********************************************************************EHEADER*/

#include "hypre_system.h"
#include "hypre_solver.h"

/*#define HYPRE_HAVE_XOM 1*/
#ifndef HYPRE_HAVE_XOM
#define hxom_DriveSolverHelp()
#define hxom_DriveSolverSolve(solver, obj_A, obj_b, obj_x)
#endif

#define DEBUG 0

#define SYSTEM_STRUCT  1
#define SYSTEM_SSTRUCT 2
#define SYSTEM_IJ      3

#define SOLVER_KRYLOV  10
#define SOLVER_STRUCT  11
#define SOLVER_SSTRUCT 12
#define SOLVER_PARCSR  13
#define SOLVER_XOM     14

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
ArgInit(
   char        *argv[],
   HYPRE_Int   *argi_ptr,
   HYPRE_Int   *argn_ptr )
{
   *argi_ptr  = 0;
   *argn_ptr  = 0;
   ArgNext(argv, argi_ptr, argn_ptr);

   return 0;
}

HYPRE_Int
ArgNext(
   char        *argv[],
   HYPRE_Int   *argi_ptr,
   HYPRE_Int   *argn_ptr )
{
   HYPRE_Int   argi, argj, nopen;

   argi  = *argi_ptr + *argn_ptr;
   argj  = argi;
   nopen = 0;

   do
   {
      if ( argv[argj] != NULL )
      {
         if ( strcmp(argv[argj], "{") == 0 )
         {
            nopen++;
         }
         else if ( strcmp(argv[argj], "}") == 0 )
         {
            nopen--;
         }
         argj++;
      }
      else
      {
         break;
      }
   }
   while (nopen > 0);

   if (nopen > 0)
   {
      hypre_printf("Error: invalid input nesting\n");
      exit(1);
   }

   *argi_ptr = argi;
   *argn_ptr = argj - argi;

   return 0;
}

HYPRE_Int
ArgStripBraces(
   char        *argv[],
   HYPRE_Int    argi,
   HYPRE_Int    argn,
   char      ***argv_ptr,
   HYPRE_Int   *argi_ptr,
   HYPRE_Int   *argn_ptr )
{
   HYPRE_Int   argj = argi + argn - 1;

   if ( !(( argv[argi] != NULL ) && ( strcmp(argv[argi], "{") == 0 )) )
   {
      hypre_printf("Error: missing brace '{'\n");
      exit(1);
   }
   if ( !(( argv[argj] != NULL ) && ( strcmp(argv[argj], "}") == 0 )) )
   {
      hypre_printf("Error: missing brace '}'\n");
      exit(1);
   }

   *argv_ptr = &argv[argi + 1];
   *argi_ptr = argi + 1;
   *argn_ptr = argn - 2;

   return 0;
}

HYPRE_Int
hypre_DriveSolverHelp(char *drivename)
{
   hypre_printf("\n");
   hypre_printf("Usage: %s [-system <SystemOptions>] [-solver <SolverOptions>]\n", drivename);
   hypre_printf("\n");
   hypre_printf("SystemOptions: <option>\n");
   hypre_printf("\n");
   hypre_printf("   struct  { <SystemStructOptions> }\n");
   hypre_printf("   sstruct { <SystemSStructOptions> }\n");
   hypre_printf("   ij      { <SystemIJOptions> }\n");
   hypre_printf("\n");
   hypre_printf("SolverOptions: <option>\n");
   hypre_printf("\n");
   hypre_printf("   struct  { <SolverStdOptions> [<KrylovOptions>] [<SolverStructOptions>] }\n");
   hypre_printf("   sstruct { <SolverStdOptions> [<KrylovOptions>] [<SolverSStructOptions>] }\n");
   hypre_printf("   parcsr  { <SolverStdOptions> [<KrylovOptions>] [<SolverParCSROptions>] }\n");
   hypre_printf("\n");
   hypre_DriveSystemStructHelp();
   hypre_DriveSystemSStructHelp();
   hypre_DriveSystemIJHelp();
   hypre_DriveSolverStdHelp();
   hypre_DriveKrylovHelp();
   hypre_DriveSolverStructHelp();
   hypre_DriveSolverSStructHelp();
   hypre_DriveSolverParCSRHelp();
   hxom_DriveSolverHelp();

   return 0;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int             system_type;
   HYPRE_Int             solver_type;
   HYPRE_Int             object_type;

   HYPRE_Int             argi; /* index into argv array */
   HYPRE_Int             argn;
                        
   char                **system_argv;
   char                **solver_argv;
   HYPRE_Int             system_argi, system_argn;
   HYPRE_Int             solver_argi, solver_argn;

   hypre_DriveSolver     krylov, precond, solver;

   HYPRE_Int             precond_bool;

   HYPRE_StructMatrix    s_A;
   HYPRE_StructVector    s_b, s_x;

   HYPRE_SStructMatrix   ss_A;
   HYPRE_SStructVector   ss_b, ss_x;

   HYPRE_IJMatrix        ij_A;
   HYPRE_IJVector        ij_b, ij_x;

   void                 *obj_A, *obj_b, *obj_x;

   HYPRE_Int             myid, time_index;

   HYPRE_Real            tol, atol;
   HYPRE_Int             max_iter;

   HYPRE_Int             num_iterations;
   HYPRE_Real            final_res_norm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   hypre_InitMemoryDebug(myid);

   /*-----------------------------------------------------------
    * Defaults
    *-----------------------------------------------------------*/

   system_type = SYSTEM_IJ;
   solver_type = SOLVER_PARCSR;
   object_type = HYPRE_PARCSR;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   system_argi = system_argn = 0;
   solver_argi = solver_argn = 0;

   ArgInit(argv, &argi, &argn);
   ArgNext(argv, &argi, &argn);
   while (argi < argc)
   {
      if ( strcmp(argv[argi], "-system") == 0 )
      {
         ArgNext(argv, &argi, &argn);
         if ( strcmp(argv[argi], "struct") == 0 )
         {
            system_type = SYSTEM_STRUCT;
         }
         else if ( strcmp(argv[argi], "sstruct") == 0 )
         {
            system_type = SYSTEM_SSTRUCT;
         }
         else if ( strcmp(argv[argi], "ij") == 0 )
         {
            system_type = SYSTEM_IJ;
         }
         ArgNext(argv, &argi, &argn);
         ArgStripBraces(argv, argi, argn, &system_argv, &system_argi, &system_argn);
      }
      else if ( strcmp(argv[argi], "-solver") == 0 )
      {
         ArgNext(argv, &argi, &argn);
         if ( strcmp(argv[argi], "struct") == 0 )
         {
            solver_type = SOLVER_STRUCT;
            object_type = HYPRE_STRUCT;
         }
         else if ( strcmp(argv[argi], "sstruct") == 0 )
         {
            solver_type = SOLVER_SSTRUCT;
            object_type = HYPRE_SSTRUCT;
         }
         else if ( strcmp(argv[argi], "parcsr") == 0 )
         {
            solver_type = SOLVER_PARCSR;
            object_type = HYPRE_PARCSR;
         }
         else if ( strcmp(argv[argi], "xom") == 0 )
         {
            solver_type = SOLVER_XOM;
            object_type = HYPRE_PARCSR;
         }
         ArgNext(argv, &argi, &argn);
         ArgStripBraces(argv, argi, argn, &solver_argv, &solver_argi, &solver_argn);
      }
      else if ( strcmp(argv[argi], "-help") == 0 )
      {
         if (myid == 0)
         {
            hypre_DriveSolverHelp(argv[0]);
         }
         exit(1);
      }
      else
      {
         if (myid == 0)
         {
            hypre_printf("Error: invalid input argument %s\n", argv[argi]);
         }
         exit(1);
      }

      ArgNext(argv, &argi, &argn);
   }

   /*-----------------------------------------------------------
    * Check input parameters
    *-----------------------------------------------------------*/

   switch (system_type)
   {
      case SYSTEM_STRUCT:
      {
         if (object_type != HYPRE_STRUCT)
         {
            if (myid == 0)
            {
               hypre_printf("Error: invalid solver type\n");
            }
            exit(1);
         }
      }
      break;

      case SYSTEM_SSTRUCT:
      {
      }
      break;

      case SYSTEM_IJ:
      {
         if (object_type != HYPRE_PARCSR)
         {
            if (myid == 0)
            {
               hypre_printf("Error: invalid solver type\n");
            }
            exit(1);
         }
      }
      break;
   }

   /*-----------------------------------------------------------
    * Print hypre drive parameters TODO
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Set up the system
    *-----------------------------------------------------------*/

   switch (system_type)
   {
      case SYSTEM_STRUCT:
      {
         hypre_DriveSystemStructCreate(system_argv, system_argn, object_type,
                                       &s_A, &s_b, &s_x);
         /* No GetObject in this interface */
      }
      break;

      case SYSTEM_SSTRUCT:
      {
         hypre_DriveSystemSStructCreate(system_argv, system_argn, object_type,
                                        &ss_A, &ss_b, &ss_x);
         HYPRE_SStructMatrixGetObject(ss_A, &obj_A);
         HYPRE_SStructVectorGetObject(ss_b, &obj_b);
         HYPRE_SStructVectorGetObject(ss_x, &obj_x);
      }
      break;

      case SYSTEM_IJ:
      {
         hypre_DriveSystemIJCreate(system_argv, system_argn, object_type,
                                   &ij_A, &ij_b, &ij_x);
         HYPRE_IJMatrixGetObject(ij_A, &obj_A);
         HYPRE_IJVectorGetObject(ij_b, &obj_b);
         HYPRE_IJVectorGetObject(ij_x, &obj_x);
      }
      break;
   }

   /*-----------------------------------------------------------
    * Set up the solver
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("Solver Setup");
   hypre_BeginTiming(time_index);

   /* Check for standard solver options */
   hypre_DriveSolverStdDefaults(&tol, &atol, &max_iter);
   hypre_DriveSolverStdOptions(solver_argv, solver_argn, &tol, &atol, &max_iter);

   /* Check for krylov method and solver/preconditioner */
   switch (solver_type)
   {
      case SOLVER_STRUCT:
      {
      }
      break;

      case SOLVER_SSTRUCT:
      {
         hypre_DriveSolverSStructCreate(solver_argv, solver_argn, &krylov, &precond);
      }
      break;

      case SOLVER_PARCSR:
      {
      }
      break;

      case SOLVER_XOM:
      {
      }
      break;
   }

   /* Set up the solver */

   precond_bool = (krylov.id != NONE);

   if (precond.id != NONE)
   {
      switch (solver_type)
      {
         case SOLVER_STRUCT:
         {
         }
         break;
         
         case SOLVER_SSTRUCT:
         {
            hypre_DriveSolverSStructSetup(precond, precond_bool, tol, atol, max_iter,
                                          obj_A, obj_b, obj_x);
         }
         break;
         
         case SOLVER_PARCSR:
         {
         }
         break;
         
         case SOLVER_XOM:
         {
         }
         break;
      }
   }

   if (krylov.id != NONE)
   {
      hypre_DriveKrylovSetup(krylov, precond, tol, atol, max_iter,
                             obj_A, obj_b, obj_x);
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   if (krylov.id != NONE)
   {
      solver = krylov;
      solver_type = SOLVER_KRYLOV;
   }
   else if (precond.id != NONE)
   {
      solver = precond;
   }
   else
   {
      /* We should at least have a default solver, so something is wrong */
      hypre_printf("Error: failed to create a solver\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("Solver Solve");
   hypre_BeginTiming(time_index);

   switch (solver_type)
   {
      case SOLVER_STRUCT:
      {
         hypre_DriveSolverStructSolve(solver, s_A, s_b, s_x);
      }
      break;

      case SOLVER_SSTRUCT:
      {
         hypre_DriveSolverSStructSolve(solver, obj_A, obj_b, obj_x);
      }
      break;

      case SOLVER_PARCSR:
      {
         hypre_DriveSolverParCSRSolve(solver, obj_A, obj_b, obj_x);
      }
      break;

      case SOLVER_XOM:
      {
         hxom_DriveSolverSolve(solver, obj_A, obj_b, obj_x);
      }
      break;

      case SOLVER_KRYLOV:
      {
         hypre_DriveKrylovSolve(solver, obj_A, obj_b, obj_x);
      }
      break;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Print results
    *-----------------------------------------------------------*/

   switch (solver_type)
   {
      case SOLVER_STRUCT:
      {
      }
      break;

      case SOLVER_SSTRUCT:
      {
         hypre_DriveSolverSStructGetStats(solver, &num_iterations, &final_res_norm);
      }
      break;

      case SOLVER_PARCSR:
      {
      }
      break;

      case SOLVER_XOM:
      {
      }
      break;

      case SOLVER_KRYLOV:
      {
         hypre_DriveKrylovGetStats(solver, &num_iterations, &final_res_norm);
      }
      break;
   }

   if (myid == 0)
   {
      hypre_printf("\n");
      hypre_printf("Iterations = %d\n", num_iterations);
      hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Destroy the solver
    *-----------------------------------------------------------*/

   switch (solver_type)
   {
      case SOLVER_STRUCT:
      {
      }
      break;
      
      case SOLVER_SSTRUCT:
      {
         hypre_DriveSolverSStructDestroy(krylov, precond);
      }
      break;
      
      case SOLVER_PARCSR:
      {
      }
      break;
      
      case SOLVER_XOM:
      {
      }
      break;
   }

   /*-----------------------------------------------------------
    * Do something with the solution and destroy the system
    *-----------------------------------------------------------*/

   switch (system_type)
   {
      case SYSTEM_STRUCT:
      {
         hypre_DriveSystemStructDestroy(s_A, s_b, s_x);
      }
      break;

      case SYSTEM_SSTRUCT:
      {
         hypre_DriveSystemSStructDestroy(ss_A, ss_b, ss_x);
      }
      break;

      case SYSTEM_IJ:
      {
         hypre_DriveSystemIJDestroy(ij_A, ij_b, ij_x);
      }
      break;
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   hypre_FinalizeMemoryDebug();

   hypre_MPI_Finalize();

   return (0);
}
