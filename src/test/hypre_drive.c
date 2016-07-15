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
  #define hxom_DriveSolve(argv, argi, argn, obj_A, obj_b, obj_x)
#endif

#define DEBUG 0

#define SYSTEM_STRUCT  1
#define SYSTEM_SSTRUCT 2
#define SYSTEM_IJ      3

#define SOLVER_STRUCT  11
#define SOLVER_SSTRUCT 12
#define SOLVER_PARCSR  13
#define SOLVER_XOM     14

static HYPRE_Int  argn_notset = 0;
static HYPRE_Int *argn_ref = &argn_notset;
#define ArgSet(argi, argn, index)  argi = argn = index; argn_ref = &argn
#define ArgInc()                   (*argn_ref)++

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int             system_type;
   HYPRE_Int             solver_type;
   HYPRE_Int             solver_pkg;

   HYPRE_Int             arg_index; /* index into argv array */
                        
   /* argi = first index in a group (i for index) */
   /* argn = first index in next group (ni for index) */
   HYPRE_Int             system_argi, system_argn;
   HYPRE_Int             solver_argi, solver_argn;

   HYPRE_StructMatrix    s_A;
   HYPRE_StructVector    s_b, s_x;

   HYPRE_SStructMatrix   ss_A;
   HYPRE_SStructVector   ss_b, ss_x;

   HYPRE_IJMatrix        ij_A;
   HYPRE_IJVector        ij_b, ij_x;

   void                 *obj_A, *obj_b, *obj_x;

   HYPRE_Int             myid;

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
   solver_type = HYPRE_PARCSR;
   solver_pkg = SOLVER_PARCSR;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   arg_index = 1;

   system_argi = system_argn = arg_index;
   solver_argi = solver_argn = arg_index;

   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-system") == 0 )
      {
         arg_index++;
         if ( strcmp(argv[arg_index], "struct") == 0 )
         {
            system_type = SYSTEM_STRUCT;
         }
         else if ( strcmp(argv[arg_index], "sstruct") == 0 )
         {
            system_type = SYSTEM_SSTRUCT;
         }
         else if ( strcmp(argv[arg_index], "ij") == 0 )
         {
            system_type = SYSTEM_IJ;
         }
         arg_index++;

         ArgSet(system_argi, system_argn, arg_index);
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         if ( strcmp(argv[arg_index], "struct") == 0 )
         {
            solver_type = HYPRE_STRUCT;
            solver_pkg = SOLVER_STRUCT;
         }
         else if ( strcmp(argv[arg_index], "sstruct") == 0 )
         {
            solver_type = HYPRE_SSTRUCT;
            solver_pkg = SOLVER_SSTRUCT;
         }
         else if ( strcmp(argv[arg_index], "parcsr") == 0 )
         {
            solver_type = HYPRE_PARCSR;
            solver_pkg = SOLVER_PARCSR;
         }
         else if ( strcmp(argv[arg_index], "xom") == 0 )
         {
            solver_type = HYPRE_PARCSR;
            solver_pkg = SOLVER_XOM;
         }
         arg_index++;

         ArgSet(solver_argi, solver_argn, arg_index);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         if (myid == 0)
         {
            hypre_printf("\n");
            hypre_printf("Usage: %s [-system <options>] [-solver <options>]\n",
                         argv[0]);
            hypre_printf("\n");
            hypre_printf("System Options: struct | sstruct | ij     <options>\n");
            hypre_printf("Solver Options: struct | sstruct | parcsr <options>\n");
            hypre_printf("\n");
            hypre_DriveSystemStructHelp();
            hypre_DriveSystemSStructHelp();
            hypre_DriveSystemIJHelp();
            hypre_DriveSolverGeneralHelp();
            hypre_DriveSolverStructHelp();
            hypre_DriveSolverSStructHelp();
            hypre_DriveSolverParCSRHelp();
            hxom_DriveSolverHelp();
         }
         exit(1);
      }
      else
      {
         arg_index++;
         ArgInc();
      }
   }

   /*-----------------------------------------------------------
    * Check input parameters
    *-----------------------------------------------------------*/

   switch (system_type)
   {
      case SYSTEM_STRUCT:
      {
         if (solver_type != HYPRE_STRUCT)
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
         if (solver_type != HYPRE_PARCSR)
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
    * Print driver parameters TODO
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
         hypre_DriveSystemStructCreate(argv, system_argi, system_argn, solver_type,
                                       &s_A, &s_b, &s_x);
         /* No GetObject in this interface */
      }
      break;

      case SYSTEM_SSTRUCT:
      {
         hypre_DriveSystemSStructCreate(argv, system_argi, system_argn, solver_type,
                                        &ss_A, &ss_b, &ss_x);
         HYPRE_SStructMatrixGetObject(ss_A, &obj_A);
         HYPRE_SStructVectorGetObject(ss_b, &obj_b);
         HYPRE_SStructVectorGetObject(ss_x, &obj_x);
      }
      break;

      case SYSTEM_IJ:
      {
         hypre_DriveSystemIJCreate(argv, system_argi, system_argn, solver_type,
                                   &ij_A, &ij_b, &ij_x);
         HYPRE_IJMatrixGetObject(ij_A, &obj_A);
         HYPRE_IJVectorGetObject(ij_b, &obj_b);
         HYPRE_IJVectorGetObject(ij_x, &obj_x);
      }
      break;
   }

   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   switch (solver_pkg)
   {
      case SOLVER_STRUCT:
      {
         hypre_DriveSolveStruct(argv, solver_argi, solver_argn,
                                s_A, s_b, s_x);
      }
      break;

      case SOLVER_SSTRUCT:
      {
         hypre_DriveSolveSStruct(argv, solver_argi, solver_argn,
                                 obj_A, obj_b, obj_x);
      }
      break;

      case SOLVER_PARCSR:
      {
         hypre_DriveSolveParCSR(argv, solver_argi, solver_argn,
                                obj_A, obj_b, obj_x);
      }
      break;

      case SOLVER_XOM:
      {
         hxom_DriveSolve(argv, solver_argi, solver_argn,
                         obj_A, obj_b, obj_x);
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
