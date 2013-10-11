/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





#include "headers.h"
#include "timing.h"


#ifdef AMG_MALLOC_DEBUG
/* malloc debug stuff */
char amg_malloclog[256];
#endif

/*--------------------------------------------------------------------------
 * Main driver for AMG
 *--------------------------------------------------------------------------*/

HYPRE_Int   main(argc, argv)
HYPRE_Int   argc;
char *argv[];
{
   char    *run_name;

   char     file_name[255];
   FILE    *fp;

   Problem *problem;
   Solver  *solver;

   hypre_Matrix  *A;
   hypre_Vector  *u;
   hypre_Vector  *f;
   HYPRE_Real   stop_tolerance;
   void    *amg_data;
   void    *wjacobi_data;
   void    *pcg_data;
   void    *gmres_data;

   amg_Clock_t time_ticks;
   amg_CPUClock_t cpu_ticks;
   amg_Clock_t start_ticks;
   amg_CPUClock_t start_cpu;
   amg_Clock_t setup_ticks;
   amg_CPUClock_t setup_cpu;
   amg_Clock_t solve_ticks;
   amg_CPUClock_t solve_cpu;


   /*-------------------------------------------------------
    * Check that the number of command args is correct
    *-------------------------------------------------------*/

   if (argc < 2)
   {
      hypre_fprintf(stderr, "Usage:  amg <run name>\n");
      exit(1);
   }

   /*-------------------------------------------------------
    * Set up debugging tools
    *-------------------------------------------------------*/

#ifdef AMG_MALLOC_DEBUG
   /* malloc debug stuff */
   malloc_logpath = amg_malloclog;
   hypre_sprintf(malloc_logpath, "malloc.log");
#endif

   /*-------------------------------------------------------
    * Set up globals 
    *-------------------------------------------------------*/

   run_name = argv[1];
   NewGlobals(run_name);

   /*-------------------------------------------------------
    * Set up the problem
    *-------------------------------------------------------*/

   hypre_sprintf(file_name, "%s.problem.strp", GlobalsInFileName);
   problem = NewProblem(file_name);

   /*-------------------------------------------------------
    * Set up the solver
    *-------------------------------------------------------*/

   hypre_sprintf(file_name, "%s.solver.strp", GlobalsInFileName);
   solver = NewSolver(file_name);

   /*-------------------------------------------------------
    * Write initial logging info
    *-------------------------------------------------------*/

   if (SolverAMGIOutDat(solver) != 0)
   {
      fp = fopen(GlobalsLogFileName, "w");
      fclose(fp);

      WriteProblem(GlobalsLogFileName, problem);
      if (SolverType(solver) != SOLVER_AMG)
      {
         WriteSolver(GlobalsLogFileName, solver);
      }
   }

   /*-------------------------------------------------------
    * Call the solver
    *-------------------------------------------------------*/

   A = ProblemA(problem);
   u = ProblemU(problem);
   f = ProblemF(problem);

   stop_tolerance = SolverStopTolerance(solver);

   amg_data     = NewAMGData(problem, solver, GlobalsLogFileName);
   wjacobi_data = NewWJacobiData(problem, solver, GlobalsLogFileName);
   pcg_data     = NewPCGData(problem, solver, GlobalsLogFileName);
   gmres_data   = NewGMRESData(problem, solver, GlobalsLogFileName);

   /* call AMG */

   HYPRE_AMGClock_init();
   start_ticks = HYPRE_AMGClock(); 
   start_cpu = HYPRE_AMGCPUClock();

   if (SolverType(solver) == SOLVER_AMG)
   {
      HYPRE_Int setup_err_flag;
      HYPRE_Int solve_err_flag;

      setup_err_flag = HYPRE_AMGSetup(A, amg_data);
      if (setup_err_flag != 0) 
      {
         hypre_printf("setup error = %d\n",setup_err_flag);
         if (setup_err_flag > 0)
         {
            return 1;
         }
         hypre_printf("Setup Error Warning. Execution Continues.\n");
      }

      setup_ticks = HYPRE_AMGClock() - start_ticks;
      setup_cpu =   HYPRE_AMGCPUClock() - start_cpu;

      solve_err_flag = HYPRE_AMGSolve(u, f, stop_tolerance, amg_data);
      if (solve_err_flag != 0) hypre_printf("solve error = %d\n",solve_err_flag);

      solve_ticks = HYPRE_AMGClock() - (start_ticks + setup_ticks);
      solve_cpu =   HYPRE_AMGCPUClock() - (start_cpu + setup_cpu);
      time_ticks =  HYPRE_AMGClock() - start_ticks;
      cpu_ticks =   HYPRE_AMGCPUClock() - start_cpu;
   }

   /* call Jacobi */
   if (SolverType(solver) == SOLVER_Jacobi)
   {
      WJacobiSetup(A, amg_data);

      WJacobi(u, f, stop_tolerance, wjacobi_data);
   }

   /* call AMG PCG */
   else if (SolverType(solver) == SOLVER_AMG_PCG)
   {
      HYPRE_AMGSetup(A, amg_data);
      PCGSetup(A, HYPRE_AMGSolve, amg_data, pcg_data);

      PCG(u, f, stop_tolerance, pcg_data);
   }

   /* call Jacobi PCG */
   else if (SolverType(solver) == SOLVER_Jacobi_PCG)
   {
      WJacobiSetup(A, wjacobi_data);
      PCGSetup(A, WJacobi, wjacobi_data, pcg_data);

      PCG(u, f, stop_tolerance, pcg_data);
   }

   /* call AMG GMRES */
   else if (SolverType(solver) == SOLVER_AMG_GMRES)
   {
      HYPRE_AMGSetup(A, amg_data);
      GMRESSetup(A, HYPRE_AMGSolve, amg_data, gmres_data);

      GMRES(u, f, stop_tolerance, gmres_data);
   }

   /* call Jacobi GMRES */
   else if (SolverType(solver) == SOLVER_Jacobi_GMRES)
   {
      WJacobiSetup(A, wjacobi_data);
      GMRESSetup(A, WJacobi, wjacobi_data, gmres_data);

      GMRES(u, f, stop_tolerance, gmres_data);
   }

/* The following should be replaced at a later data with
   a more appropriate routine.  That is, we should call
   amg_TimingOut(amg_times) where amg_times is a structure 
   carrying the timing information.  The following works 
   for now, however.  VEH 9/24/97                          */

   if (SolverAMGIOutDat(solver) >= 3)
   {
      hypre_longint AMG_CPU_TICKS_PER_SEC;

      AMG_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);

      fp = fopen(GlobalsLogFileName, "a");
 
      hypre_fprintf(fp,"\nTIMING INFORMATION\n");
      hypre_fprintf(fp,"\nSetup Time:\n");
      hypre_fprintf(fp, " wall clock time = %f seconds\n", 
                         ((HYPRE_Real) setup_ticks)/AMG_TICKS_PER_SEC);
      hypre_fprintf(fp," CPU clock time  = %f seconds\n", 
                         ((HYPRE_Real) setup_cpu)/AMG_CPU_TICKS_PER_SEC);

      hypre_fprintf(fp,"\nSolve Time:\n");
      hypre_fprintf(fp, " wall clock time = %f seconds\n", 
                         ((HYPRE_Real) solve_ticks)/AMG_TICKS_PER_SEC);
      hypre_fprintf(fp," CPU clock time  = %f seconds\n", 
                         ((HYPRE_Real) solve_cpu)/AMG_CPU_TICKS_PER_SEC);
 
      hypre_fprintf(fp,"\nOverall Time:\n");
      hypre_fprintf(fp, " wall clock time = %f seconds\n", 
                         ((HYPRE_Real) time_ticks)/AMG_TICKS_PER_SEC);
      hypre_fprintf(fp," CPU clock time  = %f seconds\n", 
                         ((HYPRE_Real) cpu_ticks)/AMG_CPU_TICKS_PER_SEC);
  
      fclose(fp);

   }


   /*-------------------------------------------------------
    * Debugging prints
    *-------------------------------------------------------*/

#ifdef AMG_MALLOC_DEBUG
   /* malloc debug stuff */
   malloc_verify(0);
   malloc_shutdown();
#endif

#if 0
   hypre_sprintf(file_name, "%s.lastu", GlobalsOutFileName);
   hypre_WriteVec(file_name, u);
#endif
#if 0
   hypre_printf("soln norm = %e\n", sqrt(hypre_InnerProd(u,u)));

   hypre_printf("rhs norm = %e\n", sqrt(hypre_InnerProd(f,f)));
   hypre_Matvec(-1.0, A, u, 1.0, f);
   hypre_sprintf(file_name, "%s.res", GlobalsOutFileName);
   hypre_WriteVec(file_name, f);

   hypre_printf("res_norm = %e\n", sqrt(hypre_InnerProd(f,f)));

   hypre_sprintf(file_name, "%s.A", GlobalsOutFileName);
   hypre_WriteYSMP(file_name, A);

   hypre_Matvec(1.0, A, u, 0.0, f);
   hypre_sprintf(file_name, "%s.Au", GlobalsOutFileName);
   hypre_WriteVec(file_name, f);
#endif


   HYPRE_AMGFinalize(amg_data);
   FreeGlobals();
   return 0;
}

