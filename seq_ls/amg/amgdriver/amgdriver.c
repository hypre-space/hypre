/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "timing.h"


#ifdef AMG_MALLOC_DEBUG
/* malloc debug stuff */
char amg_malloclog[256];
#endif

/*--------------------------------------------------------------------------
 * Main driver for AMG
 *--------------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   char    *run_name;

   char     file_name[255];
   FILE    *fp;

   Problem *problem;
   Solver  *solver;

   Matrix  *A;
   Vector  *u;
   Vector  *f;
   double   stop_tolerance;
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
      fprintf(stderr, "Usage:  amg <run name>\n");
      exit(1);
   }

   /*-------------------------------------------------------
    * Set up debugging tools
    *-------------------------------------------------------*/

#ifdef AMG_MALLOC_DEBUG
   /* malloc debug stuff */
   malloc_logpath = amg_malloclog;
   sprintf(malloc_logpath, "malloc.log");
#endif

   /*-------------------------------------------------------
    * Set up globals 
    *-------------------------------------------------------*/

   run_name = argv[1];
   NewGlobals(run_name);

   /*-------------------------------------------------------
    * Set up the problem
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.problem.strp", GlobalsInFileName);
   problem = NewProblem(file_name);

   /*-------------------------------------------------------
    * Set up the solver
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.solver.strp", GlobalsInFileName);
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

   amg_Clock_init();
   start_ticks = amg_Clock(); 
   start_cpu = amg_CPUClock();

   if (SolverType(solver) == SOLVER_AMG)
   {
      int setup_err_flag;
      int solve_err_flag;

      setup_err_flag = amg_Setup(A, amg_data);
      if (setup_err_flag != 0) 
      {
         printf("setup error = %d\n",setup_err_flag);
         if (setup_err_flag > 0)
         {
            return 1;
         }
         printf("Setup Error Warning. Execution Continues.\n");
      }

      setup_ticks = amg_Clock() - start_ticks;
      setup_cpu =   amg_CPUClock() - start_cpu;

      solve_err_flag = amg_Solve(u, f, stop_tolerance, amg_data);
      if (solve_err_flag != 0) printf("solve error = %d\n",solve_err_flag);

      solve_ticks = amg_Clock() - (start_ticks + setup_ticks);
      solve_cpu =   amg_CPUClock() - (start_cpu + setup_cpu);
      time_ticks =  amg_Clock() - start_ticks;
      cpu_ticks =   amg_CPUClock() - start_cpu;
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
      amg_Setup(A, amg_data);
      PCGSetup(A, amg_Solve, amg_data, pcg_data);

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
      amg_Setup(A, amg_data);
      GMRESSetup(A, amg_Solve, amg_data, gmres_data);

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
      long AMG_CPU_TICKS_PER_SEC;

      AMG_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);

      fp = fopen(GlobalsLogFileName, "a");
 
      fprintf(fp,"\nTIMING INFORMATION\n");
      fprintf(fp,"\nSetup Time:\n");
      fprintf(fp, " wall clock time = %f seconds\n", 
                         ((double) setup_ticks)/AMG_TICKS_PER_SEC);
      fprintf(fp," CPU clock time  = %f seconds\n", 
                         ((double) setup_cpu)/AMG_CPU_TICKS_PER_SEC);

      fprintf(fp,"\nSolve Time:\n");
      fprintf(fp, " wall clock time = %f seconds\n", 
                         ((double) solve_ticks)/AMG_TICKS_PER_SEC);
      fprintf(fp," CPU clock time  = %f seconds\n", 
                         ((double) solve_cpu)/AMG_CPU_TICKS_PER_SEC);
 
      fprintf(fp,"\nOverall Time:\n");
      fprintf(fp, " wall clock time = %f seconds\n", 
                         ((double) time_ticks)/AMG_TICKS_PER_SEC);
      fprintf(fp," CPU clock time  = %f seconds\n", 
                         ((double) cpu_ticks)/AMG_CPU_TICKS_PER_SEC);
  
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
   sprintf(file_name, "%s.lastu", GlobalsOutFileName);
   WriteVec(file_name, u);
#endif
#if 0
   printf("soln norm = %e\n", sqrt(InnerProd(u,u)));

   printf("rhs norm = %e\n", sqrt(InnerProd(f,f)));
   Matvec(-1.0, A, u, 1.0, f);
   sprintf(file_name, "%s.res", GlobalsOutFileName);
   WriteVec(file_name, f);

   printf("res_norm = %e\n", sqrt(InnerProd(f,f)));

   sprintf(file_name, "%s.A", GlobalsOutFileName);
   WriteYSMP(file_name, A);

   Matvec(1.0, A, u, 0.0, f);
   sprintf(file_name, "%s.Au", GlobalsOutFileName);
   WriteVec(file_name, f);
#endif

   return 0;
}

