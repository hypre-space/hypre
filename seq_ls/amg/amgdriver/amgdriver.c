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


   /*-------------------------------------------------------
    * Check that the number of command args is correct
    *-------------------------------------------------------*/

   if (argc < 2)
   {
      fprintf(stderr, "Usage:  amg <run name>\n");
      exit(1);
   }

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
   time_ticks = -amg_Clock();
   cpu_ticks = -amg_CPUClock();

   if (SolverType(solver) == SOLVER_AMG)
   {
      int setup_err_flag;
      int solve_err_flag;

      setup_err_flag = amg_Setup(A, amg_data);
      if (setup_err_flag != 0) printf("setup error = %d\n",setup_err_flag);

      solve_err_flag = amg_Solve(u, f, stop_tolerance, amg_data);
      if (solve_err_flag != 0) printf("solve error = %d\n",solve_err_flag);
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

   time_ticks += amg_Clock();
   cpu_ticks  += amg_CPUClock();
   PrintTiming(time_ticks,cpu_ticks);

   /*-------------------------------------------------------
    * Debugging prints
    *-------------------------------------------------------*/
#if 0
   sprintf(file_name, "%s.lastu", GlobalsOutFileName);
   WriteVec(file_name, u);

   Matvec(-1.0, A, u, 1.0, f);
   sprintf(file_name, "%s.res", GlobalsOutFileName);
   WriteVec(file_name, f);

   printf("r_norm = %e\n", sqrt(InnerProd(f,f)));
#endif

   return 0;
}

