/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"


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
    * Debugging prints
    *-------------------------------------------------------*/
#if 0
   sprintf(file_name, "%s.ysmp", GlobalsOutFileName);
   WriteYSMP(file_name, ProblemA(problem));

   sprintf(file_name, "%s.initu", GlobalsOutFileName);
   WriteVec(file_name, ProblemU(problem));

   sprintf(file_name, "%s.rhs", GlobalsOutFileName);
   WriteVec(file_name, ProblemF(problem));
#endif

   /*-------------------------------------------------------
    * Write initial logging info
    *-------------------------------------------------------*/

   fp = fopen(GlobalsLogFileName, "w");
   fclose(fp);

   WriteProblem(GlobalsLogFileName, problem);
   WriteSolver(GlobalsLogFileName, solver);

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

   /* call AMG */
   if (SolverType(solver) == 0)
   {
      amg_Setup(A, amg_data);

      amg_Solve(u, f, stop_tolerance, amg_data);
   }

   /* call AMGCG */
   else if (SolverType(solver) == 1)
   {
      amg_Setup(A, amg_data);
      PCGSetup(A, amg_Solve, amg_data, pcg_data);

      PCG(u, f, stop_tolerance, pcg_data);
   }

   /* call JCG */
   else if (SolverType(solver) == 2)
   {
      WJacobiSetup(A, wjacobi_data);
      PCGSetup(A, WJacobi, wjacobi_data, pcg_data);

      PCG(u, f, stop_tolerance, pcg_data);
   }

   /*-------------------------------------------------------
    * Debugging prints
    *-------------------------------------------------------*/
#if 0
   sprintf(file_name, "%s.lastu", GlobalsOutFileName);
   WriteVec(file_name, ProblemU(problem));

   Matvec(-1.0, ProblemA(problem), ProblemU(problem), 1.0, ProblemF(problem));
   sprintf(file_name, "%s.res", GlobalsOutFileName);
   WriteVec(file_name, ProblemF(problem));
#endif

   return 0;
}

