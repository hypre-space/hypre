/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "amg.h"


/*--------------------------------------------------------------------------
 * Main driver for AMG
 *--------------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   char    *run_name;
   char     in_file_name[255];
   char     out_file_name[255];

   char     file_name[255];
   FILE    *fp;

   Problem *problem;

   Matrix  *A;
   Vector  *f;
   Vector  *u;

   int     *ifc;
   int      isw;

   int      ndimu;
   int      ndimp;
   int      ndima;
   int      ndimb;


   /*-------------------------------------------------------
    * Check that the number of command args is correct
    *-------------------------------------------------------*/

   if (argc < 2)
   {
      fprintf(stderr, "Usage:  amg <run name>\n");
      exit(1);
   }

   /*-------------------------------------------------------
    * Set up the file names
    *-------------------------------------------------------*/

   run_name = argv[1];
   sprintf(in_file_name,  "%s.in", run_name);
   sprintf(out_file_name, "%s.out", run_name);

   /*-------------------------------------------------------
    * Set up the problem
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.problem.strp", in_file_name);
   problem = NewProblem(file_name);

   A = ProblemA(problem);
   f = ProblemF(problem);
   u = ProblemU(problem);

#if 0
   /*-------------------------------------------------------
    * Debugging prints
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.ysmp", out_file_name);
   WriteYSMP(file_name, A);

   sprintf(file_name, "%s.rhs", out_file_name);
   WriteVec(file_name, f);

   sprintf(file_name, "%s.initu", out_file_name);
   WriteVec(file_name, u);

#endif

   /*-------------------------------------------------------
    * Call AMGS01
    *-------------------------------------------------------*/

   /* Set some array dimension variables (not yet used)*/
   ndimu = NDIMU(ProblemNumVariables(problem));
   ndimp = NDIMP(ProblemNumPoints(problem));
   ndima = NDIMU(MatrixIA(A)[ProblemNumVariables(problem)]-1);
   ndimb = NDIMB(MatrixIA(A)[ProblemNumVariables(problem)]-1);

   ifc = ctalloc(int, ndimu);
   isw = 1;

   sprintf(file_name, "%s.log", out_file_name);

   amgs01(u, f, A, problem, ifc, isw, file_name);

   return 0;
}

