/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Constructors and destructors for solver structure.
 *
 *****************************************************************************/

#include "amg.h"


/*--------------------------------------------------------------------------
 * NewSolver
 *--------------------------------------------------------------------------*/

Solver  *NewSolver(file_name)
char     *file_name;
{
   Solver  *solver;

   FILE    *fp;


   /*----------------------------------------------------------
    * Allocate the solver structure
    *----------------------------------------------------------*/

   solver = ctalloc(Solver, 1);

   /*----------------------------------------------------------
    * Open the solver file
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   /*----------------------------------------------------------
    * Generic solver input
    *----------------------------------------------------------*/

   fscanf(fp, "%d", &SolverType(solver));
   fscanf(fp, "%le", &SolverStopTolerance(solver));

   /*----------------------------------------------------------
    * PCG input
    *----------------------------------------------------------*/

   SolverPCGData(solver) = ReadPCGParams(fp);

   /*----------------------------------------------------------
    * Weighted Jacobi input
    *----------------------------------------------------------*/

   SolverWJacobiData(solver) = ReadWJacobiParams(fp);

   /*----------------------------------------------------------
    * AMG input
    *----------------------------------------------------------*/

   SolverAMGS01Data(solver) = ReadAMGS01Params(fp);

   /*----------------------------------------------------------
    * Close the solver file and return
    *----------------------------------------------------------*/

   fclose(fp);

   return solver;
}

/*--------------------------------------------------------------------------
 * FreeSolver
 *--------------------------------------------------------------------------*/

void     FreeSolver(solver)
Solver  *solver;
{
   if (solver)
   {
      FreePCGData(SolverPCGData(solver));
      FreeWJacobiData(SolverPCGData(solver));
      FreeAMGS01Data(SolverAMGS01Data(solver));
      tfree(solver);
   }
}

