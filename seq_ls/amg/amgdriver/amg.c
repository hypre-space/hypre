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
 * AMG functions
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * NewAMGData
 *--------------------------------------------------------------------------*/

void     *NewAMGData(problem, solver, log_file_name)
Problem  *problem;
Solver   *solver;
char     *log_file_name;
{
   void  *amg_data;


   amg_data = amg_Initialize(NULL);

   amg_SetLevMax(SolverAMGLevMax(solver), amg_data);
   amg_SetNCG(SolverAMGNCG(solver), amg_data);
   amg_SetECG(SolverAMGECG(solver), amg_data);
   amg_SetNWT(SolverAMGNWT(solver), amg_data);
   amg_SetEWT(SolverAMGEWT(solver), amg_data);
   amg_SetNSTR(SolverAMGNSTR(solver), amg_data);
   				    
   amg_SetNCyc(SolverAMGNCyc(solver), amg_data);
   amg_SetMU(SolverAMGMU(solver), amg_data);
   amg_SetNTRLX(SolverAMGNTRLX(solver), amg_data);
   amg_SetIPRLX(SolverAMGIPRLX(solver), amg_data);
   amg_SetIERLX(SolverAMGIERLX(solver), amg_data);
   amg_SetIURLX(SolverAMGIURLX(solver), amg_data);
   				    
   amg_SetIOutDat(SolverAMGIOutDat(solver), amg_data);
   amg_SetIOutGrd(SolverAMGIOutGrd(solver), amg_data);
   amg_SetIOutMat(SolverAMGIOutMat(solver), amg_data);
   amg_SetIOutRes(SolverAMGIOutRes(solver), amg_data);
   amg_SetIOutSol(SolverAMGIOutSol(solver), amg_data);

   amg_SetLogFileName(log_file_name, amg_data);

   amg_SetNumUnknowns(ProblemNumUnknowns(problem), amg_data);
   amg_SetNumPoints(ProblemNumPoints(problem), amg_data);
   amg_SetIU(ProblemIU(problem), amg_data);
   amg_SetIP(ProblemIP(problem), amg_data);
   amg_SetIV(ProblemIV(problem), amg_data);
   amg_SetXP(ProblemXP(problem), amg_data);
   amg_SetYP(ProblemYP(problem), amg_data);
   amg_SetZP(ProblemZP(problem), amg_data);
   
   return amg_data;
}

