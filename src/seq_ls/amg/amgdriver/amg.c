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


   amg_data = HYPRE_AMGInitialize(NULL);

   HYPRE_AMGSetLevMax(SolverAMGLevMax(solver), amg_data);
   HYPRE_AMGSetNCG(SolverAMGNCG(solver), amg_data);
   HYPRE_AMGSetECG(SolverAMGECG(solver), amg_data);
   HYPRE_AMGSetNWT(SolverAMGNWT(solver), amg_data);
   HYPRE_AMGSetEWT(SolverAMGEWT(solver), amg_data);
   HYPRE_AMGSetNSTR(SolverAMGNSTR(solver), amg_data);
   				    
   HYPRE_AMGSetNCyc(SolverAMGNCyc(solver), amg_data);
   HYPRE_AMGSetMU(SolverAMGMU(solver), amg_data);
   HYPRE_AMGSetNTRLX(SolverAMGNTRLX(solver), amg_data);
   HYPRE_AMGSetIPRLX(SolverAMGIPRLX(solver), amg_data);
   				    
   HYPRE_AMGSetLogging(SolverAMGIOutDat(solver), log_file_name, amg_data);

   HYPRE_AMGSetNumUnknowns(ProblemNumUnknowns(problem), amg_data);
   HYPRE_AMGSetNumPoints(ProblemNumPoints(problem), amg_data);
   HYPRE_AMGSetIU(ProblemIU(problem), amg_data);
   HYPRE_AMGSetIP(ProblemIP(problem), amg_data);
   HYPRE_AMGSetIV(ProblemIV(problem), amg_data);
   
   return amg_data;
}

