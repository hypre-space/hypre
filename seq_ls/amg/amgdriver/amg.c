/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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

