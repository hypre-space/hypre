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
 * Constructors and destructors for solver structure.
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * NewSolver
 *--------------------------------------------------------------------------*/

Solver  *NewSolver(file_name)
char     *file_name;
{
   Solver  *solver;

   /* pcg params */
   HYPRE_Int      pcg_max_iter;
   HYPRE_Int      pcg_two_norm;

   /* gmres params */
   HYPRE_Int      gmres_max_krylov;
   HYPRE_Int      gmres_max_restarts;

   /* wjacobi params */
   double   wjacobi_weight;
   HYPRE_Int      wjacobi_max_iter;

   /* amg setup params */
   HYPRE_Int      amg_levmax;
   HYPRE_Int      amg_ncg;
   double   amg_ecg;
   HYPRE_Int      amg_nwt;
   double   amg_ewt;
   HYPRE_Int      amg_nstr;

   /* amg solve params */
   HYPRE_Int      amg_ncyc;
   HYPRE_Int     *amg_mu;
   HYPRE_Int     *amg_ntrlx;
   HYPRE_Int     *amg_iprlx;


   /* amg output params */
   HYPRE_Int      amg_ioutdat;

   FILE    *fp;
   HYPRE_Int      i;


   /*----------------------------------------------------------
    * Allocate the solver structure
    *----------------------------------------------------------*/

   solver = hypre_CTAlloc(Solver, 1);

   /*----------------------------------------------------------
    * Open the solver file
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   /*----------------------------------------------------------
    * Generic solver input
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d", &SolverType(solver));
   hypre_fscanf(fp, "%le", &SolverStopTolerance(solver));

   /*----------------------------------------------------------
    * PCG input
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d", &pcg_max_iter);
   hypre_fscanf(fp, "%d", &pcg_two_norm);

   SolverPCGMaxIter(solver) = pcg_max_iter;
   SolverPCGTwoNorm(solver) = pcg_two_norm;

   /*----------------------------------------------------------
    * GMRES input
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d", &gmres_max_krylov);
   hypre_fscanf(fp, "%d", &gmres_max_restarts);

   SolverGMRESMaxKrylov(solver)   = gmres_max_krylov;
   SolverGMRESMaxRestarts(solver) = gmres_max_restarts;

   /*----------------------------------------------------------
    * Weighted Jacobi input
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%le", &wjacobi_weight);
   hypre_fscanf(fp, "%d",  &wjacobi_max_iter);

   SolverWJacobiWeight(solver)  = wjacobi_weight;
   SolverWJacobiMaxIter(solver) = wjacobi_max_iter;

   /*----------------------------------------------------------
    * AMG input
    *----------------------------------------------------------*/

   hypre_fscanf(fp, "%d",  &amg_levmax);
   hypre_fscanf(fp, "%d",  &amg_ncg);
   hypre_fscanf(fp, "%le", &amg_ecg);
   hypre_fscanf(fp, "%d",  &amg_nwt);
   hypre_fscanf(fp, "%le", &amg_ewt);
   hypre_fscanf(fp, "%d",  &amg_nstr);
   
   hypre_fscanf(fp, "%d", &amg_ncyc);
   amg_mu = hypre_CTAlloc(HYPRE_Int, amg_levmax);
   for (i = 0; i < amg_levmax; i++)
      hypre_fscanf(fp, "%d", &amg_mu[i]);
   amg_ntrlx = hypre_CTAlloc(HYPRE_Int, 4);
   for (i = 0; i < 4; i++)
      hypre_fscanf(fp, "%d", &amg_ntrlx[i]);
   amg_iprlx = hypre_CTAlloc(HYPRE_Int, 4);
   for (i = 0; i < 4; i++)
      hypre_fscanf(fp, "%d", &amg_iprlx[i]);
   hypre_fscanf(fp, "%d", &amg_ioutdat);

   /*----------------------------------------------------------
    * Set the solver structure
    *----------------------------------------------------------*/

   SolverAMGLevMax(solver)  = amg_levmax;
   SolverAMGNCG(solver)     = amg_ncg;
   SolverAMGECG(solver)     = amg_ecg;
   SolverAMGNWT(solver)     = amg_nwt;
   SolverAMGEWT(solver)     = amg_ewt;
   SolverAMGNSTR(solver)    = amg_nstr;

   SolverAMGNCyc(solver)    = amg_ncyc;
   SolverAMGMU(solver)      = amg_mu;
   SolverAMGNTRLX(solver)   = amg_ntrlx;
   SolverAMGIPRLX(solver)   = amg_iprlx;

   SolverAMGIOutDat(solver) = amg_ioutdat;

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
      hypre_TFree(SolverAMGMU(solver));
      hypre_TFree(SolverAMGNTRLX(solver));
      hypre_TFree(SolverAMGIPRLX(solver));

      hypre_TFree(solver);
   }
}

/*--------------------------------------------------------------------------
 * WriteSolver
 *--------------------------------------------------------------------------*/

void     WriteSolver(file_name, solver)
char    *file_name;
Solver  *solver;

{
   FILE    *fp;

   HYPRE_Int      type;

   double   stop_tolerance;

   /* pcg params */
   HYPRE_Int      pcg_max_iter;
   HYPRE_Int      pcg_two_norm;

   /* gmres params */
   HYPRE_Int      gmres_max_krylov;
   HYPRE_Int      gmres_max_restarts;

   /* wjacobi params */
   double   wjacobi_weight;
   HYPRE_Int      wjacobi_max_iter;

   /* amg setup params */
   HYPRE_Int      amg_levmax;
   HYPRE_Int      amg_ncg;
   double   amg_ecg;
   HYPRE_Int      amg_nwt;
   double   amg_ewt;
   HYPRE_Int      amg_nstr;

   /* amg solve params */
   HYPRE_Int      amg_ncyc;
   HYPRE_Int     *amg_mu;
   HYPRE_Int     *amg_ntrlx;
   HYPRE_Int     *amg_iprlx;

   /* amg output params */
   HYPRE_Int      amg_ioutdat;

   HYPRE_Int      j;


   /*----------------------------------------------------------
    * Get the solver data
    *----------------------------------------------------------*/

   type = SolverType(solver);
   stop_tolerance = SolverStopTolerance(solver);

   pcg_max_iter = SolverPCGMaxIter(solver);
   pcg_two_norm = SolverPCGTwoNorm(solver);

   gmres_max_krylov   = SolverGMRESMaxKrylov(solver);
   gmres_max_restarts = SolverGMRESMaxRestarts(solver);

   wjacobi_weight   = SolverWJacobiWeight(solver);
   wjacobi_max_iter = SolverWJacobiMaxIter(solver);

   amg_levmax  = SolverAMGLevMax(solver);
   amg_ncg     = SolverAMGNCG(solver);
   amg_ecg     = SolverAMGECG(solver);
   amg_nwt     = SolverAMGNWT(solver);
   amg_ewt     = SolverAMGEWT(solver);
   amg_nstr    = SolverAMGNSTR(solver);

   amg_ncyc    = SolverAMGNCyc(solver);
   amg_mu      = SolverAMGMU(solver);
   amg_ntrlx   = SolverAMGNTRLX(solver);
   amg_iprlx   = SolverAMGIPRLX(solver);

   amg_ioutdat = SolverAMGIOutDat(solver);

   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/

   fp = fopen(file_name, "a");

   /*----------------------------------------------------------
    * Solver type
    *----------------------------------------------------------*/

   hypre_fprintf(fp,"\nSOLVER PARAMETERS:\n\n");
   hypre_fprintf(fp, "  Solver Type:  %d - ", type);

   if (type == SOLVER_AMG)
   {
      hypre_fprintf(fp, "AMG \n\n");
   }
   else if (type == SOLVER_Jacobi)
   {
      hypre_fprintf(fp, "Jacobi \n\n");
   }
   else if (type == SOLVER_AMG_PCG)
   {
      hypre_fprintf(fp, "AMG PCG \n\n");
   }
   else if (type == SOLVER_Jacobi_PCG)
   {
      hypre_fprintf(fp, "Jacobi PCG \n\n");
   }
   else if (type == SOLVER_AMG_GMRES)
   {
      hypre_fprintf(fp, "AMG GMRES \n\n");
   }
   else if (type == SOLVER_Jacobi_GMRES)
   {
      hypre_fprintf(fp, "Jacobi GMRES \n\n");
   }

   /*----------------------------------------------------------
    * PCG info
    *----------------------------------------------------------*/

   if (type == SOLVER_AMG_PCG || type == SOLVER_Jacobi_PCG)
   {
       hypre_fprintf(fp, "  Preconditioned Conjugate Gradient Parameters:\n");
       hypre_fprintf(fp, "    Solver Stop Tolerance:  %e \n", stop_tolerance);
       hypre_fprintf(fp, "    Maximum Iterations: %d \n", pcg_max_iter);
       hypre_fprintf(fp, "    Two Norm Flag: %d \n\n", pcg_two_norm);
   }

   /*----------------------------------------------------------
    * GMRES info
    *----------------------------------------------------------*/

   if (type == SOLVER_AMG_GMRES || type == SOLVER_Jacobi_GMRES)
   {
       hypre_fprintf(fp, "  Generalized Minimum Residual Parameters:\n");
       hypre_fprintf(fp, "    Solver Stop Tolerance:  %e \n", stop_tolerance);
       hypre_fprintf(fp, "    Maximum Krylov Dimension: %d \n", gmres_max_krylov);
       hypre_fprintf(fp, "    Max Number of Restarts: %d \n\n", gmres_max_restarts);
   }

   /*----------------------------------------------------------
    * Jacobi info
    *----------------------------------------------------------*/

   if (type == SOLVER_Jacobi ||
       type == SOLVER_Jacobi_PCG ||
       type == SOLVER_Jacobi_GMRES)
   {
      hypre_fprintf(fp, "  Jacobi Parameters:\n");
      hypre_fprintf(fp, "    Weight for Relaxation: %f \n", wjacobi_weight);
      hypre_fprintf(fp, "    Maximum Iterations: %d \n\n", wjacobi_max_iter);
   }

   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/

   if (type == SOLVER_AMG ||
       type == SOLVER_AMG_PCG ||
       type == SOLVER_AMG_GMRES)
   {
      hypre_fprintf(fp, "  AMG Parameters:\n");
      hypre_fprintf(fp, "    Maximum number of levels:            %d \n",
	      amg_levmax);
      hypre_fprintf(fp, "    Coarsening controls (ncg, ecg):      %d   %f \n",
	      amg_ncg, amg_ecg);
      hypre_fprintf(fp, "    Interpolation controls (nwt, ewt):   %d   %f \n",
	      amg_nwt, amg_ewt);
      hypre_fprintf(fp, "    Strong connection definition (nstr): %d \n", amg_nstr);
      hypre_fprintf(fp, "    Number and type of cycles (ncyc):    %d \n", amg_ncyc);
      hypre_fprintf(fp, "    Stopping Tolerance                   %e \n",
                   stop_tolerance); 
      hypre_fprintf(fp, "    W-cycling parameter (mu): ");
      for (j = 0; j < amg_levmax; j++)
	 hypre_fprintf(fp, "%d ", amg_mu[j]);
      hypre_fprintf(fp, "\n");
      hypre_fprintf(fp, "    Relaxation Parameters:\n");
      hypre_fprintf(fp, "       ntr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_ntrlx[0], amg_ntrlx[1], amg_ntrlx[2], amg_ntrlx[3]);
      hypre_fprintf(fp, "       ipr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_iprlx[0], amg_iprlx[1], amg_iprlx[2], amg_iprlx[3]);
 
      hypre_fprintf(fp, "    Output flag (ioutdat): %d \n", amg_ioutdat);

   }

   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

   fclose(fp);

   return;
}

