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

#include "headers.h"


/*--------------------------------------------------------------------------
 * NewSolver
 *--------------------------------------------------------------------------*/

Solver  *NewSolver(file_name)
char     *file_name;
{
   Solver  *solver;

   /* pcg params */
   int      pcg_max_iter;
   int      pcg_two_norm;

   /* gmres params */
   int      gmres_max_krylov;
   int      gmres_max_restarts;

   /* wjacobi params */
   double   wjacobi_weight;
   int      wjacobi_max_iter;

   /* amg setup params */
   int      amg_levmax;
   int      amg_ncg;
   double   amg_ecg;
   int      amg_nwt;
   double   amg_ewt;
   int      amg_nstr;

   /* amg solve params */
   int      amg_ncyc;
   int     *amg_mu;
   int     *amg_ntrlx;
   int     *amg_iprlx;
   int     *amg_ierlx;
   int     *amg_iurlx;

   /* amg output params */
   int      amg_ioutdat;
   int      amg_ioutgrd;
   int      amg_ioutmat;
   int      amg_ioutres;
   int      amg_ioutsol;

   FILE    *fp;
   int      i;


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

   fscanf(fp, "%d", &pcg_max_iter);
   fscanf(fp, "%d", &pcg_two_norm);

   SolverPCGMaxIter(solver) = pcg_max_iter;
   SolverPCGTwoNorm(solver) = pcg_two_norm;

   /*----------------------------------------------------------
    * GMRES input
    *----------------------------------------------------------*/

   fscanf(fp, "%d", &gmres_max_krylov);
   fscanf(fp, "%d", &gmres_max_restarts);

   SolverGMRESMaxKrylov(solver)   = gmres_max_krylov;
   SolverGMRESMaxRestarts(solver) = gmres_max_restarts;

   /*----------------------------------------------------------
    * Weighted Jacobi input
    *----------------------------------------------------------*/

   fscanf(fp, "%le", &wjacobi_weight);
   fscanf(fp, "%d",  &wjacobi_max_iter);

   SolverWJacobiWeight(solver)  = wjacobi_weight;
   SolverWJacobiMaxIter(solver) = wjacobi_max_iter;

   /*----------------------------------------------------------
    * AMG input
    *----------------------------------------------------------*/

   fscanf(fp, "%d",  &amg_levmax);
   fscanf(fp, "%d",  &amg_ncg);
   fscanf(fp, "%le", &amg_ecg);
   fscanf(fp, "%d",  &amg_nwt);
   fscanf(fp, "%le", &amg_ewt);
   fscanf(fp, "%d",  &amg_nstr);
   
   fscanf(fp, "%d", &amg_ncyc);
   amg_mu = ctalloc(int, amg_levmax);
   for (i = 0; i < 10; i++)
      fscanf(fp, "%d", &amg_mu[i]);
   amg_ntrlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &amg_ntrlx[i]);
   amg_iprlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &amg_iprlx[i]);
   amg_ierlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &amg_ierlx[i]);
   amg_iurlx = ctalloc(int, 4);
   for (i = 0; i < 4; i++)
      fscanf(fp, "%d", &amg_iurlx[i]);
   
   fscanf(fp, "%d", &amg_ioutdat);
   fscanf(fp, "%d", &amg_ioutgrd);
   fscanf(fp, "%d", &amg_ioutmat);
   fscanf(fp, "%d", &amg_ioutres);
   fscanf(fp, "%d", &amg_ioutsol);

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
   SolverAMGIERLX(solver)   = amg_ierlx;
   SolverAMGIURLX(solver)   = amg_iurlx;

   SolverAMGIOutDat(solver) = amg_ioutdat;
   SolverAMGIOutGrd(solver) = amg_ioutgrd;
   SolverAMGIOutMat(solver) = amg_ioutmat;
   SolverAMGIOutRes(solver) = amg_ioutres;
   SolverAMGIOutSol(solver) = amg_ioutsol;

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
      tfree(SolverAMGMU(solver));
      tfree(SolverAMGNTRLX(solver));
      tfree(SolverAMGIPRLX(solver));
      tfree(SolverAMGIERLX(solver));
      tfree(SolverAMGIURLX(solver));

      tfree(solver);
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

   int      type;

   double   stop_tolerance;

   /* pcg params */
   int      pcg_max_iter;
   int      pcg_two_norm;

   /* gmres params */
   int      gmres_max_krylov;
   int      gmres_max_restarts;

   /* wjacobi params */
   double   wjacobi_weight;
   int      wjacobi_max_iter;

   /* amg setup params */
   int      amg_levmax;
   int      amg_ncg;
   double   amg_ecg;
   int      amg_nwt;
   double   amg_ewt;
   int      amg_nstr;

   /* amg solve params */
   int      amg_ncyc;
   int     *amg_mu;
   int     *amg_ntrlx;
   int     *amg_iprlx;
   int     *amg_ierlx;
   int     *amg_iurlx;

   /* amg output params */
   int      amg_ioutdat;
   int      amg_ioutgrd;
   int      amg_ioutmat;
   int      amg_ioutres;
   int      amg_ioutsol;

   int      j;


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
   amg_ierlx   = SolverAMGIERLX(solver);
   amg_iurlx   = SolverAMGIURLX(solver);

   amg_ioutdat = SolverAMGIOutDat(solver);
   amg_ioutgrd = SolverAMGIOutGrd(solver);
   amg_ioutmat = SolverAMGIOutMat(solver);
   amg_ioutres = SolverAMGIOutRes(solver);
   amg_ioutsol = SolverAMGIOutSol(solver);

   /*----------------------------------------------------------
    * Open the output file
    *----------------------------------------------------------*/

   fp = fopen(file_name, "a");

   /*----------------------------------------------------------
    * Solver type
    *----------------------------------------------------------*/

   fprintf(fp,"\nSOLVER PARAMETERS:\n\n");
   fprintf(fp, "  Solver Type:  %d - ", type);

   if (type == SOLVER_AMG)
   {
      fprintf(fp, "AMG \n\n");
   }
   else if (type == SOLVER_Jacobi)
   {
      fprintf(fp, "Jacobi \n\n");
   }
   else if (type == SOLVER_AMG_PCG)
   {
      fprintf(fp, "AMG PCG \n\n");
   }
   else if (type == SOLVER_Jacobi_PCG)
   {
      fprintf(fp, "Jacobi PCG \n\n");
   }
   else if (type == SOLVER_AMG_GMRES)
   {
      fprintf(fp, "AMG GMRES \n\n");
   }
   else if (type == SOLVER_Jacobi_GMRES)
   {
      fprintf(fp, "Jacobi GMRES \n\n");
   }

   /*----------------------------------------------------------
    * PCG info
    *----------------------------------------------------------*/

   if (type == SOLVER_AMG_PCG || type == SOLVER_Jacobi_PCG)
   {
       fprintf(fp, "  Preconditioned Conjugate Gradient Parameters:\n");
       fprintf(fp, "    Solver Stop Tolerance:  %e \n", stop_tolerance);
       fprintf(fp, "    Maximum Iterations: %d \n", pcg_max_iter);
       fprintf(fp, "    Two Norm Flag: %d \n\n", pcg_two_norm);
   }

   /*----------------------------------------------------------
    * GMRES info
    *----------------------------------------------------------*/

   if (type == SOLVER_AMG_GMRES || type == SOLVER_Jacobi_GMRES)
   {
       fprintf(fp, "  Generalized Minimum Residual Parameters:\n");
       fprintf(fp, "    Solver Stop Tolerance:  %e \n", stop_tolerance);
       fprintf(fp, "    Maximum Krylov Dimension: %d \n", gmres_max_krylov);
       fprintf(fp, "    Max Number of Restarts: %d \n\n", gmres_max_restarts);
   }

   /*----------------------------------------------------------
    * Jacobi info
    *----------------------------------------------------------*/

   if (type == SOLVER_Jacobi ||
       type == SOLVER_Jacobi_PCG ||
       type == SOLVER_Jacobi_GMRES)
   {
      fprintf(fp, "  Jacobi Parameters:\n");
      fprintf(fp, "    Weight for Relaxation: %f \n", wjacobi_weight);
      fprintf(fp, "    Maximum Iterations: %d \n\n", wjacobi_max_iter);
   }

   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/

   if (type == SOLVER_AMG ||
       type == SOLVER_AMG_PCG ||
       type == SOLVER_AMG_GMRES)
   {
      fprintf(fp, "  AMG Parameters:\n");
      fprintf(fp, "    Maximum number of levels:            %d \n",
	      amg_levmax);
      fprintf(fp, "    Coarsening controls (ncg, ecg):      %d   %f \n",
	      amg_ncg, amg_ecg);
      fprintf(fp, "    Interpolation controls (nwt, ewt):   %d   %f \n",
	      amg_nwt, amg_ewt);
      fprintf(fp, "    Strong connection definition (nstr): %d \n", amg_nstr);
      fprintf(fp, "    Number and type of cycles (ncyc):    %d \n", amg_ncyc); 
      fprintf(fp, "    W-cycling parameter (mu): ");
      for (j = 0; j < 10; j++)
	 fprintf(fp, "%d ", amg_mu[j]);
      fprintf(fp, "\n");
      fprintf(fp, "    Relaxation Parameters:\n");
      fprintf(fp, "       ntr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_ntrlx[0], amg_ntrlx[1], amg_ntrlx[2], amg_ntrlx[3]);
      fprintf(fp, "       ipr(f,d,u,c): %d  %d  %d  %d \n",
	      amg_iprlx[0], amg_iprlx[1], amg_iprlx[2], amg_iprlx[3]);
      fprintf(fp, "       ier(f,d,u,c): %d  %d  %d  %d \n",
	      amg_ierlx[0], amg_ierlx[1], amg_ierlx[2], amg_ierlx[3]);
      fprintf(fp, "       iur(f,d,u,c): %d  %d  %d  %d \n",
	      amg_iurlx[0], amg_iurlx[1], amg_iurlx[2], amg_iurlx[3]);
 
      fprintf(fp, "    Output flag (ioutdat): %d \n", amg_ioutdat);
      fprintf(fp, "    ioutgrd: (unused) %d \n", amg_ioutgrd);
      fprintf(fp, "    Matrix output flag (ioutmat): %d \n", amg_ioutmat); 
      fprintf(fp, "    Residual report (ioutres): %d \n", amg_ioutres); 
      fprintf(fp, "    Graphical solution flag (ioutsol): %d \n", amg_ioutsol);
   }

   /*----------------------------------------------------------
    * Close the output file
    *----------------------------------------------------------*/

   fclose(fp);

   return;
}

