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
 * Header info for Solver data structures
 *
 *****************************************************************************/

#ifndef _SOLVER_HEADER
#define _SOLVER_HEADER


/*--------------------------------------------------------------------------
 * Solver types
 *--------------------------------------------------------------------------*/

#define SOLVER_AMG            0
#define SOLVER_Jacobi         1

#define SOLVER_AMG_PCG       10
#define SOLVER_Jacobi_PCG    11

#define SOLVER_AMG_GMRES     20
#define SOLVER_Jacobi_GMRES  21

/*--------------------------------------------------------------------------
 * Solver
 *--------------------------------------------------------------------------*/

typedef struct
{
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

   /* amg output params */
   int      amg_ioutdat;

} Solver;

/*--------------------------------------------------------------------------
 * Accessor functions for the Solver structure
 *--------------------------------------------------------------------------*/

#define SolverType(solver)             ((solver) -> type)

#define SolverStopTolerance(solver)    ((solver) -> stop_tolerance)

/* pcg params */
#define SolverPCGMaxIter(solver)       ((solver) -> pcg_max_iter)
#define SolverPCGTwoNorm(solver)       ((solver) -> pcg_two_norm)

/* gmres params */
#define SolverGMRESMaxKrylov(solver)   ((solver) -> gmres_max_krylov)
#define SolverGMRESMaxRestarts(solver) ((solver) -> gmres_max_restarts)

/* wjacobi params */
#define SolverWJacobiWeight(solver)    ((solver) -> wjacobi_weight)
#define SolverWJacobiMaxIter(solver)   ((solver) -> wjacobi_max_iter)

/* amg setup params */
#define SolverAMGLevMax(solver)      ((solver) -> amg_levmax)
#define SolverAMGNCG(solver)         ((solver) -> amg_ncg)
#define SolverAMGECG(solver)         ((solver) -> amg_ecg)
#define SolverAMGNWT(solver)         ((solver) -> amg_nwt)
#define SolverAMGEWT(solver)         ((solver) -> amg_ewt)
#define SolverAMGNSTR(solver)        ((solver) -> amg_nstr)
		  
/* amg solve params */
#define SolverAMGNCyc(solver)        ((solver) -> amg_ncyc)
#define SolverAMGMU(solver)          ((solver) -> amg_mu)
#define SolverAMGNTRLX(solver)       ((solver) -> amg_ntrlx)
#define SolverAMGIPRLX(solver)       ((solver) -> amg_iprlx)
		  
/* amg output params */
#define SolverAMGIOutDat(solver)     ((solver) -> amg_ioutdat)


#endif
