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
   HYPRE_Int      type;

   HYPRE_Real   stop_tolerance;

   /* pcg params */
   HYPRE_Int      pcg_max_iter;
   HYPRE_Int      pcg_two_norm;

   /* gmres params */
   HYPRE_Int      gmres_max_krylov;
   HYPRE_Int      gmres_max_restarts;

   /* wjacobi params */
   HYPRE_Real   wjacobi_weight;
   HYPRE_Int      wjacobi_max_iter;

   /* amg setup params */
   HYPRE_Int      amg_levmax;
   HYPRE_Int      amg_ncg;
   HYPRE_Real   amg_ecg;
   HYPRE_Int      amg_nwt;
   HYPRE_Real   amg_ewt;
   HYPRE_Int      amg_nstr;

   /* amg solve params */
   HYPRE_Int      amg_ncyc;
   HYPRE_Int     *amg_mu;
   HYPRE_Int     *amg_ntrlx;
   HYPRE_Int     *amg_iprlx;

   /* amg output params */
   HYPRE_Int      amg_ioutdat;

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
