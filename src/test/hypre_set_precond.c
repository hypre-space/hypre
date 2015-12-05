/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.6 $
 ***********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * Routines to set up preconditioners for use in test codes.
 * June 16, 2005
 *--------------------------------------------------------------------------*/
#include "hypre_test.h"


HYPRE_Int hypre_set_precond(HYPRE_Int matrix_id, HYPRE_Int solver_id, HYPRE_Int precond_id, void *solver,
                      void *precond)
{
  hypre_set_precond_params(precond_id, precond);


/************************************************************************
 * PARCSR MATRIX
 ***********************************************************************/
   if (matrix_id == HYPRE_PARCSR)
      {

/************************************************************************
 *     PCG Solver
 ***********************************************************************/
       if (solver_id == HYPRE_PCG)
          {
           if (precond_id == HYPRE_BOOMERAMG)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_EUCLID)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_PARASAILS)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SCHWARZ)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 *     GMRES Solver
 ***********************************************************************/
       if (solver_id == HYPRE_GMRES)
          {
           if (precond_id == HYPRE_BOOMERAMG)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_EUCLID)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_PARASAILS)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRParaSailsSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_PILUT)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SCHWARZ)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SchwarzSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 *     BiCGSTAB Solver
 ***********************************************************************/
       if (solver_id == HYPRE_BICGSTAB)
          {
           if (precond_id == HYPRE_BOOMERAMG)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_EUCLID)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_PILUT)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRPilutSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 *     CGNR Solver
 ***********************************************************************/
       if (solver_id == HYPRE_CGNR)
          {
           if (precond_id == HYPRE_BOOMERAMG)
              {
               HYPRE_CGNRSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolveT,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_CGNRSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
          }

      }


/************************************************************************
 * SSTRUCT MATRIX
 ***********************************************************************/
   if (matrix_id == HYPRE_SSTRUCT)
      {

/************************************************************************
 *     PCG Solver
 ***********************************************************************/
       if (solver_id == HYPRE_PCG)
          {
           if (precond_id == HYPRE_SPLIT)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SYSPFMG)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 * GMRES Solver
 ***********************************************************************/
       if (solver_id == HYPRE_GMRES)
          {
           if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SPLIT)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 * BiCGSTAB Solver
 ***********************************************************************/
       if (solver_id == HYPRE_BICGSTAB)
          {
           if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SPLIT)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 * CGNR Solver
 ***********************************************************************/
       if (solver_id == HYPRE_CGNR)
          {
           if (precond_id == HYPRE_BOOMERAMG)
              {
               HYPRE_CGNRSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolveT,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_CGNRSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
          }

      }

/************************************************************************
 * STRUCT MATRIX
 ***********************************************************************/
   if (matrix_id == HYPRE_STRUCT)
      {

/************************************************************************
 *     PCG Solver
 ***********************************************************************/
       if (solver_id == HYPRE_PCG)
          {
           if (precond_id == HYPRE_SMG)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_PFMG)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SPARSEMSG)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_JACOBI)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 *     HYBRID Solver
 ***********************************************************************/
       if (solver_id == HYPRE_HYBRID)
          {
           if (precond_id == HYPRE_SMG)
              {
               HYPRE_StructHybridSetPrecond( (HYPRE_StructSolver) solver,
                                    (HYPRE_PtrToStructSolverFcn) HYPRE_StructSMGSolve,
                                    (HYPRE_PtrToStructSolverFcn) HYPRE_StructSMGSetup,
                                    (HYPRE_StructSolver) precond);
              }
           else if (precond_id == HYPRE_PFMG)
              {
               HYPRE_StructHybridSetPrecond( (HYPRE_StructSolver) solver,
                                    (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSolve,
                                    (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSetup,
                                    (HYPRE_StructSolver) precond);
              }
           else if (precond_id == HYPRE_SPARSEMSG)
              {
               HYPRE_StructHybridSetPrecond( (HYPRE_StructSolver) solver,
                                    (HYPRE_PtrToStructSolverFcn) HYPRE_StructSparseMSGSolve,
                                    (HYPRE_PtrToStructSolverFcn) HYPRE_StructSparseMSGSetup,
                                    (HYPRE_StructSolver) precond);
              }
          }

/************************************************************************
 *     GMRES Solver
 ***********************************************************************/
       if (solver_id == HYPRE_GMRES)
          {
           if (precond_id == HYPRE_SMG)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_PFMG)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SPARSEMSG)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_JACOBI)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
          }

/************************************************************************
 *     BICGSTAB Solver
 ***********************************************************************/
       if (solver_id == HYPRE_BICGSTAB)
          {
           if (precond_id == HYPRE_SMG)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_PFMG)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_SPARSEMSG)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructSparseMSGSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_JACOBI)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructJacobiSetup,
                                    (HYPRE_Solver) precond);
              }
           else if (precond_id == HYPRE_DIAGSCALE)
              {
               HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructDiagScaleSetup,
                                    (HYPRE_Solver) precond);
              }
          }
      }
}


HYPRE_Int hypre_set_precond_params(HYPRE_Int precond_id, void *precond)
{
    HYPRE_Int i;
    HYPRE_Int ierr;

/* use BoomerAMG preconditioner */
    if (precond_id == HYPRE_BOOMERAMG)
       {
        HYPRE_BoomerAMGCreate(precond); 
        HYPRE_BoomerAMGSetInterpType(precond, interp_type);
        HYPRE_BoomerAMGSetNumSamples(precond, gsmg_samples);
        HYPRE_BoomerAMGSetTol(precond, pc_tol);
        HYPRE_BoomerAMGSetCoarsenType(precond, (hybrid*coarsen_type));
        HYPRE_BoomerAMGSetMeasureType(precond, measure_type);
        HYPRE_BoomerAMGSetStrongThreshold(precond, strong_threshold);
        HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor);
        HYPRE_BoomerAMGSetPrintLevel(precond, poutdat);
        HYPRE_BoomerAMGSetPrintFileName(precond, "driver.out.log");
        HYPRE_BoomerAMGSetMaxIter(precond, 1);
        HYPRE_BoomerAMGSetCycleType(precond, cycle_type);
        HYPRE_BoomerAMGSetNumGridSweeps(precond, num_grid_sweeps);
        HYPRE_BoomerAMGSetGridRelaxType(precond, grid_relax_type);
        HYPRE_BoomerAMGSetRelaxWeight(precond, relax_weight);
        HYPRE_BoomerAMGSetOmega(precond, omega);
        HYPRE_BoomerAMGSetSmoothType(precond, smooth_type);
        HYPRE_BoomerAMGSetSmoothNumLevels(precond, smooth_num_levels);
        HYPRE_BoomerAMGSetSmoothNumSweeps(precond, smooth_num_sweeps);
        HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
        HYPRE_BoomerAMGSetMaxLevels(precond, max_levels);
        HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum);
        HYPRE_BoomerAMGSetNumFunctions(precond, num_functions);
        HYPRE_BoomerAMGSetVariant(precond, variant);
        HYPRE_BoomerAMGSetOverlap(precond, overlap);
        HYPRE_BoomerAMGSetDomainType(precond, domain_type);
        HYPRE_BoomerAMGSetSchwarzRlxWeight(precond, schwarz_rlx_weight);
        if (num_functions > 1)
           HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
       }
/* use DiagScale preconditioner */
      else if (precond_id == HYPRE_DIAGSCALE)
         {
          precond = NULL;
         }
/* use ParaSails preconditioner */
      else if (precond_id == HYPRE_PARASAILS)
         {
  	  HYPRE_ParaSailsCreate(hypre_MPI_COMM_WORLD, precond);
          HYPRE_ParaSailsSetParams(precond, sai_threshold, max_levels);
          HYPRE_ParaSailsSetFilter(precond, sai_filter);
          HYPRE_ParaSailsSetLogging(precond, poutdat);
         }
/* use Schwarz preconditioner */
      else if (precond_id == HYPRE_SCHWARZ)
         {
	  HYPRE_SchwarzCreate(precond);
	  HYPRE_SchwarzSetVariant(precond, variant);
	  HYPRE_SchwarzSetOverlap(precond, overlap);
	  HYPRE_SchwarzSetDomainType(precond, domain_type);
          HYPRE_SchwarzSetRelaxWeight(precond, schwarz_rlx_weight);
         }
/* use GSMG as preconditioner */
      else if (precond_id == HYPRE_GSMG)
         {
           /* fine grid */
          num_grid_sweeps[0] = num_sweep;
          grid_relax_type[0] = relax_default;
          hypre_TFree (grid_relax_points[0]);
          grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, num_sweep);
          for (i=0; i<num_sweep; i++)
             grid_relax_points[0][i] = 0;
    
          /* down cycle */
          num_grid_sweeps[1] = num_sweep;
          grid_relax_type[1] = relax_default;
          hypre_TFree (grid_relax_points[1]);
          grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, num_sweep);
          for (i=0; i<num_sweep; i++)
             grid_relax_points[1][i] = 0;
    
          /* up cycle */
          num_grid_sweeps[2] = num_sweep;
          grid_relax_type[2] = relax_default;
          hypre_TFree (grid_relax_points[2]);
          grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, num_sweep);
          for (i=0; i<num_sweep; i++)
             grid_relax_points[2][i] = 0;
    
          /* coarsest grid */
          num_grid_sweeps[3] = 1;
          grid_relax_type[3] = 9;
          hypre_TFree (grid_relax_points[3]);
          grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
          grid_relax_points[3][0] = 0;
 
          HYPRE_BoomerAMGCreate(precond); 
          HYPRE_BoomerAMGSetGSMG(precond, 4); 
          HYPRE_BoomerAMGSetInterpType(precond, interp_type);
          HYPRE_BoomerAMGSetNumSamples(precond, gsmg_samples);
          HYPRE_BoomerAMGSetTol(precond, pc_tol);
          HYPRE_BoomerAMGSetCoarsenType(precond, (hybrid*coarsen_type));
          HYPRE_BoomerAMGSetMeasureType(precond, measure_type);
          HYPRE_BoomerAMGSetStrongThreshold(precond, strong_threshold);
          HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor);
          HYPRE_BoomerAMGSetPrintLevel(precond, poutdat);
          HYPRE_BoomerAMGSetPrintFileName(precond, "driver.out.log");
          HYPRE_BoomerAMGSetMaxIter(precond, 1);
          HYPRE_BoomerAMGSetCycleType(precond, cycle_type);
          HYPRE_BoomerAMGSetNumGridSweeps(precond, num_grid_sweeps);
          HYPRE_BoomerAMGSetGridRelaxType(precond, grid_relax_type);
          HYPRE_BoomerAMGSetRelaxWeight(precond, relax_weight);
          HYPRE_BoomerAMGSetOmega(precond, omega);
          HYPRE_BoomerAMGSetSmoothType(precond, smooth_type);
          HYPRE_BoomerAMGSetSmoothNumLevels(precond, smooth_num_levels);
          HYPRE_BoomerAMGSetSmoothNumSweeps(precond, smooth_num_sweeps);
          HYPRE_BoomerAMGSetVariant(precond, variant);
          HYPRE_BoomerAMGSetOverlap(precond, overlap);
          HYPRE_BoomerAMGSetDomainType(precond, domain_type);
          HYPRE_BoomerAMGSetSchwarzRlxWeight(precond, schwarz_rlx_weight);
          HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
          HYPRE_BoomerAMGSetMaxLevels(precond, max_levels);
          HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum);
          HYPRE_BoomerAMGSetNumFunctions(precond, num_functions);
          if (num_functions > 1)
             HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
         }

/* use PILUT as preconditioner */
      else if (precond_id == HYPRE_PILUT)
         {
          ierr = HYPRE_ParCSRPilutCreate( hypre_MPI_COMM_WORLD, precond ); 
         }
}


HYPRE_Int hypre_destroy_precond(HYPRE_Int precond_id, void *precond)
{
    
    if (precond_id == HYPRE_BICGSTAB)
        HYPRE_BiCGSTABDestroy(precond);

    else if (precond_id == HYPRE_BOOMERAMG)
        HYPRE_BoomerAMGDestroy(precond);

    else if (precond_id == HYPRE_CGNR)
        HYPRE_CGNRDestroy(precond);

    else if (precond_id == HYPRE_DIAGSCALE)
        HYPRE_Destroy(precond);

    else if (precond_id == HYPRE_EUCLID)
        HYPRE_EuclidDestroy(precond);

    else if (precond_id == HYPRE_GMRES)
        HYPRE_GMRESDestroy(precond);

    else if (precond_id == HYPRE_GSMG)
        HYPRE_BoomerAMGDestroy(precond);

    else if (precond_id == HYPRE_HYBRID)
        HYPRE_HybridDestroy(precond);

    else if (precond_id == HYPRE_JACOBI)
        HYPRE_JacobiDestroy(precond);

    else if (precond_id == HYPRE_PARASAILS)
        HYPRE_ParaSailsDestroy(precond);

    else if (precond_id == HYPRE_PCG)
        HYPRE_PCGDestroy(precond);

    else if (precond_id == HYPRE_PFMG)
        HYPRE_PFMGDestroy(precond);

    else if (precond_id == HYPRE_PILUT)
        HYPRE_PilutDestroy(precond);

    else if (precond_id == HYPRE_SCHWARZ)
        HYPRE_SchwarzDestroy(precond);

    else if (precond_id == HYPRE_SMG)
        HYPRE_SMGDestroy(precond);

    else if (precond_id == HYPRE_SPARSEMSG)
        HYPRE_SparseMSGDestroy(precond);

    else if (precond_id == HYPRE_SPLIT)
        HYPRE_SplitDestroy(precond);

    else if (precond_id == HYPRE_SPLITPFMG)
        HYPRE_SplitDestroy(precond);

    else if (precond_id == HYPRE_SPLITSMG)
        HYPRE_SplitDestroy(precond);

    else if (precond_id == HYPRE_SYSPFMG)
        HYPRE_SysPFMGDestroy(precond);
}
