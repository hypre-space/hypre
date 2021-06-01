/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 * hypre_SSAMGCoarseSolverSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGCoarseSolverSetup( void *ssamg_vdata )
{
   hypre_SSAMGData       *ssamg_data   = (hypre_SSAMGData *) ssamg_vdata;

   MPI_Comm               comm         = hypre_SSAMGDataComm(ssamg_data);
   HYPRE_Int              nparts       = hypre_SSAMGDataNParts(ssamg_data);
   HYPRE_Int              num_levels   = hypre_SSAMGDataNumLevels(ssamg_data);
   HYPRE_Int              num_crelax   = hypre_SSAMGDataNumCoarseRelax(ssamg_data);
   HYPRE_Int              csolver_type = hypre_SSAMGDataCSolverType(ssamg_data);
   HYPRE_Int              print_level  = hypre_SSAMGDataPrintLevel(ssamg_data);
   hypre_SStructGrid     *cgrid        = hypre_SSAMGDataGridl(ssamg_data)[num_levels - 1];

   void                 **relax_data_l  = (ssamg_data -> relax_data_l);
   void                 **matvec_data_l = (ssamg_data -> matvec_data_l);
   HYPRE_Int            **active_l = hypre_SSAMGDataActivel(ssamg_data);
   HYPRE_Real           **relax_weights = hypre_SSAMGDataRelaxWeights(ssamg_data);
   hypre_SStructMatrix  **A_l  = (ssamg_data -> A_l);
   hypre_SStructVector  **x_l  = (ssamg_data -> x_l);
   hypre_SStructVector  **b_l  = (ssamg_data -> b_l);
   hypre_SStructVector  **tx_l = (ssamg_data -> tx_l);

   hypre_Box             *bbox;
   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *sgrid;
   HYPRE_Solver           csolver;
   HYPRE_IJMatrix         ij_Ac;
   hypre_ParCSRMatrix    *par_Ac;
   hypre_ParVector       *par_b;
   hypre_ParVector       *par_x;

   HYPRE_Int              l, part, cmax_size, max_work;

   l = (num_levels - 1);
   HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);
   if (csolver_type == 0)
   {
      /* Compute maximum number of relaxation sweeps in the coarse grid if requested */
      if (num_crelax < 0)
      {
         /* do no more work on the coarsest grid than the cost of a V-cycle
          * (estimating roughly 4 communications per V-cycle level) */
         max_work = 4*num_levels;

         /* do sweeps proportional to the coarsest grid size */
         cmax_size = 0;
         for (part = 0; part < nparts; part++)
         {
            pgrid = hypre_SStructGridPGrid(cgrid, part);
            sgrid = hypre_SStructPGridCellSGrid(pgrid);
            bbox  = hypre_StructGridBoundingBox(sgrid);

            cmax_size = hypre_max(cmax_size, hypre_BoxMaxSize(bbox));
         }
         num_crelax = hypre_min(max_work, cmax_size);
         hypre_SSAMGDataNumCoarseRelax(ssamg_data) = num_crelax;
      }

      hypre_SStructMatvecCreate(&matvec_data_l[l]);
      hypre_SStructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
      hypre_SSAMGRelaxCreate(comm, nparts, &relax_data_l[l]);
      hypre_SSAMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_SSAMGRelaxSetWeights(relax_data_l[l], relax_weights[l]);
      hypre_SSAMGRelaxSetActiveParts(relax_data_l[l], active_l[l]);
      hypre_SSAMGRelaxSetType(relax_data_l[l], 0);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_crelax);
      hypre_SSAMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      hypre_SSAMGRelaxSetMatvecData(relax_data_l[l], matvec_data_l[l]);
      hypre_SSAMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
   }
   else if (csolver_type == 1)
   {
      /* Convert SStructMatrix to IJMatrix */
      HYPRE_SStructMatrixToIJMatrix(A_l[l], 1, &ij_Ac);
      HYPRE_IJMatrixGetObject(ij_Ac, (void **) &par_Ac);
      par_x = hypre_SStructVectorParVector(x_l[l]);
      par_b = hypre_SStructVectorParVector(b_l[l]);

      /* Use BoomerAMG */
      HYPRE_BoomerAMGCreate(&csolver);
      HYPRE_BoomerAMGSetStrongThreshold(csolver, 0.5);
      HYPRE_BoomerAMGSetPMaxElmts(csolver, 4);
      HYPRE_BoomerAMGSetInterpType(csolver, 18); /* MM ext-e interpolation */
      HYPRE_BoomerAMGSetCoarsenType(csolver, 10); /* HMIS coarsening */
      if (num_crelax > 0)
      {
         HYPRE_BoomerAMGSetMaxIter(csolver, num_crelax);
      }
      HYPRE_BoomerAMGSetMaxCoarseSize(csolver, 1000);
      HYPRE_BoomerAMGSetCycleRelaxType(csolver, 0, 3); /* Coarse solver - Jacobi */
      HYPRE_BoomerAMGSetTol(csolver, 0.0);
      HYPRE_BoomerAMGSetPrintLevel(csolver, print_level);
      HYPRE_BoomerAMGSetLogging(csolver, 1);
      HYPRE_BoomerAMGSetAggNumLevels(csolver, 1);
      HYPRE_BoomerAMGSetup(csolver, par_Ac, par_b, par_x);

      (ssamg_data -> csolver) = csolver;
      (ssamg_data -> ij_Ac)   = ij_Ac;
      (ssamg_data -> par_x)   = par_x;
      (ssamg_data -> par_b)   = par_b;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown coarse solve!\n");
   }
   HYPRE_ANNOTATE_MGLEVEL_END(l);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGCoarseSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGCoarseSolve( void *ssamg_vdata )
{
   hypre_SSAMGData       *ssamg_data    = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Int              csolver_type  = hypre_SSAMGDataCSolverType(ssamg_data);
   HYPRE_Int              num_levels    = hypre_SSAMGDataNumLevels(ssamg_data);
   HYPRE_Int            **active_l      = hypre_SSAMGDataActivel(ssamg_data);
   void                 **matvec_data_l = (ssamg_data -> matvec_data_l);
   void                 **relax_data_l  = (ssamg_data -> relax_data_l);

   hypre_SStructMatrix  **A_l     = (ssamg_data -> A_l);
   hypre_SStructVector  **x_l     = (ssamg_data -> x_l);
   hypre_SStructVector  **b_l     = (ssamg_data -> b_l);
   HYPRE_Solver           csolver = (ssamg_data -> csolver);
   HYPRE_IJMatrix         ij_Ac   = (ssamg_data -> ij_Ac);
   hypre_ParVector       *par_x   = (ssamg_data -> par_x);
   hypre_ParVector       *par_b   = (ssamg_data -> par_b);
   hypre_ParCSRMatrix    *par_Ac;

   HYPRE_Int              l;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   l = (num_levels - 1);
   if (csolver_type == 0)
   {
      /* Set active parts */
      hypre_SStructMatvecSetActiveParts(matvec_data_l[l], active_l[l]);

      /* Coarsest level solver */
      hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 1);
      hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      /* Set all parts to active */
      hypre_SStructMatvecSetAllPartsActive(matvec_data_l[l]);
   }
   else if (csolver_type == 1)
   {
      hypre_SStructVectorSetConstantValues(x_l[l], 0.0);
      HYPRE_IJMatrixGetObject(ij_Ac, (void **) &par_Ac);
      HYPRE_BoomerAMGSolve(csolver, par_Ac, par_b, par_x);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown coarse solve!\n");
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGCoarseSolverDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGCoarseSolverDestroy( void *ssamg_vdata )
{
   hypre_SSAMGData  *ssamg_data    = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Int         csolver_type  = hypre_SSAMGDataCSolverType(ssamg_data);
   HYPRE_Int         num_levels    = hypre_SSAMGDataNumLevels(ssamg_data);

   HYPRE_Int l;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   l = (num_levels - 1);
   HYPRE_SStructGridDestroy(ssamg_data -> grid_l[l]);
   HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[l]);
   HYPRE_SStructVectorDestroy(ssamg_data -> b_l[l]);
   HYPRE_SStructVectorDestroy(ssamg_data -> x_l[l]);
   HYPRE_SStructVectorDestroy(ssamg_data -> tx_l[l]);
   hypre_TFree(ssamg_data -> cdir_l[l], HYPRE_MEMORY_HOST);
   hypre_TFree(ssamg_data -> active_l[l], HYPRE_MEMORY_HOST);
   hypre_TFree(ssamg_data -> relax_weights[l], HYPRE_MEMORY_HOST);
   if (csolver_type == 0)
   {
      hypre_SSAMGRelaxDestroy(ssamg_data -> relax_data_l[l]);
      hypre_SStructMatvecDestroy(ssamg_data -> matvec_data_l[l]);
   }
   else if (csolver_type == 1)
   {
      HYPRE_BoomerAMGDestroy(ssamg_data -> csolver);
      HYPRE_IJMatrixDestroy(ssamg_data -> ij_Ac);
      (ssamg_data -> par_x) = NULL;
      (ssamg_data -> par_b) = NULL;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown coarse solve!\n");
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
