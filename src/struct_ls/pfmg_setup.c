/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetup( void               *pfmg_vdata,
                 hypre_StructMatrix *A,
                 hypre_StructVector *b,
                 hypre_StructVector *x        )
{
   hypre_PFMGData       *pfmg_data = (hypre_PFMGData *) pfmg_vdata;

   MPI_Comm              comm              = (pfmg_data -> comm);
   HYPRE_Int             relax_type        = (pfmg_data -> relax_type);
   HYPRE_Int             usr_jacobi_weight = (pfmg_data -> usr_jacobi_weight);
   HYPRE_Real            jacobi_weight     = (pfmg_data -> jacobi_weight);
   HYPRE_Int             skip_relax        = (pfmg_data -> skip_relax);
   HYPRE_Real           *dxyz              = (pfmg_data -> dxyz);
   HYPRE_Int             max_iter          = (pfmg_data -> max_iter);
   HYPRE_Int             matmult_type      = (pfmg_data -> matmult_type);
   HYPRE_Int             rap_type;
   HYPRE_Int             max_levels;
   HYPRE_Int             num_levels;

   hypre_Index           cindex;
   hypre_Index           stride;
   hypre_Index           periodic;

   HYPRE_Int            *cdir_l;
   HYPRE_Int            *active_l;
   hypre_StructGrid    **grid_l;

   HYPRE_Real           *relax_weights;

   hypre_StructMatrix  **A_l;
   hypre_StructMatrix  **P_l;
   hypre_StructMatrix  **RT_l;
   hypre_StructVector  **b_l;
   hypre_StructVector  **x_l;

   hypre_StructMatmultData **Ammdata_l;

   /* temp vectors */
   hypre_StructVector  **tx_l;
   hypre_StructVector  **r_l;
   hypre_StructVector  **e_l;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   hypre_StructGrid     *grid;

   hypre_Box            *cbox;

   HYPRE_Int             cdir, cmaxsize;
   HYPRE_Int             l;
   HYPRE_Int             dxyz_flag;

   HYPRE_Int             b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   HYPRE_Int             x_num_ghost[]  = {1, 1, 1, 1, 1, 1};
   HYPRE_Int             v_memory_mode  = 2;

   char                  region_name[1024];
#if DEBUG
   char                  filename[255];
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_sprintf(region_name, "%s", "PFMG-Init");
   HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
   hypre_GpuProfilingPushRange(region_name);
   hypre_MemoryPrintUsage(comm, hypre_HandleLogLevel(hypre_handle()), "PFMG setup begin", 0);

   /* RDF: For now, set memory mode to 0 if using R/B GS relaxation */
   if (relax_type > 1)
   {
      v_memory_mode = 0;
   }
   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid  = hypre_StructMatrixGrid(A);

   /* Initialize periodic */
   hypre_CopyIndex(hypre_StructGridPeriodic(grid), periodic);

   /* Compute a new max_levels value based on the grid */
   hypre_PFMGComputeMaxLevels(grid, &max_levels);
   if ((pfmg_data -> max_levels) > 0)
   {
      max_levels = hypre_min(max_levels, (pfmg_data -> max_levels));
   }
   (pfmg_data -> max_levels) = max_levels;

   /* compute dxyz */
   hypre_PFMGComputeDxyz(A, dxyz, &dxyz_flag);

   /* Run coarsening */
   cbox = hypre_BoxClone(hypre_StructGridBoundingBox(grid));
   hypre_PFMGCoarsen(cbox, periodic, max_levels, dxyz_flag, dxyz,
                     &cdir_l, &active_l, &relax_weights, &num_levels);
   cmaxsize = hypre_BoxMaxSize(cbox);
   hypre_BoxDestroy(cbox);

   /* set all levels active if skip_relax = 0 */
   if (!skip_relax)
   {
      for (l = 0; l < num_levels; l++)
      {
         active_l[l] = 1;
      }
   }

   (pfmg_data -> num_levels) = num_levels;
   (pfmg_data -> cdir_l)     = cdir_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   /* RDF: Rework this for vectors to avoid data copies in Resize(), similar to
    * what is being done for matrices. */

   /* Modify the rap_type if red-black Gauss-Seidel is used.  Red-black gs is
    * used only in the non-Galerkin case. */
   if (relax_type == 2 || relax_type == 3)   /* red-black gs */
   {
      (pfmg_data -> rap_type) = 1;
   }
   rap_type = (pfmg_data -> rap_type);

   grid_l = hypre_TAlloc(hypre_StructGrid *, num_levels, HYPRE_MEMORY_HOST);
   A_l    = hypre_TAlloc(hypre_StructMatrix *, num_levels, HYPRE_MEMORY_HOST);
   P_l    = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   RT_l   = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   b_l    = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   x_l    = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   tx_l   = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   r_l    = tx_l;
   e_l    = tx_l;

   Ammdata_l = hypre_TAlloc(hypre_StructMatmultData *, num_levels, HYPRE_MEMORY_HOST);

   hypre_StructGridRef(grid, &grid_l[0]);
   A_l[0] = hypre_StructMatrixRef(A);
   b_l[0] = hypre_StructVectorRef(b);
   x_l[0] = hypre_StructVectorRef(x);
   hypre_StructVectorSetMemoryMode(x_l[0], v_memory_mode);

   tx_l[0] = hypre_StructVectorCreate(comm, grid_l[0]);
   hypre_StructVectorSetMemoryMode(tx_l[0], v_memory_mode);
   hypre_StructVectorSetNumGhost(tx_l[0], x_num_ghost);
   hypre_StructVectorInitialize(tx_l[0]);
   hypre_StructVectorAssemble(tx_l[0]);

   //   /* RDF AP Debug */
   //   hypre_StructAssumedPartitionPrint("zAP", hypre_BoxManAssumedPartition(
   //                                        hypre_StructGridBoxMan(grid_l[0])));

   /* Use hypre_StructMatrixInitializeShell() and InitializeData() below to do
    * PtAPSetup() (and RTtAPSetup) first, then use MatmulMultiply() to allocate
    * data and complete the multiply.  This will avoid copies. */

   /* First set up the matrix shells (no data allocated) to adjust data spaces
    * without requiring memory copies */

   for (l = 0; l < (num_levels - 1); l++)
   {
      //HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

      cdir = cdir_l[l];

      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetStride(cdir, stride);

      /* set up interpolation and restriction operators */
      P_l[l] = hypre_PFMGCreateInterpOp(A_l[l], cdir, stride, rap_type);
      RT_l[l] = P_l[l];
#if 0 /* TODO: Allow RT != P */
      if (nonsymmetric_cycle)
      {
         RT_l[l] = hypre_PFMGCreateRestrictOp(A_l[l], cdir, stride);
      }
#endif
      HYPRE_StructMatrixSetTranspose(RT_l[l], 1);

      if (rap_type == 0)
      {
         hypre_StructMatrixInitializeShell(P_l[l]);
#if 0 /* TODO: Allow RT != P */
         if (nonsymmetric_cycle)
         {
            hypre_StructMatrixInitializeShell(RT_l[l]);
         }
#endif
         if (RT_l[l] != P_l[l])
         {
            /* If restriction is not the same as interpolation, compute RAP */
            hypre_StructMatrixRTtAPSetup(matmult_type, RT_l[l], A_l[l], P_l[l], &Ammdata_l[l + 1], &A_l[l + 1]);
         }
         else
         {
            hypre_StructMatrixPtAPSetup(matmult_type, A_l[l], P_l[l], &Ammdata_l[l + 1], &A_l[l + 1]);
         }
         hypre_StructGridRef(hypre_StructMatrixGrid(A_l[l + 1]), &grid_l[l + 1]);
      }
      else
      {
         hypre_StructMatrixInitialize(P_l[l]);
         hypre_PFMGSetupInterpOp(P_l[l], A_l[l], cdir);
#if 0 /* TODO: Allow RT != P */
         if (nonsymmetric_cycle)
         {
            hypre_StructMatrixInitialize(RT_l[l]);
            hypre_PFMGSetupRestrictOp(RT_l[l], A_l[l], cdir);
         }
#endif

         /* RDF: The coarse grid should be computed in CreateRAPOp() */
         hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l + 1]);
         hypre_StructGridAssemble(grid_l[l + 1]);

         A_l[l + 1] = hypre_PFMGCreateRAPOp(RT_l[l], A_l[l], P_l[l], grid_l[l + 1], cdir, rap_type);
         hypre_StructMatrixInitialize(A_l[l + 1]);
         hypre_PFMGSetupRAPOp(RT_l[l], A_l[l], P_l[l], cdir, cindex, stride, rap_type, A_l[l + 1]);
      }

      //      /* RDF AP Debug */
      //      hypre_StructAssumedPartitionPrint("zAP", hypre_BoxManAssumedPartition(
      //                                           hypre_StructGridBoxMan(grid_l[l+1])));

      b_l[l + 1] = hypre_StructVectorCreate(comm, grid_l[l + 1]);
      hypre_StructVectorSetNumGhost(b_l[l + 1], b_num_ghost);
      hypre_StructVectorInitialize(b_l[l + 1]);
      hypre_StructVectorAssemble(b_l[l + 1]);

      x_l[l + 1] = hypre_StructVectorCreate(comm, grid_l[l + 1]);
      hypre_StructVectorSetMemoryMode(x_l[l + 1], v_memory_mode);
      hypre_StructVectorSetNumGhost(x_l[l + 1], x_num_ghost);
      hypre_StructVectorInitialize(x_l[l + 1]);
      hypre_StructVectorAssemble(x_l[l + 1]);

      tx_l[l + 1] = hypre_StructVectorCreate(comm, grid_l[l + 1]);
      hypre_StructVectorSetMemoryMode(tx_l[l + 1], v_memory_mode);
      hypre_StructVectorSetNumGhost(tx_l[l + 1], x_num_ghost);
      hypre_StructVectorInitialize(tx_l[l + 1]);
      hypre_StructVectorAssemble(tx_l[l + 1]);

      //HYPRE_ANNOTATE_MGLEVEL_END(l);
   }
   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_REGION_END("%s", region_name);

   hypre_sprintf(region_name, "%s", "PFMG-Setup");
   HYPRE_ANNOTATE_REGION_BEGIN("%s", region_name);
   hypre_GpuProfilingPushRange(region_name);

   /* Now finish up the matrices */
   for (l = 0; l < (num_levels - 1); l++)
   {
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);
      hypre_sprintf(region_name, "%s-%d", "PFMG-Level", l);
      hypre_GpuProfilingPushRange(region_name);

      cdir = cdir_l[l];

      if (rap_type == 0)
      {
         hypre_StructMatrixInitializeData(P_l[l], NULL);
         hypre_PFMGSetupInterpOp(P_l[l], A_l[l], cdir);
#if 0 /* TODO: Allow RT != P */
         if (nonsymmetric_cycle)
         {
            hypre_StructMatrixInitializeData(RT_l[l], NULL);
            hypre_PFMGSetupRestrictOp(RT_l[l], A_l[l], cdir);
         }
#endif
         hypre_StructMatmultMultiply(Ammdata_l[l + 1]);
      }

      hypre_GpuProfilingPopRange();
      HYPRE_ANNOTATE_MGLEVEL_END(l);
   }

   hypre_TFree(Ammdata_l, HYPRE_MEMORY_HOST);

   (pfmg_data -> grid_l) = grid_l;
   (pfmg_data -> A_l)    = A_l;
   (pfmg_data -> P_l)    = P_l;
   (pfmg_data -> RT_l)   = RT_l;
   (pfmg_data -> b_l)    = b_l;
   (pfmg_data -> x_l)    = x_l;
   (pfmg_data -> tx_l)   = tx_l;
   (pfmg_data -> r_l)    = r_l;
   (pfmg_data -> e_l)    = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *-----------------------------------------------------*/

   relax_data_l    = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   matvec_data_l   = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   restrict_data_l = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   interp_data_l   = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);
      cdir = cdir_l[l];

      /* set up the interpolation operator */
      interp_data_l[l] = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup(interp_data_l[l], P_l[l], x_l[l + 1]);

      /* set up the restriction operator */
      restrict_data_l[l] = hypre_StructMatvecCreate();
      hypre_StructMatvecSetTranspose(restrict_data_l[l], 1);
      hypre_StructMatvecSetup(restrict_data_l[l], RT_l[l], r_l[l]);

      HYPRE_ANNOTATE_MGLEVEL_END(l);
   }

   /* Check for zero diagonal on coarsest grid, occurs with singular problems
    * like full Neumann or full periodic.  Note that a processor with zero
    * diagonal will set active_l = 0, other processors will not. This is OK as
    * we only want to avoid the division by zero on the one processor that owns
    * the single coarse grid point. */
   if (hypre_StructMatrixZeroDiagonal(A_l[l]))
   {
      active_l[l] = 0;
   }

   /* set up fine grid relaxation */
   relax_data_l[0] = hypre_PFMGRelaxCreate(comm);
   hypre_PFMGRelaxSetTol(relax_data_l[0], 0.0);
   if (usr_jacobi_weight)
   {
      hypre_PFMGRelaxSetJacobiWeight(relax_data_l[0], jacobi_weight);
   }
   else
   {
      hypre_PFMGRelaxSetJacobiWeight(relax_data_l[0], relax_weights[0]);
   }
   hypre_PFMGRelaxSetType(relax_data_l[0], relax_type);
   hypre_PFMGRelaxSetTempVec(relax_data_l[0], tx_l[0]);
   hypre_PFMGRelaxSetup(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
   if (num_levels > 1)
   {
      for (l = 1; l < num_levels; l++)
      {
         /* set relaxation parameters */
         if (active_l[l])
         {
            relax_data_l[l] = hypre_PFMGRelaxCreate(comm);
            hypre_PFMGRelaxSetTol(relax_data_l[l], 0.0);
            if (usr_jacobi_weight)
            {
               hypre_PFMGRelaxSetJacobiWeight(relax_data_l[l], jacobi_weight);
            }
            else
            {
               hypre_PFMGRelaxSetJacobiWeight(relax_data_l[l], relax_weights[l]);
            }
            hypre_PFMGRelaxSetType(relax_data_l[l], relax_type);
            hypre_PFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
         }
      }

      /* change coarsest grid relaxation parameters */
      l = num_levels - 1;
      if (active_l[l])
      {
         HYPRE_Int maxwork, maxiter;
         hypre_PFMGRelaxSetType(relax_data_l[l], 0);

         /* do no more work on the coarsest grid than the cost of a V-cycle
          * (estimating roughly 4 communications per V-cycle level) */
         maxwork = 4 * num_levels;

         /* do sweeps proportional to the coarsest grid size */
         maxiter = hypre_min(maxwork, cmaxsize);
         hypre_PFMGRelaxSetMaxIter(relax_data_l[l], maxiter);
      }

      /* call relax setup */
      for (l = 1; l < num_levels; l++)
      {
         if (active_l[l])
         {
            hypre_PFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }
      }
   }
   hypre_TFree(relax_weights, HYPRE_MEMORY_HOST);

   for (l = 0; l < num_levels; l++)
   {
      /* set up the residual routine */
      matvec_data_l[l] = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
   }

   (pfmg_data -> active_l)        = active_l;
   (pfmg_data -> relax_data_l)    = relax_data_l;
   (pfmg_data -> matvec_data_l)   = matvec_data_l;
   (pfmg_data -> restrict_data_l) = restrict_data_l;
   (pfmg_data -> interp_data_l)   = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((pfmg_data -> logging) > 0)
   {
      (pfmg_data -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
      (pfmg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   }

   hypre_MemoryPrintUsage(comm, hypre_HandleLogLevel(hypre_handle()), "PFMG setup end", 0);
   hypre_GpuProfilingPopRange();
   hypre_sprintf(region_name, "%s", "PFMG-Setup");
   HYPRE_ANNOTATE_REGION_END("%s", region_name);
   HYPRE_ANNOTATE_FUNC_END;

#ifdef DEBUG_SETUP
   {
      hypre_StructVector   *ones  = NULL;
      hypre_StructVector   *Pones = NULL;
      char                  filename[255];

      l = num_levels - 1;
      hypre_sprintf(filename, "pfmg_A.l%02d", l);
      hypre_StructMatrixPrint(filename, A_l[l], 0);

      for (l = 0; l < (num_levels - 1); l++)
      {
         hypre_sprintf(filename, "pfmg_A.l%02d", l);
         hypre_StructMatrixPrint(filename, A_l[l], 0);
         hypre_sprintf(filename, "pfmg_P.l%02d", l);
         hypre_StructMatrixPrint(filename, P_l[l], 0);

         /* Check if P interpolates vector of ones */
         HYPRE_StructVectorCreate(comm, hypre_StructMatrixGrid(A_l[l + 1]), &ones);
         HYPRE_StructVectorInitialize(ones);
         HYPRE_StructVectorSetConstantValues(ones, 1.0);
         HYPRE_StructVectorAssemble(ones);

         HYPRE_StructVectorCreate(comm, hypre_StructMatrixGrid(A_l[l]), &Pones);
         HYPRE_StructVectorInitialize(Pones);
         HYPRE_StructVectorAssemble(Pones);

         /* interpolate (x = P*e_c) */
         hypre_StructMatvec(1.0, P_l[l], ones, 0.0, Pones);

         hypre_sprintf(filename, "pfmg_ones.l%02d", l);
         HYPRE_StructVectorPrint(filename, ones, 0);
         hypre_sprintf(filename, "pfmg_Pones.l%02d", l);
         HYPRE_StructVectorPrint(filename, Pones, 0);

         HYPRE_StructVectorDestroy(ones);
         HYPRE_StructVectorDestroy(Pones);
      }
   }
#endif

   return hypre_error_flag;
}
