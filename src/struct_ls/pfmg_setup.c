/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

#define hypre_PFMGSetCIndex(cdir, cindex)       \
   {                                            \
      hypre_SetIndex(cindex, 0);                \
      hypre_IndexD(cindex, cdir) = 0;           \
   }

#define hypre_PFMGSetStride(cdir, stride)       \
   {                                            \
      hypre_SetIndex(stride, 1);                \
      hypre_IndexD(stride, cdir) = 2;           \
   }

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
   HYPRE_Int             rap_type;
   HYPRE_Int             max_levels;
   HYPRE_Int             num_levels;
   HYPRE_Int             resize;

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

#if DEBUG
   char                  filename[255];
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

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

   for (l = 0; l < (num_levels - 1); l++)
   {
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);
      cdir = cdir_l[l];

      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetStride(cdir, stride);

      /* set up interpolation and restriction operators */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Interpolation");
      P_l[l] = hypre_zPFMGCreateInterpOp(A_l[l], cdir, stride, rap_type);
      RT_l[l] = P_l[l];
#if 0 /* TODO: Allow RT != P */
      if (nonsymmetric_cycle)
      {
         RT_l[l] = hypre_zPFMGCreateRestrictOp(A_l[l], cdir, stride);
      }
#endif
      hypre_StructMatrixSetTranspose(RT_l[l], 1, &resize);
      hypre_StructMatrixInitialize(P_l[l]);
      hypre_zPFMGSetupInterpOp(P_l[l], A_l[l], cdir);
#if 0 /* TODO: Allow RT != P */
      if (nonsymmetric_cycle)
      {
         hypre_StructMatrixInitialize(RT_l[l]);
         hypre_zPFMGSetupRestrictOp(RT_l[l], A_l[l], cdir);
      }
#endif
      HYPRE_ANNOTATE_REGION_END("%s", "Interpolation");

      HYPRE_ANNOTATE_REGION_BEGIN("%s", "RAP");
      if (rap_type == 0)
      {
         if (RT_l[l] != P_l[l])
         {
            /* If restriction is not the same as interpolation, compute RAP */
            hypre_StructMatrixRTtAP(RT_l[l], A_l[l], P_l[l], &A_l[l + 1]);
         }
         else
         {
            hypre_StructMatrixPtAP(A_l[l], P_l[l], &A_l[l + 1]);
         }
         hypre_StructGridRef(hypre_StructMatrixGrid(A_l[l + 1]), &grid_l[l + 1]);
      }
      else
      {
         /* RDF: The coarse grid should be computed in CreateRAPOp() */
         hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l + 1]);
         hypre_StructGridAssemble(grid_l[l + 1]);

         A_l[l + 1] = hypre_PFMGCreateRAPOp(RT_l[l], A_l[l], P_l[l], grid_l[l + 1], cdir, rap_type);
         hypre_StructMatrixInitialize(A_l[l + 1]);
         hypre_PFMGSetupRAPOp(RT_l[l], A_l[l], P_l[l], cdir, cindex, stride, rap_type, A_l[l + 1]);
      }
      HYPRE_ANNOTATE_REGION_END("%s", "RAP");

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

      HYPRE_ANNOTATE_MGLEVEL_END(l);
   }

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
   if (hypre_PFMGZeroDiagonal(A_l[l]))
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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeMaxLevels( hypre_StructGrid   *grid,
                            HYPRE_Int          *max_levels_ptr )
{
   HYPRE_Int        ndim = hypre_StructGridNDim(grid);
   hypre_Box       *bbox = hypre_StructGridBoundingBox(grid);
   HYPRE_Int        max_levels, d;

   max_levels = 1;
   for (d = 0; d < ndim; d++)
   {
      max_levels += hypre_Log2(hypre_BoxSizeD(bbox, d)) + 2;
   }

   *max_levels_ptr = max_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeCxyz
 *
 * TODO: Change BoxLoopHost to BoxLoop
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_PFMGComputeCxyz( hypre_StructMatrix *A,
                       HYPRE_Real         *cxyz,
                       HYPRE_Real         *sqcxyz)
{
   hypre_StructGrid      *grid = hypre_StructMatrixGrid(A);

   hypre_Box             *A_dbox;
   HYPRE_Int              Ai;
   HYPRE_Real            *Ap;

   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
   HYPRE_Int              diag_entry;

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;

   HYPRE_Int              cte_coeff;
   HYPRE_Int              i, si, d, ndim;
   HYPRE_Real             val;
   HYPRE_Real             tcxyz[HYPRE_MAXDIM];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   ndim          = hypre_StructMatrixNDim(A);
   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   diag_entry    = hypre_StructStencilDiagEntry(stencil);
   compute_boxes = hypre_StructGridBoxes(grid);
   cte_coeff     = hypre_StructMatrixConstantCoefficient(A);
   hypre_SetIndex(stride, 1);
   for (d = 0; d < ndim; d++)
   {
      cxyz[d] = 0.0;
      sqcxyz[d] = 0.0;
   }

   /*----------------------------------------------------------
    * Compute cxyz (use arithmetic mean)
    *----------------------------------------------------------*/
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);
      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      start  = hypre_BoxIMin(compute_box);
      hypre_BoxGetStrideSize(compute_box, stride, loop_size);

      /* all coefficients constant or variable diagonal */
      if (cte_coeff)
      {
         Ai = hypre_CCBoxIndexRank(A_dbox, start);

         for (d = 0; d < ndim; d++)
         {
            tcxyz[d] = 0.0;
         }

         /* Compute tcxyz[d] */
         for (si = 0; si < stencil_size; si++)
         {
            Ap  = hypre_StructMatrixBoxData(A, i, si);
            val = Ap[Ai];

            for (d = 0; d < ndim; d++)
            {
               if (hypre_IndexD(stencil_shape[si], d))
               {
                  tcxyz[d] += val;
               }
            }
         }

         /* get sign of diagonal */
         Ap = hypre_StructMatrixBoxData(A, i, diag_entry);

         /* Update cxyz[d] */
         if (Ap[Ai] < 0)
         {
            for (d = 0; d < ndim; d++)
            {
               cxyz[d]   += tcxyz[d];
               sqcxyz[d] += tcxyz[d] * tcxyz[d];
            }
         }
         else
         {
            for (d = 0; d < ndim; d++)
            {
               cxyz[d]   -= tcxyz[d];
               sqcxyz[d] += tcxyz[d] * tcxyz[d];
            }
         }
      }

      /* constant_coefficient==0, all coefficients vary with space */
      else
      {
         hypre_BoxLoop1BeginHost(ndim, loop_size,
                                 A_dbox, start, stride, Ai);
         {
            for (d = 0; d < ndim; d++)
            {
               tcxyz[d] = 0.0;
            }

            /* Compute tcxyz[d] */
            for (si = 0; si < stencil_size; si++)
            {
               Ap  = hypre_StructMatrixBoxData(A, i, si);
               val = Ap[Ai];

               for (d = 0; d < ndim; d++)
               {
                  if (hypre_IndexD(stencil_shape[si], d))
                  {
                     tcxyz[d] += val;
                  }
               }
            }

            /* get sign of diagonal */
            Ap = hypre_StructMatrixBoxData(A, i, diag_entry);

            /* Update cxyz[d] */
            if (Ap[Ai] < 0)
            {
               for (d = 0; d < ndim; d++)
               {
                  cxyz[d]   += tcxyz[d];
                  sqcxyz[d] += tcxyz[d] * tcxyz[d];
               }
            }
            else
            {
               for (d = 0; d < ndim; d++)
               {
                  cxyz[d]   -= tcxyz[d];
                  sqcxyz[d] += tcxyz[d] * tcxyz[d];
               }
            }
         }
         hypre_BoxLoop1EndHost(Ai);
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeDxyz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz( hypre_StructMatrix *A,
                       HYPRE_Real         *dxyz,
                       HYPRE_Int          *dxyz_flag )
{
   MPI_Comm           comm = hypre_StructMatrixComm(A);
   hypre_StructGrid  *grid = hypre_StructMatrixGrid(A);

   HYPRE_Int          cte_coeff;
   HYPRE_Real         cxyz_max;
   HYPRE_Real         cxyz[HYPRE_MAXDIM];
   HYPRE_Real         sqcxyz[HYPRE_MAXDIM];
   HYPRE_Real         tcxyz[HYPRE_MAXDIM];
   HYPRE_Real         mean[HYPRE_MAXDIM];
   HYPRE_Real         deviation[HYPRE_MAXDIM];

   HYPRE_Int          d, ndim;
   HYPRE_BigInt       global_size;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*----------------------------------------------------------
    * Exit if user gives dxyz different than zero
    *----------------------------------------------------------*/

   if ((dxyz[0] != 0) && (dxyz[1] != 0) && (dxyz[2] != 0))
   {
      *dxyz_flag = 0;

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   ndim        = hypre_StructMatrixNDim(A);
   cte_coeff   = hypre_StructMatrixConstantCoefficient(A);
   global_size = hypre_StructGridGlobalSize(grid);

   /* Compute cxyz and sqcxyz arrays */
   hypre_PFMGComputeCxyz(A, cxyz, sqcxyz);

   /*----------------------------------------------------------
    * Compute dxyz
    *----------------------------------------------------------*/

   if (cte_coeff)
   {
      /* all coefficients constant or variable diagonal */
      global_size = 1;
   }
   else
   {
      /* all coefficients vary with space */
      for (d = 0; d < ndim; d++)
      {
         tcxyz[d] = cxyz[d];
      }
      hypre_MPI_Allreduce(tcxyz, cxyz, ndim, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

      for (d = 0; d < ndim; d++)
      {
         tcxyz[d] = sqcxyz[d];
      }
      hypre_MPI_Allreduce(tcxyz, sqcxyz, ndim, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   }

   for (d = 0; d < ndim; d++)
   {
      mean[d] = cxyz[d] / (HYPRE_Real) global_size;
      deviation[d] = sqcxyz[d] / (HYPRE_Real) global_size;
   }

   cxyz_max = 0.0;
   for (d = 0; d < ndim; d++)
   {
      cxyz_max = hypre_max(cxyz_max, cxyz[d]);
   }

   if (cxyz_max == 0.0)
   {
      /* Do isotropic coarsening */
      for (d = 0; d < ndim; d++)
      {
         cxyz[d] = 1.0;
      }
      cxyz_max = 1.0;
   }

   /* Set dxyz values that are scaled appropriately for the coarsening routine */
   for (d = 0; d < ndim; d++)
   {
      HYPRE_Real max_anisotropy = HYPRE_REAL_MAX / 1000;
      if (cxyz[d] > (cxyz_max / max_anisotropy))
      {
         cxyz[d] /= cxyz_max;
         dxyz[d] = hypre_sqrt(1.0 / cxyz[d]);
      }
      else
      {
         dxyz[d] = hypre_sqrt(max_anisotropy);
      }
   }

   /* Set 'dxyz_flag' if the matrix-coefficient variation is "too large".
    * This is used later to set relaxation weights for Jacobi.
    *
    * Use the "square of the coefficient of variation" = (sigma/mu)^2,
    * where sigma is the standard deviation and mu is the mean.  This is
    * equivalent to computing (d - mu^2)/mu^2 where d is the average of
    * the squares of the coefficients stored in 'deviation'.  Care is
    * taken to avoid dividing by zero when the mean is zero. */

   *dxyz_flag = 0;
   for (d = 0; d < ndim; d++)
   {
      deviation[d] -= mean[d] * mean[d];
      if ( deviation[d] > 0.1 * (mean[d]*mean[d]) )
      {
         *dxyz_flag = 1;
         break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns 1 if there is a zero on the diagonal, otherwise returns 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGZeroDiagonal( hypre_StructMatrix *A )
{
   hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;

   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;

   HYPRE_Real            *Ap;
   hypre_Box             *A_dbox;
   HYPRE_Int              i, si;

   hypre_Index            diag_offset;
   HYPRE_Real             diag_product = 1.0;
   HYPRE_Int              zero_diag = 0;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   hypre_SetIndex3(stride, 1, 1, 1);
   hypre_SetIndex3(diag_offset, 0, 0, 0);

   /* Need to modify here */
   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);
      start  = hypre_BoxIMin(compute_box);
      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      hypre_BoxGetStrideSize(compute_box, stride, loop_size);

      si = hypre_StructStencilOffsetEntry(stencil, diag_offset);
      Ap = hypre_StructMatrixBoxData(A, i, si);
      if (hypre_StructMatrixConstEntry(A, si))
      {
         diag_product = *Ap;
      }
      else
      {
         hypre_SerialBoxLoop1Begin(hypre_StructMatrixNDim(A), loop_size,
                                   A_dbox, start, stride, Ai);
         {
            diag_product *= Ap[Ai];
         }
         hypre_SerialBoxLoop1End(Ai);
      }
   }

   if (diag_product == 0)
   {
      zero_diag = 1;
   }

   return zero_diag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGCoarsen( hypre_Box     *cbox,
                   hypre_Index    periodic,
                   HYPRE_Int      max_levels,
                   HYPRE_Int      dxyz_flag,
                   HYPRE_Real    *dxyz,
                   HYPRE_Int    **cdir_l_ptr,
                   HYPRE_Int    **active_l_ptr,
                   HYPRE_Real   **relax_weights_ptr,
                   HYPRE_Int     *num_levels )
{
   HYPRE_Int      ndim = hypre_BoxNDim(cbox);
   HYPRE_Int     *cdir_l;
   HYPRE_Int     *active_l;
   HYPRE_Real    *relax_weights;

   hypre_Index    coarsen;
   hypre_Index    cindex;
   hypre_Index    stride;

   HYPRE_Real     alpha, beta, min_dxyz;
   HYPRE_Int      d, l, cdir;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Allocate data */
   cdir_l        = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   active_l      = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   relax_weights = hypre_CTAlloc(HYPRE_Real, max_levels, HYPRE_MEMORY_HOST);

   /* Force relaxation on finest grid */
   hypre_SetIndex(coarsen, 1);
   for (l = 0; l < max_levels; l++)
   {
      /* Initialize min_dxyz */
      min_dxyz = 1;
      for (d = 0; d < ndim; d++)
      {
         min_dxyz += dxyz[d];
      }

      /* Determine cdir */
      cdir = -1;
      alpha = 0.0;
      for (d = 0; d < ndim; d++)
      {
         if ((hypre_BoxIMaxD(cbox, d) > hypre_BoxIMinD(cbox, d)) &&
             (dxyz[d] < min_dxyz))
         {
            min_dxyz = dxyz[d];
            cdir = d;
         }
         alpha += 1.0 / (dxyz[d] * dxyz[d]);
      }
      relax_weights[l] = 1.0;

      /* If it's possible to coarsen, change relax_weights */
      beta = 0.0;
      if (cdir != -1)
      {
         if (dxyz_flag || (ndim == 1))
         {
            relax_weights[l] = 2.0 / 3.0;
         }
         else
         {
            for (d = 0; d < ndim; d++)
            {
               if (d != cdir)
               {
                  beta += 1.0 / (dxyz[d] * dxyz[d]);
               }
            }

            /* determine level Jacobi weights */
            relax_weights[l] = 2.0 / (3.0 - beta / alpha);
         }

         /*    don't coarsen if a periodic direction and not divisible by 2
            or don't coarsen if we've reached max_levels*/
         if (((periodic[cdir]) && (periodic[cdir] % 2)) || l == (max_levels - 1))
         {
            cdir = -1;
         }
      }

      /* stop coarsening */
      if (cdir == -1)
      {
         active_l[l] = 1; /* forces relaxation on coarsest grid */
         break;
      }

      cdir_l[l] = cdir;

      if (hypre_IndexD(coarsen, cdir) != 0)
      {
         /* coarsened previously in this direction, relax level l */
         active_l[l] = 1;
         hypre_SetIndex(coarsen, 0);
      }
      else
      {
         active_l[l] = 0;
      }
      hypre_IndexD(coarsen, cdir) = 1;

      /* set cindex and stride */
      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetStride(cdir, stride);

      /* update dxyz and coarsen cbox*/
      dxyz[cdir] *= 2;
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride, hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride, hypre_BoxIMax(cbox));

      /* update periodic */
      periodic[cdir] /= 2;
   }
   *num_levels = l + 1;

   *cdir_l_ptr        = cdir_l;
   *active_l_ptr      = active_l;
   *relax_weights_ptr = relax_weights;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* TODO (VPM): Incorporate the specialized code below for computing Dxyz */
#if 0

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz_CS( HYPRE_Int           i,
                          hypre_StructMatrix *A,
                          HYPRE_Real         *cxyz,
                          HYPRE_Real         *sqcxyz)
{
   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
   HYPRE_Int              Ai;
   HYPRE_Real            *Ap;
   HYPRE_Int              constant_coefficient;
   HYPRE_Real             tcx, tcy, tcz;
   HYPRE_Real             Adiag = 0, diag;
   HYPRE_Int              Astenc, sdiag = 0;
   HYPRE_Int              si;
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(A);

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   Ai = hypre_CCBoxIndexRank( A_dbox, start );
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   /* find diagonal stencil entry */
   for (si = 0; si < stencil_size; si++)
   {
      if ((hypre_IndexD(stencil_shape[si], 0) == 0) &&
          (hypre_IndexD(stencil_shape[si], 1) == 0) &&
          (hypre_IndexD(stencil_shape[si], 2) == 0))
      {
         sdiag = si;
         break;
      }
   }

   tcx = cxyz[0];
   tcy = cxyz[1];
   tcz = cxyz[2];

   /* get sign of diagonal */
   Ap = hypre_StructMatrixBoxData(A, i, sdiag);
   if (constant_coefficient == 1)
   {
      Adiag = Ap[Ai];
   }
   else if (constant_coefficient == 2)
   {
      hypre_TMemcpy(&Adiag, &Ap[Ai], HYPRE_Real, 1, HYPRE_MEMORY_HOST, memory_location);
   }

   diag = 1.0;
   if (Adiag < 0)
   {
      diag = -1.0;
   }

   for (si = 0; si < stencil_size; si++)
   {
      Ap = hypre_StructMatrixBoxData(A, i, si);

      /* x-direction */
      Astenc = hypre_IndexD(stencil_shape[si], 0);
      if (Astenc)
      {
         tcx -= Ap[Ai] * diag;
      }

      /* y-direction */
      Astenc = hypre_IndexD(stencil_shape[si], 1);
      if (Astenc)
      {
         tcy -= Ap[Ai] * diag;
      }

      /* z-direction */
      Astenc = hypre_IndexD(stencil_shape[si], 2);
      if (Astenc)
      {
         tcz -= Ap[Ai] * diag;
      }
   }

   cxyz[0] += tcx;
   cxyz[1] += tcy;
   cxyz[2] += tcz;

   sqcxyz[0] += tcx * tcx;
   sqcxyz[1] += tcy * tcy;
   sqcxyz[2] += tcz * tcz;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz_SS5( HYPRE_Int           bi,
                           hypre_StructMatrix *A,
                           HYPRE_Real         *cxyz,
                           HYPRE_Real         *sqcxyz)
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
   hypre_Box             *A_dbox;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;
   hypre_Index            index;
   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int              data_location = hypre_StructGridDataLocation(
                                             hypre_StructMatrixGrid(A) );
#endif

   hypre_SetIndex3(stride, 1, 1, 1);
   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   compute_box = hypre_BoxArrayBox(compute_boxes, bi);
   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), bi);
   start  = hypre_BoxIMin(compute_box);
   hypre_BoxGetStrideSize(compute_box, stride, loop_size);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient (diag)
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index,  0,  0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1,  0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index,  1,  0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index,  0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index,  0,  1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   // FIXME TODO HOW TO DO KOKKOS (WM: and SYCL) IN ONE BOXLOOP ?
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)

   HYPRE_Real cxb = cxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai]);
      cxb += tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cxb)

   HYPRE_Real cyb = cxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cn[Ai] + a_cs[Ai]);
      cyb += tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cyb)

   HYPRE_Real sqcxb = sqcxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai]);
      sqcxb += tcx * tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcxb)

   HYPRE_Real sqcyb = sqcxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cn[Ai] + a_cs[Ai]);
      sqcyb += tcy * tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcyb)

#else // #if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_RAJA)
   ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> cxb(cxyz[0]), cyb(cxyz[1]), sqcxb(sqcxyz[0]),
             sqcyb(sqcxyz[1]);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_double4 d4(cxyz[0], cxyz[1], sqcxyz[0], sqcxyz[1]);
   ReduceSum<HYPRE_double4> sum4(d4);
#else
   HYPRE_Real cxb, cyb, sqcxb, sqcyb;
   cxb = cxyz[0];
   cyb = cxyz[1];
   sqcxb = sqcxyz[0];
   sqcyb = sqcxyz[1];
#endif

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#ifdef HYPRE_USING_DEVICE_OPENMP
#define HYPRE_BOX_REDUCTION map(tofrom:cxb,cyb,sqcxb,sqcyb) reduction(+:cxb,cyb,sqcxb,sqcyb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cxb,cyb,sqcxb,sqcyb)
#endif

#define DEVICE_VAR is_device_ptr(a_cc,a_cw,a_ce,a_cn,a_cs)
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sum4);
   {
      HYPRE_Real tcx, tcy;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;

      tcx = -diag * (a_cw[Ai] + a_ce[Ai]);
      tcy = -diag * (a_cn[Ai] + a_cs[Ai]);

#if !defined(HYPRE_USING_RAJA) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
      HYPRE_double4 tmp(tcx, tcy, tcx * tcx, tcy * tcy);
      sum4 += tmp;
#else
      cxb += tcx;
      cyb += tcy;
      sqcxb += tcx * tcx;
      sqcyb += tcy * tcy;
#endif
   }
   hypre_BoxLoop1ReductionEnd(Ai, sum4)
#undef DEVICE_VAR

#endif /* kokkos */

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
   HYPRE_double4 tmp = (HYPRE_double4) sum4;
   cxyz[0]   = tmp.x;
   cxyz[1]   = tmp.y;
   sqcxyz[0] = tmp.z;
   sqcxyz[1] = tmp.w;
   //printf("1: %e %e %e %e\n", cxyz[0], cxyz[1], sqcxyz[0], sqcxyz[1]);
#else
   cxyz[0]   = (HYPRE_Real) cxb;
   cxyz[1]   = (HYPRE_Real) cyb;
   sqcxyz[0] = (HYPRE_Real) sqcxb;
   sqcxyz[1] = (HYPRE_Real) sqcyb;
#endif

   cxyz[2]   = 0;
   sqcxyz[2] = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz_SS9( HYPRE_Int bi,
                           hypre_StructMatrix *A,
                           HYPRE_Real         *cxyz,
                           HYPRE_Real         *sqcxyz)
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
   hypre_Box             *A_dbox;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;
   hypre_Index            index;
   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real            *a_csw, *a_cse, *a_cne, *a_cnw;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int              data_location = hypre_StructGridDataLocation(
                                             hypre_StructMatrixGrid(A) );
#endif

   hypre_SetIndex3(stride, 1, 1, 1);
   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   compute_box = hypre_BoxArrayBox(compute_boxes, bi);
   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), bi);
   start  = hypre_BoxIMin(compute_box);
   hypre_BoxGetStrideSize(compute_box, stride, loop_size);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, -1, 0);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, -1, 0);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 1, 0);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   // FIXME TODO HOW TO DO KOKKOS IN ONE BOXLOOP ?
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)

   HYPRE_Real cxb = cxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      cxb += tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cxb)

   HYPRE_Real cyb = cxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cs[Ai] + a_cn[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      cyb += tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cyb)

   HYPRE_Real sqcxb = sqcxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      sqcxb += tcx * tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcxb)

   HYPRE_Real sqcyb = sqcxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cs[Ai] + a_cn[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      sqcyb += tcy * tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcyb)

#else /* kokkos */

#if defined(HYPRE_USING_RAJA)
   ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> cxb(cxyz[0]), cyb(cxyz[1]), sqcxb(sqcxyz[0]),
             sqcyb(sqcxyz[1]);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_double4 d4(cxyz[0], cxyz[1], sqcxyz[0], sqcxyz[1]);
   ReduceSum<HYPRE_double4> sum4(d4);
#else
   HYPRE_Real cxb, cyb, sqcxb, sqcyb;
   cxb = cxyz[0];
   cyb = cxyz[1];
   sqcxb = sqcxyz[0];
   sqcyb = sqcxyz[1];

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#ifdef HYPRE_USING_DEVICE_OPENMP
#define HYPRE_BOX_REDUCTION map(tofrom:cxb,cyb,sqcxb,sqcyb) reduction(+:cxb,cyb,sqcxb,sqcyb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cxb,cyb,sqcxb,sqcyb)
#endif

#endif

#define DEVICE_VAR is_device_ptr(a_cc,a_cw,a_ce,a_csw,a_cse,a_cnw,a_cne,a_cs,a_cn)
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sum4)
   {
      HYPRE_Real tcx, tcy;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;

      tcx = -diag * (a_cw[Ai] + a_ce[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcy = -diag * (a_cs[Ai] + a_cn[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);

#if !defined(HYPRE_USING_RAJA) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
      HYPRE_double4 tmp(tcx, tcy, tcx * tcx, tcy * tcy);
      sum4 += tmp;
#else
      cxb += tcx;
      cyb += tcy;
      sqcxb += tcx * tcx;
      sqcyb += tcy * tcy;
#endif
   }
   hypre_BoxLoop1ReductionEnd(Ai, sum4)
#undef DEVICE_VAR

#endif /* kokkos */

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
   HYPRE_double4 tmp = (HYPRE_double4) sum4;
   cxyz[0]   = tmp.x;
   cxyz[1]   = tmp.y;
   sqcxyz[0] = tmp.z;
   sqcxyz[1] = tmp.w;
#else
   cxyz[0]   = (HYPRE_Real) cxb;
   cxyz[1]   = (HYPRE_Real) cyb;
   sqcxyz[0] = (HYPRE_Real) sqcxb;
   sqcxyz[1] = (HYPRE_Real) sqcyb;
#endif

   cxyz[2]   = 0;
   sqcxyz[2] = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz_SS7( HYPRE_Int           bi,
                           hypre_StructMatrix *A,
                           HYPRE_Real         *cxyz,
                           HYPRE_Real         *sqcxyz)
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
   hypre_Box             *A_dbox;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;
   hypre_Index            index;
   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int              data_location = hypre_StructGridDataLocation(
                                             hypre_StructMatrixGrid(A) );
#endif

   hypre_SetIndex3(stride, 1, 1, 1);
   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   compute_box = hypre_BoxArrayBox(compute_boxes, bi);
   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), bi);
   start  = hypre_BoxIMin(compute_box);
   hypre_BoxGetStrideSize(compute_box, stride, loop_size);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 0, 1);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 0, -1);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   // FIXME TODO HOW TO DO KOKKOS IN ONE BOXLOOP ?
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)

   HYPRE_Real cxb = cxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai]);
      cxb += tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cxb)

   HYPRE_Real cyb = cxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cs[Ai] + a_cn[Ai]);
      cyb += tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cyb)

   HYPRE_Real czb = cxyz[2];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, czb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcz = -diag * (a_ac[Ai] + a_bc[Ai]);
      czb += tcz;
   }
   hypre_BoxLoop1ReductionEnd(Ai, czb)

   HYPRE_Real sqcxb = sqcxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai]);
      sqcxb += tcx * tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcxb)

   HYPRE_Real sqcyb = sqcxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cs[Ai] + a_cn[Ai]);
      sqcyb += tcy * tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcyb)

   HYPRE_Real sqczb = sqcxyz[2];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqczb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcz = -diag * (a_ac[Ai] + a_bc[Ai]);
      sqczb += tcz * tcz;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqczb)

#else /* kokkos */

#if defined(HYPRE_USING_RAJA)
   ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> cxb(cxyz[0]), cyb(cxyz[1]), czb(cxyz[2]),
             sqcxb(sqcxyz[0]), sqcyb(sqcxyz[1]), sqczb(sqcxyz[2]);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_double6 d6(cxyz[0], cxyz[1], cxyz[2], sqcxyz[0], sqcxyz[1], sqcxyz[2]);
   ReduceSum<HYPRE_double6> sum6(d6);
#else
   HYPRE_Real cxb, cyb, czb, sqcxb, sqcyb, sqczb;
   cxb = cxyz[0];
   cyb = cxyz[1];
   czb = cxyz[2];
   sqcxb = sqcxyz[0];
   sqcyb = sqcxyz[1];
   sqczb = sqcxyz[2];

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#ifdef HYPRE_USING_DEVICE_OPENMP
#define HYPRE_BOX_REDUCTION map(tofrom:cxb,cyb,czb,sqcxb,sqcyb,sqczb) reduction(+:cxb,cyb,czb,sqcxb,sqcyb,sqczb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cxb,cyb,czb,sqcxb,sqcyb,sqczb)
#endif

#endif

#define DEVICE_VAR is_device_ptr(a_cc,a_cw,a_ce,a_cs,a_cn,a_ac,a_bc)
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sum6)
   {
      HYPRE_Real tcx, tcy, tcz;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;

      tcx = -diag * (a_cw[Ai] + a_ce[Ai]);
      tcy = -diag * (a_cs[Ai] + a_cn[Ai]);
      tcz = -diag * (a_ac[Ai] + a_bc[Ai]);
#if !defined(HYPRE_USING_RAJA) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
      HYPRE_double6 tmp(tcx, tcy, tcz, tcx * tcx, tcy * tcy, tcz * tcz);
      sum6 += tmp;
#else
      cxb += tcx;
      cyb += tcy;
      czb += tcz;
      sqcxb += tcx * tcx;
      sqcyb += tcy * tcy;
      sqczb += tcz * tcz;
#endif
   }
   hypre_BoxLoop1ReductionEnd(Ai, sum6)
#undef DEVICE_VAR

#endif /* kokkos */

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
   HYPRE_double6 tmp = (HYPRE_double6) sum6;
   cxyz[0]   = tmp.x;
   cxyz[1]   = tmp.y;
   cxyz[2]   = tmp.z;
   sqcxyz[0] = tmp.w;
   sqcxyz[1] = tmp.u;
   sqcxyz[2] = tmp.v;
#else
   cxyz[0]   = (HYPRE_Real) cxb;
   cxyz[1]   = (HYPRE_Real) cyb;
   cxyz[2]   = (HYPRE_Real) czb;
   sqcxyz[0] = (HYPRE_Real) sqcxb;
   sqcxyz[1] = (HYPRE_Real) sqcyb;
   sqcxyz[2] = (HYPRE_Real) sqczb;
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz_SS19( HYPRE_Int           bi,
                            hypre_StructMatrix *A,
                            HYPRE_Real         *cxyz,
                            HYPRE_Real         *sqcxyz)
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
   hypre_Box             *A_dbox;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;
   hypre_Index            index;
   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   HYPRE_Real            *a_csw, *a_cse, *a_cne, *a_cnw;
   HYPRE_Real            *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int              data_location = hypre_StructGridDataLocation(
                                             hypre_StructMatrixGrid(A) );
#endif

   hypre_SetIndex3(stride, 1, 1, 1);
   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   compute_box = hypre_BoxArrayBox(compute_boxes, bi);
   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), bi);
   start  = hypre_BoxIMin(compute_box);
   hypre_BoxGetStrideSize(compute_box, stride, loop_size);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 0, 1);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 0, -1);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, 0, 1);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, 1);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, 1);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, 1);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 0, -1);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, -1);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, -1);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, -1);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, -1, 0);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, -1, 0);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 1, 0);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   // FIXME TODO HOW TO DO KOKKOS IN ONE BOXLOOP ?
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)

   HYPRE_Real cxb = cxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai] + a_aw[Ai] + a_ae[Ai] + a_bw[Ai] + a_be[Ai] +
                                a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      cxb += tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cxb)

   HYPRE_Real cyb = cxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cs[Ai] + a_cn[Ai] + a_an[Ai] + a_as[Ai] + a_bn[Ai] + a_bs[Ai] +
                                a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      cyb += tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cyb)

   HYPRE_Real czb = cxyz[2];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, czb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcz = -diag * (a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] + a_an[Ai] + a_as[Ai] +
                                a_bw[Ai]  + a_be[Ai] +  a_bn[Ai] +  a_bs[Ai]);
      czb += tcz;
   }
   hypre_BoxLoop1ReductionEnd(Ai, czb)

   HYPRE_Real sqcxb = sqcxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcxb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcx = -diag * (a_cw[Ai] + a_ce[Ai] + a_aw[Ai] + a_ae[Ai] + a_bw[Ai] + a_be[Ai] +
                                a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      sqcxb += tcx * tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcxb)

   HYPRE_Real sqcyb = sqcxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcyb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcy = -diag * (a_cs[Ai] + a_cn[Ai] + a_an[Ai] + a_as[Ai] + a_bn[Ai] + a_bs[Ai] +
                                a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      sqcyb += tcy * tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcyb)

   HYPRE_Real sqczb = sqcxyz[2];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqczb)
   {
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      HYPRE_Real tcz = -diag * (a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] + a_an[Ai] + a_as[Ai] +
                                a_bw[Ai]  + a_be[Ai] +  a_bn[Ai] +  a_bs[Ai]);
      sqczb += tcz * tcz;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqczb)

#else /* kokkos */

#if defined(HYPRE_USING_RAJA)
   ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> cxb(cxyz[0]), cyb(cxyz[1]), czb(cxyz[2]),
             sqcxb(sqcxyz[0]), sqcyb(sqcxyz[1]), sqczb(sqcxyz[2]);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_double6 d6(cxyz[0], cxyz[1], cxyz[2], sqcxyz[0], sqcxyz[1], sqcxyz[2]);
   ReduceSum<HYPRE_double6> sum6(d6);
#else
   HYPRE_Real cxb, cyb, czb, sqcxb, sqcyb, sqczb;
   cxb = cxyz[0];
   cyb = cxyz[1];
   czb = cxyz[2];
   sqcxb = sqcxyz[0];
   sqcyb = sqcxyz[1];
   sqczb = sqcxyz[2];

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#ifdef HYPRE_USING_DEVICE_OPENMP
#define HYPRE_BOX_REDUCTION map(tofrom:cxb,cyb,czb,sqcxb,sqcyb,sqczb) reduction(+:cxb,cyb,czb,sqcxb,sqcyb,sqczb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cxb,cyb,czb,sqcxb,sqcyb,sqczb)
#endif

#endif

#define DEVICE_VAR is_device_ptr(a_cc,a_cw,a_ce,a_aw,a_ae,a_bw,a_be,a_csw,a_cse,a_cnw,a_cne,a_cs,a_cn,a_an,a_as,a_bn,a_bs,a_ac,a_bc)
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sum6)
   {
      HYPRE_Real tcx, tcy, tcz;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;

      tcx = -diag * (a_cw[Ai] + a_ce[Ai] + a_aw[Ai] + a_ae[Ai] + a_bw[Ai] + a_be[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcy = -diag * (a_cs[Ai] + a_cn[Ai] + a_an[Ai] + a_as[Ai] + a_bn[Ai] + a_bs[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcz = -diag * (a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] + a_an[Ai] + a_as[Ai] +  a_bw[Ai]  +
                     a_be[Ai] +  a_bn[Ai] +  a_bs[Ai]);

#if !defined(HYPRE_USING_RAJA) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
      HYPRE_double6 tmp(tcx, tcy, tcz, tcx * tcx, tcy * tcy, tcz * tcz);
      sum6 += tmp;
#else
      cxb += tcx;
      cyb += tcy;
      czb += tcz;
      sqcxb += tcx * tcx;
      sqcyb += tcy * tcy;
      sqczb += tcz * tcz;
#endif
   }
   hypre_BoxLoop1ReductionEnd(Ai, sum6)
#undef DEVICE_VAR

#endif /* kokkos */

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
   HYPRE_double6 tmp = (HYPRE_double6) sum6;
   cxyz[0]   = tmp.x;
   cxyz[1]   = tmp.y;
   cxyz[2]   = tmp.z;
   sqcxyz[0] = tmp.w;
   sqcxyz[1] = tmp.u;
   sqcxyz[2] = tmp.v;
#else
   cxyz[0]   = (HYPRE_Real) cxb;
   cxyz[1]   = (HYPRE_Real) cyb;
   cxyz[2]   = (HYPRE_Real) czb;
   sqcxyz[0] = (HYPRE_Real) sqcxb;
   sqcxyz[1] = (HYPRE_Real) sqcyb;
   sqcxyz[2] = (HYPRE_Real) sqczb;
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz_SS27( HYPRE_Int           bi,
                            hypre_StructMatrix *A,
                            HYPRE_Real         *cxyz,
                            HYPRE_Real         *sqcxyz)
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
   hypre_Box             *A_dbox;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;
   hypre_Index            index;

   HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   HYPRE_Real            *a_csw, *a_cse, *a_cne, *a_cnw;
   HYPRE_Real            *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real            *a_asw, *a_ase, *a_ane, *a_anw, *a_bsw, *a_bse, *a_bne, *a_bnw;

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int              data_location = hypre_StructGridDataLocation(
                                             hypre_StructMatrixGrid(A) );
#endif

   hypre_SetIndex3(stride, 1, 1, 1);
   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   compute_box = hypre_BoxArrayBox(compute_boxes, bi);
   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), bi);
   start  = hypre_BoxIMin(compute_box);
   hypre_BoxGetStrideSize(compute_box, stride, loop_size);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, 0, 0, 0);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 0, 0);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, 0);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, 0);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, 0);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 0, 1);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 0, -1);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, 0, 1);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, 1);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, 1);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, 1);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 0, -1);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 0, -1);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, -1, -1);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 0, 1, -1);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, -1, 0);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, -1, 0);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 1, 0);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point fine grid operator:
    *
    * a_asw is pointer for southwest coefficient in plane above
    * a_ase is pointer for southeast coefficient in plane above
    * a_anw is pointer for northwest coefficient in plane above
    * a_ane is pointer for northeast coefficient in plane above
    * a_bsw is pointer for southwest coefficient in plane below
    * a_bse is pointer for southeast coefficient in plane below
    * a_bnw is pointer for northwest coefficient in plane below
    * a_bne is pointer for northeast coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index, -1, -1, 1);
   a_asw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, -1, 1);
   a_ase = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 1, 1);
   a_anw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 1, 1);
   a_ane = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, -1, -1);
   a_bsw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, -1, -1);
   a_bse = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, -1, 1, -1);
   a_bnw = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   hypre_SetIndex3(index, 1, 1, -1);
   a_bne = hypre_StructMatrixExtractPointerByIndex(A, bi, index);

   // FIXME TODO HOW TO DO KOKKOS IN ONE BOXLOOP ?
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)

   HYPRE_Real cxb = cxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cxb)
   {
      HYPRE_Real tcx = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      tcx -= diag * (a_cw[Ai]  + a_ce[Ai]  +  a_aw[Ai] +  a_ae[Ai] +  a_bw[Ai] +  a_be[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcx -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);
      cxb += tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cxb)

   HYPRE_Real cyb = cxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, cyb)
   {
      HYPRE_Real tcy = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      tcy -= diag * (a_cs[Ai]  + a_cn[Ai]  +  a_an[Ai] +  a_as[Ai] +  a_bn[Ai] +  a_bs[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcy -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);
      cyb += tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, cyb)

   HYPRE_Real czb = cxyz[2];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, czb)
   {
      HYPRE_Real tcz = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      tcz -= diag * (a_ac[Ai]  +  a_bc[Ai] +  a_aw[Ai] +  a_ae[Ai] +  a_an[Ai] +  a_as[Ai] +  a_bw[Ai] +
                     a_be[Ai] + a_bn[Ai] + a_bs[Ai]);
      tcz -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);
      czb += tcz;
   }
   hypre_BoxLoop1ReductionEnd(Ai, czb)

   HYPRE_Real sqcxb = sqcxyz[0];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcxb)
   {
      HYPRE_Real tcx = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      tcx -= diag * (a_cw[Ai]  + a_ce[Ai]  +  a_aw[Ai] +  a_ae[Ai] +  a_bw[Ai] +  a_be[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcx -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);
      sqcxb += tcx * tcx;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcxb)

   HYPRE_Real sqcyb = sqcxyz[1];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqcyb);
   {
      HYPRE_Real tcy = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;

      tcy -= diag * (a_cs[Ai]  + a_cn[Ai] + a_an[Ai]  + a_as[Ai]  +
                     a_bn[Ai]  + a_bs[Ai] + a_csw[Ai] + a_cse[Ai] +
                     a_cnw[Ai] + a_cne[Ai]);
      tcy -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] +
                     a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] + a_bne[Ai]);

      sqcyb += tcy * tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcyb);

   HYPRE_Real sqczb = sqcxyz[2];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqczb)
   {
      HYPRE_Real tcz = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;

      tcz -= diag * (a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] +
                     a_an[Ai] + a_as[Ai] + a_bw[Ai] + a_be[Ai] +
                     a_bn[Ai] + a_bs[Ai]);
      tcz -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] +
                     a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] + a_bne[Ai]);

      sqczb += tcz * tcz;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqczb)

#else /* kokkos */

#if defined(HYPRE_USING_RAJA)
   ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> cxb(cxyz[0]), cyb(cxyz[1]), czb(cxyz[2]),
             sqcxb(sqcxyz[0]), sqcyb(sqcxyz[1]), sqczb(sqcxyz[2]);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_double6 d6(cxyz[0], cxyz[1], cxyz[2], sqcxyz[0], sqcxyz[1], sqcxyz[2]);
   ReduceSum<HYPRE_double6> sum6(d6);
#else
   HYPRE_Real cxb, cyb, czb, sqcxb, sqcyb, sqczb;
   cxb = cxyz[0];
   cyb = cxyz[1];
   czb = cxyz[2];
   sqcxb = sqcxyz[0];
   sqcyb = sqcxyz[1];
   sqczb = sqcxyz[2];

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#ifdef HYPRE_USING_DEVICE_OPENMP
#define HYPRE_BOX_REDUCTION map(tofrom:cxb,cyb,czb,sqcxb,sqcyb,sqczb) reduction(+:cxb,cyb,czb,sqcxb,sqcyb,sqczb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cxb,cyb,czb,sqcxb,sqcyb,sqczb)
#endif

#endif

#define DEVICE_VAR is_device_ptr(a_cc,a_cw,a_ce,a_aw,a_ae,a_bw,a_be,a_csw,a_cse,a_cnw,a_cne,a_asw,a_ase,a_anw,a_ane,a_bsw,a_bse,a_bnw,a_bne,a_cs,a_cn,a_an,a_as,a_bn,a_bs,a_ac,a_bc)
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sum6)
   {
      HYPRE_Real tcx = 0.0, tcy = 0.0, tcz = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;

      tcx -= diag * (a_cw[Ai]  + a_ce[Ai]  +  a_aw[Ai] +  a_ae[Ai] +  a_bw[Ai] +  a_be[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcx -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);

      tcy -= diag * (a_cs[Ai]  + a_cn[Ai]  +  a_an[Ai] +  a_as[Ai] +  a_bn[Ai] +  a_bs[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcy -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);

      tcz -= diag * (a_ac[Ai]  +  a_bc[Ai] +  a_aw[Ai] +  a_ae[Ai] +  a_an[Ai] +  a_as[Ai] +  a_bw[Ai] +
                     a_be[Ai] + a_bn[Ai] + a_bs[Ai]);
      tcz -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);
#if !defined(HYPRE_USING_RAJA) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
      HYPRE_double6 tmp(tcx, tcy, tcz, tcx * tcx, tcy * tcy, tcz * tcz);
      sum6 += tmp;
#else
      cxb += tcx;
      cyb += tcy;
      czb += tcz;
      sqcxb += tcx * tcx;
      sqcyb += tcy * tcy;
      sqczb += tcz * tcz;
#endif
   }
   hypre_BoxLoop1ReductionEnd(Ai, sum6)
#undef DEVICE_VAR

#endif /* kokkos */

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && \
    (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
   HYPRE_double6 tmp = (HYPRE_double6) sum6;
   cxyz[0]   = tmp.x;
   cxyz[1]   = tmp.y;
   cxyz[2]   = tmp.z;
   sqcxyz[0] = tmp.w;
   sqcxyz[1] = tmp.u;
   sqcxyz[2] = tmp.v;
#else
   cxyz[0]   = (HYPRE_Real) cxb;
   cxyz[1]   = (HYPRE_Real) cyb;
   cxyz[2]   = (HYPRE_Real) czb;
   sqcxyz[0] = (HYPRE_Real) sqcxb;
   sqcxyz[1] = (HYPRE_Real) sqcyb;
   sqcxyz[2] = (HYPRE_Real) sqczb;
#endif

   return hypre_error_flag;
}


#endif
