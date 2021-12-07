/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#define hypre_PFMGSetFIndex(cdir, findex)       \
   {                                            \
      hypre_SetIndex(findex, 0);                \
      hypre_IndexD(findex, cdir) = 1;           \
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

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int             num_level_GPU = 0;
   HYPRE_MemoryLocation  data_location = HYPRE_MEMORY_DEVICE;
   HYPRE_Int             max_box_size  = 0;
   HYPRE_Int             device_level  = (pfmg_data -> devicelevel);
   HYPRE_Int             myrank;
   hypre_MPI_Comm_rank(comm, &myrank );
#endif

#if DEBUG
   char                  filename[255];
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

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

   /*-----------------------------------------------------
    * Modify the rap_type if red-black Gauss-Seidel is used.
    * Red-black gs is used only in the non-Galerkin case.
    *-----------------------------------------------------*/
   if (relax_type == 2 || relax_type == 3)   /* red-black gs */
   {
      (pfmg_data -> rap_type) = 1;
   }
   rap_type = (pfmg_data -> rap_type);

   grid_l = hypre_TAlloc(hypre_StructGrid *, num_levels, HYPRE_MEMORY_HOST);
   A_l    = hypre_TAlloc(hypre_StructMatrix *, num_levels, HYPRE_MEMORY_HOST);
   P_l    = hypre_TAlloc(hypre_StructMatrix *, num_levels-1, HYPRE_MEMORY_HOST);
   RT_l   = hypre_TAlloc(hypre_StructMatrix *, num_levels-1, HYPRE_MEMORY_HOST);
   b_l    = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   x_l    = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   tx_l   = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   r_l    = tx_l;
   e_l    = tx_l;

   hypre_StructGridRef(grid, &grid_l[0]);
   A_l[0] = hypre_StructMatrixRef(A);
   b_l[0] = hypre_StructVectorRef(b);
   x_l[0] = hypre_StructVectorRef(x);

   tx_l[0] = hypre_StructVectorCreate(comm, grid_l[0]);
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
      HYPRE_StructMatrixSetTranspose(RT_l[l], 1);
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
            hypre_StructMatrixRTtAP(RT_l[l], A_l[l], P_l[l], &A_l[l+1]);
         }
         else
         {
            hypre_StructMatrixPtAP(A_l[l], P_l[l], &A_l[l+1]);
         }
         hypre_StructGridRef(hypre_StructMatrixGrid(A_l[l+1]), &grid_l[l+1]);
      }
      else
      {
         /* RDF: The coarse grid should be computed in CreateRAPOp() */
         hypre_PFMGSetCIndex(cdir, cindex);
         hypre_PFMGSetStride(cdir, stride);
         hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l+1]);
         hypre_StructGridAssemble(grid_l[l+1]);

         A_l[l+1] = hypre_PFMGCreateRAPOp(RT_l[l], A_l[l], P_l[l], grid_l[l+1], cdir, rap_type);
         hypre_StructMatrixInitialize(A_l[l+1]);
         hypre_PFMGSetupRAPOp(RT_l[l], A_l[l], P_l[l], cdir, cindex, stride, rap_type, A_l[l+1]);
      }
      HYPRE_ANNOTATE_REGION_END("%s", "RAP");

//      /* RDF AP Debug */
//      hypre_StructAssumedPartitionPrint("zAP", hypre_BoxManAssumedPartition(
//                                           hypre_StructGridBoxMan(grid_l[l+1])));

      b_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(b_l[l+1], b_num_ghost);
      hypre_StructVectorInitialize(b_l[l+1]);
      hypre_StructVectorAssemble(b_l[l+1]);

      x_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(x_l[l+1], x_num_ghost);
      hypre_StructVectorInitialize(x_l[l+1]);
      hypre_StructVectorAssemble(x_l[l+1]);

      tx_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(tx_l[l+1], x_num_ghost);
      hypre_StructVectorInitialize(tx_l[l+1]);
      hypre_StructVectorAssemble(tx_l[l+1]);

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
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (l == num_level_GPU)
      {
         hypre_SetDeviceOff();
      }
#endif
      cdir = cdir_l[l];

      /* set up the interpolation routine */
      interp_data_l[l] = hypre_StructMatvecCreate();
      hypre_StructMatvecSetup(interp_data_l[l], P_l[l], x_l[l+1]);

      /* set up the restriction routine */
      restrict_data_l[l] = hypre_StructMatvecCreate();
      hypre_StructMatvecSetTranspose(restrict_data_l[l], 1);
      hypre_StructMatvecSetup(restrict_data_l[l], RT_l[l], r_l[l]);
   }

   /*-----------------------------------------------------
    * Check for zero diagonal on coarsest grid, occurs with
    * singular problems like full Neumann or full periodic.
    * Note that a processor with zero diagonal will set
    * active_l =0, other processors will not. This is OK
    * as we only want to avoid the division by zero on the
    * one processor which owns the single coarse grid point.
    *-----------------------------------------------------*/

   if (hypre_ZeroDiagonal(A_l[l]))
   {
      active_l[l] = 0;
   }

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (hypre_StructGridDataLocation(grid) != HYPRE_MEMORY_HOST)
   {
      hypre_SetDeviceOn();
   }
#endif
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
         maxwork = 4*num_levels;
         /* do sweeps proportional to the coarsest grid size */
         maxiter = hypre_min(maxwork, cmaxsize);
#if 0
         hypre_printf("maxwork = %d, cmaxsize = %d, maxiter = %d\n",
                      maxwork, cmaxsize, maxiter);
#endif
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
      (pfmg_data -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter+1, HYPRE_MEMORY_HOST);
      (pfmg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter+1, HYPRE_MEMORY_HOST);
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
         HYPRE_StructVectorCreate(comm, hypre_StructMatrixGrid(A_l[l+1]), &ones);
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
 * TODO: Change SerialBoxLoop to BoxLoop
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
               sqcxyz[d] += tcxyz[d]*tcxyz[d];
            }
         }
         else
         {
            for (d = 0; d < ndim; d++)
            {
               cxyz[d]   -= tcxyz[d];
               sqcxyz[d] += tcxyz[d]*tcxyz[d];
            }
         }
       }

      /* constant_coefficient==0, all coefficients vary with space */
      else
      {
         hypre_SerialBoxLoop1Begin(ndim, loop_size,
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
                  sqcxyz[d] += tcxyz[d]*tcxyz[d];
               }
            }
            else
            {
               for (d = 0; d < ndim; d++)
               {
                  cxyz[d]   -= tcxyz[d];
                  sqcxyz[d] += tcxyz[d]*tcxyz[d];
               }
            }
         }
         hypre_SerialBoxLoop1End(Ai);
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
                       HYPRE_Int          *dxyz_flag)
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
      mean[d] = cxyz[d]/(HYPRE_Real) global_size;
      deviation[d] = sqcxyz[d]/(HYPRE_Real) global_size;
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
      HYPRE_Real max_anisotropy = HYPRE_REAL_MAX/1000;
      if (cxyz[d] > (cxyz_max/max_anisotropy))
      {
         cxyz[d] /= cxyz_max;
         dxyz[d] = sqrt(1.0 / cxyz[d]);
      }
      else
      {
         dxyz[d] = sqrt(max_anisotropy);
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
      deviation[d] -= mean[d]*mean[d];
      if ( deviation[d] > 0.1*(mean[d]*mean[d]) )
      {
         *dxyz_flag = 1;
         break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns 1 if there is a diagonal coefficient that is zero,
 * otherwise returns 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ZeroDiagonal( hypre_StructMatrix *A )
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
                   HYPRE_Int     *num_levels)
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
         alpha += 1.0/(dxyz[d]*dxyz[d]);
      }
      relax_weights[l] = 1.0;

      /* If it's possible to coarsen, change relax_weights */
      beta = 0.0;
      if (cdir != -1)
      {
         if (dxyz_flag || (ndim == 1))
         {
            relax_weights[l] = 2.0/3.0;
         }
         else
         {
            for (d = 0; d < ndim; d++)
            {
               if (d != cdir)
               {
                  beta += 1.0/(dxyz[d]*dxyz[d]);
               }
            }

            /* determine level Jacobi weights */
            relax_weights[l] = 2.0/(3.0 - beta/alpha);
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

      /* set cindex, findex, and stride */
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
