/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "sys_pfmg.h"

#define DEBUG 0

#define hypre_PFMGSetCIndex(cdir, cindex)       \
   {                                            \
      hypre_SetIndex3(cindex, 0, 0, 0);          \
      hypre_IndexD(cindex, cdir) = 0;           \
   }

#define hypre_PFMGSetStride(cdir, stride)       \
   {                                            \
      hypre_SetIndex3(stride, 1, 1, 1);          \
      hypre_IndexD(stride, cdir) = 2;           \
   }

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetup( void                 *sys_pfmg_vdata,
                    hypre_SStructMatrix  *A_in,
                    hypre_SStructVector  *b_in,
                    hypre_SStructVector  *x_in )
{
   hypre_SysPFMGData      *sys_pfmg_data = (hypre_SysPFMGData *) sys_pfmg_vdata;
   hypre_SStructPMatrix   *A             = hypre_SStructMatrixPMatrix(A_in, 0);
   hypre_SStructPVector   *b             = hypre_SStructVectorPVector(b_in, 0);
   hypre_SStructPVector   *x             = hypre_SStructVectorPVector(x_in, 0);

   MPI_Comm                comm              = (sys_pfmg_data -> comm);
   HYPRE_Int               relax_type        = (sys_pfmg_data -> relax_type);
   HYPRE_Int               usr_jacobi_weight = (sys_pfmg_data -> usr_jacobi_weight);
   HYPRE_Real              jacobi_weight     = (sys_pfmg_data -> jacobi_weight);
   HYPRE_Int               skip_relax        = (sys_pfmg_data -> skip_relax);
   HYPRE_Real             *dxyz              = (sys_pfmg_data -> dxyz);
   HYPRE_Int               max_iter          = (sys_pfmg_data -> max_iter);
   HYPRE_Int               max_levels;
   HYPRE_Int               num_levels;

   hypre_Index             cindex;
   hypre_Index             stride;
   hypre_Index             periodic;

   hypre_StructMatrix     *smatrix;
   hypre_SStructPMatrix  **A_l;
   hypre_SStructPMatrix  **P_l;
   hypre_SStructPMatrix  **RT_l;
   hypre_SStructPVector  **b_l;
   hypre_SStructPVector  **x_l;
   hypre_SStructPGrid    **grid_l;
   HYPRE_Int              *cdir_l;
   HYPRE_Int              *active_l;

   /* temp vectors */
   hypre_SStructPVector  **tx_l;
   hypre_SStructPVector  **r_l;
   hypre_SStructPVector  **e_l;

   void                  **relax_data_l;
   void                  **matvec_data_l;
   void                  **restrict_data_l;
   void                  **interp_data_l;

   hypre_SStructPGrid    *grid;
   hypre_StructGrid      *sgrid;
   HYPRE_Int              dim;
   HYPRE_Int              full_periodic;

   hypre_Box             *cbox;

   HYPRE_Real            *relax_weights;
   HYPRE_Real             alpha, beta;
   HYPRE_Int              dxyz_flag;
   HYPRE_Real             min_dxyz;
   HYPRE_Int              cdir, periodic, cmaxsize;
   HYPRE_Int              d, l;
   HYPRE_Int              i;
   HYPRE_Real             var_dxyz[3];
   HYPRE_Int              nvars;

#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid  = hypre_SStructPMatrixPGrid(A);
   sgrid = hypre_SStructPGridSGrid(grid, 0);
   nvars = hypre_SStructPGridNVars(grid);

   /* Initialize periodic */
   hypre_CopyIndex(hypre_StructGridPeriodic(sgrid), periodic);

   /* Compute a new max_levels value based on the grid */
   hypre_PFMGComputeMaxLevels(sgrid, &max_levels);
   if ((sys_pfmg_data -> max_levels) > 0)
   {
      max_levels = hypre_min(max_levels, (sys_pfmg_data -> max_levels));
   }
   (sys_pfmg_data -> max_levels) = max_levels;

   /* compute dxyz */
   for (i = 0; i < nvars; i++)
   {
      smatrix = hypre_SStructPMatrixSMatrix(A, i, i);
      hypre_PFMGComputeDxyz(smatrix, var_dxyz, &dxyz_flag);

      for (d = 0; d < 3; d++)
      {
         dxyz[d] += var_dxyz[d];
      }
   }

   /* Run coarsening */
   cbox = hypre_BoxClone(hypre_StructGridBoundingBox(sgrid));
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

   (sys_pfmg_data -> num_levels) = num_levels;
   (sys_pfmg_data -> cdir_l)     = cdir_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   grid_l = hypre_TAlloc(hypre_SStructPGrid *, max_levels, HYPRE_MEMORY_HOST);
   A_l  = hypre_TAlloc(hypre_SStructPMatrix *, num_levels, HYPRE_MEMORY_HOST);
   P_l  = hypre_TAlloc(hypre_SStructPMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   RT_l = hypre_TAlloc(hypre_SStructPMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   b_l  = hypre_TAlloc(hypre_SStructPVector *, num_levels, HYPRE_MEMORY_HOST);
   x_l  = hypre_TAlloc(hypre_SStructPVector *, num_levels, HYPRE_MEMORY_HOST);
   tx_l = hypre_TAlloc(hypre_SStructPVector *, num_levels, HYPRE_MEMORY_HOST);
   r_l  = tx_l;
   e_l  = tx_l;

   hypre_SStructPGridRef(grid, &grid_l[0]);
   hypre_SStructPMatrixRef(A, &A_l[0]);
   hypre_SStructPVectorRef(b, &b_l[0]);
   hypre_SStructPVectorRef(x, &x_l[0]);

   hypre_SStructPVectorCreate(comm, grid_l[0], &tx_l[0]);
   hypre_SStructPVectorInitialize(tx_l[0]);
   hypre_SStructPVectorAssemble(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetStride(cdir, stride);

      /* set up interpolation and restriction operators */
      P_l[l]  = hypre_SysPFMGCreateInterpOp(A_l[l], cdir, stride);
      RT_l[l] = P_l[l];
#if 0 /* TODO: Allow RT != P */
      if (nonsymmetric_cycle)
      {
         RT_l[l] = hypre_SysPFMGCreateRestrictOp(A_l[l], cdir, stride);
      }
#endif
      HYPRE_SStructPMatrixSetTranspose(RT_l[l], 1);
      hypre_SStructPMatrixInitialize(P_l[l]);
      hypre_SysPFMGSetupInterpOp(P_l[l], A_l[l], cdir);
#if 0 /* TODO: Allow RT != P */
      if (nonsymmetric_cycle)
      {
         hypre_StructMatrixInitialize(RT_l[l]);
         hypre_SysPFMGSetupRestrictOp(RT_l[l], A_l[l], cdir);
      }
#endif

      if (RT_l[l] != P_l[l])
      {
         /* If restriction is not the same as interpolation, compute RAP */
         hypre_SStructPMatrixRTtAP(RT_l[l], A_l[l], P_l[l], &A_l[l + 1]);
      }
      else
      {
         hypre_SStructPMatrixPtAP(A_l[l], P_l[l], &A_l[l + 1]);
      }
      hypre_SStructPGridRef(hypre_SStructPMatrixPGrid(A_l[l + 1]), &grid_l[l + 1]);

      hypre_SStructPVectorCreate(comm, grid_l[l + 1], &b_l[l + 1]);
      hypre_SStructPVectorInitialize(b_l[l + 1]);
      hypre_SStructPVectorAssemble(b_l[l + 1]);

      hypre_SStructPVectorCreate(comm, grid_l[l + 1], &x_l[l + 1]);
      hypre_SStructPVectorInitialize(x_l[l + 1]);
      hypre_SStructPVectorAssemble(x_l[l + 1]);

      hypre_SStructPVectorCreate(comm, grid_l[l + 1], &tx_l[l + 1]);
      hypre_SStructPVectorInitialize(tx_l[l + 1]);
      hypre_SStructPVectorAssemble(tx_l[l + 1]);
   }

   (sys_pfmg_data -> grid_l)   = grid_l;
   (sys_pfmg_data -> A_l)      = A_l;
   (sys_pfmg_data -> P_l)      = P_l;
   (sys_pfmg_data -> RT_l)     = RT_l;
   (sys_pfmg_data -> b_l)      = b_l;
   (sys_pfmg_data -> x_l)      = x_l;
   (sys_pfmg_data -> tx_l)     = tx_l;
   (sys_pfmg_data -> r_l)      = r_l;
   (sys_pfmg_data -> e_l)      = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *-----------------------------------------------------*/

   relax_data_l    = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   matvec_data_l   = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   restrict_data_l = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);
   interp_data_l   = hypre_TAlloc(void *, num_levels, HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetStride(cdir, stride);

      /* set up the interpolation operator */
      hypre_SStructPMatvecCreate(&interp_data_l[l]);
      hypre_SStructPMatvecSetup(interp_data_l[l], P_l[l], x_l[l + 1]);

      /* set up the restriction operator */
      hypre_SStructPMatvecCreate(&restrict_data_l[l]);
      hypre_SStructPMatvecSetTranspose(restrict_data_l[l], 1);
      hypre_SStructPMatvecSetup(restrict_data_l[l], RT_l[l], r_l[l]);
   }

   /* Check for zero diagonal on coarsest grid, occurs with singular problems
    * like full Neumann or full periodic.  Note that a processor with zero
    * diagonal will set active_l = 0, other processors will not. This is OK as
    * we only want to avoid the division by zero on the one processor that owns
    * the single coarse grid point. */
   if (hypre_SysPFMGZeroDiagonal(A_l[l]))
   {
      active_l[l] = 0;
   }

   /* set up fine grid relaxation */
   relax_data_l[0] = hypre_SysPFMGRelaxCreate(comm);
   hypre_SysPFMGRelaxSetTol(relax_data_l[0], 0.0);
   if (usr_jacobi_weight)
   {
      hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[0], jacobi_weight);
   }
   else
   {
      hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[0], relax_weights[0]);
   }
   hypre_SysPFMGRelaxSetType(relax_data_l[0], relax_type);
   hypre_SysPFMGRelaxSetTempVec(relax_data_l[0], tx_l[0]);
   hypre_SysPFMGRelaxSetup(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
   if (num_levels > 1)
   {
      for (l = 1; l < num_levels; l++)
      {
         /* set relaxation parameters */
         relax_data_l[l] = hypre_SysPFMGRelaxCreate(comm);
         hypre_SysPFMGRelaxSetTol(relax_data_l[l], 0.0);
         if (usr_jacobi_weight)
         {
            hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[l], jacobi_weight);
         }
         else
         {
            hypre_SysPFMGRelaxSetJacobiWeight(relax_data_l[l], relax_weights[l]);
         }
         hypre_SysPFMGRelaxSetType(relax_data_l[l], relax_type);
         hypre_SysPFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      }

      /* change coarsest grid relaxation parameters */
      l = num_levels - 1;
      if (active_l[l])
      {
         HYPRE_Int maxwork, maxiter;
         hypre_SysPFMGRelaxSetType(relax_data_l[l], 0);

         /* do no more work on the coarsest grid than the cost of a V-cycle
          * (estimating roughly 4 communications per V-cycle level) */
         maxwork = 4 * num_levels;

         /* do sweeps proportional to the coarsest grid size */
         maxiter = hypre_min(maxwork, cmaxsize);
         hypre_SysPFMGRelaxSetMaxIter(relax_data_l[l], maxiter);
      }

      /* call relax setup */
      for (l = 1; l < num_levels; l++)
      {
         if (active_l[l])
         {
            hypre_SysPFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }
      }
   }
   hypre_TFree(relax_weights, HYPRE_MEMORY_HOST);

   for (l = 0; l < num_levels; l++)
   {
      /* set up the residual routine */
      hypre_SStructPMatvecCreate(&matvec_data_l[l]);
      hypre_SStructPMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
   }

   (sys_pfmg_data -> active_l)        = active_l;
   (sys_pfmg_data -> relax_data_l)    = relax_data_l;
   (sys_pfmg_data -> matvec_data_l)   = matvec_data_l;
   (sys_pfmg_data -> restrict_data_l) = restrict_data_l;
   (sys_pfmg_data -> interp_data_l)   = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((sys_pfmg_data -> logging) > 0)
   {
      (sys_pfmg_data -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
      (sys_pfmg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   }

#if DEBUG
   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_sprintf(filename, "syspfmg_A.%02d", l);
      hypre_SStructPMatrixPrint(filename, A_l[l], 0);
      hypre_sprintf(filename, "syspfmg_P.%02d", l);
      hypre_SStructPMatrixPrint(filename, P_l[l], 0);
   }
   hypre_sprintf(filename, "syspfmg_A.%02d", l);
   hypre_SStructPMatrixPrint(filename, A_l[l], 0);
#endif

   /*-----------------------------------------------------
    * Destroy Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors).
    *-----------------------------------------------------*/
   hypre_SStructPMatrixDestroy(A);
   hypre_SStructPVectorDestroy(x);
   hypre_SStructPVectorDestroy(b);

   return hypre_error_flag;
}

#if 0  // RDF: Should be able to delete this
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysStructCoarsen( hypre_SStructPGrid  *fgrid,
                        hypre_Index          index,
                        hypre_Index          stride,
                        HYPRE_Int            prune,
                        hypre_SStructPGrid **cgrid_ptr )
{
   hypre_SStructPGrid   *cgrid;

   hypre_StructGrid     *sfgrid;
   hypre_StructGrid     *scgrid;

   MPI_Comm               comm;
   HYPRE_Int              ndim;
   HYPRE_Int              nvars;
   hypre_SStructVariable *vartypes;
   hypre_SStructVariable *new_vartypes;
   HYPRE_Int              i;
   HYPRE_Int              t;

   /*-----------------------------------------
    * Copy information from fine grid
    *-----------------------------------------*/

   comm      = hypre_SStructPGridComm(fgrid);
   ndim      = hypre_SStructPGridNDim(fgrid);
   nvars     = hypre_SStructPGridNVars(fgrid);
   vartypes  = hypre_SStructPGridVarTypes(fgrid);

   cgrid = hypre_TAlloc(hypre_SStructPGrid, 1, HYPRE_MEMORY_HOST);

   hypre_SStructPGridComm(cgrid)     = comm;
   hypre_SStructPGridNDim(cgrid)     = ndim;
   hypre_SStructPGridNVars(cgrid)    = nvars;
   new_vartypes = hypre_TAlloc(hypre_SStructVariable, nvars, HYPRE_MEMORY_HOST);
   for (i = 0; i < nvars; i++)
   {
      new_vartypes[i] = vartypes[i];
   }
   hypre_SStructPGridVarTypes(cgrid) = new_vartypes;

   for (t = 0; t < 8; t++)
   {
      hypre_SStructPGridVTPBndBoxArrayArray(cgrid, t) = NULL;
      hypre_SStructPGridVTSGrid(cgrid, t)     = NULL;
      hypre_SStructPGridVTIBoxArray(cgrid, t) = NULL;
      hypre_SStructPGridVTActive(cgrid, t)    = 1;
   }

   /*-----------------------------------------
    * Set the coarse sgrid
    *-----------------------------------------*/

   sfgrid = hypre_SStructPGridCellSGrid(fgrid);
   hypre_StructCoarsen(sfgrid, index, stride, prune, &scgrid);
   hypre_StructGridAssemble(scgrid);

   hypre_CopyIndex(hypre_StructGridPeriodic(scgrid),
                   hypre_SStructPGridPeriodic(cgrid));

   hypre_SStructPGridSetCellSGrid(cgrid, scgrid);

   hypre_SStructPGridPNeighbors(cgrid) = hypre_BoxArrayCreate(0, ndim);
   hypre_SStructPGridPNborOffsets(cgrid) = NULL;

   hypre_SStructPGridLocalSize(cgrid)  = 0;
   hypre_SStructPGridGlobalSize(cgrid) = 0;
   hypre_SStructPGridGhlocalSize(cgrid) = 0;

   hypre_SStructPGridAssemble(cgrid);

   *cgrid_ptr = cgrid;

   return hypre_error_flag;
}
#endif
