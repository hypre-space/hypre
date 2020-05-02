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

#include "_hypre_struct_ls.h"  /*Call to hypre_PFMGComputeDxyz */
#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

#define DEBUG 1

/*--------------------------------------------------------------------------
 *  TODO:
 *        1) Test full periodic problems. See what sys_pfmg does
 *        2) Implement HYPRE_SStructMatrixSetTranspose
 *        3) SetDxyz cannot be called by the user before SSAMGSetup because
 *           dxyz is being allocated here.
 *        4) Fix computation of cmaxsize. Should it vary across parts?
 *        5) Move "Initialize some data." to SSAMGCreate???
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetup( void                 *ssamg_vdata,
                  hypre_SStructMatrix  *A,
                  hypre_SStructVector  *b,
                  hypre_SStructVector  *x )
{
   hypre_SSAMGData       *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   /* Data from the SStructMatrix */
   hypre_SStructGraph    *graph      = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid     *grid       = hypre_SStructGraphGrid(graph);

   /* Solver parameters */
   MPI_Comm               comm       = hypre_SSAMGDataComm(ssamg_data);
   HYPRE_Int              max_iter   = hypre_SSAMGDataMaxIter(ssamg_data);
   HYPRE_Int              max_levels = hypre_SSAMGDataMaxLevels(ssamg_data);
   HYPRE_Int              relax_type = hypre_SSAMGDataRelaxType(ssamg_data);
   HYPRE_Real           **dxyz       = hypre_SSAMGDataDxyz(ssamg_data);
   HYPRE_Int            **cdir_l;
   hypre_SStructGrid    **grid_l;

   /* Work data structures */
   hypre_SStructMatrix  **A_l  = (ssamg_data -> A_l);
   hypre_SStructMatrix  **P_l  = (ssamg_data -> P_l);
   hypre_SStructMatrix  **RT_l = (ssamg_data -> RT_l);
   hypre_SStructVector  **b_l  = (ssamg_data -> b_l);
   hypre_SStructVector  **x_l  = (ssamg_data -> x_l);
   hypre_SStructVector  **r_l  = (ssamg_data -> r_l);
   hypre_SStructVector  **e_l  = (ssamg_data -> e_l);
   hypre_SStructVector  **tx_l = (ssamg_data -> tx_l);

   /* Data for */
   void                 **relax_data_l    = (ssamg_data -> relax_data_l);
   void                 **matvec_data_l   = (ssamg_data -> matvec_data_l);
   void                 **restrict_data_l = (ssamg_data -> restrict_data_l);
   void                 **interp_data_l   = (ssamg_data -> interp_data_l);
   HYPRE_Real           **relax_weights;

   HYPRE_Int             *dxyz_flag;
   HYPRE_Int              cmaxsize;
   HYPRE_Int              l, part, nparts;
   HYPRE_Int              num_levels;

#if DEBUG
   hypre_SStructVector   *ones  = NULL;
   hypre_SStructVector   *Pones = NULL;
   char                   filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some data.
    *-----------------------------------------------------*/
   nparts    = hypre_SStructMatrixNParts(A);
   dxyz      = hypre_TAlloc(HYPRE_Real *, nparts);
   dxyz_flag = hypre_TAlloc(HYPRE_Int, nparts);
   for (part = 0; part < nparts; part++)
   {
      dxyz[part] = hypre_TAlloc(HYPRE_Real, 3);
      dxyz[part][0] = 0.0;
      dxyz[part][1] = 0.0;
      dxyz[part][2] = 0.0;
   }
   (ssamg_data -> nparts) = nparts;
   (ssamg_data -> dxyz)   = dxyz;

   /* Compute Maximum number of multigrid levels */
   hypre_SSAMGComputeMaxLevels(grid, &max_levels);
   hypre_SSAMGDataMaxLevels(ssamg_data) = max_levels;

   /* Compute dxyz for each part */
   hypre_SSAMGComputeDxyz(A, dxyz, dxyz_flag);

   /* Compute coarsening direction and active levels for relaxation */
   hypre_SSAMGDataGridl(ssamg_data) = hypre_TAlloc(hypre_SStructGrid *, max_levels);
   grid_l = hypre_SSAMGDataGridl(ssamg_data);
   hypre_SStructGridRef(grid, &grid_l[0]);
   hypre_SSAMGCoarsen(ssamg_vdata, dxyz_flag, dxyz, &relax_weights);
   cdir_l = hypre_SSAMGDataCdir(ssamg_data);

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   num_levels = hypre_SSAMGDataNumLevels(ssamg_data);
   A_l  = hypre_TAlloc(hypre_SStructMatrix *, num_levels);
   P_l  = hypre_TAlloc(hypre_SStructMatrix *, num_levels - 1);
   RT_l = hypre_TAlloc(hypre_SStructMatrix *, num_levels - 1);
   b_l  = hypre_TAlloc(hypre_SStructVector *, num_levels);
   x_l  = hypre_TAlloc(hypre_SStructVector *, num_levels);
   tx_l = hypre_TAlloc(hypre_SStructVector *, num_levels);
   r_l  = tx_l;
   e_l  = tx_l;

   hypre_SStructMatrixRef(A, &A_l[0]);
   hypre_SStructVectorRef(b, &b_l[0]);
   hypre_SStructVectorRef(x, &x_l[0]);

   HYPRE_SStructVectorCreate(comm, grid_l[0], &tx_l[0]);
   HYPRE_SStructVectorInitialize(tx_l[0]);
   HYPRE_SStructVectorAssemble(tx_l[0]);

#if DEBUG
   hypre_sprintf(filename, "ssamg_A.%02d", 0);
   HYPRE_SStructMatrixPrint(filename, A_l[0], 0);
#endif

   /* Compute interpolation, restriction and coarse grids */
   for (l = 0; l < (num_levels - 1); l++)
   {
      // Build prolongation matrix
      P_l[l]  = hypre_SSAMGCreateInterpOp(A_l[l], grid_l[l+1], cdir_l[l]);
      //HYPRE_SStructMatrixSetTranspose(P_l[l], 1);
      hypre_SSAMGSetupInterpOp(A_l[l], cdir_l[l], P_l[l]);

#if DEBUG
      hypre_sprintf(filename, "ssamg_P.%02d", l);
      HYPRE_SStructMatrixPrint(filename, P_l[l], 0);
#endif

      // Build restriction matrix
      hypre_SStructMatrixRef(P_l[l], &RT_l[l]);

      // Compute coarse matrix
      hypre_SStructMatPtAP(P_l[l], A_l[l], &A_l[l+1]);

#if DEBUG
      hypre_sprintf(filename, "ssamg_A.%02d", l+1);
      HYPRE_SStructMatrixPrint(filename, A_l[l+1], 0);
#endif

      // Build SStructVectors
      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &b_l[l+1]);
      HYPRE_SStructVectorInitialize(b_l[l+1]);
      HYPRE_SStructVectorAssemble(b_l[l+1]);

      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &x_l[l+1]);
      HYPRE_SStructVectorInitialize(x_l[l+1]);
      HYPRE_SStructVectorAssemble(x_l[l+1]);

      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &tx_l[l+1]);
      HYPRE_SStructVectorInitialize(tx_l[l+1]);
      HYPRE_SStructVectorAssemble(tx_l[l+1]);
   }

   (ssamg_data -> A_l)  = A_l;
   (ssamg_data -> P_l)  = P_l;
   (ssamg_data -> RT_l) = RT_l;
   (ssamg_data -> b_l)  = b_l;
   (ssamg_data -> x_l)  = x_l;
   (ssamg_data -> tx_l) = tx_l;
   (ssamg_data -> r_l)  = r_l;
   (ssamg_data -> e_l)  = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *-----------------------------------------------------*/

   relax_data_l    = hypre_TAlloc(void *, num_levels);
   matvec_data_l   = hypre_TAlloc(void *, num_levels);
   restrict_data_l = hypre_TAlloc(void *, num_levels);
   interp_data_l   = hypre_TAlloc(void *, num_levels);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_SStructMatvecCreate(&matvec_data_l[l]);
      hypre_SStructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);

      hypre_SStructMatvecCreate(&interp_data_l[l]);
      hypre_SStructMatvecSetup(interp_data_l[l], P_l[l], x_l[l+1]);

      hypre_SStructMatvecCreate(&restrict_data_l[l]);
      hypre_SStructMatvecSetTranspose(restrict_data_l[l], 1);
      hypre_SStructMatvecSetup(restrict_data_l[l], RT_l[l], x_l[l]);

      hypre_SSAMGRelaxCreate(comm, nparts, &relax_data_l[l]);
      hypre_SSAMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_SSAMGRelaxSetWeight(relax_data_l[l], relax_weights[l]);
      hypre_SSAMGRelaxSetType(relax_data_l[l], relax_type);
      hypre_SSAMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      hypre_SSAMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      // Check if P interpolates vector of ones
#if DEBUG
      if (ones != NULL)
      {
         HYPRE_SStructVectorDestroy(ones);
      }
      HYPRE_SStructVectorCreate(comm, grid_l[l+1], &ones);
      HYPRE_SStructVectorInitialize(ones);
      HYPRE_SStructVectorSetConstantValues(ones, 1.0);
      HYPRE_SStructVectorAssemble(ones);

      if (Pones != NULL)
      {
         HYPRE_SStructVectorDestroy(Pones);
      }
      HYPRE_SStructVectorCreate(comm, grid_l[l], &Pones);
      HYPRE_SStructVectorInitialize(Pones);
      HYPRE_SStructVectorAssemble(Pones);

      /* interpolate error and correct (x = Pe_c) */
      hypre_SStructMatvecCompute(interp_data_l[l], 1.0, P_l[l], ones, 0.0, Pones);

      hypre_sprintf(filename, "ssamg_Pones.%02d", l);
      HYPRE_SStructVectorPrint(filename, Pones, 0);
#endif
   }

   /* set up remaining operations for the coarse grid */
   l = (num_levels - 1);
   hypre_SStructMatvecCreate(&matvec_data_l[l]);
   hypre_SStructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
   {
      HYPRE_Int             max_work, relax_max_iter;
      hypre_Box             bbox;
      hypre_SStructPGrid   *pgrid;
      hypre_StructGrid     *sgrid;

      /* do no more work on the coarsest grid than the cost of a V-cycle
       * (estimating roughly 4 communications per V-cycle level) */
      max_work = 4*num_levels;

      /* do sweeps proportional to the coarsest grid size */
      
      cmaxsize = 0;
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid_l[l], part);
         sgrid = hypre_SStructPGridCellSGrid(pgrid);
         hypre_CopyBox(hypre_StructGridBoundingBox(sgrid), &bbox);

         cmaxsize = hypre_max(cmaxsize, hypre_BoxMaxSize(&bbox));
      }
      relax_max_iter = hypre_min(max_work, cmaxsize);

      hypre_SSAMGRelaxCreate(comm, nparts, &relax_data_l[l]);
      hypre_SSAMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_SSAMGRelaxSetWeight(relax_data_l[l], relax_weights[l]);
      hypre_SSAMGRelaxSetType(relax_data_l[l], 0);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], relax_max_iter);
      hypre_SSAMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      hypre_SSAMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

#if 1
      hypre_printf("max_work = %d, cmaxsize = %d, cmax_iter = %d\n",
                   max_work, cmaxsize, relax_max_iter);
#endif
   }

   (ssamg_data -> relax_data_l)    = relax_data_l;
   (ssamg_data -> matvec_data_l)   = matvec_data_l;
   (ssamg_data -> restrict_data_l) = restrict_data_l;
   (ssamg_data -> interp_data_l)   = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((ssamg_data -> logging) > 0)
   {
      max_iter = (ssamg_data -> max_iter);
      (ssamg_data -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter);
      (ssamg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter);
   }

   /* Print statistics */
   hypre_SSAMGPrintStats(ssamg_vdata);

   /* Free memory */
   for (l = 0; l < max_levels; l++)
   {
      hypre_TFree(relax_weights[l]);
   }
   hypre_TFree(relax_weights);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeMaxLevels: maximum number of levels for multigrid.
 *   Computes max_levels for each part independently and takes the minimum
 *   over all. This may not make sense to AMR!
 *
 * TODO: Before implementing, check if this makes sense
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGComputeMaxLevels( hypre_SStructGrid  *grid,
                             HYPRE_Int          *max_levels )
{
   hypre_SStructPGrid   *pgrid;
   hypre_StructGrid     *sgrid;

   HYPRE_Int             max_levels_in;
   HYPRE_Int             max_levels_out;
   HYPRE_Int             max_levels_part;
   HYPRE_Int             part, nparts;

   nparts = hypre_SStructGridNParts(grid);

   max_levels_in  = *max_levels;
   max_levels_out = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      sgrid = hypre_SStructPGridCellSGrid(pgrid);

      hypre_PFMGComputeMaxLevels(sgrid, &max_levels_part);

      max_levels_out = hypre_max(max_levels_part, max_levels_out);
   }

   *max_levels = max_levels_out;
   if (max_levels_in > 0)
   {
      *max_levels = hypre_min(max_levels_out, max_levels_in);
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeDxyz computes dxyz for each part independently.
 *
 * TODO: Before implementing, check if this makes sense
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGComputeDxyz( hypre_SStructMatrix  *A,
                        HYPRE_Real          **dxyz,
                        HYPRE_Int            *dxyz_flag )
{
   hypre_SStructPMatrix  *pmatrix;
   hypre_StructMatrix    *smatrix;

   HYPRE_Int              d, var, part, nparts, nvars;
   HYPRE_Int              ndim;
   HYPRE_Real             sys_dxyz[3] = {0.0, 0.0, 0.0};


   /*--------------------------------------------------------
    * Allocate arrays for mesh sizes for each diagonal block
    *--------------------------------------------------------*/
   nparts = hypre_SStructMatrixNParts(A);
   ndim   = hypre_SStructMatrixNDim(A);

   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(A, part);
      nvars   = hypre_SStructPMatrixNVars(pmatrix);

      for (var = 0; var < nvars; var++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, var);
         hypre_PFMGComputeDxyz(smatrix, sys_dxyz, &dxyz_flag[part]);

         for (d = 0; d < ndim; d++)
         {
            dxyz[part][d] += sys_dxyz[d];
            sys_dxyz[d] = 0.0;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO:
 *       1) *** Handle the case where different parts have different num_levels
 *       2) relax_weights does not depend on the parts. Only the values coming from
 *          the last part in the cycle is actually used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGCoarsen( void                 *ssamg_vdata,
                    HYPRE_Int            *dxyz_flag,
                    HYPRE_Real          **dxyz,
                    HYPRE_Real         ***relax_weights_ptr)
{
   hypre_SSAMGData      *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Real            usr_relax_weight = hypre_SSAMGDataRelaxWeight(ssamg_data);
   hypre_SStructGrid   **grid_l;
   hypre_SStructPGrid   *pgrid;
   hypre_StructGrid     *sgrid;

   hypre_Box           **cbox;

   HYPRE_Int           **cdir_l;
   HYPRE_Real          **relax_weights;
   hypre_Index          *periodic;
   hypre_Index          *strides;

   hypre_Index           coarsen;
   hypre_Index           cindex;
   HYPRE_Int             num_levels, max_levels;
   HYPRE_Int             part, nparts, ndim;
   HYPRE_Int             l, d, cdir;
   HYPRE_Int             coarse;
   HYPRE_Int             cbox_imin, cbox_imax;
   HYPRE_Real            min_dxyz;
   HYPRE_Real            alpha, beta;

   /* Initalize some data */
   max_levels = hypre_SSAMGDataMaxLevels(ssamg_data);
   grid_l     = hypre_SSAMGDataGridl(ssamg_data);
   ndim       = hypre_SStructGridNDim(grid_l[0]);
   nparts     = hypre_SStructGridNParts(grid_l[0]);

   /* Allocate data */
   cdir_l        = hypre_TAlloc(HYPRE_Int *, max_levels);
   relax_weights = hypre_TAlloc(HYPRE_Real *, max_levels);
   cbox          = hypre_TAlloc(hypre_Box *, nparts);
   periodic      = hypre_TAlloc(hypre_Index, nparts);
   strides       = hypre_TAlloc(hypre_Index, nparts);
   for (l = 0; l < max_levels; l++)
   {
      relax_weights[l] = hypre_CTAlloc(HYPRE_Real, nparts);
   }
   for (part = 0; part < nparts; part++)
   {
      hypre_SetIndex(strides[part], 1);
   }

   /* Get grid bounding box and periodicity data */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid_l[0], part);
      sgrid = hypre_SStructPGridCellSGrid(pgrid);
      cbox[part] = hypre_BoxClone(hypre_StructGridBoundingBox(sgrid));

      hypre_CopyIndex(hypre_StructGridPeriodic(sgrid), periodic[part]);
   }

   /* Initialize number of levels */
   num_levels = 1;

   /* Force relaxation on finest grid */
   hypre_SetIndex(coarsen, 1);
   for (l = 0; l < max_levels; l++)
   {
      cdir_l[l] = hypre_TAlloc(HYPRE_Int, nparts);

      coarse = 0;
      for (part = 0; part < nparts; part++)
      {
         /* Initialize min_dxyz */
         min_dxyz = 1;
         for (d = 0; d < ndim; d++)
         {
            min_dxyz += dxyz[part][d];
         }

         /* Determine cdir */
         cdir = -1; alpha = 0.0;
         for (d = 0; d < ndim; d++)
         {
            cbox_imax = hypre_BoxIMaxD(cbox[part], d);
            cbox_imin = hypre_BoxIMinD(cbox[part], d);

            if ((cbox_imax > cbox_imin) && (dxyz[part][d] < min_dxyz))
            {
               min_dxyz = dxyz[part][d];
               cdir = d;
            }
            alpha += 1.0/(dxyz[part][d]*dxyz[part][d]);
         }

         /* Change relax_weights */
         if (usr_relax_weight > 0.0)
         {
            relax_weights[l][part] = usr_relax_weight;
         }
         else
         {
            beta = 0.0;
            if (dxyz_flag[part] || (ndim == 1))
            {
               relax_weights[l][part] = 2.0/3.0;
            }
            else
            {
               for (d = 0; d < ndim; d++)
               {
                  if (d != cdir)
                  {
                     beta += 1.0/(dxyz[part][d]*dxyz[part][d]);
                  }
               }

               /* determine level Jacobi weights */
               relax_weights[l][part] = 2.0/(3.0 - beta/alpha);
            }
         }

         if ((cdir > -1) && (l < (max_levels - 1)))
         {
            cdir_l[l][part] = cdir;

            /* don't coarsen if a periodic direction and not divisible by 2 */
            if ((periodic[part][cdir]) && (periodic[part][cdir] % 2))
            {
               continue;
            }

            coarse = 1;
            /* if (hypre_IndexD(coarsen, cdir) != 0) */
            /* { */
            /*    /\* coarsened previously in this direction, relax level l *\/ */
            /*    active_l[l][part] = 1; */
            /*    hypre_SetIndex(coarsen, 0); */
            /* } */
            /* hypre_IndexD(coarsen, cdir) = 1; */

            /* set cindex and stride */
            hypre_SetIndex(cindex, 0);
            hypre_SetIndex(strides[part], 1);
            hypre_IndexD(strides[part], cdir) = 2;

            /* update dxyz and coarsen cbox*/
            dxyz[part][cdir] *= 2;
            hypre_ProjectBox(cbox[part], cindex, strides[part]);
            hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox[part]), cindex, strides[part],
                                        hypre_BoxIMin(cbox[part]));
            hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox[part]), cindex, strides[part],
                                        hypre_BoxIMax(cbox[part]));

            /* update periodic */
            periodic[part][cdir] /= 2;

            /* Update number of levels */
            num_levels = l + 2; // (l + 1)?
         }
         else
         {
            relax_weights[l][part] = 1.0; // Coarse grid relax weight
         }
      } /* loop on parts */

      // If there's no part to be coarsened, exit loop
      if (!coarse) break;

      // Compute the coarsened SStructGrid object
      hypre_SStructGridCoarsen(grid_l[l], NULL, strides, periodic, 0, &grid_l[l+1]);

   } /* loop on levels */

   /* Free memory */
   for (part = 0; part < nparts; part++)
   {
      hypre_BoxDestroy(cbox[part]);
   }
   hypre_TFree(cbox);
   hypre_TFree(periodic);
   hypre_TFree(strides);

   /* Output */
   hypre_SSAMGDataNumLevels(ssamg_data) = num_levels;
   hypre_SSAMGDataCdir(ssamg_data)      = cdir_l;
   hypre_SSAMGDataGridl(ssamg_data)     = grid_l;
   *relax_weights_ptr                   = relax_weights;

   return hypre_error_flag;
}
