/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

#include <time.h>
#define DEBUG 0

#define hypre_PFMGSetCIndex(cdir, cindex)       \
   {                                            \
      hypre_SetIndex3(cindex, 0, 0, 0);         \
      hypre_IndexD(cindex, cdir) = 0;           \
   }

#define hypre_PFMGSetFIndex(cdir, findex)       \
   {                                            \
      hypre_SetIndex3(findex, 0, 0, 0);         \
      hypre_IndexD(findex, cdir) = 1;           \
   }

#define hypre_PFMGSetStride(cdir, stride)       \
   {                                            \
      hypre_SetIndex3(stride, 1, 1, 1);         \
      hypre_IndexD(stride, cdir) = 2;           \
   }

#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

HYPRE_Int hypre_StructGetNonzeroDirection(hypre_Index shape)
{
   HYPRE_Int Astenc = 0;
   /* x-direction */
   if (hypre_IndexD(shape, 0))
   {
      Astenc += 1;
   }
   /* y-direction */
   else if (hypre_IndexD(shape, 1))
   {
      Astenc += 10;
   }
   /* z-direction */
   else if (hypre_IndexD(shape, 2))
   {
      Astenc += 100;
   }
   return Astenc;
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

   MPI_Comm              comm = (pfmg_data -> comm);

   HYPRE_Int             relax_type =       (pfmg_data -> relax_type);
   HYPRE_Int             usr_jacobi_weight = (pfmg_data -> usr_jacobi_weight);
   HYPRE_Real            jacobi_weight    = (pfmg_data -> jacobi_weight);
   HYPRE_Int             skip_relax =       (pfmg_data -> skip_relax);
   HYPRE_Real           *dxyz       =       (pfmg_data -> dxyz);
   HYPRE_Int             rap_type;

   HYPRE_Int             max_iter;
   HYPRE_Int             max_levels;

   HYPRE_Int             num_levels;

   hypre_Index           cindex;
   hypre_Index           findex;
   hypre_Index           stride;

   hypre_Index           coarsen;

   HYPRE_Int            *cdir_l;
   HYPRE_Int            *active_l;
   hypre_StructGrid    **grid_l;
   hypre_StructGrid    **P_grid_l;

   HYPRE_Real           *data;
   HYPRE_Real           *data_const;
   HYPRE_Int             data_size = 0;
   HYPRE_Int             data_size_const = 0;
   HYPRE_Real           *relax_weights;
   HYPRE_Real           *mean, *deviation;
   HYPRE_Real            alpha, beta;

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
   HYPRE_Int             ndim;

   hypre_Box            *cbox;

   HYPRE_Real            min_dxyz;
   HYPRE_Int             cdir, periodic, cmaxsize;
   HYPRE_Int             d, l;
   HYPRE_Int             dxyz_flag;

   HYPRE_Int             b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   HYPRE_Int             x_num_ghost[]  = {1, 1, 1, 1, 1, 1};

#if DEBUG
   char                  filename[255];
#endif

   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(A);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid  = hypre_StructMatrixGrid(A);
   ndim  = hypre_StructGridNDim(grid);

   /* Compute a new max_levels value based on the grid */
   cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(grid));
   max_levels = 1;
   for (d = 0; d < ndim; d++)
   {
      max_levels += hypre_Log2(hypre_BoxSizeD(cbox, d)) + 2;
   }

   if ((pfmg_data -> max_levels) > 0)
   {
      max_levels = hypre_min(max_levels, (pfmg_data -> max_levels));
   }
   (pfmg_data -> max_levels) = max_levels;

   /* compute dxyz */
   dxyz_flag = 0;
   if ((dxyz[0] == 0) || (dxyz[1] == 0) || (dxyz[2] == 0))
   {
      mean = hypre_CTAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
      deviation = hypre_CTAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
      hypre_PFMGComputeDxyz(A, dxyz, mean, deviation);

      for (d = 0; d < ndim; d++)
      {
         /* Set 'dxyz_flag' if the matrix-coefficient variation is "too large".
          * This is used later to set relaxation weights for Jacobi.
          *
          * Use the "square of the coefficient of variation" = (sigma/mu)^2,
          * where sigma is the standard deviation and mu is the mean.  This is
          * equivalent to computing (d - mu^2)/mu^2 where d is the average of
          * the squares of the coefficients stored in 'deviation'.  Care is
          * taken to avoid dividing by zero when the mean is zero. */

         deviation[d] -= mean[d] * mean[d];
         if ( deviation[d] > 0.1 * (mean[d]*mean[d]) )
         {
            dxyz_flag = 1;
            break;
         }
      }

      hypre_TFree(mean,      HYPRE_MEMORY_HOST);
      hypre_TFree(deviation, HYPRE_MEMORY_HOST);
   }

   grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels, HYPRE_MEMORY_HOST);
   hypre_StructGridRef(grid, &grid_l[0]);
   P_grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels, HYPRE_MEMORY_HOST);
   P_grid_l[0] = NULL;
   cdir_l = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   active_l = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   relax_weights = hypre_CTAlloc(HYPRE_Real, max_levels, HYPRE_MEMORY_HOST);
   hypre_SetIndex3(coarsen, 1, 1, 1); /* forces relaxation on finest grid */

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   data_location = hypre_StructGridDataLocation(grid);
   if (data_location != HYPRE_MEMORY_HOST)
   {
      num_level_GPU = max_levels;
   }
   else
   {
      num_level_GPU = 0;
      device_level  = 0;
   }
#endif

   for (l = 0; ; l++)
   {
      /* determine cdir */
      min_dxyz = dxyz[0] + dxyz[1] + dxyz[2] + 1;
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
         if (dxyz_flag)
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
            if (beta == alpha)
            {
               alpha = 0.0;
            }
            else
            {
               alpha = beta / alpha;
            }

            /* determine level Jacobi weights */
            if (ndim > 1)
            {
               relax_weights[l] = 2.0 / (3.0 - alpha);
            }
            else
            {
               relax_weights[l] = 2.0 / 3.0; /* always 2/3 for 1-d */
            }
         }
      }

      if (cdir != -1)
      {
         /* don't coarsen if a periodic direction and not divisible by 2 */
         periodic = hypre_IndexD(hypre_StructGridPeriodic(grid_l[l]), cdir);
         if ((periodic) && (periodic % 2))
         {
            cdir = -1;
         }

         /* don't coarsen if we've reached max_levels */
         if (l == (max_levels - 1))
         {
            cdir = -1;
         }
      }

      /* stop coarsening */
      if (cdir == -1)
      {
         active_l[l] = 1; /* forces relaxation on coarsest grid */
         cmaxsize = 0;
         for (d = 0; d < ndim; d++)
         {
            cmaxsize = hypre_max(cmaxsize, hypre_BoxSizeD(cbox, d));
         }

         break;
      }

      cdir_l[l] = cdir;

      if (hypre_IndexD(coarsen, cdir) != 0)
      {
         /* coarsened previously in this direction, relax level l */
         active_l[l] = 1;
         hypre_SetIndex3(coarsen, 0, 0, 0);
         hypre_IndexD(coarsen, cdir) = 1;
      }
      else
      {
         active_l[l] = 0;
         hypre_IndexD(coarsen, cdir) = 1;
      }

      /* set cindex, findex, and stride */
      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetFIndex(cdir, findex);
      hypre_PFMGSetStride(cdir, stride);

      /* update dxyz and coarsen cbox*/
      dxyz[cdir] *= 2;
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
                                  hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
                                  hypre_BoxIMax(cbox));

      /* build the interpolation grid */
      hypre_StructCoarsen(grid_l[l], findex, stride, 0, &P_grid_l[l + 1]);

      /* build the coarse grid */
      hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l + 1]);
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      hypre_StructGridDataLocation(P_grid_l[l + 1]) = data_location;
      if (device_level == -1 && num_level_GPU > 0)
      {
         max_box_size = hypre_StructGridGetMaxBoxSize(grid_l[l + 1]);
         if (max_box_size < HYPRE_MIN_GPU_SIZE)
         {
            num_level_GPU = l + 1;
            data_location = HYPRE_MEMORY_HOST;
            device_level  = num_level_GPU;
            //printf("num_level_GPU = %d,device_level = %d / %d\n",num_level_GPU,device_level,num_levels);
         }
      }
      else if (l + 1 == device_level)
      {
         num_level_GPU = l + 1;
         data_location = HYPRE_MEMORY_HOST;
      }
      hypre_StructGridDataLocation(grid_l[l + 1]) = data_location;
#endif
   }

   num_levels = l + 1;

   /* free up some things */
   hypre_BoxDestroy(cbox);

   /* set all levels active if skip_relax = 0 */
   if (!skip_relax)
   {
      for (l = 0; l < num_levels; l++)
      {
         active_l[l] = 1;
      }
   }

   (pfmg_data -> num_levels)   = num_levels;
   (pfmg_data -> cdir_l)       = cdir_l;
   (pfmg_data -> grid_l)       = grid_l;
   (pfmg_data -> P_grid_l)     = P_grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   /*-----------------------------------------------------
    * Modify the rap_type if red-black Gauss-Seidel is
    * used. Red-black gs is used only in the non-Galerkin
    * case.
    *-----------------------------------------------------*/
   if (relax_type == 2 || relax_type == 3)   /* red-black gs */
   {
      (pfmg_data -> rap_type) = 1;
   }
   rap_type = (pfmg_data -> rap_type);

   A_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels, HYPRE_MEMORY_HOST);
   P_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   RT_l = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1, HYPRE_MEMORY_HOST);
   b_l  = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   x_l  = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   tx_l = hypre_TAlloc(hypre_StructVector *, num_levels, HYPRE_MEMORY_HOST);
   r_l  = tx_l;
   e_l  = tx_l;

   A_l[0] = hypre_StructMatrixRef(A);
   b_l[0] = hypre_StructVectorRef(b);
   x_l[0] = hypre_StructVectorRef(x);

   tx_l[0] = hypre_StructVectorCreate(comm, grid_l[0]);
   hypre_StructVectorSetNumGhost(tx_l[0], x_num_ghost);
   hypre_StructVectorInitializeShell(tx_l[0]);

   hypre_StructVectorSetDataSize(tx_l[0], &data_size, &data_size_const);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      P_l[l]  = hypre_PFMGCreateInterpOp(A_l[l], P_grid_l[l + 1], cdir, rap_type);
      hypre_StructMatrixInitializeShell(P_l[l]);
      data_size += hypre_StructMatrixDataSize(P_l[l]);
      data_size_const += hypre_StructMatrixDataConstSize(P_l[l]);

      if (hypre_StructMatrixSymmetric(A))
      {
         RT_l[l] = P_l[l];
      }
      else
      {
         RT_l[l] = P_l[l];
#if 0
         /* Allow RT != P for non symmetric case */
         /* NOTE: Need to create a non-pruned grid for this to work */
         RT_l[l]   = hypre_PFMGCreateRestrictOp(A_l[l], grid_l[l + 1], cdir);
         hypre_StructMatrixInitializeShell(RT_l[l]);
         data_size += hypre_StructMatrixDataSize(RT_l[l]);
         data_size_const += hypre_StructMatrixDataConstSize(RT_l[l]);
#endif
      }

      A_l[l + 1] = hypre_PFMGCreateRAPOp(RT_l[l], A_l[l], P_l[l],
                                         grid_l[l + 1], cdir, rap_type);
      hypre_StructMatrixInitializeShell(A_l[l + 1]);
      data_size += hypre_StructMatrixDataSize(A_l[l + 1]);
      data_size_const += hypre_StructMatrixDataConstSize(A_l[l + 1]);

      b_l[l + 1] = hypre_StructVectorCreate(comm, grid_l[l + 1]);
      hypre_StructVectorSetNumGhost(b_l[l + 1], b_num_ghost);
      hypre_StructVectorInitializeShell(b_l[l + 1]);
      hypre_StructVectorSetDataSize(b_l[l + 1], &data_size, &data_size_const);

      x_l[l + 1] = hypre_StructVectorCreate(comm, grid_l[l + 1]);
      hypre_StructVectorSetNumGhost(x_l[l + 1], x_num_ghost);
      hypre_StructVectorInitializeShell(x_l[l + 1]);
      hypre_StructVectorSetDataSize(x_l[l + 1], &data_size, &data_size_const);

      tx_l[l + 1] = hypre_StructVectorCreate(comm, grid_l[l + 1]);
      hypre_StructVectorSetNumGhost(tx_l[l + 1], x_num_ghost);
      hypre_StructVectorInitializeShell(tx_l[l + 1]);
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (l + 1 == num_level_GPU)
      {
         hypre_StructVectorSetDataSize(tx_l[l + 1], &data_size, &data_size_const);
      }
#endif
   }

   data = hypre_CTAlloc(HYPRE_Real, data_size, memory_location);
   data_const = hypre_CTAlloc(HYPRE_Real, data_size_const, HYPRE_MEMORY_HOST);
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   //hypre_printf("num_level_GPU = %d,device_level = %d / %d\n",num_level_GPU,device_level,num_levels);
#endif

   (pfmg_data -> memory_location) = memory_location;
   (pfmg_data -> data) = data;
   (pfmg_data -> data_const) = data_const;

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   data_location = hypre_StructGridDataLocation(grid_l[0]);
   if (data_location != HYPRE_MEMORY_HOST)
   {
      hypre_StructVectorInitializeData(tx_l[0], data);
      hypre_StructVectorAssemble(tx_l[0]);
      data += hypre_StructVectorDataSize(tx_l[0]);
   }
   else
   {
      hypre_StructVectorInitializeData(tx_l[0], data_const);
      hypre_StructVectorAssemble(tx_l[0]);
      data_const += hypre_StructVectorDataSize(tx_l[0]);
   }
#else
   hypre_StructVectorInitializeData(tx_l[0], data);
   hypre_StructVectorAssemble(tx_l[0]);
   data += hypre_StructVectorDataSize(tx_l[0]);
#endif

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_StructMatrixInitializeData(P_l[l], data, data_const);
      data += hypre_StructMatrixDataSize(P_l[l]);
      data_const += hypre_StructMatrixDataConstSize(P_l[l]);

#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
      {
         hypre_StructMatrixInitializeData(RT_l[l], data, data_const);
         data += hypre_StructMatrixDataSize(RT_l[l]);
         data_const += hypre_StructMatrixDataConstSize(RT_l[l]);
      }
#endif

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (l + 1 == num_level_GPU)
      {
         data_location = HYPRE_MEMORY_HOST;
      }
#endif

      hypre_StructMatrixInitializeData(A_l[l + 1], data, data_const);
      data += hypre_StructMatrixDataSize(A_l[l + 1]);
      data_const += hypre_StructMatrixDataConstSize(A_l[l + 1]);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (data_location != HYPRE_MEMORY_HOST)
      {
         hypre_StructVectorInitializeData(b_l[l + 1], data);
         hypre_StructVectorAssemble(b_l[l + 1]);
         data += hypre_StructVectorDataSize(b_l[l + 1]);

         hypre_StructVectorInitializeData(x_l[l + 1], data);
         hypre_StructVectorAssemble(x_l[l + 1]);
         data += hypre_StructVectorDataSize(x_l[l + 1]);
         hypre_StructVectorInitializeData(tx_l[l + 1],
                                          hypre_StructVectorData(tx_l[0]));
         hypre_StructVectorAssemble(tx_l[l + 1]);
      }
      else
      {
         hypre_StructVectorInitializeData(b_l[l + 1], data_const);
         hypre_StructVectorAssemble(b_l[l + 1]);
         data_const += hypre_StructVectorDataSize(b_l[l + 1]);

         hypre_StructVectorInitializeData(x_l[l + 1], data_const);
         hypre_StructVectorAssemble(x_l[l + 1]);
         data_const += hypre_StructVectorDataSize(x_l[l + 1]);
         if (l + 1 == num_level_GPU)
         {
            hypre_StructVectorInitializeData(tx_l[l + 1], data_const);
            hypre_StructVectorAssemble(tx_l[l + 1]);
            data_const += hypre_StructVectorDataSize(tx_l[l + 1]);
         }
         hypre_StructVectorInitializeData(tx_l[l + 1], hypre_StructVectorData(tx_l[num_level_GPU]));
         hypre_StructVectorAssemble(tx_l[l + 1]);
      }
#else
      hypre_StructVectorInitializeData(b_l[l + 1], data);
      hypre_StructVectorAssemble(b_l[l + 1]);
      data += hypre_StructVectorDataSize(b_l[l + 1]);

      hypre_StructVectorInitializeData(x_l[l + 1], data);
      hypre_StructVectorAssemble(x_l[l + 1]);
      data += hypre_StructVectorDataSize(x_l[l + 1]);

      hypre_StructVectorInitializeData(tx_l[l + 1],
                                       hypre_StructVectorData(tx_l[0]));
      hypre_StructVectorAssemble(tx_l[l + 1]);
#endif
   }

   (pfmg_data -> A_l)  = A_l;
   (pfmg_data -> P_l)  = P_l;
   (pfmg_data -> RT_l) = RT_l;
   (pfmg_data -> b_l)  = b_l;
   (pfmg_data -> x_l)  = x_l;
   (pfmg_data -> tx_l) = tx_l;
   (pfmg_data -> r_l)  = r_l;
   (pfmg_data -> e_l)  = e_l;

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

      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetFIndex(cdir, findex);
      hypre_PFMGSetStride(cdir, stride);

      /* set up interpolation operator */
      hypre_PFMGSetupInterpOp(A_l[l], cdir, findex, stride, P_l[l], rap_type);

      /* set up the restriction operator */
#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
         hypre_PFMGSetupRestrictOp(A_l[l], tx_l[l],
                                   cdir, cindex, stride, RT_l[l]);
#endif

      /* set up the coarse grid operator */
      hypre_PFMGSetupRAPOp(RT_l[l], A_l[l], P_l[l],
                           cdir, cindex, stride, rap_type, A_l[l + 1]);

      /* set up the interpolation routine */
      interp_data_l[l] = hypre_SemiInterpCreate();
      hypre_SemiInterpSetup(interp_data_l[l], P_l[l], 0, x_l[l + 1], e_l[l],
                            cindex, findex, stride);

      /* set up the restriction routine */
      restrict_data_l[l] = hypre_SemiRestrictCreate();
      hypre_SemiRestrictSetup(restrict_data_l[l], RT_l[l], 1, r_l[l], b_l[l + 1],
                              cindex, findex, stride);
   }

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (l == num_level_GPU)
   {
      hypre_SetDeviceOff();
   }
#endif

   /*-----------------------------------------------------
    * Check for zero diagonal on coarsest grid, occurs with
    * singular problems like full Neumann or full periodic.
    * Note that a processor with zero diagonal will set
    * active_l =0, other processors will not. This is OK
    * as we only want to avoid the division by zero on the
    * one processor which owns the single coarse grid
    * point.
    *-----------------------------------------------------*/

   if ( hypre_ZeroDiagonal(A_l[l]))
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
         maxwork = 4 * num_levels;
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
      max_iter = (pfmg_data -> max_iter);
      (pfmg_data -> norms)     = hypre_TAlloc(HYPRE_Real, max_iter, HYPRE_MEMORY_HOST);
      (pfmg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real, max_iter, HYPRE_MEMORY_HOST);
   }

#if DEBUG
   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_sprintf(filename, "zout_A.%02d", l);
      hypre_StructMatrixPrint(filename, A_l[l], 0);
      hypre_sprintf(filename, "zout_P.%02d", l);
      hypre_StructMatrixPrint(filename, P_l[l], 0);
   }
   hypre_sprintf(filename, "zout_A.%02d", l);
   hypre_StructMatrixPrint(filename, A_l[l], 0);
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz( hypre_StructMatrix *A,
                       HYPRE_Real         *dxyz,
                       HYPRE_Real         *mean,
                       HYPRE_Real         *deviation)
{
   hypre_BoxArray        *compute_boxes;
   HYPRE_Real             cxyz[3], sqcxyz[3], tcxyz[3];
   HYPRE_Real             cxyz_max;
   HYPRE_Int              tot_size;
   hypre_StructStencil   *stencil;
   //hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
   HYPRE_Int              constant_coefficient;
   HYPRE_Int              i, d;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/
   stencil       = hypre_StructMatrixStencil(A);
   //stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   /*----------------------------------------------------------
    * Compute cxyz (use arithmetic mean)
    *----------------------------------------------------------*/
   cxyz[0] = cxyz[1] = cxyz[2] = 0.0;
   sqcxyz[0] = sqcxyz[1] = sqcxyz[2] = 0.0;

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   tot_size = hypre_StructGridGlobalSize(hypre_StructMatrixGrid(A));

   hypre_ForBoxI(i, compute_boxes)
   {
      /* all coefficients constant or variable diagonal */
      if ( constant_coefficient )
      {
         hypre_PFMGComputeDxyz_CS(i, A, cxyz, sqcxyz);
      }
      /* constant_coefficient==0, all coefficients vary with space */
      else
      {
         switch (stencil_size)
         {
            case 5:
               hypre_PFMGComputeDxyz_SS5 (i, A, cxyz, sqcxyz);
               break;
            case 9:
               hypre_PFMGComputeDxyz_SS9 (i, A, cxyz, sqcxyz);
               break;
            case 7:
               hypre_PFMGComputeDxyz_SS7 (i, A, cxyz, sqcxyz);
               break;
            case 19:
               hypre_PFMGComputeDxyz_SS19(i, A, cxyz, sqcxyz);
               break;
            case 27:
               hypre_PFMGComputeDxyz_SS27(i, A, cxyz, sqcxyz);
               break;
            default:
               hypre_printf("hypre error: unsupported stencil size %d\n", stencil_size);
               hypre_MPI_Abort(hypre_MPI_COMM_WORLD, 1);
         }
      }
   }

   /*----------------------------------------------------------
    * Compute dxyz
    *----------------------------------------------------------*/

   /* all coefficients constant or variable diagonal */
   if ( constant_coefficient )
   {
      for (d = 0; d < 3; d++)
      {
         mean[d] = cxyz[d];
         deviation[d] = sqcxyz[d];
      }
   }
   /* constant_coefficient==0, all coefficients vary with space */
   else
   {
      tcxyz[0] = cxyz[0];
      tcxyz[1] = cxyz[1];
      tcxyz[2] = cxyz[2];
      hypre_MPI_Allreduce(tcxyz, cxyz, 3, HYPRE_MPI_REAL, hypre_MPI_SUM,
                          hypre_StructMatrixComm(A));

      tcxyz[0] = sqcxyz[0];
      tcxyz[1] = sqcxyz[1];
      tcxyz[2] = sqcxyz[2];
      hypre_MPI_Allreduce(tcxyz, sqcxyz, 3, HYPRE_MPI_REAL, hypre_MPI_SUM,
                          hypre_StructMatrixComm(A));

      for (d = 0; d < 3; d++)
      {
         mean[d] = cxyz[d] / tot_size;
         deviation[d] = sqcxyz[d] / tot_size;
      }
   }

   cxyz_max = 0.0;
   for (d = 0; d < 3; d++)
   {
      cxyz_max = hypre_max(cxyz_max, cxyz[d]);
   }
   if (cxyz_max == 0.0)
   {
      /* Do isotropic coarsening */
      for (d = 0; d < 3; d++)
      {
         cxyz[d] = 1.0;
      }
      cxyz_max = 1.0;
   }

   /* Set dxyz values that are scaled appropriately for the coarsening routine */
   for (d = 0; d < 3; d++)
   {
      HYPRE_Real  max_anisotropy = HYPRE_REAL_MAX / 1000;
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

   return hypre_error_flag;
}

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
      tcy -= diag * (a_cs[Ai]  + a_cn[Ai]  +  a_an[Ai] +  a_as[Ai] +  a_bn[Ai] +  a_bs[Ai] + a_csw[Ai] +
                     a_cse[Ai] + a_cnw[Ai] + a_cne[Ai]);
      tcy -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);
      sqcyb += tcy * tcy;
   }
   hypre_BoxLoop1ReductionEnd(Ai, sqcyb);

   HYPRE_Real sqczb = sqcxyz[2];
   hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                A_dbox, start, stride, Ai, sqczb)
   {
      HYPRE_Real tcz = 0.0;
      HYPRE_Real diag = a_cc[Ai] < 0.0 ? -1.0 : 1.0;
      tcz -= diag * (a_ac[Ai]  +  a_bc[Ai] +  a_aw[Ai] +  a_ae[Ai] +  a_an[Ai] +  a_as[Ai] +  a_bw[Ai] +
                     a_be[Ai] + a_bn[Ai] + a_bs[Ai]);
      tcz -= diag * (a_asw[Ai] + a_ase[Ai] + a_anw[Ai] + a_ane[Ai] + a_bsw[Ai] + a_bse[Ai] + a_bnw[Ai] +
                     a_bne[Ai]);
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
 * Returns 1 if there is a diagonal coefficient that is zero,
 * otherwise returns 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ZeroDiagonal( hypre_StructMatrix *A )
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;

   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;

   HYPRE_Real            *Ap;
   hypre_Box             *A_dbox;
   HYPRE_Int              Ai;

   HYPRE_Int              i;

   hypre_Index            diag_index;
   HYPRE_Real             diag_product = 0.0;
   HYPRE_Int              zero_diag = 0;

   HYPRE_Int              constant_coefficient;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int              data_location = hypre_StructGridDataLocation(hypre_StructMatrixGrid(A));
#endif

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   hypre_SetIndex3(stride, 1, 1, 1);
   hypre_SetIndex3(diag_index, 0, 0, 0);

   /* Need to modify here */
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);
      start  = hypre_BoxIMin(compute_box);
      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      Ap = hypre_StructMatrixExtractPointerByIndex(A, i, diag_index);
      hypre_BoxGetStrideSize(compute_box, stride, loop_size);

      if ( constant_coefficient == 1 )
      {
         Ai = hypre_CCBoxIndexRank( A_dbox, start );
         diag_product += Ap[Ai] == 0 ? 1 : 0;
      }
      else
      {
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)
         HYPRE_Real diag_product_local = diag_product;
#elif defined(HYPRE_USING_RAJA)
         ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> diag_product_local(diag_product);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         ReduceSum<HYPRE_Real> diag_product_local(diag_product);
#else
         HYPRE_Real diag_product_local = diag_product;
#endif

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_BOX_REDUCTION map(tofrom:diag_product_local) reduction(+:diag_product_local)
#else
#define HYPRE_BOX_REDUCTION reduction(+:diag_product_local)
#endif

#define DEVICE_VAR is_device_ptr(Ap)
         hypre_BoxLoop1ReductionBegin(hypre_StructMatrixNDim(A), loop_size,
                                      A_dbox, start, stride, Ai, diag_product_local);
         {
            HYPRE_Real one  = 1.0;
            HYPRE_Real zero = 0.0;
            if (Ap[Ai] == 0.0)
            {
               diag_product_local += one;
            }
            else
            {
               diag_product_local += zero;
            }
         }
         hypre_BoxLoop1ReductionEnd(Ai, diag_product_local);

         diag_product += (HYPRE_Real) diag_product_local;
      }
   }

   if (diag_product > 0)
   {
      zero_diag = 1;
   }

   return zero_diag;
}
