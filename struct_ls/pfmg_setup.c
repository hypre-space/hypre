/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

#define DEBUG 0

#define hypre_PFMGSetCIndex(cdir, cindex) \
{\
   hypre_SetIndex(cindex, 0, 0, 0);\
   hypre_IndexD(cindex, cdir) = 0;\
}

#define hypre_PFMGSetFIndex(cdir, findex) \
{\
   hypre_SetIndex(findex, 0, 0, 0);\
   hypre_IndexD(findex, cdir) = 1;\
}

#define hypre_PFMGSetStride(cdir, stride) \
{\
   hypre_SetIndex(stride, 1, 1, 1);\
   hypre_IndexD(stride, cdir) = 2;\
}


/*--------------------------------------------------------------------------
 * hypre_PFMGSetup
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSetup( void               *pfmg_vdata,
                 hypre_StructMatrix *A,
                 hypre_StructVector *b,
                 hypre_StructVector *x        )
{
   hypre_PFMGData       *pfmg_data = pfmg_vdata;

   MPI_Comm              comm = (pfmg_data -> comm);
                     
   int                   relax_type =       (pfmg_data -> relax_type);
   int                   usr_jacobi_weight= (pfmg_data -> usr_jacobi_weight);
   double                jacobi_weight    = (pfmg_data -> jacobi_weight);
   int                   skip_relax =       (pfmg_data -> skip_relax);
   double               *dxyz       =       (pfmg_data -> dxyz);
   int                   rap_type;
                     
   int                   max_iter;
   int                   max_levels;
                      
   int                   num_levels;
                     
   hypre_Index           cindex;
   hypre_Index           findex;
   hypre_Index           stride;

   hypre_Index           coarsen;

   int                  *cdir_l;
   int                  *active_l;
   hypre_StructGrid    **grid_l;
   hypre_StructGrid    **P_grid_l;
                    
   double               *data;
   int                   data_size = 0;
   double               *relax_weights;
   double               *mean, *deviation;
   double                alpha, beta;

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
   int                   dim;

   hypre_Box            *cbox;

   double                min_dxyz;
   int                   cdir;
   int                   d, l;
   int                   dxyz_flag;
                       
   int                   b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   int                   x_num_ghost[]  = {1, 1, 1, 1, 1, 1};

   int                   ierr = 0;
#if DEBUG
   char                  filename[255];
#endif


   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid  = hypre_StructMatrixGrid(A);
   dim   = hypre_StructGridDim(grid);

   /* Compute a new max_levels value based on the grid */
   cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(grid));
   max_levels =
      hypre_Log2(hypre_BoxSizeD(cbox, 0)) + 2 +
      hypre_Log2(hypre_BoxSizeD(cbox, 1)) + 2 +
      hypre_Log2(hypre_BoxSizeD(cbox, 2)) + 2;
   if ((pfmg_data -> max_levels) > 0)
   {
      max_levels = hypre_min(max_levels, (pfmg_data -> max_levels));
   }
   (pfmg_data -> max_levels) = max_levels;

   /* compute dxyz */
   if ((dxyz[0] == 0) || (dxyz[1] == 0) || (dxyz[2] == 0))
   {
     
      mean= hypre_CTAlloc(double, 3);
      deviation= hypre_CTAlloc(double, 3);
      hypre_PFMGComputeDxyz(A, dxyz, mean, deviation);
        
      dxyz_flag= 0;
      for (d= 0; d< dim; d++)
      {
         deviation[d]-= mean[d]*mean[d];
         if (deviation[d]/(mean[d]*mean[d]) > .1)  /* square of coeff. of variation */
         {
            dxyz_flag= 1;
            break;
         }
      }
      hypre_TFree(mean);
      hypre_TFree(deviation);
   }

   grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
   hypre_StructGridRef(grid, &grid_l[0]);
   P_grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
   P_grid_l[0] = NULL;
   cdir_l = hypre_TAlloc(int, max_levels);
   active_l = hypre_TAlloc(int, max_levels);
   relax_weights= hypre_CTAlloc(double, max_levels);
   hypre_SetIndex(coarsen, 1, 1, 1); /* forces relaxation on finest grid */
   for (l = 0; ; l++)
   {
      /* determine cdir */
      min_dxyz = dxyz[0] + dxyz[1] + dxyz[2] + 1;
      cdir = -1;
      alpha= 0.0;
      for (d = 0; d < dim; d++)
      {
         if ((hypre_BoxIMaxD(cbox, d) > hypre_BoxIMinD(cbox, d)) &&
             (dxyz[d] < min_dxyz))
         {
            min_dxyz = dxyz[d];
            cdir = d;
         }
         alpha+= 1.0/(dxyz[d]*dxyz[d]);
      }
      relax_weights[l]= 2.0/3.0;
                                                                                                                               
      beta= 0.0;
      if (cdir != -1)
      {
         if (dxyz_flag)
         {
            relax_weights[l]= 2.0/3.0;
         }

         else
         {
            for (d = 0; d < dim; d++)
            {
               if (d != cdir)
               {
                  beta+= 1.0/(dxyz[d]*dxyz[d]);
               }
            }
            if (beta == alpha)
            {
               alpha= 0.0;
            }
            else
            {
               alpha= beta/alpha;
            }

            /* determine level Jacobi weights */
            if (dim > 1)
            {
               relax_weights[l]= 2.0/(3.0 - alpha);
            }
            else
            {
               relax_weights[l]= 2.0/3.0; /* always 2/3 for 1-d */
            }
         }
      }

      /* if cannot coarsen in any direction, stop */
      if ( (cdir == -1) || (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         active_l[l] = 1; /* forces relaxation on coarsest grid */
         break;
      }

      cdir_l[l] = cdir;

      if (hypre_IndexD(coarsen, cdir) != 0)
      {
         /* coarsened previously in this direction, relax level l */
         active_l[l] = 1;
         hypre_SetIndex(coarsen, 0, 0, 0);
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
      hypre_StructCoarsen(grid_l[l], findex, stride, 0, &P_grid_l[l+1]);

      /* build the coarse grid */
      hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l+1]);
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
      (pfmg_data -> rap_type)= 1;
   }
   rap_type = (pfmg_data -> rap_type);

   A_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels);
   P_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
   RT_l = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
   b_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
   x_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
   tx_l = hypre_TAlloc(hypre_StructVector *, num_levels);
   r_l  = tx_l;
   e_l  = tx_l;

   A_l[0] = hypre_StructMatrixRef(A);
   b_l[0] = hypre_StructVectorRef(b);
   x_l[0] = hypre_StructVectorRef(x);

   tx_l[0] = hypre_StructVectorCreate(comm, grid_l[0]);
   hypre_StructVectorSetNumGhost(tx_l[0], x_num_ghost);
   hypre_StructVectorInitializeShell(tx_l[0]);
   data_size += hypre_StructVectorDataSize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      P_l[l]  = hypre_PFMGCreateInterpOp(A_l[l], P_grid_l[l+1], cdir, rap_type);
      hypre_StructMatrixInitializeShell(P_l[l]);
      data_size += hypre_StructMatrixDataSize(P_l[l]);

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
         RT_l[l]   = hypre_PFMGCreateRestrictOp(A_l[l], grid_l[l+1], cdir);
         hypre_StructMatrixInitializeShell(RT_l[l]);
         data_size += hypre_StructMatrixDataSize(RT_l[l]);
#endif
      }

      A_l[l+1] = hypre_PFMGCreateRAPOp(RT_l[l], A_l[l], P_l[l],
                                       grid_l[l+1], cdir, rap_type);
      hypre_StructMatrixInitializeShell(A_l[l+1]);
      data_size += hypre_StructMatrixDataSize(A_l[l+1]);

      b_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(b_l[l+1], b_num_ghost);
      hypre_StructVectorInitializeShell(b_l[l+1]);
      data_size += hypre_StructVectorDataSize(b_l[l+1]);

      x_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(x_l[l+1], x_num_ghost);
      hypre_StructVectorInitializeShell(x_l[l+1]);
      data_size += hypre_StructVectorDataSize(x_l[l+1]);

      tx_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(tx_l[l+1], x_num_ghost);
      hypre_StructVectorInitializeShell(tx_l[l+1]);
   }

   data = hypre_SharedCTAlloc(double, data_size);
   (pfmg_data -> data) = data;

   hypre_StructVectorInitializeData(tx_l[0], data);
   hypre_StructVectorAssemble(tx_l[0]);
   data += hypre_StructVectorDataSize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_StructMatrixInitializeData(P_l[l], data);
      data += hypre_StructMatrixDataSize(P_l[l]);

#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
      {
         hypre_StructMatrixInitializeData(RT_l[l], data);
         data += hypre_StructMatrixDataSize(RT_l[l]);
      }
#endif

      hypre_StructMatrixInitializeData(A_l[l+1], data);
      data += hypre_StructMatrixDataSize(A_l[l+1]);

      hypre_StructVectorInitializeData(b_l[l+1], data);
      hypre_StructVectorAssemble(b_l[l+1]);
      data += hypre_StructVectorDataSize(b_l[l+1]);

      hypre_StructVectorInitializeData(x_l[l+1], data);
      hypre_StructVectorAssemble(x_l[l+1]);
      data += hypre_StructVectorDataSize(x_l[l+1]);

      hypre_StructVectorInitializeData(tx_l[l+1],
                                       hypre_StructVectorData(tx_l[0]));
      hypre_StructVectorAssemble(tx_l[l+1]);
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

   relax_data_l    = hypre_TAlloc(void *, num_levels);
   matvec_data_l   = hypre_TAlloc(void *, num_levels);
   restrict_data_l = hypre_TAlloc(void *, num_levels);
   interp_data_l   = hypre_TAlloc(void *, num_levels);

   for (l = 0; l < (num_levels - 1); l++)
   {
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
                           cdir, cindex, stride, rap_type, A_l[l+1]);

      /* set up the interpolation routine */
      interp_data_l[l] = hypre_SemiInterpCreate();
      hypre_SemiInterpSetup(interp_data_l[l], P_l[l], 0, x_l[l+1], e_l[l],
                            cindex, findex, stride);

      /* set up the restriction routine */
      restrict_data_l[l] = hypre_SemiRestrictCreate();
      hypre_SemiRestrictSetup(restrict_data_l[l], RT_l[l], 1, r_l[l], b_l[l+1],
                              cindex, findex, stride);
   }

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
      for (l = 1; l < (num_levels - 1); l++)
      {
         /* set up relaxation */
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
            hypre_PFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }
      }
      /* set up coarsest grid relaxation */
      if (active_l[l])
      {
         relax_data_l[l] = hypre_PFMGRelaxCreate(comm);
         hypre_PFMGRelaxSetTol(relax_data_l[l], 0.0);
         hypre_PFMGRelaxSetMaxIter(relax_data_l[l], 1);
         if (usr_jacobi_weight)
         {
            hypre_PFMGRelaxSetJacobiWeight(relax_data_l[l], jacobi_weight);
         }
         else
         {
            hypre_PFMGRelaxSetJacobiWeight(relax_data_l[l], 1.0);
         }
         hypre_PFMGRelaxSetType(relax_data_l[l], 0);
         hypre_PFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
         hypre_PFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
      }
   }
   hypre_TFree(relax_weights);

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
      (pfmg_data -> norms)     = hypre_TAlloc(double, max_iter);
      (pfmg_data -> rel_norms) = hypre_TAlloc(double, max_iter);
   }

#if DEBUG
   for (l = 0; l < (num_levels - 1); l++)
   {
      sprintf(filename, "zout_A.%02d", l);
      hypre_StructMatrixPrint(filename, A_l[l], 0);
      sprintf(filename, "zout_P.%02d", l);
      hypre_StructMatrixPrint(filename, P_l[l], 0);
   }
   sprintf(filename, "zout_A.%02d", l);
   hypre_StructMatrixPrint(filename, A_l[l], 0);
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeDxyz
 *--------------------------------------------------------------------------*/

int
hypre_PFMGComputeDxyz( hypre_StructMatrix *A,
                       double             *dxyz,
                       double             *mean,
                       double             *deviation)
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *A_dbox;
                        
   int                    Ai;
                        
   double                *Ap;
   double                 cxyz[3];
   double                 sqcxyz[3];
   double                 tcxyz[3];
   double                 cxyz_max;

   int                    tot_size; 

   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   int                    stencil_size;

   int                    constant_coefficient;
                        
   int                    Astenc;
                        
   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;
   hypre_Index            diag_index;
                        
   int                    i, si, d;
   int                    loopi, loopj, loopk;
   int                    diag_exists, diag_rank;

   int                    ierr = 0;
   double                 cx, cy, cz;
   double                 sqcx, sqcy, sqcz;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   hypre_SetIndex(stride, 1, 1, 1);

   /*----------------------------------------------------------
    * Compute cxyz (use arithmetic mean)
    *----------------------------------------------------------*/

   cxyz[0] = 0.0;
   cxyz[1] = 0.0;
   cxyz[2] = 0.0;

   sqcxyz[0] = 0.0;
   sqcxyz[1] = 0.0;
   sqcxyz[2] = 0.0;

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient==2 )
   {
      hypre_SetIndex(diag_index, 0, 0, 0);
      diag_rank = hypre_StructStencilElementRank(stencil, diag_index);
   }

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));

   tot_size= hypre_StructGridGlobalSize(hypre_StructMatrixGrid(A));

   hypre_ForBoxI(i, compute_boxes)
      {
         compute_box = hypre_BoxArrayBox(compute_boxes, i);


         A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);

         start  = hypre_BoxIMin(compute_box);

         hypre_BoxGetStrideSize(compute_box, stride, loop_size);

         cx = cxyz[0];
         cy = cxyz[1];
         cz = cxyz[2];

         sqcx = sqcxyz[0];
         sqcy = sqcxyz[1];
         sqcz = sqcxyz[2];

         if ( constant_coefficient==1 )
            /* all coefficients constant */
         {
            Ai = hypre_CCBoxIndexRank( A_dbox, start );

            tcxyz[0] = 0.0;
            tcxyz[1] = 0.0;
            tcxyz[2] = 0.0;

            for (si = 0; si < stencil_size; si++)
            {
               Ap = hypre_StructMatrixBoxData(A, i, si);

               /* x-direction */
               Astenc = hypre_IndexD(stencil_shape[si], 0);
               if (Astenc)
               {
                  tcxyz[0] -= Ap[Ai];
               }

               /* y-direction */
               Astenc = hypre_IndexD(stencil_shape[si], 1);
               if (Astenc)
               {
                  tcxyz[1] -= Ap[Ai];
               }

               /* z-direction */
               Astenc = hypre_IndexD(stencil_shape[si], 2);
               if (Astenc)
               {
                  tcxyz[2] -= Ap[Ai];
               }
            }

            cx += tcxyz[0];
            cy += tcxyz[1];
            cz += tcxyz[2];

            sqcx += (tcxyz[0]*tcxyz[0]);
            sqcy += (tcxyz[1]*tcxyz[1]);
            sqcz += (tcxyz[2]*tcxyz[2]);
         }
         else if ( constant_coefficient==2 )
            /* variable diagonal, other coefficients constant */
         {
            Ai = hypre_CCBoxIndexRank( A_dbox, start );

            tcxyz[0] = 0.0;
            tcxyz[1] = 0.0;
            tcxyz[2] = 0.0;

            diag_exists = 0;
            for (si = 0; si < stencil_size; si++)
            {
               if (si == diag_rank)
               {
                  diag_exists = 1;
               }
               else
               {
                  /* constant coeff part of matrix, same as constant_coefficient==1 */
                  Ap = hypre_StructMatrixBoxData(A, i, si);

                  /* x-direction */
                  Astenc = hypre_IndexD(stencil_shape[si], 0);
                  if (Astenc)
                  {
                     tcxyz[0] -= Ap[Ai];
                  }

                  /* y-direction */
                  Astenc = hypre_IndexD(stencil_shape[si], 1);
                  if (Astenc)
                  {
                     tcxyz[1] -= Ap[Ai];
                  }

                  /* z-direction */
                  Astenc = hypre_IndexD(stencil_shape[si], 2);
                  if (Astenc)
                  {
                     tcxyz[2] -= Ap[Ai];
                  }
               }
            }

            /* variable diagonal part is same as constant_coefficient==0 */
            if ( diag_exists )
            {
               hypre_BoxLoop1Begin(loop_size, A_dbox, start, stride, Ai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai
#define HYPRE_SMP_REDUCTION_OP +
#define HYPRE_SMP_REDUCTION_VARS cx,cy,cz
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop1For(loopi, loopj, loopk, Ai)
                  {
                     tcxyz[0] = 0.0;
                     tcxyz[1] = 0.0;
                     tcxyz[2] = 0.0;

                     si = diag_rank;
                     Ap = hypre_StructMatrixBoxData(A, i, si);

                     /* x-direction */
                     Astenc = hypre_IndexD(stencil_shape[si], 0);
                     if (Astenc)
                     {
                        tcxyz[0] -= Ap[Ai];
                     }

                     /* y-direction */
                     Astenc = hypre_IndexD(stencil_shape[si], 1);
                     if (Astenc)
                     {
                        tcxyz[1] -= Ap[Ai];
                     }

                     /* z-direction */
                     Astenc = hypre_IndexD(stencil_shape[si], 2);
                     if (Astenc)
                     {
                        tcxyz[2] -= Ap[Ai];
                     }

                     cx += tcxyz[0];
                     cy += tcxyz[1];
                     cz += tcxyz[2];

                     sqcx += (tcxyz[0]*tcxyz[0]);
                     sqcy += (tcxyz[1]*tcxyz[1]);
                     sqcz += (tcxyz[2]*tcxyz[2]);
                  }
               hypre_BoxLoop1End(Ai);
            }

            cx += tcxyz[0];
            cy += tcxyz[1];
            cz += tcxyz[2];

            sqcx += (tcxyz[0]*tcxyz[0]);
            sqcy += (tcxyz[1]*tcxyz[1]);
            sqcz += (tcxyz[2]*tcxyz[2]);
         }
         else
            /* constant_coefficient==0, all coefficients vary with space */
         {
            hypre_BoxLoop1Begin(loop_size, A_dbox, start, stride, Ai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai
#define HYPRE_SMP_REDUCTION_OP +
#define HYPRE_SMP_REDUCTION_VARS cx,cy,cz
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop1For(loopi, loopj, loopk, Ai)
               {
                  tcxyz[0] = 0.0;
                  tcxyz[1] = 0.0;
                  tcxyz[2] = 0.0;

                  for (si = 0; si < stencil_size; si++)
                  {
                     Ap = hypre_StructMatrixBoxData(A, i, si);

                     /* x-direction */
                     Astenc = hypre_IndexD(stencil_shape[si], 0);
                     if (Astenc)
                     {
                        tcxyz[0] -= Ap[Ai];
                     }

                     /* y-direction */
                     Astenc = hypre_IndexD(stencil_shape[si], 1);
                     if (Astenc)
                     {
                        tcxyz[1] -= Ap[Ai];
                     }

                     /* z-direction */
                     Astenc = hypre_IndexD(stencil_shape[si], 2);
                     if (Astenc)
                     {
                        tcxyz[2] -= Ap[Ai];
                     }
                  }

                  cx += tcxyz[0];
                  cy += tcxyz[1];
                  cz += tcxyz[2];

                  sqcx += (tcxyz[0]*tcxyz[0]);
                  sqcy += (tcxyz[1]*tcxyz[1]);
                  sqcz += (tcxyz[2]*tcxyz[2]);
               }
            hypre_BoxLoop1End(Ai);

         }

         cxyz[0] = cx;
         cxyz[1] = cy;
         cxyz[2] = cz;

         sqcxyz[0] = sqcx;
         sqcxyz[1] = sqcy;
         sqcxyz[2] = sqcz;
      }

   /*----------------------------------------------------------
    * Compute dxyz
    *----------------------------------------------------------*/

   tcxyz[0] = cxyz[0];
   tcxyz[1] = cxyz[1];
   tcxyz[2] = cxyz[2];
   MPI_Allreduce(tcxyz, cxyz, 3, MPI_DOUBLE, MPI_SUM,
                 hypre_StructMatrixComm(A));

   tcxyz[0] = sqcxyz[0];
   tcxyz[1] = sqcxyz[1];
   tcxyz[2] = sqcxyz[2];
   MPI_Allreduce(tcxyz, sqcxyz, 3, MPI_DOUBLE, MPI_SUM,
                 hypre_StructMatrixComm(A));

   for (d= 0; d< 3; d++)
   {
      mean[d]= cxyz[d]/tot_size;
      deviation[d]= sqcxyz[d]/tot_size;
   }
     
   cxyz_max = 0.0;
   for (d = 0; d < 3; d++)
   {
      cxyz_max = hypre_max(cxyz_max, cxyz[d]);
   }
   if (cxyz_max == 0.0)
   {
      cxyz_max = 1.0;
   }

   for (d = 0; d < 3; d++)
   {
      if (cxyz[d] > 0)
      {
         cxyz[d] /= cxyz_max;
         dxyz[d] = sqrt(1.0 / cxyz[d]);
      }
      else
      {
         dxyz[d] = 1.0e+123;
      }
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ZeroDiagonal
 *
 * Returns 1 if there is a diagonal coefficient that is zero,
 * otherwise returns 0.
 *--------------------------------------------------------------------------*/

int
hypre_ZeroDiagonal( hypre_StructMatrix *A )
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;

   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            stride;

   double                *Ap;
   hypre_Box             *A_dbox;
   int                    Ai;

   int                    i;
   int                    loopi, loopj, loopk;

   hypre_Index            diag_index;
   double                 diag_product = 1.0;
   int                    zero_diag = 0;

   int                    constant_coefficient; 

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);
   hypre_SetIndex(diag_index, 0, 0, 0);

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

         if ( constant_coefficient )
         {
            Ai = hypre_CCBoxIndexRank( A_dbox, start );
            diag_product *= Ap[Ai];
         }
         else
         {

            hypre_BoxLoop1Begin(loop_size, A_dbox, start, stride, Ai);
            hypre_BoxLoop1For(loopi, loopj, loopk, Ai)
               {
                  diag_product *= Ap[Ai];
               }
            hypre_BoxLoop1End(Ai);
         }

      }

   if (diag_product == 0)
   {
      zero_diag = 1;
   }
   
   return zero_diag;
}





