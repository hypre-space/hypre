/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

#define DEBUG 0

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
                     
   int                   relax_type = (pfmg_data -> relax_type);
   int                   n_pre      = (pfmg_data -> num_pre_relax);
   int                   n_post     = (pfmg_data -> num_post_relax);
   double               *dxyz       = (pfmg_data -> dxyz);
                     
   int                   max_iter;
   int                   max_levels;
                      
   int                   num_levels;
                     
   int                  *cdirs;

   hypre_Index           cindex;
   hypre_Index           findex;
   hypre_Index           stride;

   int                  *cdir_l;
   hypre_StructGrid    **grid_l;
   hypre_StructGrid    **P_grid_l;
                    
   double               *data;
   int                   data_size = 0;
   hypre_StructMatrix  **A_l;
   hypre_StructMatrix  **P_l;
   hypre_StructMatrix  **RT_l;
   hypre_StructVector  **b_l;
   hypre_StructVector  **x_l;

   /* temp vectors */
   hypre_StructVector  **tx_l;
   hypre_StructVector  **r_l;
   hypre_StructVector  **e_l;
   double               *b_data;
   double               *x_data;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;
   hypre_BoxArray       *all_boxes;
   int                  *processes;
   int                  *box_ranks;
   hypre_BoxArray       *base_all_boxes;
   hypre_Index           pindex;
   hypre_Index           pstride;

   hypre_BoxArray       *P_all_boxes;
   hypre_Index           P_pindex;

   int                   num_boxes;
   int                   num_all_boxes;

   hypre_Box            *box;
   hypre_Box            *cbox;

   double                min_dxyz;
   int                   cdir;
   int                   idmin, idmax;
   int                   i, d, l;
                       
   int                   b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   int                   x_num_ghost[]  = {1, 1, 1, 1, 1, 1};

   int                   ierr = 0;
#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid           = hypre_StructMatrixGrid(A);
   boxes          = hypre_StructGridBoxes(grid);
   all_boxes      = hypre_StructGridAllBoxes(grid);
   processes      = hypre_StructGridProcesses(grid);
   box_ranks      = hypre_StructGridBoxRanks(grid);
   base_all_boxes = hypre_StructGridBaseAllBoxes(grid);
   hypre_CopyIndex(hypre_StructGridPIndex(grid),  pindex);
   hypre_CopyIndex(hypre_StructGridPStride(grid), pstride);
   num_boxes      = hypre_BoxArraySize(boxes);
   num_all_boxes  = hypre_BoxArraySize(all_boxes);

   /* compute all_boxes from base_all_boxes */
   hypre_ForBoxI(i, all_boxes)
      {
         box = hypre_BoxArrayBox(all_boxes, i);
         hypre_CopyBox(hypre_BoxArrayBox(base_all_boxes, i), box);
         hypre_ProjectBox(box, pindex, pstride);
         hypre_PFMGMapFineToCoarse(hypre_BoxIMin(box), pindex, pstride,
                                   hypre_BoxIMin(box));
         hypre_PFMGMapFineToCoarse(hypre_BoxIMax(box), pindex, pstride,
                                   hypre_BoxIMax(box));
      }

   /* allocate P_all_boxes */
   P_all_boxes = hypre_NewBoxArray(num_all_boxes);

   /* Compute a bounding box (cbox) used to determine cdir */
   cbox = hypre_NewBox();
   for (d = 0; d < 3; d++)
   {
      idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), d);
      idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), d);
      for (i = 0; i < num_all_boxes; i++)
      {
         idmin = min(idmin,
                     hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), d));
         idmax = max(idmax,
                     hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), d));
      }
      hypre_BoxIMinD(cbox, d) = idmin;
      hypre_BoxIMaxD(cbox, d) = idmax;
   }

   /* Compute a new max_levels value based on the grid */
   max_levels =
      hypre_Log2(hypre_BoxSizeD(cbox, 0)) + 2 +
      hypre_Log2(hypre_BoxSizeD(cbox, 1)) + 2 +
      hypre_Log2(hypre_BoxSizeD(cbox, 2)) + 2;
   if ((pfmg_data -> max_levels) > 0)
   {
      max_levels = min(max_levels, (pfmg_data -> max_levels));
   }
   (pfmg_data -> max_levels) = max_levels;

   cdir_l = hypre_TAlloc(int, max_levels);
   grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
   grid_l[0] = grid;
   P_grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
   P_grid_l[0] = NULL;
   for (l = 0; ; l++)
   {
      /* determine cdir */
      min_dxyz = dxyz[0] + dxyz[1] + dxyz[2] + 1;
      cdir = -1;
      for (d = 0; d < 3; d++)
      {
         idmin = hypre_BoxIMinD(cbox, d);
         idmax = hypre_BoxIMaxD(cbox, d);
         if ((idmax > idmin) && (dxyz[d] < min_dxyz))
         {
            min_dxyz = dxyz[d];
            cdir = d;
         }
      }

      /* if cannot coarsen in any direction, stop */
      if ( (cdir == -1) || (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         break;
      }

      cdir_l[l] = cdir;

      /* set cindex, findex, and stride */
      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetFIndex(cdir, findex);
      hypre_PFMGSetStride(cdir, stride);

      /* compute new P_pindex, pindex, and pstride */
      for (d = 0; d < 3; d++)
      {
         hypre_IndexD(P_pindex, d) = hypre_IndexD(pindex, d) +
            hypre_IndexD(findex, d) * hypre_IndexD(pstride, d);
         hypre_IndexD(pindex, d) +=
            hypre_IndexD(cindex, d) * hypre_IndexD(pstride, d);
         hypre_IndexD(pstride, d) *= hypre_IndexD(stride, d);
      }

      /* update dxyz and coarsen cbox*/
      dxyz[cdir] *= 2;
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_PFMGMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
                                hypre_BoxIMin(cbox));
      hypre_PFMGMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
                                hypre_BoxIMax(cbox));

      /*---------------------------------------
       * build the grid for interpolation P
       *---------------------------------------*/

      /* build from all_boxes (reduces communication) */
      for (i = 0; i < num_all_boxes; i++)
      {
         hypre_CopyBox(hypre_BoxArrayBox(all_boxes, i),
                       hypre_BoxArrayBox(P_all_boxes, i));
      }
      hypre_ProjectBoxArray(P_all_boxes, findex, stride);
      for (i = 0; i < num_all_boxes; i++)
      {
         box = hypre_BoxArrayBox(P_all_boxes, i);
         hypre_PFMGMapFineToCoarse(hypre_BoxIMin(box), findex, stride,
                                   hypre_BoxIMin(box));
         hypre_PFMGMapFineToCoarse(hypre_BoxIMax(box), findex, stride,
                                   hypre_BoxIMax(box));
      }

      /* compute local boxes */
      boxes = hypre_NewBoxArray(num_boxes);
      for (i = 0; i < num_boxes; i++)
      {
         hypre_CopyBox(hypre_BoxArrayBox(P_all_boxes, box_ranks[i]),
                       hypre_BoxArrayBox(boxes, i));
      }

      P_grid_l[l+1] = hypre_NewStructGrid(comm, hypre_StructGridDim(grid));
      hypre_SetStructGridBoxes(P_grid_l[l+1], boxes);
      hypre_SetStructGridGlobalInfo(P_grid_l[l+1],
                                    P_all_boxes, processes, box_ranks,
                                    base_all_boxes, P_pindex, pstride);
      hypre_AssembleStructGrid(P_grid_l[l+1]);

      /*---------------------------------------
       * build the coarse grid
       *---------------------------------------*/

      /* coarsen the grid by coarsening all_boxes (reduces communication) */
      hypre_ProjectBoxArray(all_boxes, cindex, stride);
      for (i = 0; i < num_all_boxes; i++)
      {
         box = hypre_BoxArrayBox(all_boxes, i);
         hypre_PFMGMapFineToCoarse(hypre_BoxIMin(box), cindex, stride,
                                   hypre_BoxIMin(box));
         hypre_PFMGMapFineToCoarse(hypre_BoxIMax(box), cindex, stride,
                                   hypre_BoxIMax(box));
      }

      /* compute local boxes */
      boxes = hypre_NewBoxArray(num_boxes);
      for (i = 0; i < num_boxes; i++)
      {
         hypre_CopyBox(hypre_BoxArrayBox(all_boxes, box_ranks[i]),
                       hypre_BoxArrayBox(boxes, i));
      }

      grid_l[l+1] = hypre_NewStructGrid(comm, hypre_StructGridDim(grid));
      hypre_SetStructGridBoxes(grid_l[l+1], boxes);
      hypre_SetStructGridGlobalInfo(grid_l[l+1],
                                    all_boxes, processes, box_ranks,
                                    base_all_boxes, pindex, pstride);
      hypre_AssembleStructGrid(grid_l[l+1]);
   }
   num_levels = l + 1;

   /* free up some things */
   hypre_FreeBoxArray(P_all_boxes);
   hypre_FreeBox(cbox);

   (pfmg_data -> num_levels) = num_levels;
   (pfmg_data -> cdir_l)     = cdir_l;
   (pfmg_data -> grid_l)     = grid_l;
   (pfmg_data -> P_grid_l)   = P_grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels);
   P_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
   RT_l = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
   b_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
   x_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
   tx_l = hypre_TAlloc(hypre_StructVector *, num_levels);
   r_l  = tx_l;
   e_l  = tx_l;

   A_l[0] = A;
   b_l[0] = b;
   x_l[0] = x;

   tx_l[0] = hypre_NewStructVector(comm, grid_l[0]);
   hypre_SetStructVectorNumGhost(tx_l[0], x_num_ghost);
   hypre_InitializeStructVectorShell(tx_l[0]);
   data_size += hypre_StructVectorDataSize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      cdir = cdir_l[l];

      P_l[l]  = hypre_PFMGNewInterpOp(A_l[l], P_grid_l[l+1], cdir);
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
         RT_l[l]   = hypre_PFMGNewRestrictOp(A_l[l], grid_l[l+1], cdir);
         data_size += hypre_StructMatrixDataSize(RT_l[l]);
#endif
      }

      A_l[l+1] = hypre_PFMGNewRAPOp(RT_l[l], A_l[l], P_l[l],
                                    grid_l[l+1], cdir);
      data_size += hypre_StructMatrixDataSize(A_l[l+1]);

      b_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(b_l[l+1], b_num_ghost);
      hypre_InitializeStructVectorShell(b_l[l+1]);
      data_size += hypre_StructVectorDataSize(b_l[l+1]);

      x_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      hypre_InitializeStructVectorShell(x_l[l+1]);
      data_size += hypre_StructVectorDataSize(x_l[l+1]);

      tx_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(tx_l[l+1], x_num_ghost);
      hypre_InitializeStructVectorShell(tx_l[l+1]);
   }

   data = hypre_SharedCTAlloc(double, data_size);
   (pfmg_data -> data) = data;

   hypre_InitializeStructVectorData(tx_l[0], data);
   hypre_AssembleStructVector(tx_l[0]);
   data += hypre_StructVectorDataSize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_InitializeStructMatrixData(P_l[l], data);
      data += hypre_StructMatrixDataSize(P_l[l]);

#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
      {
         hypre_InitializeStructMatrixData(RT_l[l], data);
         data += hypre_StructMatrixDataSize(RT_l[l]);
      }
#endif

      hypre_InitializeStructMatrixData(A_l[l+1], data);
      data += hypre_StructMatrixDataSize(A_l[l+1]);

      hypre_InitializeStructVectorData(b_l[l+1], data);
      hypre_AssembleStructVector(b_l[l+1]);
      data += hypre_StructVectorDataSize(b_l[l+1]);

      hypre_InitializeStructVectorData(x_l[l+1], data);
      hypre_AssembleStructVector(x_l[l+1]);
      data += hypre_StructVectorDataSize(x_l[l+1]);

      hypre_InitializeStructVectorData(tx_l[l+1],
                                       hypre_StructVectorData(tx_l[0]));
      hypre_AssembleStructVector(tx_l[l+1]);
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
      hypre_PFMGSetupInterpOp(A_l[l], cdir, findex, stride, P_l[l]);

      /* set up the restriction operator */
#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
         hypre_PFMGSetupRestrictOp(A_l[l], tx_l[l],
                                   cdir, cindex, stride, RT_l[l]);
#endif

      /* set up the coarse grid operator */
      hypre_PFMGSetupRAPOp(RT_l[l], A_l[l], P_l[l],
                           cdir, cindex, stride, A_l[l+1]);

      /* set up the interpolation routine */
      interp_data_l[l] = hypre_PFMGInterpInitialize();
      hypre_PFMGInterpSetup(interp_data_l[l], P_l[l], x_l[l+1], e_l[l],
                            cindex, findex, stride);

      /* set up the restriction routine */
      restrict_data_l[l] = hypre_PFMGRestrictInitialize();
      hypre_PFMGRestrictSetup(restrict_data_l[l], RT_l[l], r_l[l], b_l[l+1],
                              cindex, findex, stride);
   }

   /* set up fine grid relaxation */
   relax_data_l[0] = hypre_PFMGRelaxInitialize(comm);
   hypre_PFMGRelaxSetTol(relax_data_l[0], 0.0);
   hypre_PFMGRelaxSetType(relax_data_l[0], relax_type);
   hypre_PFMGRelaxSetTempVec(relax_data_l[0], tx_l[0]);
   hypre_PFMGRelaxSetup(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
   if (num_levels > 1)
   {
      for (l = 1; l < (num_levels - 1); l++)
      {
         /* set up relaxation */
         relax_data_l[l] = hypre_PFMGRelaxInitialize(comm);
         hypre_PFMGRelaxSetTol(relax_data_l[l], 0.0);
         hypre_PFMGRelaxSetType(relax_data_l[l], relax_type);
         hypre_PFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
         hypre_PFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
      }
      /* set up coarsest grid relaxation */
      relax_data_l[l] = hypre_PFMGRelaxInitialize(comm);
      hypre_PFMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_PFMGRelaxSetMaxIter(relax_data_l[l], 1);
      hypre_PFMGRelaxSetType(relax_data_l[l], 0);
      hypre_PFMGRelaxSetTempVec(relax_data_l[l], tx_l[l]);
      hypre_PFMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
   }

   for (l = 0; l < num_levels; l++)
   {
      /* set up the residual routine */
      matvec_data_l[l] = hypre_StructMatvecInitialize();
      hypre_StructMatvecSetup(matvec_data_l[l], A_l[l], x_l[l]);
   }

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
      hypre_PrintStructMatrix(filename, A_l[l], 0);
      sprintf(filename, "zout_P.%02d", l);
      hypre_PrintStructMatrix(filename, P_l[l], 0);
   }
   sprintf(filename, "zout_A.%02d", l);
   hypre_PrintStructMatrix(filename, A_l[l], 0);
#endif

   return ierr;
}

