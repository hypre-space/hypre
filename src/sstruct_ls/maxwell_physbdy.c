/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   cnt
 *
 ******************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * Finds the physical boundary boxes for all levels. Since the coarse grid's
 * boundary may not be on the physical bdry, we need to compare the coarse
 * grid to the finest level boundary boxes. All boxes of the coarse grids
 * must be checked, not just the bounding box.
 *    Algo:
 *         1) obtain boundary boxes for the finest grid
 *             i) mark the fboxes that have boundary elements. 
 *         2) loop over coarse levels
 *             i) for a cbox that maps to a fbox that has boundary layers
 *                a) refine the cbox 
 *                b) intersect with the cell boundary layers of the fbox
 *                c) coarsen the intersection 
 *            ii) determine the var boxes
 *           iii) mark the coarse box 
 *
 * Concerns: Checking an individual pgrid may give artificial physical 
 * boundaries. Need to check if any other pgrid is adjacent to it. 
 * We omit this case and assume only one part for now.
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_Maxwell_PhysBdy( hypre_SStructGrid      **grid_l,
                       HYPRE_Int                num_levels,
                       hypre_Index              rfactors,
                       HYPRE_Int             ***BdryRanksl_ptr, 
                       HYPRE_Int              **BdryRanksCntsl_ptr )
{

   MPI_Comm                comm= (grid_l[0]-> comm);

   HYPRE_Int             **BdryRanks_l;
   HYPRE_Int              *BdryRanksCnts_l;

   HYPRE_Int              *npts;
   HYPRE_Int              *ranks, *upper_rank, *lower_rank;
   hypre_BoxManEntry      *boxman_entry;

   hypre_SStructGrid      *grid;
   hypre_SStructPGrid     *pgrid;
   hypre_StructGrid       *cell_fgrid, *cell_cgrid, *sgrid;

   hypre_BoxArrayArray ****bdry;
   hypre_BoxArrayArray    *fbdry;
   hypre_BoxArrayArray    *cbdry;

   hypre_BoxArray         *box_array;
   hypre_BoxArray         *fboxes, *cboxes;

   hypre_Box              *fbox, *cbox;
   hypre_Box              *box, *contract_fbox, rbox;
   hypre_Box               intersect;

   HYPRE_Int             **cbox_mapping, **fbox_mapping;
   HYPRE_Int             **boxes_with_bdry;

   HYPRE_Int               ndim, nvars;
   HYPRE_Int               nboxes, nfboxes;
   HYPRE_Int               boxi;
   
   hypre_Index             zero_shift, upper_shift, lower_shift;
   hypre_Index             loop_size, start, index;
   HYPRE_Int               loopi, loopj, loopk;

   HYPRE_Int               i, j, k, l, m, n, p;
   HYPRE_Int               d;
   HYPRE_Int               cnt;

   HYPRE_Int               part= 0;  /* NOTE, ASSUMING ONE PART */
   HYPRE_Int               matrix_type= HYPRE_PARCSR;
   HYPRE_Int               myproc;

   HYPRE_Int               ierr= 0;

   hypre_MPI_Comm_rank(comm, &myproc);

   ndim= hypre_SStructGridNDim(grid_l[0]);
   hypre_SetIndex(zero_shift, 0, 0, 0);

  /* bounding global ranks of this processor & allocate boundary box markers. */
   upper_rank= hypre_CTAlloc(HYPRE_Int, num_levels);
   lower_rank= hypre_CTAlloc(HYPRE_Int, num_levels);

   boxes_with_bdry= hypre_TAlloc(HYPRE_Int *, num_levels);
   for (i= 0; i< num_levels; i++)
   {
      grid = grid_l[i];
      lower_rank[i]= hypre_SStructGridStartRank(grid);

     /* note we are assuming only one part */
      pgrid= hypre_SStructGridPGrid(grid, part);
      nvars= hypre_SStructPGridNVars(pgrid);
      sgrid= hypre_SStructPGridSGrid(pgrid, nvars-1);
      box_array= hypre_StructGridBoxes(sgrid);
      box  = hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array)-1);

      hypre_SStructGridBoxProcFindBoxManEntry(grid, part, nvars-1,
                                              hypre_BoxArraySize(box_array)-1, myproc, &boxman_entry);
      hypre_SStructBoxManEntryGetGlobalCSRank(boxman_entry, hypre_BoxIMax(box), 
                                              &upper_rank[i]);

      sgrid= hypre_SStructPGridCellSGrid(pgrid);
      box_array= hypre_StructGridBoxes(sgrid);
      boxes_with_bdry[i]= hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(box_array));
   }
 
  /*-----------------------------------------------------------------------------
   * construct box_number mapping between levels, and offset strides because of 
   * projection coarsening. Note: from the way the coarse boxes are created and
   * numbered, to determine the coarse box that matches the fbox, we need to
   * only check the tail end of the list of cboxes. In fact, given fbox_i,
   * if it's coarsened extents do not interesect with the first coarse box of the
   * tail end, then this fbox vanishes in the coarsening.
   *   c/fbox_mapping gives the fine/coarse box mapping between two consecutive levels
   *   of the multilevel hierarchy. 
   *-----------------------------------------------------------------------------*/
   if (num_levels > 1)
   {
      cbox_mapping= hypre_CTAlloc(HYPRE_Int *, num_levels);
      fbox_mapping= hypre_CTAlloc(HYPRE_Int *, num_levels);
   }
   for (i= 0; i< (num_levels-1); i++)
   {
      grid = grid_l[i];
      pgrid= hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_fgrid= hypre_SStructPGridCellSGrid(pgrid);
      fboxes= hypre_StructGridBoxes(cell_fgrid);
      nfboxes= hypre_BoxArraySize(hypre_StructGridBoxes(cell_fgrid));
      fbox_mapping[i]= hypre_CTAlloc(HYPRE_Int, nfboxes);

      grid = grid_l[i+1];
      pgrid= hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid= hypre_SStructPGridCellSGrid(pgrid);
      cboxes= hypre_StructGridBoxes(cell_cgrid);
      nboxes= hypre_BoxArraySize(hypre_StructGridBoxes(cell_cgrid));

      cbox_mapping[i+1]= hypre_CTAlloc(HYPRE_Int, nboxes);

     /* assuming if i1 > i2 and (box j1) is coarsened from (box i1)
        and (box j2) from (box i2), then j1 > j2. */
      k= 0;
      hypre_ForBoxI(j, fboxes)
      {
         fbox= hypre_BoxArrayBox(fboxes, j);
         hypre_CopyBox(fbox, &rbox);
         hypre_ProjectBox(&rbox, zero_shift, rfactors);
         hypre_StructMapFineToCoarse(hypre_BoxIMin(&rbox), zero_shift, 
                                     rfactors, hypre_BoxIMin(&rbox));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(&rbox), zero_shift, 
                                     rfactors, hypre_BoxIMax(&rbox));

        /* since the ordering of the cboxes was determined by the fbox
           ordering, we only have to check if the first cbox in the
           list intersects with rbox. If not, this fbox vanished in the
           coarsening. */
         cbox= hypre_BoxArrayBox(cboxes, k);
         hypre_IntersectBoxes(&rbox, cbox, &rbox);
         if (hypre_BoxVolume(&rbox))
         {
            cbox_mapping[i+1][k]= j;
            fbox_mapping[i][j]= k;
            k++;
         }  /* if (hypre_BoxVolume(&rbox)) */
      }     /* hypre_ForBoxI(j, fboxes) */
   }        /* for (i= 0; i< (num_levels-1); i++) */
         
   bdry= hypre_TAlloc(hypre_BoxArrayArray ***, num_levels);
   npts= hypre_CTAlloc(HYPRE_Int, num_levels);

  /* finest level boundary determination */
   grid = grid_l[0];
   pgrid= hypre_SStructGridPGrid(grid, 0); /* assuming one part */
   nvars= hypre_SStructPGridNVars(pgrid);
   cell_fgrid= hypre_SStructPGridCellSGrid(pgrid);
   nboxes= hypre_BoxArraySize(hypre_StructGridBoxes(cell_fgrid));

   hypre_Maxwell_PNedelec_Bdy(cell_fgrid, pgrid, &bdry[0]);
   for (i= 0; i< nboxes; i++)
   {
     if (bdry[0][i])  /* boundary layers on box[i] */
     {
        for (j= 0; j< nvars; j++)
        {
           fbdry= bdry[0][i][j+1]; /*(j+1) since j= 0 stores cell-centred boxes*/
           hypre_ForBoxArrayI(k, fbdry)
           {
              box_array= hypre_BoxArrayArrayBoxArray(fbdry, k);
              hypre_ForBoxI(p, box_array)
              {
                 box= hypre_BoxArrayBox(box_array, p);
                 npts[0]+= hypre_BoxVolume(box);
              }
           }
        }  /* for (j= 0; j< nvars; j++) */
        
        boxes_with_bdry[0][i]= 1; /* mark this box as containing boundary layers */
     }  /* if (bdry[0][i]) */
   }
   nfboxes= nboxes;
  
  /* coarser levels */
   for (i= 1; i< num_levels; i++)
   {
      grid = grid_l[i-1];
      pgrid= hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_fgrid= hypre_SStructPGridCellSGrid(pgrid);
      fboxes= hypre_StructGridBoxes(cell_fgrid);

      grid = grid_l[i];
      pgrid= hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid= hypre_SStructPGridCellSGrid(pgrid);
      nvars= hypre_SStructPGridNVars(pgrid);
      cboxes= hypre_StructGridBoxes(cell_cgrid);
      nboxes= hypre_BoxArraySize(hypre_StructGridBoxes(cell_cgrid));

      bdry[i]= hypre_TAlloc(hypre_BoxArrayArray **, nboxes);
      p= 2*(ndim-1);
      for (j= 0; j< nboxes; j++)
      {
         bdry[i][j]= hypre_TAlloc(hypre_BoxArrayArray *, nvars+1);

         /* cell grid boxarrayarray */
         bdry[i][j][0]= hypre_BoxArrayArrayCreate(2*ndim);

         /* var grid boxarrayarrays */
         for (k= 0; k< nvars; k++)
         {
            bdry[i][j][k+1]= hypre_BoxArrayArrayCreate(p);
         }
      }
   
     /* check if there are boundary points from the previous level */
      for (j= 0; j< nfboxes; j++)
      {
        /* see if the j box of level (i-1) has any boundary layers */
         if (boxes_with_bdry[i-1][j])
         {
            boxi= fbox_mapping[i-1][j];
            cbox= hypre_BoxArrayBox(cboxes, boxi);
            fbox= hypre_BoxArrayBox(fboxes, j);

           /* contract the fbox so that divisible in rfactor */
            contract_fbox= hypre_BoxContraction(fbox, cell_fgrid, rfactors);

           /* refine the cbox. Expand the refined cbox so that the complete 
              chunk of the fine box that coarsened to it is included. This
              requires some offsets */
            hypre_ClearIndex(upper_shift);
            hypre_ClearIndex(lower_shift);
            for (k= 0; k< ndim; k++)
            {
               m= hypre_BoxIMin(contract_fbox)[k];
               p= m%rfactors[k];

               if (p > 0 && m > 0)
               {
                  upper_shift[k]= p-1;
                  lower_shift[k]= p-rfactors[k];
               }
               else
               {
                  upper_shift[k]= rfactors[k]-p-1;
                  lower_shift[k]=-p;
               }
            }
            hypre_BoxDestroy(contract_fbox);

            hypre_CopyBox(cbox, &rbox);
            hypre_StructMapCoarseToFine(hypre_BoxIMin(&rbox), zero_shift, 
                                        rfactors, hypre_BoxIMin(&rbox));
            hypre_StructMapCoarseToFine(hypre_BoxIMax(&rbox), zero_shift, 
                                        rfactors, hypre_BoxIMax(&rbox));

            hypre_AddIndex(lower_shift, hypre_BoxIMin(&rbox), hypre_BoxIMin(&rbox));
            hypre_AddIndex(upper_shift, hypre_BoxIMax(&rbox), hypre_BoxIMax(&rbox));

           /* Determine, if any, boundary layers for this rbox. Since the 
              boundaries of the coarser levels may not be physical, we cannot
              use hypre_BoxBoundaryDG. But accomplished through intersecting
              with the finer level boundary boxes. */
            fbdry= bdry[i-1][j][0]; /* cell-centred boundary layers of level (i-1) */
            cbdry= bdry[i][boxi][0]; /* cell-centred boundary layers of level i */

           /* fbdry is the cell-centred box_arrayarray. Contains an array of (2*ndim)
              boxarrays, one for each direction. */
            cnt= 0;
            hypre_ForBoxArrayI(l, fbdry)
            {
               /* determine which boundary side we are doing. Depending on the
                  boundary, when we coarsen the refined boundary layer, the
                  extents may need to be changed, 
                         e.g., index[lower,j,k]= index[upper,j,k]. */
               switch(l)
               {
                  case 0:  /* lower x direction, x_upper= x_lower */ 
                  {
                     n= 1; /* n flags whether upper or lower to be replaced */
                     d= 0; /* x component */
                     break;
                  }
                  case 1:  /* upper x direction, x_lower= x_upper */ 
                  {
                     n= 0; /* n flags whether upper or lower to be replaced */
                     d= 0; /* x component */
                     break;
                  }
                  case 2:  /* lower y direction, y_upper= y_lower */ 
                  {
                     n= 1; /* n flags whether upper or lower to be replaced */
                     d= 1; /* y component */
                     break;
                  }
                  case 3:  /* upper y direction, y_lower= y_upper */ 
                  {
                     n= 0; /* n flags whether upper or lower to be replaced */
                     d= 1; /* y component */
                     break;
                  }
                  case 4:  /* lower z direction, z_lower= z_upper */ 
                  {
                     n= 1; /* n flags whether upper or lower to be replaced */
                     d= 2; /* z component */
                     break;
                  }
                  case 5:  /* upper z direction, z_upper= z_lower */ 
                  {
                     n= 0; /* n flags whether upper or lower to be replaced */
                     d= 2; /* z component */
                     break;
                  }
               }
                    
               box_array= hypre_BoxArrayArrayBoxArray(fbdry, l);
               hypre_ForBoxI(p, box_array) 
               {
                  hypre_IntersectBoxes(hypre_BoxArrayBox(box_array, p), &rbox,
                                       &intersect);
                  if (hypre_BoxVolume(&intersect))
                  {
                     /* coarsen the refined boundary box and append it to
                        boxarray hypre_BoxArrayArrayBoxArray(cbdry, l) */
                      hypre_ProjectBox(&intersect, zero_shift, rfactors);
                      hypre_StructMapFineToCoarse(hypre_BoxIMin(&intersect), 
                              zero_shift, rfactors, hypre_BoxIMin(&intersect));
                      hypre_StructMapFineToCoarse(hypre_BoxIMax(&intersect), 
                              zero_shift, rfactors, hypre_BoxIMax(&intersect));

                     /* the coarsened intersect box may be incorrect because
                        of the box projecting formulas. */
                      if (n) /* replace upper by lower */
                      { 
                         hypre_BoxIMax(&intersect)[d]= hypre_BoxIMin(&intersect)[d];
                      }
                      else   /* replace lower by upper */
                      { 
                         hypre_BoxIMin(&intersect)[d]= hypre_BoxIMax(&intersect)[d];
                      }
                   
                      hypre_AppendBox(&intersect,
                                       hypre_BoxArrayArrayBoxArray(cbdry, l));
                      cnt++; /* counter to signal boundary layers for cbox boxi */
                  }   /* if (hypre_BoxVolume(&intersect)) */
               }      /* hypre_ForBoxI(p, box_array) */
            }         /* hypre_ForBoxArrayI(l, fbdry) */
            
           /* All the boundary box_arrayarrays have been checked for coarse boxi.
              Now get the variable boundary layers if any, count the number of 
              boundary points, and appropriately mark boxi. */
            if (cnt)
            {
               hypre_Maxwell_VarBdy(pgrid, bdry[i][boxi]);

               for (p= 0; p< nvars; p++)
               {
                  cbdry= bdry[i][boxi][p+1];
                  hypre_ForBoxArrayI(l, cbdry)
                  {
                     box_array= hypre_BoxArrayArrayBoxArray(cbdry, l);
                     hypre_ForBoxI(m, box_array) 
                     {
                        cbox= hypre_BoxArrayBox(box_array, m);
                        npts[i]+= hypre_BoxVolume(cbox);
                     }
                  }
               }

               boxes_with_bdry[i][boxi]= 1; /* mark as containing boundary */
            }
 
         }  /* if (boxes_with_bdry[i-1][j]) */
      }     /* for (j= 0; j< nfboxes; j++) */

      nfboxes= nboxes;
   }  /* for (i= 1; i< num_levels; i++) */

  /* de-allocate objects that are not needed anymore */
   for (i= 0; i< (num_levels-1); i++)
   {
      if (fbox_mapping[i])
      {
         hypre_TFree(fbox_mapping[i]);
      }
      if (cbox_mapping[i+1])
      {
         hypre_TFree(cbox_mapping[i+1]);
      }

      grid = grid_l[i+1];
      pgrid= hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid= hypre_SStructPGridCellSGrid(pgrid);
      cboxes= hypre_StructGridBoxes(cell_cgrid);
      nboxes= hypre_BoxArraySize(hypre_StructGridBoxes(cell_cgrid));
   }
   if (num_levels > 1)
   {
      hypre_TFree(fbox_mapping);
      hypre_TFree(cbox_mapping);
   }

  /* find the ranks for the boundary points */
   BdryRanks_l    = hypre_TAlloc(HYPRE_Int *, num_levels);
   BdryRanksCnts_l= hypre_TAlloc(HYPRE_Int  , num_levels);

  /* loop over levels and extract boundary ranks. Only extract unique
     ranks */
   for (i= 0; i< num_levels; i++)
   {
      grid= grid_l[i];
      pgrid= hypre_SStructGridPGrid(grid, 0); /* assuming one part */
      cell_cgrid= hypre_SStructPGridCellSGrid(pgrid);
      nvars= hypre_SStructPGridNVars(pgrid);
      cboxes= hypre_StructGridBoxes(cell_cgrid);
      nboxes= hypre_BoxArraySize(hypre_StructGridBoxes(cell_cgrid));
 
      ranks= hypre_TAlloc(HYPRE_Int, npts[i]);
      cnt= 0;
      for (j= 0; j< nboxes; j++)
      {
         if (boxes_with_bdry[i][j])
         {
            for (k= 0; k< nvars; k++)
            {
               fbdry= bdry[i][j][k+1];
  
               hypre_ForBoxArrayI(m, fbdry)
               {
                  box_array= hypre_BoxArrayArrayBoxArray(fbdry, m);
                  hypre_ForBoxI(p, box_array)
                  {
                     box= hypre_BoxArrayBox(box_array, p);
                     hypre_BoxGetSize(box, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(box), start);
      
                     hypre_BoxLoop0Begin(loop_size);
#if 0
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,index,boxman_entry,cnt
#include "hypre_box_smp_forloop.h"
#else
                     hypre_BoxLoopSetOneBlock();
#endif
                     hypre_BoxLoop0For(loopi, loopj, loopk)
                     {
                        hypre_SetIndex(index, loopi, loopj, loopk);
                        hypre_AddIndex(index, start, index);

                        hypre_SStructGridFindBoxManEntry(grid, part, index,
                                                      k, &boxman_entry);
                        hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index,
                                                          &ranks[cnt], matrix_type);
                        cnt++;

                     }
                     hypre_BoxLoop0End();
                  }  /* hypre_ForBoxI(p, box_array) */
               }     /* hypre_ForBoxArrayI(m, fbdry) */

            }  /* for (k= 0; k< nvars; k++) */
         } /* if (boxes_with_bdry[i][j]) */

         for (k= 0; k< nvars; k++)
         {
            hypre_BoxArrayArrayDestroy(bdry[i][j][k+1]);
         }
         hypre_BoxArrayArrayDestroy(bdry[i][j][0]);
         hypre_TFree(bdry[i][j]);
        
      }  /* for (j= 0; j< nboxes; j++) */
      hypre_TFree(bdry[i]);

     /* mark all ranks that are outside this processor to -1 */
      for (j= 0; j< cnt; j++)
      {
         if ( (ranks[j] < lower_rank[i]) || (ranks[j] > upper_rank[i]) )
         {
            ranks[j]= -1;
         }
      }

     /* sort the ranks & extract the unique ones */
      if (cnt)  /* recall that some may not have bdry pts */
      {
         qsort0(ranks, 0, cnt-1);

         k= 0;
         if (ranks[0] < 0) /* remove the off-processor markers */
         {
            for (j= 1; j< cnt; j++)
            {
               if (ranks[j] > -1)
               {
                  k= j;
                  break;
               }
            }
         }

         l= 1;
         for (j= k+1; j< cnt; j++)
         {
            if (ranks[j] != ranks[j-1])
            {
               l++;
            }
         }
         BdryRanks_l[i]= hypre_TAlloc(HYPRE_Int, l);
         BdryRanksCnts_l[i]= l;

         l= 0;
         BdryRanks_l[i][l]= ranks[k]-lower_rank[i];
         for (j= k+1; j< cnt; j++)
         {
            if (ranks[j] != ranks[j-1])
            {
               l++;
               BdryRanks_l[i][l]= ranks[j]-lower_rank[i]; /* store local ranks */
            }
         }
      }

      else /* set BdryRanks_l[i] to be null */   
      {
         BdryRanks_l[i]= NULL;
         BdryRanksCnts_l[i]= 0;
      }

      hypre_TFree(ranks);
      hypre_TFree(boxes_with_bdry[i]);

   }  /* for (i= 0; i< num_levels; i++) */

   hypre_TFree(boxes_with_bdry);
   hypre_TFree(lower_rank);
   hypre_TFree(upper_rank);

   hypre_TFree(bdry);
   hypre_TFree(npts);

  *BdryRanksl_ptr    = BdryRanks_l;
  *BdryRanksCntsl_ptr= BdryRanksCnts_l;

   return ierr;
}

/*-----------------------------------------------------------------------------
 * Determine the variable boundary layers using the cell-centred boundary
 * layers. The cell-centred boundary layers are located in bdry[0], a
 * hypre_BoxArrayArray of size 2*ndim, one array for the upper side and one
 * for the lower side, for each direction.
 *-----------------------------------------------------------------------------*/
HYPRE_Int
hypre_Maxwell_VarBdy( hypre_SStructPGrid       *pgrid,
                      hypre_BoxArrayArray     **bdry )
{
   HYPRE_Int              ierr = 0;
   HYPRE_Int              nvars= hypre_SStructPGridNVars(pgrid);

   hypre_BoxArrayArray   *cell_bdry= bdry[0];
   hypre_BoxArray        *box_array, *box_array2;
   hypre_Box             *bdy_box, *shifted_box;

   HYPRE_SStructVariable *vartypes = hypre_SStructPGridVarTypes(pgrid);
   hypre_Index            varoffset, ishift, jshift, kshift;
   hypre_Index            lower, upper;

   HYPRE_Int              ndim = hypre_SStructPGridNDim(pgrid);
   HYPRE_Int              i, k, t;

   hypre_SetIndex(ishift, 1, 0, 0);
   hypre_SetIndex(jshift, 0, 1, 0);
   hypre_SetIndex(kshift, 0, 0, 1);

   shifted_box= hypre_BoxCreate();
   for (i= 0; i< nvars; i++)
   {
      t= vartypes[i];
      hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);
      switch(t)
      {
         case 2: /* xface, boundary i= lower, upper */
         {
           /* boundary i= lower */
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 0);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 0);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, varoffset, upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary i= upper */
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 1);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 1);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }
            break;
         }

         case 3: /* yface, boundary j= lower, upper */
         {
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 2);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 0);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, varoffset, upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 3);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 1);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }
            break;
         }

         case 5: /* xedge, boundary z_faces & y_faces */
         {
           /* boundary k= lower zface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 4);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 0);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, kshift, upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary k= upper zface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 5);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 1);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, jshift, lower);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary j= lower yface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 2);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 2);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, jshift, upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

            /* boundary j= upper yface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 3);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 3);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, kshift, lower);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }
            break;
         }

         case 6: /* yedge, boundary z_faces & x_faces */
         {
           /* boundary k= lower zface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 4);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 0);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, kshift, upper);
                   
                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary k= upper zface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 5);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 1);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, ishift, lower);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary i= lower xface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 0);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 2);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, ishift, upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary i= upper xface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 1);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 3);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, kshift, lower);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }
            break;
         }

         case 7: /* zedge, boundary y_faces & x_faces */
         {
           /* boundary j= lower yface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 2);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 0);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, jshift, upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary j= upper yface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 3);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 1);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, ishift, lower);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary i= lower xface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 0);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 2);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, varoffset, lower);
                   hypre_SubtractIndex(upper, ishift, upper);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }

           /* boundary i= upper xface*/
            box_array= hypre_BoxArrayArrayBoxArray(cell_bdry, 1);
            if (hypre_BoxArraySize(box_array))
            {
                box_array2= hypre_BoxArrayArrayBoxArray(bdry[i+1], 3);
                hypre_ForBoxI(k, box_array)
                {
                   bdy_box= hypre_BoxArrayBox(box_array, k);

                  /* bdry boxes */
                   hypre_CopyIndex(hypre_BoxIMin(bdy_box), lower);
                   hypre_CopyIndex(hypre_BoxIMax(bdy_box), upper);
                   hypre_SubtractIndex(lower, jshift, lower);

                   hypre_BoxSetExtents(shifted_box, lower, upper);
                   hypre_AppendBox(shifted_box, box_array2);
                }
            }
            break;
         }

      }  /* switch(t) */
   }     /* for (i= 0; i< nvars; i++) */

   hypre_BoxDestroy(shifted_box);

   return ierr;
}

