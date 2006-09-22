/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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



#include "headers.h"

int
hypre_SStructIndexScaleF_C( hypre_Index findex,
                            hypre_Index index,
                            hypre_Index stride,
                            hypre_Index cindex )
{
   hypre_IndexX(cindex) =
      (hypre_IndexX(findex) - hypre_IndexX(index)) / hypre_IndexX(stride);
   hypre_IndexY(cindex) =
      (hypre_IndexY(findex) - hypre_IndexY(index)) / hypre_IndexY(stride);
   hypre_IndexZ(cindex) =
      (hypre_IndexZ(findex) - hypre_IndexZ(index)) / hypre_IndexZ(stride);

   return 0;
}


int
hypre_SStructIndexScaleC_F( hypre_Index cindex,
                            hypre_Index index,
                            hypre_Index stride,
                            hypre_Index findex )
{
   hypre_IndexX(findex) =
      hypre_IndexX(cindex) * hypre_IndexX(stride) + hypre_IndexX(index);
   hypre_IndexY(findex) =
      hypre_IndexY(cindex) * hypre_IndexY(stride) + hypre_IndexY(index);
   hypre_IndexZ(findex) =
      hypre_IndexZ(cindex) * hypre_IndexZ(stride) + hypre_IndexZ(index);

   return 0;
}
/*--------------------------------------------------------------------------
 * hypre_SStructOwnInfo: Given a fgrid, coarsen each fbox and find the
 * coarsened boxes that belong on my current processor. These are my own_boxes.
 *--------------------------------------------------------------------------*/

hypre_SStructOwnInfoData *
hypre_SStructOwnInfo( hypre_StructGrid  *fgrid,
                      hypre_StructGrid  *cgrid,
                      hypre_BoxMap      *cmap,
                      hypre_BoxMap      *fmap,
                      hypre_Index        rfactor )
{
   hypre_SStructOwnInfoData *owninfo_data;

   MPI_Comm                  comm= hypre_SStructVectorComm(fgrid);
   int                       ndim= hypre_StructGridDim(fgrid);

   hypre_BoxArray           *grid_boxes;
   hypre_BoxArray           *intersect_boxes;
   hypre_BoxArray           *tmp_boxarray;

   hypre_Box                *grid_box, scaled_box;
   hypre_Box                 map_entry_box;

   hypre_BoxMapEntry       **map_entries;
   int                       nmap_entries;

   hypre_BoxArrayArray      *own_boxes;
   int                     **own_cboxnums;

   hypre_BoxArrayArray      *own_composite_cboxes;

   hypre_Index               ilower, iupper, index;

   int                       myproc, proc;

   int                       cnt;
   int                       i, j, k, mod;

   hypre_ClearIndex(index); 
   MPI_Comm_rank(comm, &myproc);

   owninfo_data= hypre_CTAlloc(hypre_SStructOwnInfoData, 1);

   /*------------------------------------------------------------------------
    * Create the structured ownbox patterns. 
    *
    *   own_boxes are obtained by intersecting this proc's fgrid boxes
    *   with cgrid's box_map. Intersecting BoxMapEntries on this proc
    *   will give the own_boxes.
    *------------------------------------------------------------------------*/
   grid_boxes    = hypre_StructGridBoxes(fgrid);

   own_boxes   = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes));
   own_cboxnums= hypre_CTAlloc(int *, hypre_BoxArraySize(grid_boxes));

   hypre_ForBoxI(i, grid_boxes)
   {
       grid_box= hypre_BoxArrayBox(grid_boxes, i);

       /*---------------------------------------------------------------------
        * Find the boxarray that is owned. BoxMapIntersect returns
        * the full extents of the boxes that intersect with the given box.
        * We further need to intersect each box in the list with the given
        * box to determine the actual box that is owned.
        *---------------------------------------------------------------------*/
       hypre_SStructIndexScaleF_C(hypre_BoxIMin(grid_box), index,
                                  rfactor, hypre_BoxIMin(&scaled_box));
       hypre_SStructIndexScaleF_C(hypre_BoxIMax(grid_box), index,
                                  rfactor, hypre_BoxIMax(&scaled_box));

       hypre_BoxMapIntersect(cmap, hypre_BoxIMin(&scaled_box), 
                             hypre_BoxIMax(&scaled_box), &map_entries,
                            &nmap_entries);

       cnt= 0;
       for (j= 0; j< nmap_entries; j++)
       {
          hypre_SStructMapEntryGetProcess(map_entries[j], &proc);
          if (proc == myproc)
          {
             cnt++;
          }
       }
       own_cboxnums[i]= hypre_CTAlloc(int, cnt);

       cnt= 0;
       for (j= 0; j< nmap_entries; j++)
       {
          hypre_SStructMapEntryGetProcess(map_entries[j], &proc);

          /* determine the chunk of the map_entries[j] box that is needed */
          hypre_BoxMapEntryGetExtents(map_entries[j], ilower, iupper);
          hypre_BoxSetExtents(&map_entry_box, ilower, iupper);
          hypre_IntersectBoxes(&map_entry_box, &scaled_box, &map_entry_box);

          if (proc == myproc)
          {
             hypre_SStructMapEntryGetBox(map_entries[j], &own_cboxnums[i][cnt]);
             hypre_AppendBox(&map_entry_box, 
                              hypre_BoxArrayArrayBoxArray(own_boxes, i));
             cnt++;
          }
      } 
      hypre_TFree(map_entries);
   }  /* hypre_ForBoxI(i, grid_boxes) */ 

   (owninfo_data -> size)     = hypre_BoxArraySize(grid_boxes);
   (owninfo_data -> own_boxes)= own_boxes;
   (owninfo_data -> own_cboxnums)= own_cboxnums;

   /*------------------------------------------------------------------------
    *   own_composite_cboxes are obtained by intersecting this proc's cgrid 
    *   boxes with fgrid's box_map. For each cbox, subtracting all the 
    *   intersecting boxes from all processors will give the 
    *   own_composite_cboxes.
    *------------------------------------------------------------------------*/
   grid_boxes= hypre_StructGridBoxes(cgrid);
   own_composite_cboxes= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes));
  (owninfo_data -> own_composite_size)= hypre_BoxArraySize(grid_boxes);

   tmp_boxarray = hypre_BoxArrayCreate(0);
   hypre_ForBoxI(i, grid_boxes)
   {
       grid_box= hypre_BoxArrayBox(grid_boxes, i);
       hypre_AppendBox(grid_box,
                       hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i));

       hypre_ClearIndex(index); 
       hypre_SStructIndexScaleC_F(hypre_BoxIMin(grid_box), index,
                                  rfactor, hypre_BoxIMin(&scaled_box));
       hypre_SetIndex(index, rfactor[0]-1, rfactor[1]-1, rfactor[2]-1); 
       hypre_SStructIndexScaleC_F(hypre_BoxIMax(grid_box), index,
                                  rfactor, hypre_BoxIMax(&scaled_box));

       hypre_BoxMapIntersect(fmap, hypre_BoxIMin(&scaled_box),
                             hypre_BoxIMax(&scaled_box), &map_entries,
                            &nmap_entries);
       
       hypre_ClearIndex(index); 
       intersect_boxes= hypre_BoxArrayCreate(0);
       for (j= 0; j< nmap_entries; j++)
       {
          hypre_BoxMapEntryGetExtents(map_entries[j], ilower, iupper);
          hypre_BoxSetExtents(&map_entry_box, ilower, iupper);
          hypre_IntersectBoxes(&map_entry_box, &scaled_box, &map_entry_box);

         /* contract the intersection box so that only the cnodes in the 
            intersection box are included. */
          for (k= 0; k< ndim; k++)
          {
             mod= hypre_BoxIMin(&map_entry_box)[k] % rfactor[k];
             if (mod)
             {
                hypre_BoxIMin(&map_entry_box)[k]+= rfactor[k] - mod;
             }
          }
 
          hypre_SStructIndexScaleF_C(hypre_BoxIMin(&map_entry_box), index,
                                     rfactor, hypre_BoxIMin(&map_entry_box));
          hypre_SStructIndexScaleF_C(hypre_BoxIMax(&map_entry_box), index,
                                     rfactor, hypre_BoxIMax(&map_entry_box));
          hypre_AppendBox(&map_entry_box, intersect_boxes);
       }

       hypre_SubtractBoxArrays(hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i),
                               intersect_boxes, tmp_boxarray);
       hypre_MinUnionBoxes(hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i));

       hypre_TFree(map_entries);
       hypre_BoxArrayDestroy(intersect_boxes);
   }
   hypre_BoxArrayDestroy(tmp_boxarray);
       
  (owninfo_data -> own_composite_cboxes)= own_composite_cboxes;

   return owninfo_data;
}

/*--------------------------------------------------------------------------
 * hypre_SStructOwnInfoDataDestroy
 *--------------------------------------------------------------------------*/
int
hypre_SStructOwnInfoDataDestroy(hypre_SStructOwnInfoData *owninfo_data)
{
   int ierr = 0;
   int i;

   if (owninfo_data)
   {
      if (owninfo_data -> own_boxes)
      {
         hypre_BoxArrayArrayDestroy( (owninfo_data -> own_boxes) );
      }

      for (i= 0; i< (owninfo_data -> size); i++)
      {
         if (owninfo_data -> own_cboxnums[i])
         {
             hypre_TFree(owninfo_data -> own_cboxnums[i]);
         }
      }
      hypre_TFree(owninfo_data -> own_cboxnums);

      if (owninfo_data -> own_composite_cboxes)
      {
         hypre_BoxArrayArrayDestroy( (owninfo_data -> own_composite_cboxes) );
      }
   }

   hypre_TFree(owninfo_data);

   return ierr;
}

