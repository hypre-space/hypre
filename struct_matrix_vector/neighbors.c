/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routines for finding "neighboring" boxes.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_FindBoxNeighbors:
 *
 *   Finds boxes in the `all_boxes' zzz_BoxArray that form a neighborhood
 *   of the `boxes' zzz_BoxArray.  This neighborhood is determined by the
 *   shape of the stencil passed in and represents the minimum number
 *   of boxes touched by the stencil elements.
 *
 *   The routine returns a zzz_BoxArray called `neighbors' that contains
 *   pointers to boxes in `all_boxes' that are in the neighborhood.  An
 *   int array called `neighbor_ranks' is also returned and contains the
 *   indices into the `all_boxes' array of each of the boxes in `neighbors'.
 *   This is done so that additional information (such as process number)
 *   can be extracted.
 *--------------------------------------------------------------------------*/

void
zzz_FindBoxNeighbors( zzz_BoxArray       *boxes,
                      zzz_BoxArray       *all_boxes,
                      zzz_StructStencil  *stencil,
                      zzz_BoxArray      **neighbors_ptr,
                      int               **neighbor_ranks_ptr )
{
   zzz_BoxArray   *neighbors;
   int            *neighbor_ranks;
   int            *tmp_neighbor_ranks;
                  
   int            *neighbor_flags;
                  
   zzz_Box        *box;
   zzz_Box        *shift_box;
   zzz_Box        *all_box;
   zzz_Box        *tmp_box;
                  
   int             i, j, d, s;
                
   zzz_Index     **stencil_shape = zzz_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Determine `neighbors' and `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   neighbors = zzz_NewBoxArray();
   neighbor_ranks = zzz_CTAlloc(int, zzz_BoxArraySize(all_boxes));
   neighbor_flags = zzz_CTAlloc(int, zzz_BoxArraySize(all_boxes));

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);
      shift_box = zzz_DuplicateBox(box);

      for (s = 0; s < zzz_StructStencilSize(stencil); s++)
      {
         for (d = 0; d < 3; d++)
         {
            zzz_BoxIMinD(shift_box, d) =
               zzz_BoxIMinD(box, d) + zzz_IndexD(stencil_shape[s], d);
            zzz_BoxIMaxD(shift_box, d) =
               zzz_BoxIMaxD(box, d) + zzz_IndexD(stencil_shape[s], d);
         }

         zzz_ForBoxI(j, all_boxes)
         {
            all_box = zzz_BoxArrayBox(all_boxes, j);

            tmp_box = zzz_IntersectBoxes(shift_box, all_box);
            if (tmp_box)
            {
               if (!neighbor_flags[j])
               {
                  neighbor_flags[j] = 1;
                  neighbor_ranks[zzz_BoxArraySize(neighbors)] = j;
                  zzz_AppendBox(all_box, neighbors);
               }

               zzz_FreeBox(tmp_box);
            }
         }
      }

      zzz_FreeBox(shift_box);
   }

   zzz_TFree(neighbor_flags);

   /*-----------------------------------------------------------------------
    * Compress size of `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   tmp_neighbor_ranks = neighbor_ranks;
   neighbor_ranks = zzz_CTAlloc(int, zzz_BoxArraySize(neighbors));
   zzz_ForBoxI(i, neighbors)
      neighbor_ranks[i] = tmp_neighbor_ranks[i];
   zzz_TFree(tmp_neighbor_ranks);

   *neighbors_ptr = neighbors;
   *neighbor_ranks_ptr = neighbor_ranks;
}

/*--------------------------------------------------------------------------
 * zzz_FindBoxApproxNeighbors:
 *
 *   Finds boxes in the `all_boxes' zzz_BoxArray that form an approximate
 *   neighborhood of the `boxes' zzz_BoxArray.  This neighborhood is
 *   determined by the min and max shape offsets of the stencil passed in.
 *   It contains the neighborhood computed by zzz_FindBoxNeighbors.
 *
 *   The routine returns a zzz_BoxArray called `neighbors' that contains
 *   pointers to boxes in `all_boxes' that are in the neighborhood.  An
 *   int array called `neighbor_ranks' is also returned and contains the
 *   indices into the `all_boxes' array of each of the boxes in `neighbors'.
 *   This is done so that additional information (such as process number)
 *   can be extracted.
 *--------------------------------------------------------------------------*/

void
zzz_FindBoxApproxNeighbors( zzz_BoxArray       *boxes,
                            zzz_BoxArray       *all_boxes,
                            zzz_StructStencil  *stencil,
                            zzz_BoxArray      **neighbors_ptr,
                            int               **neighbor_ranks_ptr )
{
   zzz_BoxArray   *neighbors;
   int            *neighbor_ranks;
   int            *tmp_neighbor_ranks;
                  
   int            *neighbor_flags;
                  
   zzz_Box        *box;
   zzz_Box        *grow_box;
   zzz_Box        *all_box;
   zzz_Box        *tmp_box;
                  
   int             min_offset[3], max_offset[3];
                  
   int             i, j, d, s;

   zzz_Index     **stencil_shape = zzz_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Compute min and max stencil offsets
    *-----------------------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      min_offset[d] = 0;
      max_offset[d] = 0;
   }

   for (s = 0; s < zzz_StructStencilSize(stencil); s++)
   {
      for (d = 0; d < 3; d++)
      {
         min_offset[d] = min(min_offset[d], zzz_IndexD(stencil_shape[s], d));
         max_offset[d] = max(max_offset[d], zzz_IndexD(stencil_shape[s], d));
      }
   }

   /*-----------------------------------------------------------------------
    * Determine `neighbors' and `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   neighbors = zzz_NewBoxArray();
   neighbor_ranks = zzz_CTAlloc(int, zzz_BoxArraySize(all_boxes));
   neighbor_flags = zzz_CTAlloc(int, zzz_BoxArraySize(all_boxes));

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);

      /* grow the box */
      grow_box = zzz_DuplicateBox(box);
      for (d = 0; d < 3; d++)
      {
         zzz_BoxIMinD(grow_box, d) += min_offset[d];
         zzz_BoxIMaxD(grow_box, d) += max_offset[d];
      }

      zzz_ForBoxI(j, all_boxes)
      {
         all_box = zzz_BoxArrayBox(all_boxes, j);

         tmp_box = zzz_IntersectBoxes(grow_box, all_box);
         if (tmp_box)
         {
            if (!neighbor_flags[j])
            {
               neighbor_flags[j] = 1;
               neighbor_ranks[zzz_BoxArraySize(neighbors)] = j;
               zzz_AppendBox(all_box, neighbors);
            }

            zzz_FreeBox(tmp_box);
         }
      }

      zzz_FreeBox(grow_box);
   }

   zzz_TFree(neighbor_flags);

   /*-----------------------------------------------------------------------
    * Compress size of `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   tmp_neighbor_ranks = neighbor_ranks;
   neighbor_ranks = zzz_CTAlloc(int, zzz_BoxArraySize(neighbors));
   zzz_ForBoxI(i, neighbors)
      neighbor_ranks[i] = tmp_neighbor_ranks[i];
   zzz_TFree(tmp_neighbor_ranks);

   *neighbors_ptr = neighbors;
   *neighbor_ranks_ptr = neighbor_ranks;
}
