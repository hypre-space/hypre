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
 * hypre_FindBoxNeighbors:
 *
 *   Finds boxes in the `all_boxes' hypre_BoxArray that form a neighborhood
 *   of the `boxes' hypre_BoxArray.  This neighborhood is determined by the
 *   shape of the stencil passed in and represents the minimum number
 *   of boxes touched by the stencil elements.
 *
 *   The routine returns a hypre_BoxArray called `neighbors' that contains
 *   pointers to boxes in `all_boxes' that are in the neighborhood.  An
 *   int array called `neighbor_ranks' is also returned and contains the
 *   indices into the `all_boxes' array of each of the boxes in `neighbors'.
 *   This is done so that additional information (such as process number)
 *   can be extracted.
 *--------------------------------------------------------------------------*/

void
hypre_FindBoxNeighbors( hypre_BoxArray       *boxes,
                      hypre_BoxArray       *all_boxes,
                      hypre_StructStencil  *stencil,
                      int                 transpose,
                      hypre_BoxArray      **neighbors_ptr,
                      int               **neighbor_ranks_ptr )
{
   hypre_BoxArray   *neighbors;
   int            *neighbor_ranks;
   int            *tmp_neighbor_ranks;
                  
   int            *neighbor_flags;
                  
   hypre_Box        *box;
   hypre_Box        *shift_box;
   hypre_Box        *all_box;
   hypre_Box        *tmp_box;
                  
   int             i, j, d, s;
                
   hypre_Index      *stencil_shape = hypre_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Determine `neighbors' and `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   neighbors = hypre_NewBoxArray();
   neighbor_ranks = hypre_CTAlloc(int, hypre_BoxArraySize(all_boxes));
   neighbor_flags = hypre_CTAlloc(int, hypre_BoxArraySize(all_boxes));

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      shift_box = hypre_DuplicateBox(box);

      for (s = 0; s < hypre_StructStencilSize(stencil); s++)
      {
         if (transpose)
         {
            for (d = 0; d < 3; d++)
            {
               hypre_BoxIMinD(shift_box, d) =
                  hypre_BoxIMinD(box, d) - hypre_IndexD(stencil_shape[s], d);
               hypre_BoxIMaxD(shift_box, d) =
                  hypre_BoxIMaxD(box, d) - hypre_IndexD(stencil_shape[s], d);
            }
         }
         else
         {
            for (d = 0; d < 3; d++)
            {
               hypre_BoxIMinD(shift_box, d) =
                  hypre_BoxIMinD(box, d) + hypre_IndexD(stencil_shape[s], d);
               hypre_BoxIMaxD(shift_box, d) =
                  hypre_BoxIMaxD(box, d) + hypre_IndexD(stencil_shape[s], d);
            }
         }

         hypre_ForBoxI(j, all_boxes)
         {
            all_box = hypre_BoxArrayBox(all_boxes, j);

            tmp_box = hypre_IntersectBoxes(shift_box, all_box);
            if (tmp_box)
            {
               if (!neighbor_flags[j])
               {
                  neighbor_flags[j] = 1;
                  neighbor_ranks[hypre_BoxArraySize(neighbors)] = j;
                  hypre_AppendBox(all_box, neighbors);
               }

               hypre_FreeBox(tmp_box);
            }
         }
      }

      hypre_FreeBox(shift_box);
   }

   hypre_TFree(neighbor_flags);

   /*-----------------------------------------------------------------------
    * Compress size of `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   tmp_neighbor_ranks = neighbor_ranks;
   neighbor_ranks = hypre_CTAlloc(int, hypre_BoxArraySize(neighbors));
   hypre_ForBoxI(i, neighbors)
      neighbor_ranks[i] = tmp_neighbor_ranks[i];
   hypre_TFree(tmp_neighbor_ranks);

   *neighbors_ptr = neighbors;
   *neighbor_ranks_ptr = neighbor_ranks;
}

/*--------------------------------------------------------------------------
 * hypre_FindBoxApproxNeighbors:
 *
 *   Finds boxes in the `all_boxes' hypre_BoxArray that form an approximate
 *   neighborhood of the `boxes' hypre_BoxArray.  This neighborhood is
 *   determined by the min and max shape offsets of the stencil passed in.
 *   It contains the neighborhood computed by hypre_FindBoxNeighbors.
 *
 *   The routine returns a hypre_BoxArray called `neighbors' that contains
 *   pointers to boxes in `all_boxes' that are in the neighborhood.  An
 *   int array called `neighbor_ranks' is also returned and contains the
 *   indices into the `all_boxes' array of each of the boxes in `neighbors'.
 *   This is done so that additional information (such as process number)
 *   can be extracted.
 *--------------------------------------------------------------------------*/

void
hypre_FindBoxApproxNeighbors( hypre_BoxArray       *boxes,
                            hypre_BoxArray       *all_boxes,
                            hypre_StructStencil  *stencil,
                            int                 transpose,
                            hypre_BoxArray      **neighbors_ptr,
                            int               **neighbor_ranks_ptr )
{
   hypre_BoxArray   *neighbors;
   int            *neighbor_ranks;
   int            *tmp_neighbor_ranks;
                  
   int            *neighbor_flags;
                  
   hypre_Box        *box;
   hypre_Box        *grow_box;
   hypre_Box        *all_box;
   hypre_Box        *tmp_box;
                  
   int             min_offset[3], max_offset[3];
                  
   int             i, j, d, s;

   hypre_Index      *stencil_shape = hypre_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Compute min and max stencil offsets
    *-----------------------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      min_offset[d] = 0;
      max_offset[d] = 0;
   }

   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      if (transpose)
      {
         for (d = 0; d < 3; d++)
         {
            min_offset[d] =
               min(min_offset[d], -hypre_IndexD(stencil_shape[s], d));
            max_offset[d] =
               max(max_offset[d], -hypre_IndexD(stencil_shape[s], d));
         }
      }
      else
      {
         for (d = 0; d < 3; d++)
         {
            min_offset[d] =
               min(min_offset[d], hypre_IndexD(stencil_shape[s], d));
            max_offset[d] =
               max(max_offset[d], hypre_IndexD(stencil_shape[s], d));
         }
      }
   }

   /*-----------------------------------------------------------------------
    * Determine `neighbors' and `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   neighbors = hypre_NewBoxArray();
   neighbor_ranks = hypre_CTAlloc(int, hypre_BoxArraySize(all_boxes));
   neighbor_flags = hypre_CTAlloc(int, hypre_BoxArraySize(all_boxes));

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      /* grow the box */
      grow_box = hypre_DuplicateBox(box);
      for (d = 0; d < 3; d++)
      {
         hypre_BoxIMinD(grow_box, d) += min_offset[d];
         hypre_BoxIMaxD(grow_box, d) += max_offset[d];
      }

      hypre_ForBoxI(j, all_boxes)
      {
         all_box = hypre_BoxArrayBox(all_boxes, j);

         tmp_box = hypre_IntersectBoxes(grow_box, all_box);
         if (tmp_box)
         {
            if (!neighbor_flags[j])
            {
               neighbor_flags[j] = 1;
               neighbor_ranks[hypre_BoxArraySize(neighbors)] = j;
               hypre_AppendBox(all_box, neighbors);
            }

            hypre_FreeBox(tmp_box);
         }
      }

      hypre_FreeBox(grow_box);
   }

   hypre_TFree(neighbor_flags);

   /*-----------------------------------------------------------------------
    * Compress size of `neighbor_ranks'
    *-----------------------------------------------------------------------*/

   tmp_neighbor_ranks = neighbor_ranks;
   neighbor_ranks = hypre_CTAlloc(int, hypre_BoxArraySize(neighbors));
   hypre_ForBoxI(i, neighbors)
      neighbor_ranks[i] = tmp_neighbor_ranks[i];
   hypre_TFree(tmp_neighbor_ranks);

   *neighbors_ptr = neighbors;
   *neighbor_ranks_ptr = neighbor_ranks;
}
