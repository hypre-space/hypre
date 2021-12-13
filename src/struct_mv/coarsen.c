/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"

#define DEBUG 0
#define TIME_DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
static HYPRE_Int debug_count = 0;
#endif

#if TIME_DEBUG
static HYPRE_Int s_coarsen_num = 0;
#endif

/*--------------------------------------------------------------------------
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MapToCoarseIndex( hypre_Index    index,
                        hypre_IndexRef origin,
                        hypre_Index    stride,
                        HYPRE_Int      ndim )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      if (origin != NULL)
      {
         index[d] -= origin[d];
      }
      if ((index[d]%stride[d]) != 0)
      {
         /* This index doesn't map directly to a coarse index */
         hypre_error(HYPRE_ERROR_GENERIC);
      }
      index[d] /= stride[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MapToFineIndex( hypre_Index    index,
                      hypre_IndexRef origin,
                      hypre_Index    stride,
                      HYPRE_Int      ndim )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      index[d] *= stride[d];
      if (origin != NULL)
      {
         index[d] += origin[d];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO: Start phasing out the following two routines.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMapFineToCoarse( hypre_Index findex,
                             hypre_Index origin,
                             hypre_Index stride,
                             hypre_Index cindex )
{
   hypre_CopyToIndex(findex, 3, cindex);
   hypre_MapToCoarseIndex(cindex, origin, stride, 3);

   return hypre_error_flag;
}

HYPRE_Int
hypre_StructMapCoarseToFine( hypre_Index cindex,
                             hypre_Index origin,
                             hypre_Index stride,
                             hypre_Index findex )
{
   hypre_CopyToIndex(cindex, 3, findex);
   hypre_MapToFineIndex(findex, origin, stride, 3);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute a new coarse (origin, stride) pair from an old (origin, stride) pair.
 * This new pair can be used with the RefineBox() function to map back to the
 * finest grid with just one call, i.e., the following two lines of code are
 * equivalent:
 *
 *    RefineBox(box, origin_new, stride_new);
 *    RefineBox(box, origin, stride); RefineBox(box, origin_old, stride_old);
 *
 * NOTE: Need to check to see if this holds for CoarsenBox() also.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeCoarseOriginStride( hypre_Index     coarse_origin,
                                 hypre_Index     coarse_stride,
                                 hypre_IndexRef  origin,
                                 hypre_Index     stride,
                                 HYPRE_Int       ndim )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      if (origin != NULL)
      {
         coarse_origin[d] += origin[d] * coarse_stride[d];
      }
      coarse_stride[d] *= stride[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This may produce an empty box, i.e., one with volume 0.
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CoarsenBox( hypre_Box      *box,
                  hypre_IndexRef  origin,
                  hypre_Index     stride )
{
   hypre_ProjectBox(box, origin, stride);
   hypre_MapToCoarseIndex(hypre_BoxIMin(box), origin, stride, hypre_BoxNDim(box));
   hypre_MapToCoarseIndex(hypre_BoxIMax(box), origin, stride, hypre_BoxNDim(box));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CoarsenBoxNeg
 *
 * This functions snaps in place the box extents of 'box' in the negative
 * direction to the nearest point in the index space given by (origin, stride).
 * Then, it coarsens 'box' to such index space. Lastly, it intersects the
 * resulting box with a reference box living in the strided index space and
 * checks if the volume of the intersected region is positive. If it isn't,
 * the output box is set to have zero volume.
 *
 * 2D Example:
 *
 *       *-------*-------*-------*-------*             *-------*-------*
 *       |       |       |       |       |             |       |       |
 *       |   *   |   *   |   *   |   X   |             |   *   |   +   |
 *       |       |       |       |       |             |       |       |
 *       *-------*-------*-------*-------*     -->     *-------*-------*
 *       |       |       |       |       |             |       |       |
 *       | (0,0) |   *   |   *   |   X   |             | (0,0) |   +   |
 *       |       |       |       |       |             |       |       |
 *       *-------*-------*-------*-------*             *-------*-------*
 *
 *    input box: (3,0) x (3,1) (denoted by X)
 *    refbox: (0,0) x (1,1)
 *    origin: (0,0)
 *    stride: (2,1)
 *    output box: (1,0) x (1,1) (denoted by +)
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CoarsenBoxNeg( hypre_Box      *box,
                     hypre_Box      *refbox,
                     hypre_IndexRef  origin,
                     hypre_Index     stride )
{
   HYPRE_Int   ndim = hypre_BoxNDim(box);
   hypre_Box  *int_box;

   /* Snap indices in negative direction */
   hypre_SnapIndexNeg(hypre_BoxIMin(box), origin, stride, ndim);
   hypre_SnapIndexNeg(hypre_BoxIMax(box), origin, stride, ndim);

   /* Map to coarse index space */
   hypre_MapToCoarseIndex(hypre_BoxIMin(box), origin, stride, ndim);
   hypre_MapToCoarseIndex(hypre_BoxIMax(box), origin, stride, ndim);

   /* Find box intersection */
   int_box = hypre_BoxCreate(ndim);
   hypre_IntersectBoxes(box, refbox, int_box);

   /* If the intersection has zero volume, then box will have zero volume also */
   if (!hypre_BoxVolume(int_box))
   {
      hypre_BoxIMinD(box, 0) = 1;
      hypre_BoxIMaxD(box, 0) = 0;
   }

   hypre_BoxDestroy(int_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RefineBox( hypre_Box      *box,
                 hypre_IndexRef  origin,
                 hypre_Index     stride )
{
   hypre_MapToFineIndex(hypre_BoxIMin(box), origin, stride, hypre_BoxNDim(box));
   hypre_MapToFineIndex(hypre_BoxIMax(box), origin, stride, hypre_BoxNDim(box));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * The dimensions of the modified box array are not changed.
 * It is possible to have boxes with volume 0.
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CoarsenBoxArray( hypre_BoxArray  *box_array,
                       hypre_IndexRef   origin,
                       hypre_Index      stride )
{
   hypre_Box  *box;
   HYPRE_Int   i;

   hypre_ForBoxI(i, box_array)
   {
      box = hypre_BoxArrayBox(box_array, i);
      hypre_CoarsenBox(box, origin, stride);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * The dimensions of the modified box array-array are not changed.
 * It is possible to have boxes with volume 0.
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CoarsenBoxArrayArray( hypre_BoxArrayArray  *box_array_array,
                            hypre_IndexRef        origin,
                            hypre_Index           stride )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;
   HYPRE_Int        i, j;

   hypre_ForBoxArrayI(i, box_array_array)
   {
      box_array = hypre_BoxArrayArrayBoxArray(box_array_array, i);
      hypre_ForBoxI(j, box_array)
      {
         box = hypre_BoxArrayBox(box_array, j);
         hypre_CoarsenBox(box, origin, stride);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * The dimensions of the new BoxArrayArray can be changed.
 * hypre_CoarsenBoxNeg is used to coarsen the boxes.
 * It is not possible to have boxes with volume 0.
 * If 'origin' is NULL, a zero origin is used.
 *
 * TODO: Substitute the BinarySearch for a loop over the largest BoxArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CoarsenBoxArrayArrayNeg( hypre_BoxArrayArray   *boxaa,
                               hypre_BoxArray        *refboxa,
                               hypre_IndexRef         origin,
                               hypre_Index            stride,
                               hypre_BoxArrayArray  **new_boxaa_ptr )
{
   HYPRE_Int              ndim         = hypre_BoxArrayArrayNDim(boxaa);
   HYPRE_Int              num_refboxes = hypre_BoxArraySize(refboxa);
   HYPRE_Int             *refbox_ids   = hypre_BoxArrayIDs(refboxa);

   hypre_Box             *box;
   hypre_Box             *refbox;
   hypre_BoxArray        *boxa;
   hypre_BoxArray        *new_boxa;
   hypre_BoxArrayArray   *new_boxaa;

   HYPRE_Int              count_box;
   HYPRE_Int              count_boxa;
   HYPRE_Int              box_id;
   HYPRE_Int              i, ii, j;

   /* Allocate box */
   box = hypre_BoxCreate(ndim);

   /* Find out how many BoxArrays will be coarsened */
   count_boxa = 0;
   hypre_ForBoxArrayI(i, boxaa)
   {
      boxa   = hypre_BoxArrayArrayBoxArray(boxaa, i);
      box_id = hypre_BoxArrayArrayID(boxaa, i);

      ii = hypre_BinarySearch(refbox_ids, box_id, num_refboxes);
      hypre_assert(ii > -1);
      refbox = hypre_BoxArrayBox(refboxa, ii);

      hypre_ForBoxI(j, boxa)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxa, j), box);
         hypre_CoarsenBoxNeg(box, refbox, origin, stride);
         if (hypre_BoxVolume(box))
         {
            count_boxa++;
            break;
         }
      }
   }

   /* Allocate memory */
   new_boxaa = hypre_BoxArrayArrayCreate(count_boxa, ndim);

   /* Coarsen BoxArrayArray */
   count_boxa = 0;
   hypre_ForBoxArrayI(i, boxaa)
   {
      boxa   = hypre_BoxArrayArrayBoxArray(boxaa, i);
      box_id = hypre_BoxArrayArrayID(boxaa, i);

      ii = hypre_BinarySearch(refbox_ids, box_id, num_refboxes);
      refbox = hypre_BoxArrayBox(refboxa, ii);

      count_box = 0;
      hypre_ForBoxI(j, boxa)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxa, j), box);
         hypre_CoarsenBoxNeg(box, refbox, origin, stride);
         if (hypre_BoxVolume(box))
         {
            count_box++;
         }
      }

      if (count_box)
      {
         new_boxa = hypre_BoxArrayArrayBoxArray(new_boxaa, count_boxa);
         hypre_BoxArraySetSize(new_boxa, count_box);

         count_box  = 0;
         hypre_ForBoxI(j, boxa)
         {
            hypre_CopyBox(hypre_BoxArrayBox(boxa, j), box);
            hypre_CoarsenBoxNeg(box, refbox, origin, stride);
            if (hypre_BoxVolume(box))
            {
               hypre_CopyBox(box, hypre_BoxArrayBox(new_boxa, count_box));
               count_box++;
            }
         }
         hypre_BoxArrayArrayID(new_boxaa, count_boxa) = hypre_BoxArrayArrayID(boxaa, i);
         count_boxa++;
      }
   }

   /* Free memory */
   hypre_BoxDestroy(box);

   /* Set pointer to new_boxaa */
   *new_boxaa_ptr = new_boxaa;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine coarsens the grid, 'fgrid', by the coarsening factor, 'stride',
 * using the index mapping in 'hypre_MapToCoarseIndex'.
 *
 *  1. A coarse grid is created with boxes that result from coarsening the fine
 *  grid boxes, bounding box, and periodicity information.
 *
 *  2. If "sufficient" neighbor information exists in the fine grid to be
 *  transferred to the coarse grid, then the coarse grid box manager can be
 *  created by simply coarsening all of the entries in the fine grid manager.
 *  ("Sufficient" is determined by checking max_distance in the fine grid.)
 *
 *  3. Otherwise, neighbor information will be collected during the
 *  StructGridAssemble according to the chosen value of max_distance for the
 *  coarse grid.
 *
 *  4. We do not need a separate version for the assumed partition case
 *
 *  5. The coarse grid is returned in unassembled format to provide more
 *  control on MPI collective calls from the semi-struct interface. Thus,
 *  a grid created with this function must be assembled later.
 *
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructCoarsen( hypre_StructGrid  *fgrid,
                     hypre_IndexRef     origin,
                     hypre_Index        stride,
                     HYPRE_Int          prune,
                     hypre_StructGrid **cgrid_ptr )
{
   hypre_StructGrid *cgrid;

   MPI_Comm          comm;
   HYPRE_Int         ndim;

   hypre_BoxArray   *fboxes;
   hypre_BoxArray   *cboxes;

   hypre_Index       periodic;
   hypre_Index       ilower, iupper;

   hypre_Box        *box;
   hypre_Box        *new_box;
   hypre_Box        *bounding_box;

   HYPRE_Int         i, j, myid, count;
   HYPRE_Int         info_size, max_nentries;
   HYPRE_Int         num_entries;
   HYPRE_Int        *fids, *cids;
   hypre_Index       new_dist;
   hypre_IndexRef    max_distance;
   HYPRE_Int         proc, id;
   HYPRE_Int         coarsen_factor, known;
   HYPRE_Int         num, last_proc;
   hypre_BoxManager *fboxman, *cboxman;

   hypre_BoxManEntry *entries;
   hypre_BoxManEntry *entry;

   void              *entry_info = NULL;

#if TIME_DEBUG
   HYPRE_Int tindex;
   char new_title[80];
   hypre_sprintf(new_title,"Coarsen.%d",s_coarsen_num);
   tindex = hypre_InitializeTiming(new_title);
   s_coarsen_num++;

   hypre_BeginTiming(tindex);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SetIndex(ilower, 0);
   hypre_SetIndex(iupper, 0);

   /* get relevant information from the fine grid */
   fboxes  = hypre_StructGridBoxes(fgrid);
   fids    = hypre_StructGridIDs(fgrid);
   fboxman = hypre_StructGridBoxMan(fgrid);
   comm    = hypre_StructGridComm(fgrid);
   ndim    = hypre_StructGridNDim(fgrid);
   max_distance = hypre_StructGridMaxDistance(fgrid);

   /* initial */
   hypre_MPI_Comm_rank(comm, &myid );

   /* create new coarse grid */
   hypre_StructGridCreate(comm, ndim, &cgrid);

   /* Set global size to a number different than zero to
      avoid its computation on hypre_StructGridAssemble. */
   hypre_StructGridGlobalSize(cgrid) = -1;

   /* RDF TODO: Inherit num ghost from fgrid here... */

   /* coarsen boxes and create the coarse grid ids (same as fgrid) */
   /* TODO: Move this to hypre_CoarsenBoxArray */
   if (prune)
   {
      /* Compute number of active boxes in the coarse grid */
      box = hypre_BoxCreate(ndim);
      count = 0;
      hypre_ForBoxI(i, fboxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(fboxes, i), box);
         hypre_CoarsenBox(box, origin, stride);
         if (hypre_BoxVolume(box))
         {
           count++;
         }
      }

      cboxes = hypre_BoxArrayCreate(count, ndim);
      cids   = hypre_TAlloc(HYPRE_Int, count, HYPRE_MEMORY_HOST);
      count = 0;
      hypre_ForBoxI(i, fboxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(fboxes, i), box);
         hypre_CoarsenBox(box, origin, stride);
         if (hypre_BoxVolume(box))
         {
            hypre_CopyBox(box, hypre_BoxArrayBox(cboxes, count));
            cids[count] = fids[i];
            hypre_BoxArrayID(cboxes, count) = fids[i];
            count++;
         }
      }
      hypre_BoxDestroy(box);
   }
   else
   {
      /* number of boxes in coarse and fine grids are equal */
      cboxes = hypre_BoxArrayClone(fboxes);
      hypre_CoarsenBoxArray(cboxes, origin, stride);
      cids   = hypre_TAlloc(HYPRE_Int, hypre_BoxArraySize(fboxes), HYPRE_MEMORY_HOST);
      hypre_ForBoxI(i, fboxes)
      {
         cids[i] = fids[i];
      }
   }

   /* set coarse grid boxes */
   hypre_StructGridSetBoxes(cgrid, cboxes);

   /* set coarse grid ids */
   hypre_StructGridSetIDs(cgrid, cids);

   /* adjust periodicity and set for the coarse grid */
   hypre_CopyIndex(hypre_StructGridPeriodic(fgrid), periodic);
   for (i = 0; i < ndim; i++)
   {
      hypre_IndexD(periodic,i) /= hypre_IndexD(stride,i);
   }
   hypre_StructGridSetPeriodic(cgrid, periodic);

   /* Check the max_distance value of the fine grid to determine whether we will
      need to re-gather information in the assemble.  If we need to re-gather,
      then the max_distance will be set to (0,0,0).  Either way, we will create
      and populate the box manager with the information from the fine grid.

      Note: if all global info is already known for a grid, the we do not need
      to re-gather regardless of the max_distance values. */

   // hypre_SetIndex(new_dist, 2); /* RDF: Is this needed with new MAXDIM stuff? */
   for (i = 0; i < ndim; i++)
   {
      coarsen_factor = hypre_IndexD(stride,i);
      hypre_IndexD(new_dist, i) = hypre_IndexD(max_distance,i)/coarsen_factor;
   }

   hypre_BoxManGetAllGlobalKnown(fboxman, &known);

   if ((hypre_IndexMin(new_dist, ndim) > 1) || known)
   {
      /* large enough - don't need to re-gather */
      if (!known)
      {
         /* update new max distance value if global info is not known */
         hypre_StructGridSetMaxDistance(cgrid, new_dist);
      }
   }
   else
   {
      /* not large enough - set max_distance to 0 -
         neighbor info will be collected during the assemble */
      hypre_SetIndex(new_dist, 0);
      hypre_StructGridSetMaxDistance(cgrid, new_dist);
   }

   /* update the new bounding box */
   bounding_box = hypre_BoxClone(hypre_StructGridBoundingBox(fgrid));
   hypre_CoarsenBox(bounding_box, origin, stride);

   hypre_StructGridSetBoundingBox(cgrid, bounding_box);

   /* create a box manager for the coarse grid */
   info_size = hypre_BoxManEntryInfoSize(fboxman);
   max_nentries =  hypre_BoxManMaxNEntries(fboxman);
   hypre_BoxManCreate(max_nentries, info_size, ndim, bounding_box,
                      comm, &cboxman);

   hypre_BoxDestroy(bounding_box);

   /* update all global known */
   hypre_BoxManSetAllGlobalKnown(cboxman, known );

   /* now get the entries from the fgrid box manager, coarsen, and add to the
      coarse grid box manager (note: my boxes have already been coarsened) */
   hypre_BoxManGetAllEntries(fboxman, &num_entries, &entries);

   new_box = hypre_BoxCreate(ndim);
   num = 0;
   last_proc = -1;

   /* entries are sorted by (proc, id) pairs - may not have entries for all
      processors, but for each processor represented, we do have all of its
      boxes.  We will keep them sorted in the new box manager - to avoid
      re-sorting */
   for (i = 0; i < num_entries; i++)
   {
      entry = &entries[i];
      proc = hypre_BoxManEntryProc(entry);

      if  (proc != myid) /* not my boxes */
      {
         hypre_BoxManEntryGetExtents(entry, ilower, iupper);
         hypre_BoxSetExtents(new_box, ilower, iupper);
         hypre_CoarsenBox(new_box, origin, stride);
         id =  hypre_BoxManEntryId(entry);
         /* if there is pruning we need to adjust the ids if any boxes drop out
            (we want these ids sequential - no gaps) - and zero boxes are not
            kept in the box manager */
         if (prune)
         {
            if (proc != last_proc)
            {
               num = 0;
               last_proc = proc;
            }
            if (hypre_BoxVolume(new_box))
            {
               hypre_BoxManAddEntry(cboxman, hypre_BoxIMin(new_box),
                                    hypre_BoxIMax(new_box), proc, num, entry_info);
               num++;
            }
         }
         else /* no pruning - just use id (note that size zero boxes will not be
                 saved in the box manager, so we will have gaps in the box
                 numbers) */
         {
            hypre_BoxManAddEntry(cboxman, hypre_BoxIMin(new_box),
                                 hypre_BoxIMax(new_box), proc, id, entry_info);
         }
      }
      else /* my boxes */
           /* add my coarse grid boxes to the coarse grid box manager (have
              already been pruned if necessary) - re-number the entry ids to be
              sequential (this is the box number, really) */
      {
         if (proc != last_proc) /* just do this once (the first myid) */
         {
            hypre_ForBoxI(j, cboxes)
            {
               box = hypre_BoxArrayBox(cboxes, j);
               hypre_BoxManAddEntry(cboxman, hypre_BoxIMin(box),
                                    hypre_BoxIMax(box), myid, j, entry_info );
            }
            last_proc = proc;
         }
      }
   } /* loop through entries */

   /* these entries are sorted */
   hypre_BoxManSetIsEntriesSort(cboxman, 1);

   hypre_BoxDestroy(new_box);

   // TODO: StructCoarsenAP breaks the regression tests
#if 0
   /* if there is an assumed partition in the fg, then coarsen those boxes as
      well and add to cg */
   hypre_BoxManGetAssumedPartition(fboxman, &fap);

   if (fap)
   {
      /* coarsen fap to get cap */
      hypre_StructCoarsenAP(fap, origin, stride, &cap);

      /* set cap */
      hypre_BoxManSetAssumedPartition(cboxman, cap);
   }
#endif

   /* assign new box manager */
   hypre_StructGridSetBoxManager(cgrid, cboxman);

   /* return the coarse grid */
   *cgrid_ptr = cgrid;

#if TIME_DEBUG
   hypre_EndTiming(tindex);
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
