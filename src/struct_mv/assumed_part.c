/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* This is code for the struct assumed partition - AHB 6/05 */

#include "_hypre_struct_mv.h"

/* these are for debugging */
#define REGION_STAT 0
#define NO_REFINE   0
#define REFINE_INFO 0

/* Note: Functions used only in this file (not elsewhere) to determine the
 * partition have names that start with hypre_AP */

/*--------------------------------------------------------------------------
 * Computes the product of the first ndim index values.  Returns 1 if ndim = 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexProd( hypre_Index  index,
                 HYPRE_Int    ndim )
{
   HYPRE_Int  d, prod;

   prod = 1;
   for (d = 0; d < ndim; d++)
   {
      prod *= hypre_IndexD(index, d);
   }

   return prod;
}

/*--------------------------------------------------------------------------
 * Computes an index into a multi-D box of size bsize[0] x bsize[1] x ... from a
 * rank where the assumed ordering is dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexFromRank( HYPRE_Int    rank,
                     hypre_Index  bsize,
                     hypre_Index  index,
                     HYPRE_Int    ndim )
{
   HYPRE_Int  d, r, s;

   r = rank;
   for (d = ndim - 1; d >= 0; d--)
   {
      s = hypre_IndexProd(bsize, d);
      hypre_IndexD(index, d) = r / s;
      r = r % s;
   }

   return hypre_error_flag;
}

/******************************************************************************
 * Given a region, subdivide the region equally a specified number of times.
 * For dimension d, each "level" is a subdivison of 2^d.  The box_array is
 * adjusted to have space for l(2^d)^level boxes.  We are bisecting each
 * dimension (level) times.
 *
 * We may want to add min size parameter for dimension of results regions
 * (currently 2), i.e., don't bisect a dimension if it will be smaller than 2
 * grid points, for example.
 *****************************************************************************/

HYPRE_Int
hypre_APSubdivideRegion( hypre_Box      *region,
                         HYPRE_Int       ndim,
                         HYPRE_Int       level,
                         hypre_BoxArray *box_array,
                         HYPRE_Int      *num_new_boxes )
{
   HYPRE_Int    i, j,  width, sz, dv, total;
   HYPRE_Int    extra, points, count;
   HYPRE_Int   *partition[HYPRE_MAXDIM];

   HYPRE_Int    min_gridpts; /* This should probably be an input parameter */

   hypre_Index  isize, index, div;
   hypre_Box   *box;

   /* Initialize div */
   hypre_SetIndex(div, 0);

   /* if level = 0 then no dividing */
   if (!level)
   {
      hypre_BoxArraySetSize(box_array, 1);
      hypre_CopyBox(region, hypre_BoxArrayBox(box_array, 0));
      *num_new_boxes = 1;
      return hypre_error_flag;
   }

   /* Get the size of the box in each dimension */
   hypre_BoxGetSize(region, isize);

   /* div = num of regions in each dimension */

   /* Figure out the number of regions.  Make sure the sizes will contain the
      min number of gridpoints, or divide less in that dimension.  We require at
      least min_gridpts in a region dimension. */

   min_gridpts = 4;

   total = 1;
   for (i = 0; i < ndim; i++)
   {
      dv = 1;
      sz = hypre_IndexD(isize, i);
      for (j = 0; j < level; j++)
      {
         if (sz >= 2 * dv * min_gridpts) /* Cut each dim in half */
         {
            dv *= 2;
         }
      }

      /* Space for each partition */
      partition[i] = hypre_TAlloc(HYPRE_Int,  dv + 1, HYPRE_MEMORY_HOST);
      /* Total number of regions to create */
      total = total * dv;

      hypre_IndexD(div, i) = dv;
   }
   *num_new_boxes = total;

   /* Prepare box array */
   hypre_BoxArraySetSize(box_array, total);

   /* Divide each dimension */
   for (i = 0; i < ndim; i++)
   {
      dv = hypre_IndexD(div, i);
      partition[i][0] =  hypre_BoxIMinD(region, i);
      /* Count grid points */
      points = hypre_IndexD(isize, i);
      width =  points / dv;
      extra =  points % dv;
      for (j = 1; j < dv; j++)
      {
         partition[i][j] = partition[i][j - 1] + width;
         if (j <= extra)
         {
            partition[i][j]++;
         }
      }
      partition[i][dv] = hypre_BoxIMaxD(region, i) + 1;
   }

   count = 0;
   hypre_SerialBoxLoop0Begin(ndim, div);
   {
      box = hypre_BoxArrayBox(box_array, count);
      zypre_BoxLoopGetIndex(index);
      for (i = 0; i < ndim; i++)
      {
         j = hypre_IndexD(index, i);
         hypre_BoxIMinD(box, i) = partition[i][j];
         hypre_BoxIMaxD(box, i) = partition[i][j + 1] - 1;
      }
      count++;
   }
   hypre_SerialBoxLoop0End();

   /* clean up */
   for (i = 0; i < ndim; i++)
   {
      hypre_TFree(partition[i], HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, find out how many of *my* boxes are contained in
 * each region.
 *****************************************************************************/

HYPRE_Int
hypre_APFindMyBoxesInRegions( hypre_BoxArray *region_array,
                              hypre_BoxArray *my_box_array,
                              HYPRE_Int     **p_count_array,
                              HYPRE_Real    **p_vol_array )
{
   HYPRE_Int      ndim = hypre_BoxArrayNDim(region_array);
   HYPRE_Int      i, j, d;
   HYPRE_Int      num_boxes, num_regions;
   HYPRE_Int     *count_array;
   HYPRE_Real    *vol_array;
   hypre_Box     *my_box, *result_box, *grow_box, *region;
   hypre_Index    grow_index;

   num_boxes =  hypre_BoxArraySize(my_box_array);
   num_regions = hypre_BoxArraySize(region_array);

   count_array = *p_count_array;
   vol_array = *p_vol_array;

   /* May need to add some sorting to make this more efficient, though we
      shouldn't have many regions */

   /* Note: a box can be in more than one region */

   result_box = hypre_BoxCreate(ndim);
   grow_box = hypre_BoxCreate(ndim);

   for (i = 0; i < num_regions; i++)
   {
      count_array[i] = 0;
      vol_array[i] = 0.0;

      region = hypre_BoxArrayBox(region_array, i);

      for (j = 0; j < num_boxes; j++)
      {
         my_box = hypre_BoxArrayBox(my_box_array, j);
         /* Check if its a zero volume box.  If so, it still need to be counted,
            so expand until volume is non-zero, then intersect. */
         if (hypre_BoxVolume(my_box) == 0)
         {
            hypre_CopyBox(my_box, grow_box);
            for (d = 0; d < ndim; d++)
            {
               if (!hypre_BoxSizeD(my_box, d))
               {
                  hypre_IndexD(grow_index, d) =
                     (hypre_BoxIMinD(my_box, d) - hypre_BoxIMaxD(my_box, d) + 1) / 2;
               }
               else
               {
                  hypre_IndexD(grow_index, d) = 0;
               }
            }
            /* Expand the grow box (leave our box untouched) */
            hypre_BoxGrowByIndex(grow_box, grow_index);
            /* Do they intersect? */
            hypre_IntersectBoxes(grow_box, region, result_box);
         }
         else
         {
            /* Do they intersect? */
            hypre_IntersectBoxes(my_box, region, result_box);
         }
         if (hypre_BoxVolume(result_box) > 0)
         {
            count_array[i]++;
            vol_array[i] += (HYPRE_Real) hypre_BoxVolume(result_box);
         }
      }
   }

   /* clean up */
   hypre_BoxDestroy(result_box);
   hypre_BoxDestroy(grow_box);

   /* output */
   *p_count_array = count_array;
   *p_vol_array = vol_array;

   return hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, find out how many global boxes are contained in each
 * region.  Assumes that p_count_array and p_vol_array have been allocated.
 *****************************************************************************/

HYPRE_Int
hypre_APGetAllBoxesInRegions( hypre_BoxArray *region_array,
                              hypre_BoxArray *my_box_array,
                              HYPRE_Int     **p_count_array,
                              HYPRE_Real    **p_vol_array,
                              MPI_Comm        comm )
{
   HYPRE_Int    i;
   HYPRE_Int   *count_array;
   HYPRE_Int    num_regions;
   HYPRE_Int   *send_buf_count;
   HYPRE_Real  *send_buf_vol;
   HYPRE_Real  *vol_array;
   HYPRE_Real  *dbl_vol_and_count;

   count_array = *p_count_array;
   vol_array = *p_vol_array;

   /* First get a count and volume of my boxes in each region */
   num_regions = hypre_BoxArraySize(region_array);

   send_buf_count = hypre_CTAlloc(HYPRE_Int,  num_regions, HYPRE_MEMORY_HOST);
   send_buf_vol = hypre_CTAlloc(HYPRE_Real,  num_regions * 2,
                                HYPRE_MEMORY_HOST); /* allocate HYPRE_Real */

   dbl_vol_and_count =  hypre_CTAlloc(HYPRE_Real,  num_regions * 2,
                                      HYPRE_MEMORY_HOST); /* allocate HYPRE_Real */

   hypre_APFindMyBoxesInRegions( region_array, my_box_array, &send_buf_count,
                                 &send_buf_vol);


   /* Copy ints to doubles so we can do one Allreduce */
   for (i = 0; i < num_regions; i++)
   {
      send_buf_vol[num_regions + i] = (HYPRE_Real) send_buf_count[i];
   }

   hypre_MPI_Allreduce(send_buf_vol, dbl_vol_and_count, num_regions * 2,
                       HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   /* Unpack */
   for (i = 0; i < num_regions; i++)
   {
      vol_array[i] = dbl_vol_and_count[i];
      count_array[i] = (HYPRE_Int) dbl_vol_and_count[num_regions + i];
   }

   /* Clean up */
   hypre_TFree(send_buf_count, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_vol, HYPRE_MEMORY_HOST);
   hypre_TFree(dbl_vol_and_count, HYPRE_MEMORY_HOST);

   /* Output */
   *p_count_array = count_array;
   *p_vol_array = vol_array;

   return hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, shrink regions according to min and max extents.
 * These regions should all be non-empty at the global level.
 *****************************************************************************/

HYPRE_Int
hypre_APShrinkRegions( hypre_BoxArray *region_array,
                       hypre_BoxArray *my_box_array,
                       MPI_Comm        comm )
{
   HYPRE_Int     ndim, ndim2;
   HYPRE_Int     i, j, d, ii;
   HYPRE_Int     num_boxes, num_regions;
   HYPRE_Int    *indices, *recvbuf;
   HYPRE_Int     count = 0;

   hypre_Box    *my_box, *result_box, *grow_box, *region;
   hypre_Index   grow_index, imin, imax;

   ndim  = hypre_BoxArrayNDim(my_box_array);
   ndim2 = 2 * ndim;

   num_boxes   = hypre_BoxArraySize(my_box_array);
   num_regions = hypre_BoxArraySize(region_array);

   indices = hypre_CTAlloc(HYPRE_Int,  num_regions * ndim2, HYPRE_MEMORY_HOST);
   recvbuf = hypre_CTAlloc(HYPRE_Int,  num_regions * ndim2, HYPRE_MEMORY_HOST);

   result_box = hypre_BoxCreate(ndim);

   /* Allocate a grow box */
   grow_box = hypre_BoxCreate(ndim);

   /* Look locally at my boxes */
   /* For each region */
   for (i = 0; i < num_regions; i++)
   {
      count = 0; /* Number of my boxes in this region */

      /* Get the region box */
      region = hypre_BoxArrayBox(region_array, i);

      /* Go through each of my local boxes */
      for (j = 0; j < num_boxes; j++)
      {
         my_box = hypre_BoxArrayBox(my_box_array, j);

         /* Check if its a zero volume box.  If so, it still needs to be
            checked, so expand until volume is nonzero, then intersect. */
         if (hypre_BoxVolume(my_box) == 0)
         {
            hypre_CopyBox(my_box, grow_box);
            for (d = 0; d < ndim; d++)
            {
               if (!hypre_BoxSizeD(my_box, d))
               {
                  hypre_IndexD(grow_index, d) =
                     (hypre_BoxIMinD(my_box, d) - hypre_BoxIMaxD(my_box, d) + 1) / 2;
               }
               else
               {
                  hypre_IndexD(grow_index, d) = 0;
               }
            }
            /* Grow the grow box (leave our box untouched) */
            hypre_BoxGrowByIndex(grow_box, grow_index);
            /* Do they intersect? */
            hypre_IntersectBoxes(grow_box, region, result_box);
         }
         else
         {
            /* Do they intersect? */
            hypre_IntersectBoxes( my_box, region, result_box);
         }

         if (hypre_BoxVolume(result_box) > 0) /* They intersect */
         {
            if (!count) /* Set min and max for first box */
            {
               ii = i * ndim2;
               for (d = 0; d < ndim; d++)
               {
                  indices[ii + d] = hypre_BoxIMinD(result_box, d);
                  indices[ii + ndim + d] = hypre_BoxIMaxD(result_box, d);
               }
            }

            count++;

            /* Boxes intersect, so get max and min extents of the result box
               (this keeps the bounds inside the region) */
            ii = i * ndim2;
            for (d = 0; d < ndim; d++)
            {
               indices[ii + d] = hypre_min(indices[ii + d],
                                           hypre_BoxIMinD(result_box, d));
               indices[ii + ndim + d] = hypre_max(indices[ii + ndim + d],
                                                  hypre_BoxIMaxD(result_box, d));
            }
         }
      }

      /* If we had no boxes in that region, set the min to the max extents of
         the region and the max to the min! */
      if (!count)
      {
         ii = i * ndim2;
         for (d = 0; d < ndim; d++)
         {
            indices[ii + d] = hypre_BoxIMaxD(region, d);
            indices[ii + ndim + d] = hypre_BoxIMinD(region, d);
         }
      }

      /* Negate max indices for the Allreduce */
      /* Note: min(x)= -max(-x) */
      ii = i * ndim2;
      for (d = 0; d < ndim; d++)
      {
         indices[ii + ndim + d] = -indices[ii + ndim + d];
      }
   }

   /* Do an Allreduce on size and volume to get the global information */
   hypre_MPI_Allreduce(indices, recvbuf, num_regions * ndim2, HYPRE_MPI_INT,
                       hypre_MPI_MIN, comm);

   /* Unpack the "shrunk" regions */
   /* For each region */
   for (i = 0; i < num_regions; i++)
   {
      /* Get the region box */
      region = hypre_BoxArrayBox(region_array, i);

      /* Resize the box */
      ii = i * ndim2;
      for (d = 0; d < ndim; d++)
      {
         hypre_IndexD(imin, d) =  recvbuf[ii + d];
         hypre_IndexD(imax, d) = -recvbuf[ii + ndim + d];
      }

      hypre_BoxSetExtents(region, imin, imax );

      /* Add: check to see whether any shrinking is actually occuring */
   }

   /* Clean up */
   hypre_TFree(recvbuf, HYPRE_MEMORY_HOST);
   hypre_TFree(indices, HYPRE_MEMORY_HOST);
   hypre_BoxDestroy(result_box);
   hypre_BoxDestroy(grow_box);

   return hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, eliminate empty regions.
 *
 * region_array = assumed partition regions
 * count_array  = number of global boxes in each region
 *****************************************************************************/

HYPRE_Int
hypre_APPruneRegions( hypre_BoxArray *region_array,
                      HYPRE_Int     **p_count_array,
                      HYPRE_Real    **p_vol_array )
{
   HYPRE_Int   i, j;
   HYPRE_Int   num_regions;
   HYPRE_Int   count;
   HYPRE_Int   *delete_indices;

   HYPRE_Int   *count_array;
   HYPRE_Real  *vol_array;

   count_array = *p_count_array;
   vol_array = *p_vol_array;

   num_regions = hypre_BoxArraySize(region_array);
   delete_indices = hypre_CTAlloc(HYPRE_Int,  num_regions, HYPRE_MEMORY_HOST);
   count = 0;

   /* Delete regions with zero elements */
   for (i = 0; i < num_regions; i++)
   {
      if (count_array[i] == 0)
      {
         delete_indices[count++] = i;
      }
   }

   hypre_DeleteMultipleBoxes(region_array, delete_indices, count);

   /* Adjust count and volume arrays */
   if (count > 0)
   {
      j = 0;
      for (i = delete_indices[0]; (i + j) < num_regions; i++)
      {
         if (j < count)
         {
            while ((i + j) == delete_indices[j])
            {
               j++; /* Increase the shift */
               if (j == count) { break; }
            }
         }
         vol_array[i] = vol_array[i + j];
         count_array[i] = count_array[i + j];
      }
   }

   /* Clean up */
   hypre_TFree(delete_indices, HYPRE_MEMORY_HOST);

   /* Return variables */
   *p_count_array = count_array;
   *p_vol_array = vol_array;

   return hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, and corresponding volumes contained in regions
 * subdivide some of the regions that are not full enough.
 *****************************************************************************/

HYPRE_Int
hypre_APRefineRegionsByVol( hypre_BoxArray *region_array,
                            HYPRE_Real     *vol_array,
                            HYPRE_Int       max_regions,
                            HYPRE_Real      gamma,
                            HYPRE_Int       ndim,
                            HYPRE_Int      *return_code,
                            MPI_Comm        comm )
{
   HYPRE_Int          i, count, loop;
   HYPRE_Int          num_regions, init_num_regions;
   HYPRE_Int         *delete_indices;

   HYPRE_Real        *fraction_full;
   HYPRE_Int         *order;
   HYPRE_Int          myid, num_procs, est_size;
   HYPRE_Int          new1;

   hypre_BoxArray    *tmp_array;
   hypre_Box         *box;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

   num_regions = hypre_BoxArraySize(region_array);

   if (!num_regions)
   {
      /* No regions, so no subdividing */
      *return_code = 1;
      return hypre_error_flag;
   }

   fraction_full = hypre_CTAlloc(HYPRE_Real,   num_regions, HYPRE_MEMORY_HOST);
   order = hypre_CTAlloc(HYPRE_Int,   num_regions, HYPRE_MEMORY_HOST);
   delete_indices = hypre_CTAlloc(HYPRE_Int,   num_regions, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_regions; i++)
   {
      box = hypre_BoxArrayBox(region_array, i);
      fraction_full[i] = vol_array[i] / hypre_doubleBoxVolume(box);
      order[i] = i; /* This is what order to access the boxes */
   }

   /* Want to refine the regions starting with those that are the least full */
   /* Sort the fraction AND the index */
   hypre_qsort2(order, fraction_full, 0, num_regions - 1);

   /* Now we can subdivide any that are not full enough */
   /* When this is called, we know that size < max_regions */
   /* It is ok to subdivde such that we have slightly more regions than
      max_region, but we do not want more regions than processors */

   tmp_array = hypre_BoxArrayCreate(0, ndim);
   count = 0; /* How many regions subdivided */
   loop = 0; /* Counts the loop number */
   init_num_regions = num_regions;
   /* All regions are at least gamma full and no subdividing occured */
   *return_code = 1;

   while (fraction_full[loop] < gamma)
   {
      /* Some subdividing occurred */
      *return_code = 2;

      /* We can't let the number of regions exceed the number of processors.
         Only an issue for small proc numbers. */
      est_size = num_regions + hypre_pow2(ndim) - 1;
      if (est_size > num_procs)
      {
         if (loop == 0)
         {
            /* Some are less than gamma full, but we cannot further subdivide
               due to max processors limit (no subdividing occured) */
            *return_code = 4;
         }

         else
         {
            /* Some subdividing occured, but there are some regions less than
               gamma full (max reached) that were not subdivided */
            *return_code = 3;
         }

         break;
      }

      box = hypre_BoxArrayBox(region_array, order[loop]);
      hypre_APSubdivideRegion(box, ndim, 1, tmp_array, &new1);

      if (new1 > 1) /* If new = 1, then no subdividing occured */
      {
         num_regions = num_regions + new1 - 1; /* The orginal will be deleted */

         delete_indices[count] = order[loop];
         count++; /* Number of regions subdivided */

         /* Append tmp_array to region_array */
         hypre_AppendBoxArray(tmp_array, region_array);
      }

      /* If we are on the last region */
      if  ((loop + 1) == init_num_regions)
      {
         break;
      }

      /* Clear tmp_array for next loop */
      hypre_BoxArraySetSize(tmp_array, 0);

      /* If we now have too many regions, don't want to subdivide anymore */
      if (num_regions >= max_regions)
      {
         /* See if next regions satifies gamma */
         if (fraction_full[order[loop + 1]] > gamma)
         {
            /* All regions less than gamma full have been subdivided (and we
               have reached max) */
            *return_code = 5;
         }
         else
         {
            /* Some regions less than gamma full (but max is reached) */
            *return_code = 3;
         }
         break;
      }

      loop++; /* Increment to repeat loop */
   }

   if (count == 0 )
   {
      /* No refining occured so don't do any more */
      *return_code = 1;
   }
   else
   {
      /* We subdivided count regions */
      /* Delete the old regions */
      hypre_qsort0(delete_indices, 0, count - 1); /* Put deleted indices in asc order */
      hypre_DeleteMultipleBoxes( region_array, delete_indices, count );
   }

   /* TO DO: number of regions intact (beginning of region array is intact) -
      may return this eventually */
   /* regions_intact = init_num_regions - count; */

   /* Clean up */
   hypre_TFree(fraction_full, HYPRE_MEMORY_HOST);
   hypre_TFree(order, HYPRE_MEMORY_HOST);
   hypre_TFree(delete_indices, HYPRE_MEMORY_HOST);
   hypre_BoxArrayDestroy(tmp_array);

   return hypre_error_flag;
}

/******************************************************************************
 * Construct an assumed partition
 *
 * 8/06 - Changed the assumption that the local boxes have boxnums 0 to
 * num(local_boxes)-1 (now need to pass in ids).
 *
 * 10/06 - Changed.  No longer need to deal with negative boxes as this is used
 * through the box manager.
 *
 * 3/6 - Don't allow more regions than boxes (unless global boxes = 0) and don't
 * partition into more procs than global number of boxes.
 *****************************************************************************/

HYPRE_Int
hypre_StructAssumedPartitionCreate(
   HYPRE_Int                 ndim,
   hypre_Box                *bounding_box,
   HYPRE_Real                global_boxes_size,
   HYPRE_Int                 global_num_boxes,
   hypre_BoxArray           *local_boxes,
   HYPRE_Int                *local_boxnums,
   HYPRE_Int                 max_regions,
   HYPRE_Int                 max_refinements,
   HYPRE_Real                gamma,
   MPI_Comm                  comm,
   hypre_StructAssumedPart **p_assumed_partition )
{
   HYPRE_Int          i, j, d;
   HYPRE_Int          size;
   HYPRE_Int          myid, num_procs;
   HYPRE_Int          num_proc_partitions;
   HYPRE_Int          count_array_size;
   HYPRE_Int         *count_array = NULL;
   HYPRE_Real        *vol_array = NULL, one_volume, dbl_vol;
   HYPRE_Int          return_code;
   HYPRE_Int          num_refine;
   HYPRE_Int          total_boxes, proc_count, max_position;
   HYPRE_Int         *proc_array = NULL;
   HYPRE_Int          initial_level;
   HYPRE_Int          dmax;
   HYPRE_Real         width, wmin, wmax;
   HYPRE_Real         rn_cubes, rn_cube_procs, rn_cube_divs, rdiv;

   hypre_Index        div_index;
   hypre_BoxArray    *region_array;
   hypre_Box         *box, *grow_box;

   hypre_StructAssumedPart *assumed_part;

   HYPRE_Int   proc_alloc, count, box_count;
   HYPRE_Int   max_response_size;
   HYPRE_Int  *response_buf = NULL, *response_buf_starts = NULL;
   HYPRE_Int  *tmp_proc_ids = NULL, *tmp_box_nums = NULL, *tmp_box_inds = NULL;
   HYPRE_Int  *proc_array_starts = NULL;

   hypre_BoxArray              *my_partition;
   hypre_DataExchangeResponse  response_obj;

   HYPRE_Int  *contact_boxinfo;
   HYPRE_Int  index;


   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   /* Special case where there are no boxes in the grid */
   if (global_num_boxes == 0)
   {
      region_array = hypre_BoxArrayCreate(0, ndim);
      assumed_part = hypre_TAlloc(hypre_StructAssumedPart,  1, HYPRE_MEMORY_HOST);

      hypre_StructAssumedPartNDim(assumed_part) = ndim;
      hypre_StructAssumedPartRegions(assumed_part) = region_array;
      hypre_StructAssumedPartNumRegions(assumed_part) = 0;
      hypre_StructAssumedPartDivisions(assumed_part) =  NULL;
      hypre_StructAssumedPartProcPartitions(assumed_part) =
         hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
      hypre_StructAssumedPartProcPartition(assumed_part, 0) = 0;
      hypre_StructAssumedPartMyPartition(assumed_part) =  NULL;
      hypre_StructAssumedPartMyPartitionBoxes(assumed_part)
         = hypre_BoxArrayCreate(0, ndim);
      hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = 0;
      hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = 0;
      hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part) = 0;
      hypre_StructAssumedPartMyPartitionBoxnums(assumed_part) = NULL;
      hypre_StructAssumedPartMyPartitionProcIds(assumed_part) = NULL;
      *p_assumed_partition = assumed_part;

      return hypre_error_flag;
   }
   /* End special case of zero boxes */

   /* FIRST DO ALL THE GLOBAL PARTITION INFO */

   /* Initially divide the bounding box */

   if (!hypre_BoxVolume(bounding_box) && global_num_boxes)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Bounding box has zero volume AND there are grid boxes");
   }

   /* First modify any input parameters if necessary */

   /* Don't want the number of regions exceeding the number of processors */
   /* Note: This doesn't change the value in the caller's code */
   max_regions = hypre_min(num_procs, max_regions);

   /* Don't want more regions than boxes either */
   if (global_num_boxes) { max_regions = hypre_min(global_num_boxes, max_regions); }

   /* Start with a region array of size 0 */
   region_array = hypre_BoxArrayCreate(0, ndim);

   /* If the bounding box is sufficiently covered by boxes, then we will just
      have one region (the bounding box), otherwise we will subdivide */

   one_volume = hypre_doubleBoxVolume(bounding_box);

   if ( ((global_boxes_size / one_volume) > gamma) ||
        (global_num_boxes > one_volume) || (global_num_boxes == 0) )
   {
      /* Don't bother with any refinements.  We are full enough, or we have a
         small bounding box and we are not full because of empty boxes */
      initial_level = 0;
      max_refinements = 0;
   }
   else
   {
      /* Could be an input parameter, but 1 division is probably sufficient */
      initial_level = 1;

      /* Start with the specified intial_levels for the original domain, unless
         we have a smaller number of procs */
      for (i = 0; i < initial_level; i++)
      {
         if ( hypre_pow2(initial_level * ndim) > num_procs) { initial_level --; }

         /* Not be able to do any refinements due to the number of processors */
         if (!initial_level) { max_refinements = 0; }
      }
   }

#if NO_REFINE
   max_refinements = 0;
   initial_level = 0;
#endif

#if REFINE_INFO
   if (myid == 0)
   {
      hypre_printf("gamma =  %g\n", gamma);
      hypre_printf("max_regions =  %d\n", max_regions);
      hypre_printf("max_refinements =  %d\n", max_refinements);
      hypre_printf("initial level =  %d\n", initial_level);
   }
#endif

   /* Divide the bounding box */
   hypre_APSubdivideRegion(bounding_box, ndim, initial_level, region_array, &size);
   /* If no subdividing occured (because too small) then don't try to refine */
   if (initial_level > 0 && size == 1) { max_refinements = 0; }

   /* Need space for count and volume */
   size = hypre_BoxArraySize(region_array);
   count_array_size = size; /* Memory allocation size */
   count_array = hypre_CTAlloc(HYPRE_Int,   size, HYPRE_MEMORY_HOST);
   vol_array =  hypre_CTAlloc(HYPRE_Real,   size, HYPRE_MEMORY_HOST);

   /* How many boxes are in each region (global count) and what is the volume */
   hypre_APGetAllBoxesInRegions(region_array, local_boxes, &count_array,
                                &vol_array, comm);

   /* Don't do any initial prune and shrink if we have only one region and we
      can't do any refinements */

   if ( !(size == 1 && max_refinements == 0))
   {
      /* Get rid of regions with no boxes (and adjust count and vol arrays) */
      hypre_APPruneRegions( region_array, &count_array, &vol_array);

      /* Shrink the extents */
      hypre_APShrinkRegions( region_array, local_boxes, comm);
   }

   /* Keep track of refinements */
   num_refine = 0;

   /* Now we can keep refining by dividing the regions that are not full enough
      and eliminating empty regions */
   while ( (hypre_BoxArraySize(region_array) < max_regions) &&
           (num_refine < max_refinements) )
   {
      num_refine++;

      /* Calculate how full the regions are and subdivide the least full */

      size = hypre_BoxArraySize(region_array);

      /* Divide regions that are not full enough */
      hypre_APRefineRegionsByVol(region_array, vol_array, max_regions,
                                 gamma, ndim, &return_code, comm);

      /* 1 = all regions are at least gamma full - no subdividing occured */
      /* 4 = no subdividing occured due to num_procs limit on regions */
      if (return_code == 1 || return_code == 4)
      {
         break;
      }
      /* This is extraneous I think */
      if (size == hypre_BoxArraySize(region_array))
      {
         /* No dividing occured - exit the loop */
         break;
      }

      size = hypre_BoxArraySize(region_array);
      if (size >  count_array_size)
      {
         count_array = hypre_TReAlloc(count_array,  HYPRE_Int,   size, HYPRE_MEMORY_HOST);
         vol_array =  hypre_TReAlloc(vol_array,  HYPRE_Real,   size, HYPRE_MEMORY_HOST);
         count_array_size = size;
      }

      /* FUTURE MOD: Just count and prune and shrink in the modified regions
         from refineRegionsByVol. These are the last regions in the array. */

      /* Num boxes are in each region (global count) and what the volume is */
      hypre_APGetAllBoxesInRegions(region_array, local_boxes, &count_array,
                                   &vol_array, comm);

      /* Get rid of regions with no boxes (and adjust count and vol arrays) */
      hypre_APPruneRegions(region_array, &count_array, &vol_array);

      /* Shrink the extents */
      hypre_APShrinkRegions(region_array, local_boxes, comm);

      /* These may be ok after pruning, but if no pruning then exit the loop */
      /* 5 = all regions < gamma full were subdivided and max reached */
      /* 3 = some regions were divided (not all that needed) and max reached */
      if ( (return_code == 3 || return_code == 5)
           && size == hypre_BoxArraySize(region_array) )
      {
         break;
      }

   }
   /* End of refinements */

   /* Error checking */
   if (global_num_boxes)
   {
      hypre_ForBoxI(i, region_array)
      {
         if (hypre_BoxVolume(hypre_BoxArrayBox(region_array, i)) == 0)
         {
            hypre_error(HYPRE_ERROR_GENERIC);
            hypre_error_w_msg(
               HYPRE_ERROR_GENERIC,
               "A region has zero volume (this should never happen)!");
         }
      }
   }

#if REGION_STAT
   if (myid == 0)
   {
      hypre_printf("myid = %d, %d REGIONS (after refining %d times\n",
                   myid, hypre_BoxArraySize(region_array), num_refine);

      hypre_ForBoxI(i, region_array)
      {
         box = hypre_BoxArrayBox(region_array, i);
         hypre_printf("myid = %d, %d:  (%d, %d, %d)  x  (%d, %d, %d)\n",
                      myid, i,
                      hypre_BoxIMinX(box),
                      hypre_BoxIMinY(box),
                      hypre_BoxIMinZ(box),
                      hypre_BoxIMaxX(box),
                      hypre_BoxIMaxY(box),
                      hypre_BoxIMaxZ(box));
      }
   }
#endif

   hypre_TFree(vol_array, HYPRE_MEMORY_HOST);

   /* ------------------------------------------------------------------------*/

   /* Now we have the regions - construct the assumed partition */

   size = hypre_BoxArraySize(region_array);
   assumed_part = hypre_TAlloc(hypre_StructAssumedPart,  1, HYPRE_MEMORY_HOST);
   hypre_StructAssumedPartNDim(assumed_part) = ndim;
   hypre_StructAssumedPartRegions(assumed_part) = region_array;
   /* The above is aliased, so don't destroy region_array in this function */
   hypre_StructAssumedPartNumRegions(assumed_part) = size;
   hypre_StructAssumedPartDivisions(assumed_part) =
      hypre_CTAlloc(hypre_Index,  size, HYPRE_MEMORY_HOST);

   /* First determine which processors (how many) to assign to each region */
   proc_array = hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);
   /* This is different than the total number of boxes as some boxes can be in
      more than one region */
   total_boxes = 0;
   proc_count = 0;
   d = -1;
   max_position = -1;
   /* Calculate total number of boxes in the regions */
   for (i = 0; i < size; i++)
   {
      total_boxes += count_array[i];
   }
   /* Calculate the fraction of actual boxes in each region, multiplied by total
      number of proc partitons desired, put result in proc_array to assign each
      region a number of processors proportional to the fraction of boxes */

   /* 3/6 - Limit the number of proc partitions to no larger than the total
      boxes in the regions (at coarse levels, may be many more procs than boxes,
      so this should minimize some communication). */
   num_proc_partitions = hypre_min(num_procs, total_boxes);

   for (i = 0; i < size; i++)
   {
      if (!total_boxes) /* In case there are no boxes in a grid */
      {
         proc_array[i] = 0;
      }
      else
      {
         proc_array[i] = (HYPRE_Int)
                         hypre_round( ((HYPRE_Real)count_array[i] / (HYPRE_Real)total_boxes) *
                                      (HYPRE_Real) num_proc_partitions );
      }

      box =  hypre_BoxArrayBox(region_array, i);
      dbl_vol = hypre_doubleBoxVolume(box);

      /* Can't have any zeros! */
      if (!proc_array[i]) { proc_array[i] = 1; }

      if (dbl_vol < (HYPRE_Real) proc_array[i])
      {
         /* Don't let the number of procs be greater than the volume.  If true,
            then safe to cast back to HYPRE_Int and vol doesn't overflow. */
         proc_array[i] = (HYPRE_Int) dbl_vol;
      }

      proc_count += proc_array[i];
      if (d < proc_array[i])
      {
         d = proc_array[i];
         max_position = i;
      }

      /*If (myid == 0) hypre_printf("proc array[%d] = %d\n", i, proc_array[i]);*/
   }

   hypre_TFree(count_array, HYPRE_MEMORY_HOST);

   /* Adjust such that num_proc_partitions = proc_count (they should be close) */
   /* A processor is only assigned to ONE region */

   /* If we need a few more processors assigned in proc_array for proc_count to
      equal num_proc_partitions (it is ok if we have fewer procs in proc_array
      due to volume constraints) */
   while (num_proc_partitions > proc_count)
   {
      proc_array[max_position]++;

      if ( (HYPRE_Real) proc_array[max_position] >
           hypre_doubleBoxVolume(hypre_BoxArrayBox(region_array, max_position)) )
      {
         proc_array[max_position]--;
         break; /* Some processors won't get assigned partitions */
      }
      proc_count++;
   }

   /* If we we need fewer processors in proc_array */
   i = 0;
   while (num_proc_partitions < proc_count)
   {
      if (proc_array[max_position] != 1)
      {
         proc_array[max_position]--;
      }
      else
      {
         while (i < size && proc_array[i] <= 1) /* size is the number of regions */
         {
            i++;
         }
         proc_array[i]--;
      }
      proc_count--;
   }
   /* The above logic would be flawed IF we allowed more regions than
      processors, but this is not allowed! */

   /* Now we have the number of processors in each region so create the
      processor partition */
   /* size = # of regions */
   hypre_StructAssumedPartProcPartitions(assumed_part) =
      hypre_CTAlloc(HYPRE_Int,  size + 1, HYPRE_MEMORY_HOST);
   hypre_StructAssumedPartProcPartition(assumed_part, 0) = 0;
   for (i = 0; i < size; i++)
   {
      hypre_StructAssumedPartProcPartition(assumed_part, i + 1) =
         hypre_StructAssumedPartProcPartition(assumed_part, i) + proc_array[i];
   }

   /* Now determine the NUMBER of divisions in the x, y amd z dir according
      to the number or processors assigned to the region */

   /* FOR EACH REGION */
   for (i = 0; i < size; i++)
   {
      proc_count = proc_array[i];
      box = hypre_BoxArrayBox(region_array, i);

      /* Find min width and max width dimensions */
      dmax = 0;
      wmin = wmax = hypre_BoxSizeD(box, 0);
      for (d = 1; d < ndim; d++)
      {
         width = hypre_BoxSizeD(box, d);
         if (width < wmin)
         {
            wmin = width;
         }
         else if (width > wmax)
         {
            dmax = d;
            wmax = width;
         }
      }

      /* Notation (all real numbers):
         rn_cubes      - number of wmin-width cubes in the region
         rn_cube_procs - number of procs per wmin-width cube
         rn_cube_divs  - number of divs per wmin-width cube */

      /* After computing the above, each div_index[d] is set by first flooring
         rn_cube_divs, then div_index[dmax] is incremented until we have more
         partitions than processors. */

      rn_cubes = hypre_doubleBoxVolume(box) / hypre_pow(wmin, ndim);
      rn_cube_procs = proc_count / rn_cubes;
      rn_cube_divs = hypre_pow(rn_cube_procs, (1.0 / (HYPRE_Real)ndim));

      for (d = 0; d < ndim; d++)
      {
         width = hypre_BoxSizeD(box, d);
         rdiv = rn_cube_divs * (width / wmin);
         /* Add a small number to compensate for roundoff issues */
         hypre_IndexD(div_index, d) = (HYPRE_Int) hypre_floor(rdiv + 1.0e-6);
         /* Make sure div_index[d] is at least 1 */
         hypre_IndexD(div_index, d) = hypre_max(hypre_IndexD(div_index, d), 1);
      }

      /* Decrease div_index to ensure no more than 2 partitions per processor.
       * This is only needed when div_index[d] is adjusted to 1 above. */
      while (hypre_IndexProd(div_index, ndim) >= 2 * proc_count)
      {
         /* Decrease the max dimension by a factor of 2 without going below 1 */
         hypre_IndexD(div_index, dmax) = (hypre_IndexD(div_index, dmax) + 1) / 2;
         for (d = 0; d < ndim; d++)
         {
            if (hypre_IndexD(div_index, d) > hypre_IndexD(div_index, dmax))
            {
               dmax = d;
            }
         }
      }

      /* Increment div_index[dmax] to ensure more partitions than processors.
         This can never result in more than 2 partitions per processor. */
      while (hypre_IndexProd(div_index, ndim) < proc_count)
      {
         hypre_IndexD(div_index, dmax) ++;
      }

      hypre_CopyIndex(div_index, hypre_StructAssumedPartDivision(assumed_part, i));

#if REGION_STAT
      if ( myid == 0 )
      {
         hypre_printf("region = %d, proc_count = %d, divisions = [", i, proc_count);
         for (d = 0; d < ndim; d++)
         {
            hypre_printf(" %d", hypre_IndexD(div_index, d));
         }
         hypre_printf("]\n");
      }
#endif
   } /* End of FOR EACH REGION loop */

   /* NOW WE HAVE COMPLETED GLOBAL INFO - START FILLING IN LOCAL INFO */

   /* We need to populate the assumed partition object with info specific to
      each processor, like which assumed partition we own, which boxes are in
      that region, etc. */

   /* Figure out my partition region and put it in the assumed_part structure */
   hypre_StructAssumedPartMyPartition(assumed_part) = hypre_BoxArrayCreate(2, ndim);
   my_partition = hypre_StructAssumedPartMyPartition(assumed_part);
   hypre_StructAssumedPartitionGetRegionsFromProc(assumed_part, myid, my_partition);
#if 0
   hypre_ForBoxI(i, my_partition)
   {
      box = hypre_BoxArrayBox(my_partition, i);
      hypre_printf("myid = %d: MY ASSUMED Partitions (%d):  (%d, %d, %d)  x  "
                   "(%d, %d, %d)\n",
                   myid, i,
                   hypre_BoxIMinX(box),
                   hypre_BoxIMinY(box),
                   hypre_BoxIMinZ(box),
                   hypre_BoxIMaxX(box),
                   hypre_BoxIMaxY(box),
                   hypre_BoxIMaxZ(box));
   }
#endif

   /* Find out which boxes are in my partition: Look through my boxes, figure
      out which assumed parition (AP) they fall in and contact that processor.
      Use the exchange data functionality for this. */

   proc_alloc = hypre_pow2(ndim);
   proc_array = hypre_TReAlloc(proc_array,  HYPRE_Int,  proc_alloc, HYPRE_MEMORY_HOST);

   /* Probably there will mostly be one proc per box */
   /* Don't want to allocate too much memory here */
   size = (HYPRE_Int)(1.2 * hypre_BoxArraySize(local_boxes));

   /* Each local box may live on multiple procs in the assumed partition */
   tmp_proc_ids = hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST); /* local box proc ids */
   tmp_box_nums = hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST); /* local box boxnum */
   tmp_box_inds = hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST); /* local box array index */

   proc_count = 0;
   count = 0; /* Current number of procs */
   grow_box = hypre_BoxCreate(ndim);

   hypre_ForBoxI(i, local_boxes)
   {
      box = hypre_BoxArrayBox(local_boxes, i);

      hypre_StructAssumedPartitionGetProcsFromBox(
         assumed_part, box, &proc_count, &proc_alloc, &proc_array);
      /* Do we need more storage? */
      if ((count + proc_count) > size)
      {
         size = (HYPRE_Int)(size + proc_count + 1.2 * (hypre_BoxArraySize(local_boxes) - i));
         /* hypre_printf("myid = %d, *adjust* alloc size = %d\n", myid, size);*/
         tmp_proc_ids = hypre_TReAlloc(tmp_proc_ids,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
         tmp_box_nums = hypre_TReAlloc(tmp_box_nums,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
         tmp_box_inds = hypre_TReAlloc(tmp_box_inds,  HYPRE_Int,  size, HYPRE_MEMORY_HOST);
      }
      for (j = 0; j < proc_count; j++)
      {
         tmp_proc_ids[count] = proc_array[j];
         tmp_box_nums[count] = local_boxnums[i];
         tmp_box_inds[count] = i;
         count++;
      }
   }

   hypre_BoxDestroy(grow_box);

   /* Now we have two arrays: tmp_proc_ids and tmp_box_nums.  These are
      corresponding box numbers and proc ids.  We need to sort the processor ids
      and then create a new buffer to send to the exchange data function. */

   /* Sort the proc_ids */
   hypre_qsort3i(tmp_proc_ids, tmp_box_nums, tmp_box_inds, 0, count - 1);

   /* Use proc_array for the processor ids to contact.  Use box array to get our
      boxes and then pass the array only (not the structure) to exchange data. */
   box_count = count;

   contact_boxinfo = hypre_CTAlloc(HYPRE_Int,  box_count * (1 + 2 * ndim), HYPRE_MEMORY_HOST);

   proc_array = hypre_TReAlloc(proc_array,  HYPRE_Int,  box_count, HYPRE_MEMORY_HOST);
   proc_array_starts = hypre_CTAlloc(HYPRE_Int,  box_count + 1, HYPRE_MEMORY_HOST);
   proc_array_starts[0] = 0;

   proc_count = 0;
   index = 0;

   if (box_count)
   {
      proc_array[0] = tmp_proc_ids[0];

      contact_boxinfo[index++] = tmp_box_nums[0];
      box = hypre_BoxArrayBox(local_boxes, tmp_box_inds[0]);
      for (d = 0; d < ndim; d++)
      {
         contact_boxinfo[index++] = hypre_BoxIMinD(box, d);
         contact_boxinfo[index++] = hypre_BoxIMaxD(box, d);
      }
      proc_count++;
   }

   for (i = 1; i < box_count; i++)
   {
      if (tmp_proc_ids[i]  != proc_array[proc_count - 1])
      {
         proc_array[proc_count] = tmp_proc_ids[i];
         proc_array_starts[proc_count] = i;
         proc_count++;
      }

      /* These boxes are not copied in a particular order */

      contact_boxinfo[index++] = tmp_box_nums[i];
      box = hypre_BoxArrayBox(local_boxes, tmp_box_inds[i]);
      for (d = 0; d < ndim; d++)
      {
         contact_boxinfo[index++] = hypre_BoxIMinD(box, d);
         contact_boxinfo[index++] = hypre_BoxIMaxD(box, d);
      }
   }
   proc_array_starts[proc_count] = box_count;

   /* Clean up */
   hypre_TFree(tmp_proc_ids, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_box_nums, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_box_inds, HYPRE_MEMORY_HOST);

   /* EXCHANGE DATA */

   /* Prepare to populate the local info in the assumed partition */
   hypre_StructAssumedPartMyPartitionBoxes(assumed_part)
      = hypre_BoxArrayCreate(box_count, ndim);
   hypre_BoxArraySetSize(hypre_StructAssumedPartMyPartitionBoxes(assumed_part), 0);
   hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = 0;
   hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = box_count;
   hypre_StructAssumedPartMyPartitionProcIds(assumed_part)
      = hypre_CTAlloc(HYPRE_Int,  box_count, HYPRE_MEMORY_HOST);
   hypre_StructAssumedPartMyPartitionBoxnums(assumed_part)
      = hypre_CTAlloc(HYPRE_Int,  box_count, HYPRE_MEMORY_HOST);
   hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part) = 0;

   /* Set up for exchanging data */
   /* The response we expect is just a confirmation */
   response_buf = NULL;
   response_buf_starts = NULL;

   /* Response object */
   response_obj.fill_response = hypre_APFillResponseStructAssumedPart;
   response_obj.data1 = assumed_part; /* Where we keep info from contacts */
   response_obj.data2 = NULL;

   max_response_size = 0; /* No response data - just confirmation */

   hypre_DataExchangeList(proc_count, proc_array,
                          contact_boxinfo, proc_array_starts,
                          (1 + 2 * ndim)*sizeof(HYPRE_Int),
                          sizeof(HYPRE_Int), &response_obj, max_response_size, 1,
                          comm, (void**) &response_buf, &response_buf_starts);

   hypre_TFree(proc_array, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_array_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(contact_boxinfo, HYPRE_MEMORY_HOST);

   /* Return vars */
   *p_assumed_partition = assumed_part;

   return hypre_error_flag;
}

/******************************************************************************
 * Destroy the assumed partition.
 *****************************************************************************/

HYPRE_Int
hypre_StructAssumedPartitionDestroy( hypre_StructAssumedPart *assumed_part )
{
   if (assumed_part)
   {
      hypre_BoxArrayDestroy( hypre_StructAssumedPartRegions(assumed_part));
      hypre_TFree(hypre_StructAssumedPartProcPartitions(assumed_part), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_StructAssumedPartDivisions(assumed_part), HYPRE_MEMORY_HOST);
      hypre_BoxArrayDestroy( hypre_StructAssumedPartMyPartition(assumed_part));
      hypre_BoxArrayDestroy( hypre_StructAssumedPartMyPartitionBoxes(assumed_part));
      hypre_TFree(hypre_StructAssumedPartMyPartitionProcIds(assumed_part), HYPRE_MEMORY_HOST);
      hypre_TFree( hypre_StructAssumedPartMyPartitionBoxnums(assumed_part), HYPRE_MEMORY_HOST);

      /* This goes last! */
      hypre_TFree(assumed_part, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/******************************************************************************
 * fillResponseStructAssumedPart
 *****************************************************************************/

HYPRE_Int
hypre_APFillResponseStructAssumedPart(void      *p_recv_contact_buf,
                                      HYPRE_Int  contact_size,
                                      HYPRE_Int  contact_proc,
                                      void      *ro,
                                      MPI_Comm   comm,
                                      void     **p_send_response_buf,
                                      HYPRE_Int *response_message_size )
{
   HYPRE_UNUSED_VAR(p_send_response_buf);

   HYPRE_Int    ndim, size, alloc_size, myid, i, d, index;
   HYPRE_Int   *ids, *boxnums;
   HYPRE_Int   *recv_contact_buf;

   hypre_Box   *box;

   hypre_BoxArray              *part_boxes;
   hypre_DataExchangeResponse  *response_obj = (hypre_DataExchangeResponse  *)ro;
   hypre_StructAssumedPart     *assumed_part = (hypre_StructAssumedPart     *)response_obj->data1;

   /* Initialize stuff */
   hypre_MPI_Comm_rank(comm, &myid );

   ndim = hypre_StructAssumedPartNDim(assumed_part);
   part_boxes =  hypre_StructAssumedPartMyPartitionBoxes(assumed_part);
   ids = hypre_StructAssumedPartMyPartitionProcIds(assumed_part);
   boxnums = hypre_StructAssumedPartMyPartitionBoxnums(assumed_part);

   size =  hypre_StructAssumedPartMyPartitionIdsSize(assumed_part);
   alloc_size = hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part);

   recv_contact_buf = (HYPRE_Int * ) p_recv_contact_buf;

   /* Increment how many procs have contacted us */
   hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part)++;

   /* Check to see if we need to allocate more space for ids and boxnums */
   if ((size + contact_size) > alloc_size)
   {
      alloc_size = size + contact_size;
      ids = hypre_TReAlloc(ids,  HYPRE_Int,  alloc_size, HYPRE_MEMORY_HOST);
      boxnums = hypre_TReAlloc(boxnums,  HYPRE_Int,  alloc_size, HYPRE_MEMORY_HOST);
      hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = alloc_size;
   }

   box = hypre_BoxCreate(ndim);

   /* Populate our assumed partition according to boxes received */
   index = 0;
   for (i = 0; i < contact_size; i++)
   {
      ids[size + i] = contact_proc; /* Set the proc id */
      boxnums[size + i] = recv_contact_buf[index++];
      for (d = 0; d < ndim; d++)
      {
         hypre_BoxIMinD(box, d) = recv_contact_buf[index++];
         hypre_BoxIMaxD(box, d) = recv_contact_buf[index++];
      }

      hypre_AppendBox(box, part_boxes);
   }
   /* Adjust the size of the proc ids*/
   hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = size + contact_size;

   /* In case more memory was allocated we have to assign these pointers back */
   hypre_StructAssumedPartMyPartitionBoxes(assumed_part) = part_boxes;
   hypre_StructAssumedPartMyPartitionProcIds(assumed_part) = ids;
   hypre_StructAssumedPartMyPartitionBoxnums(assumed_part) = boxnums;

   /* Output - no message to return (confirmation) */
   *response_message_size = 0;

   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/******************************************************************************
 * Given a processor id, get that processor's assumed region(s).
 *
 * At most a processor has 2 assumed regions.  Pass in a BoxArray of size 2.
 *****************************************************************************/

HYPRE_Int
hypre_StructAssumedPartitionGetRegionsFromProc(
   hypre_StructAssumedPart *assumed_part,
   HYPRE_Int                proc_id,
   hypre_BoxArray          *assumed_regions )
{
   HYPRE_Int   *proc_partitions;
   HYPRE_Int    ndim, i, d;
   HYPRE_Int    in_region, proc_count, proc_start, num_partitions;
   HYPRE_Int    part_num, width, extra;
   HYPRE_Int    adj_proc_id;
   HYPRE_Int    num_assumed, num_regions;

   hypre_Box   *region, *box;
   hypre_Index  div, divindex, rsize, imin, imax;
   HYPRE_Int    divi;

   ndim = hypre_StructAssumedPartNDim(assumed_part);
   num_regions = hypre_StructAssumedPartNumRegions(assumed_part);
   proc_partitions = hypre_StructAssumedPartProcPartitions(assumed_part);

   /* Check if this processor owns an assumed region.  It is rare that it won't
      (only if # procs > bounding box or # procs > global #boxes). */

   if (proc_id >= proc_partitions[num_regions])
   {
      /* Owns no boxes */
      num_assumed = 0;
   }
   else
   {
      /* Which partition region am I in? */
      in_region = 0;
      if (num_regions > 1)
      {
         while (proc_id >= proc_partitions[in_region + 1])
         {
            in_region++;
         }
      }

      /* First processor in the range */
      proc_start = proc_partitions[in_region];
      /* How many processors in that region? */
      proc_count = proc_partitions[in_region + 1] - proc_partitions[in_region];
      /* Get the region */
      region = hypre_BoxArrayBox(hypre_StructAssumedPartRegions(assumed_part),
                                 in_region);
      /* Size of the regions */
      hypre_BoxGetSize(region, rsize);
      /* Get the divisions in each dimension */
      hypre_CopyIndex(hypre_StructAssumedPartDivision(assumed_part, in_region),
                      div);

      /* Calculate the assumed partition(s) (at most 2) that I own */

      num_partitions = hypre_IndexProd(div, ndim);
      /* How many procs have 2 partitions instead of one*/
      extra =  num_partitions % proc_count;

      /* Adjust the proc number to range from 0 to (proc_count-1) */
      adj_proc_id = proc_id - proc_start;

      /* The region is divided into num_partitions partitions according to the
         number of divisions in each direction.  Some processors may own more
         than one partition (up to 2).  These partitions are numbered by
         dimension 0 first, then dimension 1, etc.  From the partition number,
         we can calculate the processor id. */

      /* Get my partition number */
      if (adj_proc_id < extra)
      {
         part_num = adj_proc_id * 2;
         num_assumed = 2;
      }
      else
      {
         part_num = extra + adj_proc_id;
         num_assumed = 1;
      }
   }

   /* Make sure BoxArray has been allocated for num_assumed boxes */
   hypre_BoxArraySetSize(assumed_regions, num_assumed);

   for (i = 0; i < num_assumed; i++)
   {
      hypre_IndexFromRank(part_num + i, div, divindex, ndim);

      for (d = ndim - 1; d >= 0; d--)
      {
         width = hypre_IndexD(rsize, d) / hypre_IndexD(div, d);
         extra = hypre_IndexD(rsize, d) % hypre_IndexD(div, d);

         divi = hypre_IndexD(divindex, d);
         hypre_IndexD(imin, d) = divi * width + hypre_min(divi, extra);
         divi = hypre_IndexD(divindex, d) + 1;
         hypre_IndexD(imax, d) = divi * width + hypre_min(divi, extra) - 1;

         /* Change relative coordinates to absolute */
         hypre_IndexD(imin, d) +=  hypre_BoxIMinD(region, d);
         hypre_IndexD(imax, d) +=  hypre_BoxIMinD(region, d);
      }

      /* Set the assumed region*/
      box = hypre_BoxArrayBox(assumed_regions, i);
      hypre_BoxSetExtents(box, imin, imax);
   }

   return hypre_error_flag;
}

/******************************************************************************
 * Given a box, which processor(s) assumed partition does the box intersect.
 *
 * proc_array should be allocated to size_alloc_proc_array
 *****************************************************************************/

HYPRE_Int
hypre_StructAssumedPartitionGetProcsFromBox(
   hypre_StructAssumedPart *assumed_part,
   hypre_Box               *box,
   HYPRE_Int               *num_proc_array,
   HYPRE_Int               *size_alloc_proc_array,
   HYPRE_Int              **p_proc_array )
{
   HYPRE_Int       ndim = hypre_StructAssumedPartNDim(assumed_part);

   HYPRE_Int       i, d, p, q, r, myid;
   HYPRE_Int       num_regions, in_regions, this_region, proc_count, proc_start;
   HYPRE_Int       adj_proc_id, extra, num_partitions;
   HYPRE_Int       width;

   HYPRE_Int      *proc_array, proc_array_count;
   HYPRE_Int      *which_regions;
   HYPRE_Int      *proc_ids, num_proc_ids, size_proc_ids, ncorners;

   hypre_Box      *region;
   hypre_Box      *result_box, *part_box, *part_dbox;
   hypre_Index     div, rsize, stride, loop_size;
   hypre_IndexRef  start;
   hypre_BoxArray *region_array;
   HYPRE_Int      *proc_partitions;

   /* Need myid only for the hypre_printf statement */
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   proc_array = *p_proc_array;
   region_array = hypre_StructAssumedPartRegions(assumed_part);
   num_regions = hypre_StructAssumedPartNumRegions(assumed_part);
   proc_partitions = hypre_StructAssumedPartProcPartitions(assumed_part);

   /* First intersect the box to find out which region(s) it lies in, then
      determine which processor owns the assumed part of these regions(s) */

   result_box = hypre_BoxCreate(ndim);
   part_box = hypre_BoxCreate(ndim);
   part_dbox = hypre_BoxCreate(ndim);
   which_regions = hypre_CTAlloc(HYPRE_Int,  num_regions, HYPRE_MEMORY_HOST);

   /* The number of corners in a box is a good initial size for proc_ids */
   ncorners = hypre_pow2(ndim);
   size_proc_ids = ncorners;
   proc_ids = hypre_CTAlloc(HYPRE_Int,  size_proc_ids, HYPRE_MEMORY_HOST);
   num_proc_ids = 0;

   /* which partition region(s) am i in? */
   in_regions = 0;
   for (i = 0; i < num_regions; i++)
   {
      region = hypre_BoxArrayBox(region_array, i);
      hypre_IntersectBoxes(box, region, result_box);
      if (  hypre_BoxVolume(result_box) > 0 )
      {
         which_regions[in_regions] = i;
         in_regions++;
      }
   }

#if 0
   if (in_regions == 0)
   {
      /* 9/16/10 - In hypre_SStructGridAssembleBoxManagers we grow boxes by 1
         before we gather boxes because of shared variables, so we can get the
         situation that the gather box is outside of the assumed region. */

      if (hypre_BoxVolume(box) > 0)
      {
         hypre_error(HYPRE_ERROR_GENERIC);
         hypre_printf("MY_ID = %d Error: positive volume box (%d, %d, %d) x "
                      "(%d, %d, %d)  not in any assumed regions! (this should never"
                      " happen)\n",
                      myid,
                      hypre_BoxIMinX(box),
                      hypre_BoxIMinY(box),
                      hypre_BoxIMinZ(box),
                      hypre_BoxIMaxX(box),
                      hypre_BoxIMaxY(box),
                      hypre_BoxIMaxZ(box));
      }
   }
#endif

   /* For each region, who is assumed to own this box?  Add the proc number to
      proc array. */
   for (r = 0; r < in_regions; r++)
   {
      /* Initialization for this particular region */
      this_region = which_regions[r];
      region = hypre_BoxArrayBox(region_array, this_region);
      /* First processor in the range */
      proc_start = proc_partitions[this_region];
      /* How many processors in that region? */
      proc_count = proc_partitions[this_region + 1] - proc_start;
      /* Size of the regions */
      hypre_BoxGetSize(region, rsize);
      /* Get the divisons in each dimension */
      hypre_CopyIndex(hypre_StructAssumedPartDivision(assumed_part, this_region),
                      div);

      /* Intersect box with region */
      hypre_IntersectBoxes(box, region, result_box);

      /* Compute part_box (the intersected assumed partitions) from result_box.
         Start part index number from 1 for convenience in BoxLoop below. */
      for (d = 0; d < ndim; d++)
      {
         width = hypre_IndexD(rsize, d) / hypre_IndexD(div, d);
         extra = hypre_IndexD(rsize, d) % hypre_IndexD(div, d);

         /* imin component, shifted by region imin */
         i = hypre_BoxIMinD(result_box, d) - hypre_BoxIMinD(region, d);
         p = i / (width + 1);
         if (p < extra)
         {
            hypre_BoxIMinD(part_box, d) = p + 1;
         }
         else
         {
            q = (i - extra * (width + 1)) / width;
            hypre_BoxIMinD(part_box, d) = extra + q + 1;
         }

         /* imax component, shifted by region imin  */
         i = hypre_BoxIMaxD(result_box, d) - hypre_BoxIMinD(region, d);
         p = i / (width + 1);
         if (p < extra)
         {
            hypre_BoxIMaxD(part_box, d) = p + 1;
         }
         else
         {
            q = (i - extra * (width + 1)) / width;
            hypre_BoxIMaxD(part_box, d) = extra + q + 1;
         }
      }

      /* Number of partitions in this region? */
      num_partitions = hypre_IndexProd(div, ndim);
      /* How many procs have 2 partitions instead of one*/
      extra =  num_partitions % proc_count;

      /* Compute part_num for each index in part_box and get proc_ids */
      start = hypre_BoxIMin(part_box);
      hypre_SetIndex(stride, 1);
      hypre_BoxGetSize(part_box, loop_size);
      hypre_BoxSetExtents(part_dbox, stride, div);
      hypre_SerialBoxLoop1Begin(ndim, loop_size, part_dbox, start, stride, part_num);
      {
         /*convert the partition number to a processor number*/
         if (part_num < (2 * extra))
         {
            adj_proc_id = part_num / 2 ;
         }
         else
         {
            adj_proc_id =  extra + (part_num - 2 * extra);
         }

         if (num_proc_ids == size_proc_ids)
         {
            size_proc_ids += ncorners;
            proc_ids = hypre_TReAlloc(proc_ids,  HYPRE_Int,  size_proc_ids, HYPRE_MEMORY_HOST);
         }

         proc_ids[num_proc_ids] = adj_proc_id + proc_start;
         num_proc_ids++;
      }
      hypre_SerialBoxLoop1End(part_num);

   } /*end of for each region loop*/

   if (in_regions)
   {
      /* Determine unique proc_ids (could be duplicates due to a processor
         owning more than one partiton in a region).  Sort the array. */
      hypre_qsort0(proc_ids, 0, num_proc_ids - 1);

      /* Make sure we have enough space from proc_array */
      if (*size_alloc_proc_array < num_proc_ids)
      {
         proc_array = hypre_TReAlloc(proc_array,  HYPRE_Int,  num_proc_ids, HYPRE_MEMORY_HOST);
         *size_alloc_proc_array = num_proc_ids;
      }

      /* Put unique values in proc_array */
      proc_array[0] = proc_ids[0]; /* There will be at least one processor id */
      proc_array_count = 1;
      for (i = 1; i < num_proc_ids; i++)
      {
         if  (proc_ids[i] != proc_array[proc_array_count - 1])
         {
            proc_array[proc_array_count] = proc_ids[i];
            proc_array_count++;
         }
      }
   }
   else /* No processors for this box */
   {
      proc_array_count = 0;
   }

   /* Return variables */
   *p_proc_array = proc_array;
   *num_proc_array = proc_array_count;

   /* Clean up*/
   hypre_BoxDestroy(result_box);
   hypre_BoxDestroy(part_box);
   hypre_BoxDestroy(part_dbox);
   hypre_TFree(which_regions, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_ids, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

#if 0
/******************************************************************************
 * UNFINISHED
 *
 * Create a new assumed partition by coarsening another assumed partition.
 *
 * Unfinished because of a problem: Can't figure out what the new id is since
 * the zero boxes drop out, and we don't have all of the boxes from a particular
 * processor in the AP.  This may not be a problem any longer (see [issue708]).
 *****************************************************************************/

HYPRE_Int
hypre_StructCoarsenAP(hypre_StructAssumedPart  *ap,
                      hypre_Index               index,
                      hypre_Index               stride,
                      hypre_StructAssumedPart **new_ap_ptr )
{
   HYPRE_Int num_regions;

   hypre_BoxArray *coarse_boxes;
   hypre_BoxArray *fine_boxes;
   hypre_BoxArray *regions_array;
   hypre_Box      *box, *new_box;

   hypre_StructAssumedPartition *new_ap;

   /* Create new ap and copy global description information */
   new_ap = hypre_TAlloc(hypre_StructAssumedPart,  1, HYPRE_MEMORY_HOST);

   num_regions = hypre_StructAssumedPartNumRegions(ap);
   regions_array = hypre_BoxArrayCreate(num_regions, ndim);

   hypre_StructAssumedPartRegions(new_ap) = regions_array;
   hypre_StructAssumedPartNumRegions(new_ap) = num_regions;
   hypre_StructAssumedPartProcPartitions(new_ap) =
      hypre_CTAlloc(HYPRE_Int,  num_regions + 1, HYPRE_MEMORY_HOST);
   hypre_StructAssumedPartDivisions(new_ap) =
      hypre_CTAlloc(HYPRE_Int,  num_regions, HYPRE_MEMORY_HOST);

   hypre_StructAssumedPartProcPartitions(new_ap)[0] = 0;

   for (i = 0; i < num_regions; i++)
   {
      box =  hypre_BoxArrayBox(hypre_StructAssumedPartRegions(ap), i);

      hypre_CopyBox(box, hypre_BoxArrayBox(regions_array, i));

      hypre_StructAssumedPartDivision(new_ap, i) =
         hypre_StructAssumedPartDivision(new_ap, i);

      hypre_StructAssumedPartProcPartition(new_ap, i + 1) =
         hypre_StructAssumedPartProcPartition(ap, i + 1);
   }

   /* Copy my partition (at most 2 boxes)*/
   hypre_StructAssumedPartMyPartition(new_ap) = hypre_BoxArrayCreate(2, ndim);
   for (i = 0; i < 2; i++)
   {
      box     = hypre_BoxArrayBox(hypre_StructAssumedPartMyPartition(ap), i);
      new_box = hypre_BoxArrayBox(hypre_StructAssumedPartMyPartition(new_ap), i);
      hypre_CopyBox(box, new_box);
   }

   /* Create space for the boxes, ids and boxnums */
   size = hypre_StructAssumedPartMyPartitionIdsSize(ap);

   hypre_StructAssumedPartMyPartitionProcIds(new_ap) =
      hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);
   hypre_StructAssumedPartMyPartitionBoxnums(new_ap) =
      hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);

   hypre_StructAssumedPartMyPartitionBoxes(new_ap)
      = hypre_BoxArrayCreate(size, ndim);

   hypre_StructAssumedPartMyPartitionIdsAlloc(new_ap) = size;
   hypre_StructAssumedPartMyPartitionIdsSize(new_ap) = size;

   /* Coarsen and copy the boxes.  Need to prune size 0 boxes. */
   coarse_boxes = hypre_StructAssumedPartMyPartitionBoxes(new_ap);
   fine_boxes =  hypre_StructAssumedPartMyPartitionBoxes(ap);

   new_box = hypre_BoxCreate(ndim);

   hypre_ForBoxI(i, fine_boxes)
   {
      box =  hypre_BoxArrayBox(fine_boxes, i);
      hypre_CopyBox(box, new_box);
      hypre_StructCoarsenBox(new_box, index, stride);
   }

   /* Unfinished because of a problem: Can't figure out what the new id is since
      the zero boxes drop out, and we don't have all of the boxes from a
      particular processor in the AP */

   /* hypre_StructAssumedPartMyPartitionNumDistinctProcs(new_ap) */

   *new_ap_ptr = new_ap;

   return hypre_error_flag;
}
#endif
