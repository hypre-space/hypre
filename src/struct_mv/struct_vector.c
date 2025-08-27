/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_StructVector class.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorMapDataIndex( hypre_StructVector *vector,
                                hypre_Index         dindex )
{
   hypre_MapToCoarseIndex(dindex, NULL,
                          hypre_StructVectorStride(vector), hypre_StructVectorNDim(vector));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorUnMapDataIndex( hypre_StructVector *vector,
                                  hypre_Index         dindex )
{
   hypre_MapToFineIndex(dindex, NULL,
                        hypre_StructVectorStride(vector), hypre_StructVectorNDim(vector));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorMapDataBox( hypre_StructVector *vector,
                              hypre_Box          *dbox )
{
   hypre_CoarsenBox(dbox, NULL, hypre_StructVectorStride(vector));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorUnMapDataBox( hypre_StructVector *vector,
                                hypre_Box          *dbox )
{
   hypre_RefineBox(dbox, NULL, hypre_StructVectorStride(vector));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorMapDataStride( hypre_StructVector *vector,
                                 hypre_Index         dstride )
{
   hypre_StructVectorMapDataIndex(vector, dstride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorUnMapDataStride( hypre_StructVector *vector,
                                   hypre_Index         dstride )
{
   hypre_StructVectorUnMapDataIndex(vector, dstride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorMapCommInfo( hypre_StructVector *vector,
                               hypre_CommInfo     *comm_info )
{
   HYPRE_Int  ndim = hypre_StructVectorNDim(vector);

   /* Map the comm_info boxes only if the vector does not have a unit stride */
   if (!hypre_IndexEqual(hypre_StructVectorStride(vector), 1, ndim))
   {
      hypre_CommInfoCoarsen(comm_info, NULL, hypre_StructVectorStride(vector));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorCreate( MPI_Comm          comm,
                          hypre_StructGrid *grid )
{
   HYPRE_Int            ndim = hypre_StructGridNDim(grid);
   hypre_StructVector  *vector;
   HYPRE_Int            i;

   vector = hypre_CTAlloc(hypre_StructVector, 1, HYPRE_MEMORY_HOST);

   hypre_StructVectorComm(vector)           = comm;
   hypre_StructGridRef(grid, &hypre_StructVectorGrid(vector));
   hypre_StructVectorSetStride(vector, NULL);                       /* Set default stride */
   hypre_StructVectorSetMemoryMode(vector, 0);
   hypre_StructVectorDataAlloced(vector)    = 0;
   hypre_StructVectorBGhostNotClear(vector) = 0;
   hypre_StructVectorRefCount(vector)       = 1;

   /* set defaults */
   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructVectorNumGhost(vector)[i] = hypre_StructGridNumGhost(grid)[i];
   }

   hypre_StructVectorMemoryLocation(vector) = hypre_HandleMemoryLocation(hypre_handle());

#if defined(HYPRE_MIXED_PRECISION)
   hypre_StructVectorPrecision(vector) = HYPRE_OBJECT_PRECISION;
#endif

   return vector;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorRef( hypre_StructVector *vector )
{
   hypre_StructVectorRefCount(vector) ++;

   return vector;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorDestroy( hypre_StructVector *vector )
{
   if (vector)
   {
      hypre_StructVectorRefCount(vector) --;
      if (hypre_StructVectorRefCount(vector) == 0)
      {
         if (hypre_StructVectorDataAlloced(vector) == 1)
         {
            hypre_TFree(hypre_StructVectorData(vector), hypre_StructVectorMemoryLocation(vector));
         }

         hypre_TFree(hypre_StructVectorDataIndices(vector), HYPRE_MEMORY_HOST);
         hypre_BoxArrayDestroy(hypre_StructVectorDataSpace(vector));
         hypre_TFree(hypre_StructVectorBoxnums(vector), HYPRE_MEMORY_HOST);
         hypre_StructGridDestroy(hypre_StructVectorGrid(vector));
         hypre_StructVectorForget(vector);
         hypre_TFree(vector, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set vector stride, nboxes, and boxnums.
 * If stride == NULL, set default values.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetStride( hypre_StructVector *vector,
                             hypre_IndexRef      stride )
{
   HYPRE_Int  nboxes, *boxnums, ndim = hypre_StructVectorNDim(vector);

   if (stride != NULL)
   {
      hypre_CopyToIndex(stride, ndim, hypre_StructVectorStride(vector));
   }
   else
   {
      /* set default stride of 1 */
      hypre_SetIndex(hypre_StructVectorStride(vector), 1);
   }
   hypre_StructGridComputeBoxnums(hypre_StructVectorGrid(vector), 0, NULL,
                                  hypre_StructVectorStride(vector), &nboxes, &boxnums);
   hypre_TFree(hypre_StructVectorBoxnums(vector), HYPRE_MEMORY_HOST);
   hypre_StructVectorNBoxes(vector)  = nboxes;
   hypre_StructVectorBoxnums(vector) = boxnums;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set memory management approach, where valid memory_mode values are:
 *   0 - set data space exactly, allow restore (default)
 *   1 - set data space exactly, always forget
 *   2 - only grow the data space, always forget
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetMemoryMode( hypre_StructVector *vector,
                                 HYPRE_Int           memory_mode )
{
   hypre_StructVectorMemoryMode(vector) = memory_mode;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine changes the base grid.  Before the vector can actually be used,
 * hypre_StructVectorResize() must be called with an appropriate data space.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorRebase( hypre_StructVector *vector,
                          hypre_StructGrid   *grid,
                          hypre_Index         stride )
{
   hypre_StructGrid *old_grid   = hypre_StructVectorGrid(vector);
   hypre_IndexRef    old_stride = hypre_StructVectorStride(vector);

   if (hypre_StructVectorSaveGrid(vector) != NULL)
   {
      /* Call Restore or Forget first */
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Rebase has already been called");
      return hypre_error_flag;
   }

   if (hypre_StructVectorMemoryMode(vector) == 2)
   {
      /* If stride is not greater than the current stride, keep the current base grid */
      if ( !hypre_IndexesGreaterEqual(stride, old_stride, hypre_StructVectorNDim(vector)) )
      {
         return hypre_error_flag;
      }
   }
   hypre_StructVectorSaveGrid(vector) = old_grid;
   hypre_CopyIndex(old_stride, hypre_StructVectorSaveStride(vector));
   hypre_StructGridRef(grid, &hypre_StructVectorGrid(vector));
   hypre_StructVectorSetStride(vector, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This computes a vector data space from a num_ghost array.  The num_ghost
 * array applies to the coarse index space indicated by stride.  If the stride
 * argument is NULL, a unit stride is used.  If the num_ghost argument is NULL,
 * the vector num_ghost is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorComputeDataSpace( hypre_StructVector *vector,
                                    hypre_IndexRef      stride,
                                    HYPRE_Int          *num_ghost,
                                    hypre_BoxArray    **data_space_ptr )
{
   HYPRE_Int          ndim      = hypre_StructVectorNDim(vector);
   hypre_StructGrid  *grid      = hypre_StructVectorGrid(vector);
   hypre_BoxArray    *data_space, *cdata_space;
   hypre_Box         *data_box;
   HYPRE_Int          i, d;

   if (num_ghost == NULL)
   {
      /* Use the vector num_ghost */
      num_ghost = hypre_StructVectorNumGhost(vector);
   }

   /* Add ghost layers and map the data space */
   data_space = hypre_BoxArrayClone(hypre_StructGridBoxes(grid));
   hypre_ForBoxI(i, data_space)
   {
      data_box = hypre_BoxArrayBox(data_space, i);
      if (stride != NULL)
      {
         hypre_CoarsenBox(data_box, NULL, stride);
      }
      for (d = 0; d < ndim; d++)
      {
         hypre_BoxIMinD(data_box, d) -= num_ghost[2 * d];
         hypre_BoxIMaxD(data_box, d) += num_ghost[2 * d + 1];
      }
      if (stride != NULL)
      {
         hypre_RefineBox(data_box, NULL, stride);
      }
      hypre_StructVectorMapDataBox(vector, data_box);
   }

   /* Assume that the index space for the vector base grid (grid) is at least as
    * fine as that of the saved base grid (call it sgrid).  Rebase ensures this.
    * In other words, since the vector's grid is given by both (grid, stride)
    * and (sgrid, sstride), we assume that stride >= sstride.  As a consequence,
    * we can assume that all box ids in cdata_space (the current data space) are
    * also in data_space, and that cdata_space has at most as many boxes as
    * data_space. */
   cdata_space = hypre_StructVectorDataSpace(vector);
   if ((hypre_StructVectorMemoryMode(vector) == 2) && (cdata_space != NULL))
   {
      HYPRE_Int  *cids = hypre_BoxArrayIDs(cdata_space);
      HYPRE_Int  *ids  = hypre_BoxArrayIDs(data_space);
      HYPRE_Int   ci;

      i = 0;
      hypre_ForBoxI(ci, cdata_space)
      {
         while (ids[i] != cids[ci])
         {
            i++;
         }
         hypre_BoxGrowByBox(hypre_BoxArrayBox(data_space, i),
                            hypre_BoxArrayBox(cdata_space, ci));
      }
   }

   *data_space_ptr = data_space;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Check to see if a resize is needed
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorNeedResize( hypre_StructVector *vector,
                              hypre_BoxArray     *data_space )
{
   hypre_BoxArray  *cdata_space = hypre_StructVectorDataSpace(vector);
   hypre_IndexRef   stride      = hypre_StructVectorStride(vector);
   hypre_IndexRef   sstride     = hypre_StructVectorSaveStride(vector);
   HYPRE_Int        ndim        = hypre_StructVectorNDim(vector);
   HYPRE_Int        need_resize = 0;

   if (cdata_space == NULL)
   {
      /* resize if no current data space (cdata_space) */
      need_resize = 1;
   }
   else if ( !hypre_IndexesEqual(stride, sstride, ndim) )
   {
      /* resize if stride and saved stride are different */
      need_resize = 1;
   }
   else if ( !hypre_BoxArrayInBoxArray(data_space, cdata_space) )
   {
      /* resize if data_space is not contained in cdata_space */
      need_resize = 1;
   }

   return need_resize;
}

/*--------------------------------------------------------------------------
 * This routine takes new data space information and recomputes entries in the
 * vector that depend on it (e.g., data_indices and data_size).  The routine
 * will also re-allocate the vector data if there was data to begin with.
 *
 * The boxes in the data_space argument may be larger (but not smaller) than
 * those computed by the routine VectorComputeDataSpace().
 *
 * This routine serves to both "size" and a "re-size" a vector, so it can also
 * be used to set the initial data space information.
 *
 * RDF NOTE: I'm not sure if this works correctly when DataAlloced = 0 because
 * of the check for it in the second 'if' below.  Look at this later.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorResize( hypre_StructVector *vector,
                          hypre_BoxArray     *data_space )
{
   HYPRE_Complex        *old_data         = hypre_StructVectorData(vector);
   hypre_BoxArray       *old_data_space   = hypre_StructVectorDataSpace(vector);
   HYPRE_Int             old_data_size    = hypre_StructVectorDataSize(vector);
   HYPRE_Int            *old_data_indices = hypre_StructVectorDataIndices(vector);
   HYPRE_Int             ndim             = hypre_StructVectorNDim(vector);
   HYPRE_MemoryLocation  memory_location  = hypre_StructVectorMemoryLocation(vector);

   HYPRE_Complex        *data = NULL;
   HYPRE_Int             data_size;
   HYPRE_Int            *data_indices;

   hypre_Box            *data_box;
   HYPRE_Int             i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (hypre_StructVectorSaveDataSpace(vector) != NULL)
   {
      /* Call Restore or Forget first */
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Resize has already been called");

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   /* Set up data_indices and data_size */
   data_indices = hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(data_space), HYPRE_MEMORY_HOST);
   data_size = 0;
   hypre_ForBoxI(i, data_space)
   {
      data_box = hypre_BoxArrayBox(data_space, i);

      data_indices[i] = data_size;
      data_size += hypre_BoxVolume(data_box);
   }

   /* Copy old_data to data and save old data; only do this if both the old data
    * space and old data have been initialized */
   if ( (old_data_space != NULL) && (hypre_StructVectorDataAlloced(vector)) )
   {
      /* If Rebase() has not been called, mimic it by saving a copy of grid/stride */
      if (hypre_StructVectorSaveGrid(vector) == NULL)
      {
         hypre_StructGridRef(hypre_StructVectorGrid(vector), &hypre_StructVectorSaveGrid(vector));
         hypre_CopyIndex(hypre_StructVectorStride(vector), hypre_StructVectorSaveStride(vector));
      }

      /* This will return NULL if data_size = 0  */
      data = hypre_CTAlloc(HYPRE_Complex, data_size, memory_location);

      /* Copy old_data to data */
      if ((old_data != NULL) && (data != NULL))
      {
         /* At this point we have either called or mimiced Rebase(), so the
          * saved grid corresponds to the old data */
         hypre_StructDataCopy(old_data, old_data_space, data, data_space, ndim, 1);
      }

      /* Free up some things */
      if (hypre_StructVectorDataAlloced(vector) == 1)
      {
         hypre_TFree(old_data, memory_location);
      }
      hypre_TFree(old_data_indices, HYPRE_MEMORY_HOST);

      /* Save old data */
      hypre_StructVectorSaveData(vector)      = old_data;
      hypre_StructVectorSaveDataSpace(vector) = old_data_space;
      hypre_StructVectorSaveDataSize(vector)  = old_data_size;
   }
   else if (old_data_space != NULL)
   {
      hypre_TFree(old_data_indices, HYPRE_MEMORY_HOST);
      hypre_BoxArrayDestroy(old_data_space);
   }

   hypre_StructVectorData(vector)        = data;
   hypre_StructVectorDataSpace(vector)   = data_space;
   hypre_StructVectorDataSize(vector)    = data_size;
   hypre_StructVectorDataIndices(vector) = data_indices;

   /* For (MemoryMode > 0), don't restore */
   if (hypre_StructVectorMemoryMode(vector) > 0)
   {
      hypre_StructVectorForget(vector);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine restores the old data and data space.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorRestore( hypre_StructVector *vector )
{
   hypre_StructGrid      *old_grid        = hypre_StructVectorGrid(vector);
   HYPRE_Complex         *old_data        = hypre_StructVectorData(vector);
   hypre_BoxArray        *old_data_space  = hypre_StructVectorDataSpace(vector);
   hypre_StructGrid      *grid            = hypre_StructVectorSaveGrid(vector);
   HYPRE_Complex         *data            = hypre_StructVectorSaveData(vector);
   hypre_IndexRef         stride          = hypre_StructVectorSaveStride(vector);
   hypre_BoxArray        *data_space      = hypre_StructVectorSaveDataSpace(vector);
   HYPRE_Int              data_size       = hypre_StructVectorSaveDataSize(vector);
   HYPRE_MemoryLocation   memory_location = hypre_StructVectorMemoryLocation(vector);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (data_space != NULL)
   {
      HYPRE_Int   ndim = hypre_StructVectorNDim(vector);

      /* Move old_data to data */
      if (hypre_StructVectorDataAlloced(vector) == 1)
      {
         data = hypre_CTAlloc(HYPRE_Complex, data_size, memory_location);
      }
      hypre_StructDataCopy(old_data, old_data_space, data, data_space, ndim, 1);
      hypre_TFree(old_data, memory_location);

      /* Reset certain fields to enable the Resize call below */
      hypre_StructVectorSaveGrid(vector)      = NULL;
      hypre_StructVectorSaveData(vector)      = NULL;
      hypre_StructVectorSaveDataSpace(vector) = NULL;
      hypre_StructVectorSaveDataSize(vector)  = 0;

      /* Set the grid and boxnums */
      hypre_StructGridDestroy(old_grid);
      hypre_StructVectorGrid(vector) = grid;
      hypre_StructVectorSetStride(vector, stride);

      /* Set the data space and recompute data_indices, etc. */
      {
         HYPRE_Int  data_alloced = hypre_StructVectorDataAlloced(vector);

         hypre_StructVectorDataAlloced(vector) = 0;
         hypre_StructVectorResize(vector, data_space);
         hypre_StructVectorDataAlloced(vector) = data_alloced;
         hypre_StructVectorForget(vector);
      }

      /* Set the data pointer */
      hypre_StructVectorData(vector) = data;
   }
   else if ((old_grid != NULL) && (grid != NULL))
   {
      /* Only a Rebase was called */
      hypre_StructVectorSaveGrid(vector) = NULL;
      hypre_StructGridDestroy(old_grid);
      hypre_StructVectorGrid(vector) = grid;
      hypre_StructVectorSetStride(vector, stride);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine will clear data needed to do a Restore
 *
 * Note: If InitializeData() was used to set the original data pointer, the
 * pointer will be lost after a Resize()-and-Forget() and more memory will be
 * used than is needed.  Consider removing InitializeData() usage altogether.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorForget( hypre_StructVector *vector )
{
   hypre_StructGrid *save_grid       = hypre_StructVectorSaveGrid(vector);
   hypre_BoxArray   *save_data_space = hypre_StructVectorSaveDataSpace(vector);

   if (save_data_space != NULL)
   {
      /* Only forget a Rebase if the companion Resize was also called */
      if (save_grid != NULL)
      {
         hypre_StructGridDestroy(save_grid);
         hypre_StructVectorSaveGrid(vector) = NULL;
      }

      hypre_BoxArrayDestroy(save_data_space);
      hypre_StructVectorSaveData(vector)      = NULL;
      hypre_StructVectorSaveDataSpace(vector) = NULL;
      hypre_StructVectorSaveDataSize(vector)  = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorInitializeShell( hypre_StructVector *vector )
{
   hypre_StructGrid     *grid       = hypre_StructVectorGrid(vector);
   hypre_BoxArray       *data_space;

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   hypre_StructVectorGlobalSize(vector) = hypre_StructGridGlobalSize(grid);

   /*-----------------------------------------------------------------------
    * Set up information related to the data space and data storage
    *-----------------------------------------------------------------------*/

   if (hypre_StructVectorDataSpace(vector) == NULL)
   {
      hypre_StructVectorComputeDataSpace(vector, NULL, NULL, &data_space);
      hypre_StructVectorResize(vector, data_space);
      hypre_StructVectorForget(vector);
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorInitializeData( hypre_StructVector *vector,
                                  HYPRE_Complex      *data   )
{
   hypre_StructVectorData(vector) = data;
   hypre_StructVectorDataAlloced(vector) = 2;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorInitialize( hypre_StructVector *vector,
                              HYPRE_Int           zero_init )
{
   HYPRE_Complex *data;

   hypre_StructVectorInitializeShell(vector);

   if (zero_init)
   {
      data = hypre_CTAlloc(HYPRE_Complex,
                           hypre_StructVectorDataSize(vector),
                           hypre_StructVectorMemoryLocation(vector));
   }
   else
   {
      /* TODO (VPM): TAlloc leads to regressions on TEST_struct/emptyproc */
      data = hypre_CTAlloc(HYPRE_Complex,
                           hypre_StructVectorDataSize(vector),
                           hypre_StructVectorMemoryLocation(vector));
   }

   hypre_StructVectorInitializeData(vector, data);
   hypre_StructVectorDataAlloced(vector) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * (outside > 0): set values possibly outside of the grid extents
 * (outside = 0): set values only inside the grid extents
 *
 * NOTE: Getting and setting values outside of the grid extents requires care,
 * as these values may be stored in multiple ghost zone locations.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetValues( hypre_StructVector *vector,
                             hypre_Index         grid_index,
                             HYPRE_Complex      *values,
                             HYPRE_Int           action,
                             HYPRE_Int           boxnum,
                             HYPRE_Int           outside    )
{
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   HYPRE_Complex       *vecp;
   HYPRE_Int            i, istart, istop;
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(vector);
#endif

   if (outside > 0)
   {
      grid_boxes = hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   }

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);

      if (hypre_IndexInBox(grid_index, grid_box))
      {
         vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);

#if defined(HYPRE_USING_GPU)
         if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
         {
            if (action > 0)
            {
#define DEVICE_VAR is_device_ptr(vecp,values)
               hypre_LoopBegin(1, k)
               {
                  *vecp += *values;
               }
               hypre_LoopEnd()
#undef DEVICE_VAR
            }
            else if (action > -1)
            {
               hypre_TMemcpy(vecp, values, HYPRE_Complex, 1, memory_location, memory_location);
            }
            else /* action < 0 */
            {
               hypre_TMemcpy(values, vecp, HYPRE_Complex, 1, memory_location, memory_location);
            }
         }
         else
#endif
         {
            if (action > 0)
            {
               *vecp += *values;
            }
            else if (action > -1)
            {
               *vecp = *values;
            }
            else /* action < 0 */
            {
               *values = *vecp;
            }
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * (outside > 0): set values possibly outside of the grid extents
 * (outside = 0): set values only inside the grid extents
 *
 * NOTE: Getting and setting values outside of the grid extents requires care,
 * as these values may be stored in multiple ghost zone locations.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetBoxValues( hypre_StructVector *vector,
                                hypre_Box          *set_box,
                                hypre_Box          *value_box,
                                HYPRE_Complex      *values,
                                HYPRE_Int           action,
                                HYPRE_Int           boxnum,
                                HYPRE_Int           outside )
{
   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;
   hypre_Box          *int_box;

   hypre_BoxArray     *data_space;
   hypre_Box          *data_box;
   hypre_IndexRef      data_start;
   hypre_Index         data_stride;
   HYPRE_Complex      *datap;

   hypre_Box          *dval_box;
   hypre_Index         dval_start;
   hypre_Index         dval_stride;

   hypre_Index         loop_size;

   HYPRE_Int           i, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   }
   data_space = hypre_StructVectorDataSpace(vector);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(data_stride, 1);

   int_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   dval_box = hypre_BoxClone(value_box);
   hypre_SetIndex(dval_stride, 1);

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_IntersectBoxes(set_box, grid_box, int_box);

      /* if there was an intersection */
      if (hypre_BoxVolume(int_box))
      {
         data_start = hypre_BoxIMin(int_box);
         hypre_CopyIndex(data_start, dval_start);

         datap = hypre_StructVectorBoxData(vector, i);

         hypre_BoxGetSize(int_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap, values)
         if (action > 0)
         {
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(vector), loop_size,
                                data_box, data_start, data_stride, datai,
                                dval_box, dval_start, dval_stride, dvali);
            {
               datap[datai] += values[dvali];
            }
            hypre_BoxLoop2End(datai, dvali);
         }
         else if (action > -1)
         {
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(vector), loop_size,
                                data_box, data_start, data_stride, datai,
                                dval_box, dval_start, dval_stride, dvali);
            {
               datap[datai] = values[dvali];
            }
            hypre_BoxLoop2End(datai, dvali);
         }
         else /* action < 0 */
         {
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(vector), loop_size,
                                data_box, data_start, data_stride, datai,
                                dval_box, dval_start, dval_stride, dvali);
            {
               values[dvali] = datap[datai];
            }
            hypre_BoxLoop2End(datai, dvali);
         }
#undef DEVICE_VAR
      }
   }

   hypre_BoxDestroy(int_box);
   hypre_BoxDestroy(dval_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorClearValues( hypre_StructVector *vector,
                               hypre_Index         grid_index,
                               HYPRE_Int           boxnum,
                               HYPRE_Int           outside    )
{
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   HYPRE_Complex       *vecp;
   HYPRE_Int            i, istart, istop;
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(vector);
#endif

   if (outside > 0)
   {
      grid_boxes = hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   }

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);

      if (hypre_IndexInBox(grid_index, grid_box))
      {
         vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);

#if defined(HYPRE_USING_GPU)
         if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
         {
#define DEVICE_VAR is_device_ptr(vecp)
            hypre_LoopBegin(1, k)
            {
               *vecp = 0.0;
            }
            hypre_LoopEnd()
#undef DEVICE_VAR
         }
         else
#endif
         {
            *vecp = 0.0;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorClearBoxValues( hypre_StructVector *vector,
                                  hypre_Box          *clear_box,
                                  HYPRE_Int           boxnum,
                                  HYPRE_Int           outside )
{
   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;
   hypre_Box          *int_box;

   hypre_BoxArray     *data_space;
   hypre_Box          *data_box;
   hypre_IndexRef      data_start;
   hypre_Index         data_stride;
   HYPRE_Complex      *datap;

   hypre_Index         loop_size;

   HYPRE_Int           i, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   }
   data_space = hypre_StructVectorDataSpace(vector);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(data_stride, 1);

   int_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_IntersectBoxes(clear_box, grid_box, int_box);

      /* if there was an intersection */
      if (hypre_BoxVolume(int_box))
      {
         data_start = hypre_BoxIMin(int_box);

         datap = hypre_StructVectorBoxData(vector, i);

         hypre_BoxGetSize(int_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap)
         hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                             data_box, data_start, data_stride, datai);
         {
            datap[datai] = 0.0;
         }
         hypre_BoxLoop1End(datai);
#undef DEVICE_VAR
      }
   }

   hypre_BoxDestroy(int_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorClearAllValues( hypre_StructVector *vector )
{
   HYPRE_Complex *data      = hypre_StructVectorData(vector);
   HYPRE_Int      data_size = hypre_StructVectorDataSize(vector);
   HYPRE_Int      i;

#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(vector);
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      hypre_Memset((void*) data, 0.0,
                   (size_t) (data_size) * sizeof(HYPRE_Complex),
                   memory_location);
   }
   else
#endif
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < data_size; i++)
      {
         data[i] = 0.0;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetNumGhost( hypre_StructVector *vector,
                               HYPRE_Int          *num_ghost )
{
   HYPRE_Int  d, ndim = hypre_StructVectorNDim(vector);

   for (d = 0; d < ndim; d++)
   {
      hypre_StructVectorNumGhost(vector)[2 * d]     = num_ghost[2 * d];
      hypre_StructVectorNumGhost(vector)[2 * d + 1] = num_ghost[2 * d + 1];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetDataSize(hypre_StructVector *vector,
                              HYPRE_Int          *data_size,
                              HYPRE_Int          *data_host_size)
{
   HYPRE_UNUSED_VAR(data_host_size);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_StructGrid     *grid = hypre_StructVectorGrid(vector);
   if (hypre_StructGridDataLocation(grid) != HYPRE_MEMORY_HOST)
   {
      *data_size += hypre_StructVectorDataSize(vector);
   }
   else
   {
      *data_host_size += hypre_StructVectorDataSize(vector);
   }
#else
   *data_size += hypre_StructVectorDataSize(vector);
#endif
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorAssemble( hypre_StructVector *vector )
{
   HYPRE_UNUSED_VAR(vector);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetConstantValues( hypre_StructVector *vector,
                                     HYPRE_Complex       value )
{
   HYPRE_Int           ndim = hypre_StructVectorNDim(vector);

   hypre_Box          *dbox;
   HYPRE_Complex      *vp;

   HYPRE_Int           nboxes;
   hypre_Box          *loop_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         ustride;
   HYPRE_Int           i;

   nboxes = hypre_StructVectorNBoxes(vector);

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   for (i = 0; i < nboxes; i++)
   {
      hypre_StructVectorGridBoxCopy(vector, i, loop_box);
      start = hypre_BoxIMin(loop_box);

      dbox = hypre_StructVectorGridDataBox(vector, i);
      vp = hypre_StructVectorGridData(vector, i);

      hypre_BoxGetSize(loop_box, loop_size);

#define DEVICE_VAR is_device_ptr(vp)
      hypre_BoxLoop1Begin(ndim, loop_size, dbox, start, ustride, vi);
      {
         vp[vi] = value;
      }
      hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
   }

   hypre_BoxDestroy(loop_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetRandomValues
 *
 * builds a StructVector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetRandomValues( hypre_StructVector *vector,
                                   HYPRE_Int           seed )
{
   HYPRE_Int           ndim = hypre_StructVectorNDim(vector);

   hypre_Box          *dbox;
   HYPRE_Complex      *vp;

   HYPRE_Int           nboxes;
   hypre_Box          *loop_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         ustride;
   HYPRE_Int           i;

   hypre_SeedRand(seed);

   nboxes = hypre_StructVectorNBoxes(vector);

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   for (i = 0; i < nboxes; i++)
   {
      hypre_StructVectorGridBoxCopy(vector, i, loop_box);
      start = hypre_BoxIMin(loop_box);

      dbox = hypre_StructVectorGridDataBox(vector, i);
      vp = hypre_StructVectorGridData(vector, i);

      hypre_BoxGetSize(loop_box, loop_size);

      /* TODO: generate on host and copy to device. FIX? */
#if defined(HYPRE_USING_GPU)
      HYPRE_Int loop_n = 1, ii;
      for (ii = 0; ii < hypre_StructVectorNDim(vector); ii++)
      {
         loop_n *= loop_size[ii];
      }

      HYPRE_Real *rand_host   = hypre_TAlloc(HYPRE_Real, loop_n, HYPRE_MEMORY_HOST);
      HYPRE_Real *rand_device = hypre_TAlloc(HYPRE_Real, loop_n, HYPRE_MEMORY_DEVICE);

      ii = 0;
      hypre_SerialBoxLoop0Begin(hypre_StructVectorNDim(vector), loop_size)
      {
         rand_host[ii++] = 2.0 * hypre_Rand() - 1.0;
      }
      hypre_SerialBoxLoop0End()
      hypre_TMemcpy(rand_device, rand_host, HYPRE_Real, loop_n,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
#endif

#define DEVICE_VAR is_device_ptr(vp, rand_device)
      hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                          dbox, start, ustride, vi);
      {
#if defined(HYPRE_USING_GPU)
         vp[vi] = rand_device[idx];
#else
         vp[vi] = 2.0 * hypre_Rand() - 1.0;
#endif
      }
      hypre_BoxLoop1End(vi);
#undef DEVICE_VAR

#if defined(HYPRE_USING_GPU)
      hypre_TFree(rand_device, HYPRE_MEMORY_DEVICE);
      hypre_TFree(rand_host, HYPRE_MEMORY_HOST);
#endif
   }

   hypre_BoxDestroy(loop_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * RDF TODO: Extend to allow non-unitary stride
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorClearGhostValues( hypre_StructVector *vector )
{
   HYPRE_Int           ndim = hypre_StructVectorNDim(vector);
   hypre_Box          *v_data_box;

   HYPRE_Complex      *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_BoxArray     *diff_boxes;
   hypre_Box          *diff_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         ustride;

   HYPRE_Int           i, j;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(ustride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   diff_boxes = hypre_BoxArrayCreate(0, ndim);
   hypre_ForBoxI(i, boxes)
   {
      box        = hypre_BoxArrayBox(boxes, i);
      v_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
      hypre_BoxArraySetSize(diff_boxes, 0);
      hypre_SubtractBoxes(v_data_box, box, diff_boxes);

      vp = hypre_StructVectorBoxData(vector, i);
      hypre_ForBoxI(j, diff_boxes)
      {
         diff_box = hypre_BoxArrayBox(diff_boxes, j);
         start = hypre_BoxIMin(diff_box);

         hypre_BoxGetSize(diff_box, loop_size);

#define DEVICE_VAR is_device_ptr(vp)
         hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                             v_data_box, start, ustride, vi);
         {
            vp[vi] = 0.0;
         }
#undef DEVICE_VAR
         hypre_BoxLoop1End(vi);
      }
   }
   hypre_BoxArrayDestroy(diff_boxes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * RDF TODO: Extend to allow non-unitary stride
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorClearBoundGhostValues( hypre_StructVector *vector,
                                         HYPRE_Int           force )
{
   HYPRE_Int           ndim = hypre_StructVectorNDim(vector);
   HYPRE_Complex      *vp;
   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Box          *v_data_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         stride;
   hypre_Box          *bbox;
   hypre_StructGrid   *grid;
   hypre_BoxArray     *boundary_boxes;
   hypre_BoxArray     *array_of_box;
   hypre_BoxArray     *work_boxarray;

   HYPRE_Int           i, i2;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   /* Only clear if not clear already or if force argument is set */
   if (hypre_StructVectorBGhostNotClear(vector) || force)
   {
      grid = hypre_StructVectorGrid(vector);
      boxes = hypre_StructGridBoxes(grid);
      hypre_SetIndex(stride, 1);

      hypre_ForBoxI(i, boxes)
      {
         box        = hypre_BoxArrayBox(boxes, i);
         boundary_boxes = hypre_BoxArrayCreate( 0, ndim );
         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         hypre_BoxBoundaryG( v_data_box, grid, boundary_boxes );
         vp = hypre_StructVectorBoxData(vector, i);

         /* box is a grid box, no ghost zones.
            v_data_box is vector data box, may or may not have ghost zones
            To get only ghost zones, subtract box from boundary_boxes.   */
         work_boxarray = hypre_BoxArrayCreate( 0, ndim );
         array_of_box = hypre_BoxArrayCreate( 1, ndim );
         hypre_BoxArrayBoxes(array_of_box)[0] = *box;
         hypre_SubtractBoxArrays( boundary_boxes, array_of_box, work_boxarray );

         hypre_ForBoxI(i2, boundary_boxes)
         {
            bbox       = hypre_BoxArrayBox(boundary_boxes, i2);
            hypre_BoxGetSize(bbox, loop_size);
            start = hypre_BoxIMin(bbox);

#define DEVICE_VAR is_device_ptr(vp)
            hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                                v_data_box, start, stride, vi);
            {
               vp[vi] = 0.0;
            }
            hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
         }
         hypre_BoxArrayDestroy(boundary_boxes);
         hypre_BoxArrayDestroy(work_boxarray);
         hypre_BoxArrayDestroy(array_of_box);
      }

      hypre_StructVectorBGhostNotClear(vector) = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorScaleValues( hypre_StructVector *vector,
                               HYPRE_Complex       factor )
{
   HYPRE_Complex    *data;

   hypre_Index       imin;
   hypre_Index       imax;
   hypre_Box        *box;
   hypre_Index       loop_size;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   hypre_SetIndex(imin, 1);
   hypre_SetIndex(imax, 1);
   hypre_IndexD(imax, 0) = hypre_StructVectorDataSize(vector);
   hypre_BoxSetExtents(box, imin, imax);
   data = hypre_StructVectorData(vector);
   hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(data)
   hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                       box, imin, imin, datai);
   {
      data[datai] *= factor;
   }
   hypre_BoxLoop1End(datai);
#undef DEVICE_VAR

   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_CommPkg *
hypre_StructVectorGetMigrateCommPkg( hypre_StructVector *from_vector,
                                     hypre_StructVector *to_vector   )
{
   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;

   /*------------------------------------------------------
    * Set up hypre_CommPkg
    *------------------------------------------------------*/

   hypre_CreateCommInfoFromGrids(hypre_StructVectorGrid(from_vector),
                                 hypre_StructVectorGrid(to_vector),
                                 &comm_info);
   hypre_CommPkgCreate(comm_info,
                       hypre_StructVectorDataSpace(from_vector),
                       hypre_StructVectorDataSpace(to_vector), 1, NULL, 0,
                       hypre_StructVectorComm(from_vector),
                       hypre_StructVectorMemoryLocation(from_vector),
                       &comm_pkg);
   hypre_CommInfoDestroy(comm_info);
   /* is this correct for periodic? */

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorMigrate( hypre_CommPkg      *comm_pkg,
                           hypre_StructVector *fr_vector,
                           hypre_StructVector *to_vector )
{
   hypre_CommHandle  *comm_handle;
   HYPRE_Complex     *fr_data = hypre_StructVectorData(fr_vector);
   HYPRE_Complex     *to_data = hypre_StructVectorData(to_vector);

   /*-----------------------------------------------------------------------
    * Migrate the vector data
    *-----------------------------------------------------------------------*/

   hypre_StructCommunicationInitialize(comm_pkg, &fr_data, &to_data, 0, 0, &comm_handle);
   hypre_StructCommunicationFinalize(comm_handle);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorPrintData
 *
 * RDF TODO: Update to allow for non-unitary stride?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorPrintData( FILE               *file,
                             hypre_StructVector *vector,
                             HYPRE_Int           all )
{
   HYPRE_Int             ndim            = hypre_StructVectorNDim(vector);
   hypre_StructGrid     *grid            = hypre_StructVectorGrid(vector);
   hypre_BoxArray       *grid_boxes      = hypre_StructGridBoxes(grid);
   hypre_BoxArray       *data_space      = hypre_StructVectorDataSpace(vector);
   HYPRE_Int             data_size       = hypre_StructVectorDataSize(vector);
   HYPRE_Complex        *data            = hypre_StructVectorData(vector);
   HYPRE_MemoryLocation  memory_location = hypre_StructVectorMemoryLocation(vector);

   HYPRE_Int             num_values      = 1;
   HYPRE_Int             value_ids[1]    = {0};

   hypre_BoxArray       *boxes;
   HYPRE_Complex        *h_data;

   /* Allocate/Point to data on the host memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      h_data = hypre_CTAlloc(HYPRE_Complex, data_size, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(h_data, data, HYPRE_Complex, data_size,
                    HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_data = data;
   }

   /* Print ghost data (all) also or only real data? */
   boxes = (all) ? data_space : grid_boxes;

   /* Print data to file */
   hypre_PrintBoxArrayData(file, ndim, boxes, data_space, num_values, value_ids, h_data);

   /* Free memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorReadData
 *
 * RDF TODO: Update to allow for non-unitary stride?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorReadData( FILE               *file,
                            hypre_StructVector *vector )
{
   HYPRE_Int             ndim  = hypre_StructVectorNDim(vector);
   hypre_StructGrid     *grid  = hypre_StructVectorGrid(vector);
   hypre_BoxArray       *boxes = hypre_StructGridBoxes(grid);

   HYPRE_MemoryLocation  memory_location = hypre_StructVectorMemoryLocation(vector);

   hypre_Box            *box;
   HYPRE_Int             num_values;
   HYPRE_Complex        *h_values, *values;
   HYPRE_Int            *value_ids;
   HYPRE_Int             i, vi;

   /* Read data from file */
   hypre_fscanf(file, "\nData:\n");
   hypre_ReadBoxArrayData(file, ndim, boxes, &num_values, &value_ids, &h_values);

   /* Move data to the device memory if necessary and free host data */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      vi = hypre_BoxArrayVolume(boxes) * num_values;
      values = hypre_TAlloc(HYPRE_Complex, vi, memory_location);
      hypre_TMemcpy(values, h_values, HYPRE_Complex, num_values,
                    memory_location, HYPRE_MEMORY_HOST);
      hypre_TFree(h_values, HYPRE_MEMORY_HOST);
   }
   else
   {
      values = h_values;
   }

   /* Set vector values */
   hypre_StructVectorInitialize(vector, 1);
   vi = 0;
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      HYPRE_StructVectorSetBoxValues(vector,
                                     hypre_BoxIMin(box),
                                     hypre_BoxIMax(box),
                                     &values[vi]);
      vi += hypre_BoxVolume(box);
   }

   /* Clean up */
   hypre_TFree(value_ids, HYPRE_MEMORY_HOST);
   hypre_TFree(values, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorPrint( const char         *filename,
                         hypre_StructVector *vector,
                         HYPRE_Int           all      )
{
   FILE              *file;
   char               new_filename[255];

   hypre_StructGrid  *grid;
   HYPRE_Int          myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(hypre_StructVectorComm(vector), &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   hypre_fprintf(file, "StructVector\n");

   /* print grid info */
   hypre_fprintf(file, "\nGrid:\n");
   grid = hypre_StructVectorGrid(vector);
   hypre_StructGridPrint(file, grid);

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   hypre_fprintf(file, "\nData:\n");
   hypre_StructVectorPrintData(file, vector, all);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fflush(file);
   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorRead
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorRead( MPI_Comm    comm,
                        const char *filename,
                        HYPRE_Int  *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];

   hypre_StructVector   *vector;
   hypre_StructGrid     *grid;

   HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_printf("Error: can't open input file %s\n", new_filename);
      hypre_error_in_arg(2);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   hypre_fscanf(file, "StructVector\n");

   /* read grid info */
   hypre_fscanf(file, "\nGrid:\n");
   hypre_StructGridRead(comm, file, &grid);

   /*----------------------------------------
    * Initialize the vector
    *----------------------------------------*/

   HYPRE_StructVectorCreate(comm, grid, &vector);
   hypre_StructVectorSetNumGhost(vector, num_ghost);  // RDF Is this needed?

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   hypre_StructVectorReadData(file, vector);

   /*----------------------------------------
    * Assemble the vector
    *----------------------------------------*/

   hypre_StructVectorAssemble(vector);

   /*----------------------------------------
    * Clean up and close file
    *----------------------------------------*/

   HYPRE_StructGridDestroy(grid);

   fclose(file);

   return vector;
}

/*--------------------------------------------------------------------------
 * The following is used only as a debugging aid.
 *--------------------------------------------------------------------------*/

// TODO (VPM): Fix for GPU support
#if 0
HYPRE_Int
hypre_StructVectorMaxValue( hypre_StructVector *vector,
                            HYPRE_Real *max_value, HYPRE_Int *max_index,
                            hypre_Index max_xyz_index )
/* Input: vector, and pointers to where to put returned data.
   Return value: error flag, 0 means ok.
   Finds the maximum value in a vector, puts it in max_value.
   The corresponding index is put in max_index.
   A hypre_Index corresponding to max_index is put in max_xyz_index.
   We assume that there is only one box to deal with. */
{
   HYPRE_Real       *data;

   hypre_Index       imin;
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_Index       ustride;

   HYPRE_Int         i, ndim;
   HYPRE_Real        maxvalue;
   HYPRE_Int         maxindex;

   ndim = hypre_StructVectorNDim(vector);
   boxes = hypre_StructVectorDataSpace(vector);
   if ( hypre_BoxArraySize(boxes) != 1 )
   {
      /* if more than one box, the return system max_xyz_index is too simple
         if needed, fix later */
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }
   hypre_SetIndex(ustride, 1);
   hypre_ForBoxI(i, boxes)
   {
      box  = hypre_BoxArrayBox(boxes, i);
      /*v_data_box =
        hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);*/
      data = hypre_StructVectorBoxData(vector, i);
      hypre_BoxGetSize(box, loop_size);
      hypre_CopyIndex( hypre_BoxIMin(box), imin );

      maxindex = hypre_BoxIndexRank( box, imin );
      maxvalue = data[maxindex];
      hypre_SetIndex(max_xyz_index, 0);

      hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                box, imin, ustride, datai);
      {
         if ( data[datai] > maxvalue )
         {
            maxvalue = data[datai];
            maxindex = datai;
            hypre_BoxLoopGetIndex(max_xyz_index);
         }
      }
      hypre_SerialBoxLoop1End(datai);
      hypre_AddIndexes(max_xyz_index, imin, ndim, max_xyz_index);
   }

   *max_value = maxvalue;
   *max_index = maxindex;

   return hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 * hypre_StructVectorClone
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorClone( hypre_StructVector *x )
{
   MPI_Comm              comm            = hypre_StructVectorComm(x);
   hypre_StructGrid     *grid            = hypre_StructVectorGrid(x);
   HYPRE_MemoryLocation  memory_location = hypre_StructVectorMemoryLocation(x);
   hypre_BoxArray       *data_space      = hypre_StructVectorDataSpace(x);
   HYPRE_Int            *data_indices    = hypre_StructVectorDataIndices(x);
   HYPRE_Int             data_size       = hypre_StructVectorDataSize(x);
   HYPRE_Int             ndim            = hypre_StructGridNDim(grid);
   HYPRE_Int             data_space_size = hypre_BoxArraySize(data_space);

   hypre_StructVector   *y;
   HYPRE_Int             i;

   y = hypre_StructVectorCreate(comm, grid);
   hypre_StructVectorSetStride(y, hypre_StructVectorStride(x));

   hypre_StructVectorDataSize(y)    = data_size;
   hypre_StructVectorDataSpace(y)   = hypre_BoxArrayClone(data_space);
   hypre_StructVectorDataAlloced(y) = 1;
   hypre_StructVectorData(y)        = hypre_CTAlloc(HYPRE_Complex, data_size, memory_location);
   hypre_StructVectorDataIndices(y) = hypre_CTAlloc(HYPRE_Int, data_space_size,
                                                    HYPRE_MEMORY_HOST);
   for (i = 0; i < data_space_size; i++)
   {
      hypre_StructVectorDataIndices(y)[i] = data_indices[i];
   }
   hypre_StructCopy(x, y);

   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructVectorNumGhost(y)[i] = hypre_StructVectorNumGhost(x)[i];
   }
   hypre_StructVectorBGhostNotClear(y) = hypre_StructVectorBGhostNotClear(x);
   hypre_StructVectorGlobalSize(y)     = hypre_StructVectorGlobalSize(x);

   return y;
}
