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
   hypre_StructVectorDataAlloced(vector)    = 1;
   hypre_StructVectorBGhostNotClear(vector) = 0;
   hypre_StructVectorRefCount(vector)       = 1;

   /* set defaults */
   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructVectorNumGhost(vector)[i] = hypre_StructGridNumGhost(grid)[i];
   }

   hypre_StructVectorMemoryLocation(vector) = hypre_HandleMemoryLocation(hypre_handle());

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
         if (hypre_StructVectorDataAlloced(vector))
         {
            hypre_TFree(hypre_StructVectorData(vector), hypre_StructVectorMemoryLocation(vector));
         }

         hypre_TFree(hypre_StructVectorDataIndices(vector), HYPRE_MEMORY_HOST);
         hypre_BoxArrayDestroy(hypre_StructVectorDataSpace(vector));
         hypre_StructGridDestroy(hypre_StructVectorGrid(vector));
         hypre_TFree(vector, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorInitializeShell( hypre_StructVector *vector )
{
   HYPRE_Int             ndim = hypre_StructVectorNDim(vector);
   hypre_StructGrid     *grid;

   HYPRE_Int            *num_ghost;

   hypre_BoxArray       *data_space;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Box            *data_box;

   HYPRE_Int            *data_indices;
   HYPRE_Int             data_size;

   HYPRE_Int             i, d;

   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   grid = hypre_StructVectorGrid(vector);

   if (hypre_StructVectorDataSpace(vector) == NULL)
   {
      num_ghost = hypre_StructVectorNumGhost(vector);

      boxes = hypre_StructGridBoxes(grid);
      data_space = hypre_BoxArrayCreate(hypre_BoxArraySize(boxes), ndim);

      hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         hypre_CopyBox(box, data_box);
         for (d = 0; d < ndim; d++)
         {
            hypre_BoxIMinD(data_box, d) -= num_ghost[2 * d];
            hypre_BoxIMaxD(data_box, d) += num_ghost[2 * d + 1];
         }
      }

      hypre_StructVectorDataSpace(vector) = data_space;
   }

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data_size
    *-----------------------------------------------------------------------*/

   if (hypre_StructVectorDataIndices(vector) == NULL)
   {
      data_space = hypre_StructVectorDataSpace(vector);
      data_indices = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(data_space), HYPRE_MEMORY_HOST);

      data_size = 0;
      hypre_ForBoxI(i, data_space)
      {
         data_box = hypre_BoxArrayBox(data_space, i);

         data_indices[i] = data_size;
         data_size += hypre_BoxVolume(data_box);
      }

      hypre_StructVectorDataIndices(vector) = data_indices;

      hypre_StructVectorDataSize(vector)    = data_size;

   }

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   hypre_StructVectorGlobalSize(vector) = hypre_StructGridGlobalSize(grid);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorInitializeData( hypre_StructVector *vector,
                                  HYPRE_Complex      *data)
{
   hypre_StructVectorData(vector) = data;
   hypre_StructVectorDataAlloced(vector) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorInitialize( hypre_StructVector *vector )
{
   HYPRE_Complex *data;

   hypre_StructVectorInitializeShell(vector);

   data = hypre_CTAlloc(HYPRE_Complex, hypre_StructVectorDataSize(vector),
                        hypre_StructVectorMemoryLocation(vector));

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
   dval_box = hypre_BoxDuplicate(value_box);
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

#define DEVICE_VAR is_device_ptr(datap,values)
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
   hypre_Index    imin, imax;
   hypre_Box     *box;

   box = hypre_BoxCreate(1);
   hypre_IndexD(imin, 0) = 1;
   hypre_IndexD(imax, 0) = data_size;
   hypre_BoxSetExtents(box, imin, imax);

#define DEVICE_VAR is_device_ptr(data)
   hypre_BoxLoop1Begin(1, imax,
                       box, imin, imin, datai);
   {
      data[datai] = 0.0;
   }
   hypre_BoxLoop1End(datai);
#undef DEVICE_VAR

   hypre_BoxDestroy(box);

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
 * copies data from x to y
 * y has its own data array, so this is a deep copy in that sense.
 * The grid and other size information are not copied - they are
 * assumed to have already been set up to be consistent.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorCopy( hypre_StructVector *x,
                        hypre_StructVector *y )
{
   hypre_Box          *x_data_box;

   HYPRE_Complex      *xp, *yp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1);

   boxes = hypre_StructGridBoxes( hypre_StructVectorGrid(x) );
   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      x_data_box =
         hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
      hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, unit_stride, vi);
      {
         yp[vi] = xp[vi];
      }
      hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorSetConstantValues( hypre_StructVector *vector,
                                     HYPRE_Complex       values )
{
   hypre_Box          *v_data_box;

   HYPRE_Complex      *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
   {
      box      = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      v_data_box =
         hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
      vp = hypre_StructVectorBoxData(vector, i);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(vp)
      hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, unit_stride, vi);
      {
         vp[vi] = values;
      }
      hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Takes a function pointer of the form:  HYPRE_Complex  f(i,j,k)
 * RDF: This function doesn't appear to be used anywhere.
 *--------------------------------------------------------------------------*/

/* ONLY3D */

HYPRE_Int
hypre_StructVectorSetFunctionValues( hypre_StructVector *vector,
                                     HYPRE_Complex     (*fcn)(HYPRE_Int, HYPRE_Int, HYPRE_Int) )
{
   hypre_Box          *v_data_box;

   HYPRE_Complex      *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   HYPRE_Int           b, i, j, k;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(b, boxes)
   {
      box      = hypre_BoxArrayBox(boxes, b);
      start = hypre_BoxIMin(box);

      v_data_box =
         hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b);
      vp = hypre_StructVectorBoxData(vector, b);

      hypre_BoxGetSize(box, loop_size);

      i = hypre_IndexD(start, 0);
      j = hypre_IndexD(start, 1);
      k = hypre_IndexD(start, 2);

      hypre_SerialBoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                                v_data_box, start, unit_stride, vi)
      {
         vp[vi] = fcn(i, j, k);
         i++;
         j++;
         k++;
      }
      hypre_SerialBoxLoop1End(vi)
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
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
   hypre_Index         unit_stride;

   HYPRE_Int           i, j;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1);

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
                             v_data_box, start, unit_stride, vi);
         {
            vp[vi] = 0.0;
         }
         hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
      }
   }
   hypre_BoxArrayDestroy(diff_boxes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * clears vector values on the physical boundaries
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
hypre_StructVectorScaleValues( hypre_StructVector *vector, HYPRE_Complex factor )
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
                       hypre_StructVectorComm(from_vector), &comm_pkg);
   hypre_CommInfoDestroy(comm_info);
   /* is this correct for periodic? */

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorMigrate( hypre_CommPkg      *comm_pkg,
                           hypre_StructVector *from_vector,
                           hypre_StructVector *to_vector   )
{
   hypre_CommHandle      *comm_handle;

   /*-----------------------------------------------------------------------
    * Migrate the vector data
    *-----------------------------------------------------------------------*/

   hypre_InitializeCommunication(comm_pkg,
                                 hypre_StructVectorData(from_vector),
                                 hypre_StructVectorData(to_vector), 0, 0,
                                 &comm_handle);
   hypre_FinalizeCommunication(comm_handle);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorPrintData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorPrintData( FILE               *file,
                             hypre_StructVector *vector,
                             HYPRE_Int           all )
{
   HYPRE_Int            ndim            = hypre_StructVectorNDim(vector);
   hypre_StructGrid    *grid            = hypre_StructVectorGrid(vector);
   hypre_BoxArray      *grid_boxes      = hypre_StructGridBoxes(grid);
   hypre_BoxArray      *data_space      = hypre_StructVectorDataSpace(vector);
   HYPRE_Int            data_size       = hypre_StructVectorDataSize(vector);
   HYPRE_Complex       *data            = hypre_StructVectorData(vector);
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(vector);
   hypre_BoxArray      *boxes;
   HYPRE_Complex       *h_data;

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
   hypre_PrintBoxArrayData(file, boxes, data_space, 1, ndim, h_data);

   /* Free memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorReadData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorReadData( FILE               *file,
                            hypre_StructVector *vector )
{
   HYPRE_Int            ndim            = hypre_StructVectorNDim(vector);
   hypre_StructGrid    *grid            = hypre_StructVectorGrid(vector);
   hypre_BoxArray      *grid_boxes      = hypre_StructGridBoxes(grid);
   hypre_BoxArray      *data_space      = hypre_StructVectorDataSpace(vector);
   HYPRE_Int            data_size       = hypre_StructVectorDataSize(vector);
   HYPRE_Complex       *data            = hypre_StructVectorData(vector);
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(vector);
   HYPRE_Complex       *h_data;

   /* Allocate/Point to data on the host memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      h_data = hypre_CTAlloc(HYPRE_Complex, data_size, HYPRE_MEMORY_HOST);
   }
   else
   {
      h_data = data;
   }

   /* Read data from file */
   hypre_ReadBoxArrayData(file, grid_boxes, data_space, 1, ndim, h_data);

   /* Move data to the device memory if necessary and free host data */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      hypre_TMemcpy(data, h_data, HYPRE_Complex, data_size,
                    memory_location, HYPRE_MEMORY_HOST);
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

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

   vector = hypre_StructVectorCreate(comm, grid);
   hypre_StructVectorSetNumGhost(vector, num_ghost);
   hypre_StructVectorInitialize(vector);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   hypre_fscanf(file, "\nData:\n");
   hypre_StructVectorReadData(file, vector);

   /*----------------------------------------
    * Assemble the vector
    *----------------------------------------*/

   hypre_StructVectorAssemble(vector);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fclose(file);

   return vector;
}

/*--------------------------------------------------------------------------
 * The following is used only as a debugging aid.
 *
 * hypre_StructVectorClone
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorClone(hypre_StructVector *x)
{
   MPI_Comm             comm            = hypre_StructVectorComm(x);
   hypre_StructGrid    *grid            = hypre_StructVectorGrid(x);
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(x);
   hypre_BoxArray      *data_space      = hypre_StructVectorDataSpace(x);
   HYPRE_Int           *data_indices    = hypre_StructVectorDataIndices(x);
   HYPRE_Int            data_size       = hypre_StructVectorDataSize(x);
   HYPRE_Int            ndim            = hypre_StructGridNDim(grid);
   HYPRE_Int            data_space_size = hypre_BoxArraySize(data_space);
   hypre_StructVector  *y               = hypre_StructVectorCreate(comm, grid);
   HYPRE_Int            i;

   hypre_StructVectorDataSize(y)    = data_size;
   hypre_StructVectorDataSpace(y)   = hypre_BoxArrayDuplicate(data_space);
   hypre_StructVectorData(y)        = hypre_CTAlloc(HYPRE_Complex, data_size, memory_location);
   hypre_StructVectorDataIndices(y) = hypre_CTAlloc(HYPRE_Int, data_space_size, HYPRE_MEMORY_HOST);

   for (i = 0; i < data_space_size; i++)
   {
      hypre_StructVectorDataIndices(y)[i] = data_indices[i];
   }

   hypre_StructVectorCopy( x, y );

   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructVectorNumGhost(y)[i] = hypre_StructVectorNumGhost(x)[i];
   }

   hypre_StructVectorBGhostNotClear(y) = hypre_StructVectorBGhostNotClear(x);
   hypre_StructVectorGlobalSize(y) = hypre_StructVectorGlobalSize(x);

   return y;
}
