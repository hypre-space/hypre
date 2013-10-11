/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Member functions for hypre_StructVector class.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorCreate( MPI_Comm          comm,
                          hypre_StructGrid *grid )
{
   HYPRE_Int            ndim = hypre_StructGridNDim(grid);
   hypre_StructVector  *vector;
   HYPRE_Int            i;

   vector = hypre_CTAlloc(hypre_StructVector, 1);

   hypre_StructVectorComm(vector)           = comm;
   hypre_StructGridRef(grid, &hypre_StructVectorGrid(vector));
   hypre_StructVectorDataAlloced(vector)    = 1;
   hypre_StructVectorBGhostNotClear(vector) = 0;
   hypre_StructVectorRefCount(vector)       = 1;

   /* set defaults */
   for (i = 0; i < 2*ndim; i++)
   {
      hypre_StructVectorNumGhost(vector)[i] = hypre_StructGridNumGhost(grid)[i];
   }

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
            hypre_SharedTFree(hypre_StructVectorData(vector));
         }
         hypre_TFree(hypre_StructVectorDataIndices(vector));
         hypre_BoxArrayDestroy(hypre_StructVectorDataSpace(vector));
         hypre_StructGridDestroy(hypre_StructVectorGrid(vector));
         hypre_TFree(vector);
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
            hypre_BoxIMinD(data_box, d) -= num_ghost[2*d];
            hypre_BoxIMaxD(data_box, d) += num_ghost[2*d + 1];
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
      data_indices = hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(data_space));

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
                                  HYPRE_Complex      *data   )
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

   data = hypre_SharedCTAlloc(HYPRE_Complex, hypre_StructVectorDataSize(vector));
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
   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;

   HYPRE_Complex      *vecp;

   HYPRE_Int           i, istart, istop;

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
   HYPRE_Int           datai;
   HYPRE_Complex      *datap;

   hypre_Box          *dval_box;
   hypre_Index         dval_start;
   hypre_Index         dval_stride;
   HYPRE_Int           dvali;

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

         if (action > 0)
         {
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(vector), loop_size,
                                data_box,data_start,data_stride,datai,
                                dval_box,dval_start,dval_stride,dvali);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,datai,dvali) HYPRE_SMP_SCHEDULE
#endif
            hypre_BoxLoop2For(datai, dvali)
            {
               datap[datai] += values[dvali];
            }
            hypre_BoxLoop2End(datai, dvali);
         }
         else if (action > -1)
         {
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(vector), loop_size,
                                data_box,data_start,data_stride,datai,
                                dval_box,dval_start,dval_stride,dvali);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,datai,dvali) HYPRE_SMP_SCHEDULE
#endif
            hypre_BoxLoop2For(datai, dvali)
            {
               datap[datai] = values[dvali];
            }
            hypre_BoxLoop2End(datai, dvali);
         }
         else /* action < 0 */
         {
            hypre_BoxLoop2Begin(hypre_StructVectorNDim(vector), loop_size,
                                data_box,data_start,data_stride,datai,
                                dval_box,dval_start,dval_stride,dvali);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,datai,dvali) HYPRE_SMP_SCHEDULE
#endif
            hypre_BoxLoop2For(datai, dvali)
            {
               values[dvali] = datap[datai];
            }
            hypre_BoxLoop2End(datai, dvali);
         }
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
   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;

   HYPRE_Complex      *vecp;

   HYPRE_Int           i, istart, istop;

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
         *vecp = 0.0;
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
   HYPRE_Int           datai;
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

         hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                             data_box,data_start,data_stride,datai);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,datai) HYPRE_SMP_SCHEDULE
#endif
         hypre_BoxLoop1For(datai)
         {
            datap[datai] = 0.0;
         }
         hypre_BoxLoop1End(datai);
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

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < data_size; i++)
   {
      data[i] = 0.0;
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
      hypre_StructVectorNumGhost(vector)[2*d]     = num_ghost[2*d];
      hypre_StructVectorNumGhost(vector)[2*d + 1] = num_ghost[2*d + 1];
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_StructVectorAssemble( hypre_StructVector *vector )
{
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
                    
   HYPRE_Int           vi;
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
      box      = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      x_data_box =
         hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);
 
      hypre_BoxGetSize(box, loop_size);

      hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, unit_stride, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,vi ) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop1For(vi)
      {
         yp[vi] = xp[vi];
      }
      hypre_BoxLoop1End(vi);
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
                    
   HYPRE_Int           vi;
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

      hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, unit_stride, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,vi ) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop1For(vi)
      {
         vp[vi] = values;
      }
      hypre_BoxLoop1End(vi);
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
                                     HYPRE_Complex     (*fcn)() )
{
   hypre_Box          *v_data_box;
                    
   HYPRE_Int           vi;
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

      hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, unit_stride, vi);
      i = hypre_IndexD(start, 0);
      j = hypre_IndexD(start, 1);
      k = hypre_IndexD(start, 2);
/* RDF: This won't work as written with threading on */
#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,vi ) HYPRE_SMP_SCHEDULE
#endif
#else
      hypre_BoxLoopSetOneBlock();
#endif
      hypre_BoxLoop1For(vi)
      {
         vp[vi] = fcn(i, j, k);
         i++;
         j++;
         k++;
      }
      hypre_BoxLoop1End(vi);
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
                    
   HYPRE_Int           vi;
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

         hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                             v_data_box, start, unit_stride, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,vi ) HYPRE_SMP_SCHEDULE
#endif
         hypre_BoxLoop1For(vi)
         {
            vp[vi] = 0.0;
         }
         hypre_BoxLoop1End(vi);
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
   HYPRE_Int           vi;
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
            hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                                v_data_box, start, stride, vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,vi ) HYPRE_SMP_SCHEDULE
#endif
            hypre_BoxLoop1For(vi)
            {
               vp[vi] = 0.0;
            }
            hypre_BoxLoop1End(vi);
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
   HYPRE_Int         datai;
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

   hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                       box, imin, imin, datai);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,datai) HYPRE_SMP_SCHEDULE
#endif
   hypre_BoxLoop1For(datai)
   {
      data[datai] *= factor;
   }
   hypre_BoxLoop1End(datai);

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
   hypre_BoxArray    *boxes;

   hypre_BoxArray    *data_space;

   HYPRE_Int          myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(hypre_StructVectorComm(vector), &myid );

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
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

   data_space = hypre_StructVectorDataSpace(vector);

   if (all)
      boxes = data_space;
   else
      boxes = hypre_StructGridBoxes(grid);

   hypre_fprintf(file, "\nData:\n");
   hypre_PrintBoxArrayData(file, boxes, data_space, 1,
                           hypre_StructGridNDim(grid),
                           hypre_StructVectorData(vector));
 
   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
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
   hypre_BoxArray       *boxes;

   hypre_BoxArray       *data_space;

   HYPRE_Int             myid;
 
   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid );

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   hypre_fscanf(file, "StructVector\n");

   /* read grid info */
   hypre_fscanf(file, "\nGrid:\n");
   hypre_StructGridRead(comm,file,&grid);

   /*----------------------------------------
    * Initialize the vector
    *----------------------------------------*/

   vector = hypre_StructVectorCreate(comm, grid);
   hypre_StructVectorSetNumGhost(vector, num_ghost);
   hypre_StructVectorInitialize(vector);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   boxes      = hypre_StructGridBoxes(grid);
   data_space = hypre_StructVectorDataSpace(vector);
 
   hypre_fscanf(file, "\nData:\n");
   hypre_ReadBoxArrayData(file, boxes, data_space, 1,
                          hypre_StructGridNDim(grid),
                          hypre_StructVectorData(vector));

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
 *--------------------------------------------------------------------------*/

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
   HYPRE_Int         datai;
   HYPRE_Real       *data;

   hypre_Index       imin;
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_Index       unit_stride;

   HYPRE_Int         i, ndim;
   HYPRE_Real        maxvalue;
   HYPRE_Int         maxindex;

   ndim = hypre_StructVectorNDim(vector);
   boxes = hypre_StructVectorDataSpace(vector);
   if ( hypre_BoxArraySize(boxes)!=1 )
   {
      /* if more than one box, the return system max_xyz_index is too simple
         if needed, fix later */
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }
   hypre_SetIndex(unit_stride, 1);
   hypre_ForBoxI(i, boxes)
   {
      box  = hypre_BoxArrayBox(boxes, i);
      /*v_data_box =
        hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);*/
      data = hypre_StructVectorBoxData(vector, i);
      hypre_BoxGetSize(box, loop_size);
      hypre_CopyIndex( hypre_BoxIMin(box), imin );

      hypre_BoxLoop1Begin(ndim, loop_size,
                          box, imin, unit_stride, datai);
      maxindex = hypre_BoxIndexRank( box, imin );
      maxvalue = data[maxindex];
      hypre_SetIndex(max_xyz_index, 0);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,datai) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop1For(datai)
      {
         if ( data[datai] > maxvalue )
         {
            maxvalue = data[datai];
            maxindex = datai;
            hypre_BoxLoopGetIndex(max_xyz_index);
         }
      }
      hypre_BoxLoop1End(datai);
      hypre_AddIndexes(max_xyz_index, imin, ndim, max_xyz_index);
   }

   *max_value = maxvalue;
   *max_index = maxindex;

   return hypre_error_flag;
}

