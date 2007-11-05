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


/******************************************************************************
 *
 * Member functions for hypre_StructVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_StructVectorCreate
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorCreate( MPI_Comm          comm,
                          hypre_StructGrid *grid )
{
   hypre_StructVector  *vector;

   int                  i;

   vector = hypre_CTAlloc(hypre_StructVector, 1);

   hypre_StructVectorComm(vector)        = comm;
   hypre_StructGridRef(grid, &hypre_StructVectorGrid(vector));
   hypre_StructVectorDataAlloced(vector) = 1;
   hypre_StructVectorOffProcAdd(vector)  = 0;
   hypre_StructVectorRefCount(vector)    = 1;

   /* set defaults */
   for (i = 0; i < 6; i++)
   {
      hypre_StructVectorNumGhost(vector)[i] = 1;
      hypre_StructVectorAddNumGhost(vector)[i] = 1;
   }

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorRef
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorRef( hypre_StructVector *vector )
{
   hypre_StructVectorRefCount(vector) ++;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorDestroy( hypre_StructVector *vector )
{
   int  ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorInitializeShell
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorInitializeShell( hypre_StructVector *vector )
{
   int    ierr = 0;

   hypre_StructGrid     *grid;

   int                  *num_ghost;
 
   hypre_BoxArray       *data_space;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Box            *data_box;

   int                  *data_indices;
   int                   data_size;
                     
   int                   i, d;
 
   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   grid = hypre_StructVectorGrid(vector);

   if (hypre_StructVectorDataSpace(vector) == NULL)
   {
      num_ghost = hypre_StructVectorNumGhost(vector);

      boxes = hypre_StructGridBoxes(grid);
      data_space = hypre_BoxArrayCreate(hypre_BoxArraySize(boxes));

      hypre_ForBoxI(i, boxes)
         {
            box = hypre_BoxArrayBox(boxes, i);
            data_box = hypre_BoxArrayBox(data_space, i);

            hypre_CopyBox(box, data_box);
            for (d = 0; d < 3; d++)
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
      data_indices = hypre_CTAlloc(int, hypre_BoxArraySize(data_space));

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorInitializeData
 *--------------------------------------------------------------------------*/

int
hypre_StructVectorInitializeData( hypre_StructVector *vector,
                                  double             *data   )
{
   int ierr = 0;

   hypre_StructVectorData(vector) = data;
   hypre_StructVectorDataAlloced(vector) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorInitialize( hypre_StructVector *vector )
{
   int    ierr = 0;

   double *data;

   ierr = hypre_StructVectorInitializeShell(vector);

   data = hypre_SharedCTAlloc(double, hypre_StructVectorDataSize(vector));
   hypre_StructVectorInitializeData(vector, data);
   hypre_StructVectorDataAlloced(vector) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorSetValues( hypre_StructVector *vector,
                             hypre_Index         grid_index,
                             double              values,
                             int                 add_to     )
{
   int    ierr = 0;

   MPI_Comm            comm=  hypre_StructVectorComm(vector);
   hypre_BoxArray     *boxes;
   hypre_Box          *box;

   double             *vecp;

   int                 i, found;
   int                 true = 1;
   int                 false= 0;

   int                 nprocs;
   
   MPI_Comm_size(comm, &nprocs);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));

   found= false;
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);

         if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
             (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
             (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
             (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
             (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
             (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
         {
            vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);
            if (add_to)
            {
               *vecp += values;
            }
            else
            {
               *vecp = values;
            }
            found= true;
         }
      }

   /* to permit ADD values off myproc, but only glayers away from myproc's
      grid, use the data_space boxes of vector instead of the grid boxes. */
   if ((!found) && (add_to) && (nprocs > 1))
   {
      hypre_Box  *orig_box;

      int        *add_num_ghost= hypre_StructVectorAddNumGhost(vector);
      int         j;

      hypre_ForBoxI(i, boxes)
      {
         orig_box = hypre_BoxArrayBox(boxes, i);
         box      = hypre_BoxDuplicate(orig_box );
         for (j= 0; j< 3; j++)
         {
            hypre_BoxIMin(box)[j]-= add_num_ghost[2*j];
            hypre_BoxIMax(box)[j]+= add_num_ghost[2*j+1];
         }

         if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
             (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
             (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
             (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
             (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
             (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
         {
            vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);
            if (add_to)
            {
               *vecp += values;
            }
            else
            {
               *vecp = values;
            }
            found= true;
         }
         hypre_BoxDestroy(box);

         if (found) break;
      }

      /* set OffProcAdd for communication. Note that
         we have an Add if only one point switches this on. */
      if (found)
      {
         if (add_to)
         {
            hypre_StructVectorOffProcAdd(vector)= 1;
         }
      }
      else
      {
         printf("not found- grid_index off the extended vector grid\n");
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorSetBoxValues( hypre_StructVector *vector,
                                hypre_Box          *value_box,
                                double             *values,
                                int                 add_to    )
{
   int    ierr = 0;

   MPI_Comm            comm         = hypre_StructVectorComm(vector);
   int                *add_num_ghost= hypre_StructVectorAddNumGhost(vector);

   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;
   hypre_BoxArray     *box_array1, *box_array2, *tmp_box_array;
   hypre_BoxArray     *value_boxarray;
   hypre_BoxArrayArray *box_aarray;
   
   hypre_Box          *box, *tmp_box, *orig_box, *vbox;

   hypre_BoxArray     *data_space;
   hypre_Box          *data_box;
   hypre_IndexRef      data_start;
   hypre_Index         data_stride;
   int                 datai;
   double             *datap;

   hypre_Box          *dval_box;
   hypre_Index         dval_start;
   hypre_Index         dval_stride;
   int                 dvali;

   hypre_Index         loop_size;

   int                 i, j, k, vol_vbox, vol_iboxes, vol_offproc;
   int                 loopi, loopj, loopk;
   int                 nprocs;
   
   MPI_Comm_size(comm, &nprocs);

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes
    *-----------------------------------------------------------------------*/

   /* Find the intersecting boxes of the grid with value_box. Record the
      volumes of the intersections for possible off_proc settings. */
   vol_vbox  = hypre_BoxVolume(value_box);
   vol_iboxes= 0;

   grid_boxes= hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   box_array1= hypre_BoxArrayCreate(hypre_BoxArraySize(grid_boxes));
   box = hypre_BoxCreate();
   hypre_ForBoxI(i, grid_boxes)
   {
       grid_box = hypre_BoxArrayBox(grid_boxes, i);
       hypre_IntersectBoxes(value_box, grid_box, box);
       hypre_CopyBox(box, hypre_BoxArrayBox(box_array1, i));
       vol_iboxes+= hypre_BoxVolume(box);
   }

   /* Check if possible off_proc setting */
   vol_offproc= 0;
   if ((vol_vbox > vol_iboxes) && (add_to) && (nprocs > 1))
   {
      box_aarray= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes));

      /* to prevent overlapping intersected boxes, we subtract the intersected
         boxes from value_box. This requires a box_array structure. */
      value_boxarray= hypre_BoxArrayCreate(0);
      hypre_AppendBox(value_box, value_boxarray);

      hypre_ForBoxI(i, grid_boxes)
      {
         tmp_box_array= hypre_BoxArrayCreate(0);

         /* get ghostlayer boxes */
         orig_box= hypre_BoxArrayBox(grid_boxes, i);
         tmp_box  = hypre_BoxDuplicate(orig_box );
         for (j= 0; j< 3; j++)
         {
            hypre_BoxIMin(tmp_box)[j]-= add_num_ghost[2*j];
            hypre_BoxIMax(tmp_box)[j]+= add_num_ghost[2*j+1];
         }
         hypre_SubtractBoxes(tmp_box, orig_box, tmp_box_array);
         hypre_BoxDestroy(tmp_box);
          
         box_array2= hypre_BoxArrayArrayBoxArray(box_aarray, i);
         /* intersect the value_box and the ghostlayer boxes */
         hypre_ForBoxI(j, tmp_box_array)
         {
            tmp_box= hypre_BoxArrayBox(tmp_box_array, j);
            hypre_ForBoxI(k, value_boxarray)
            {
               vbox= hypre_BoxArrayBox(value_boxarray, k);
               hypre_IntersectBoxes(vbox, tmp_box, box);
               hypre_AppendBox(box, box_array2);

               vol_offproc+= hypre_BoxVolume(box);
            }
         }

         /* eliminate intersected boxes so that we do not get overlapping */
         hypre_SubtractBoxArrays(value_boxarray, box_array2, tmp_box_array);
         hypre_BoxArrayDestroy(tmp_box_array);

      }  /* hypre_ForBoxI(i, grid_boxes) */

      /* if vol_offproc= 0, trying to set values away from ghostlayer */
      if (!vol_offproc)
      {
         hypre_BoxArrayArrayDestroy(box_aarray);
      }
      else
      {
         /* set OffProcAdd for communication. Note that
            we have an Add if only one point switches this on. */
         hypre_StructVectorOffProcAdd(vector)= 1;
      }
      hypre_BoxArrayDestroy(value_boxarray);
      
   }
   hypre_BoxDestroy(box);

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   if (box_array1)
   {
      data_space = hypre_StructVectorDataSpace(vector);
      hypre_SetIndex(data_stride, 1, 1, 1);
 
      dval_box = hypre_BoxDuplicate(value_box);
      hypre_SetIndex(dval_stride, 1, 1, 1);
 
      hypre_ForBoxI(i, box_array1)
      {
         box      = hypre_BoxArrayBox(box_array1, i);
         data_box = hypre_BoxArrayBox(data_space, i);
 
         /* if there was an intersection */
         if (box)
         {
            data_start = hypre_BoxIMin(box);
            hypre_CopyIndex(data_start, dval_start);
 
            datap = hypre_StructVectorBoxData(vector, i);
 
            hypre_BoxGetSize(box, loop_size);

            if (add_to)
            {
               hypre_BoxLoop2Begin(loop_size,
                                   data_box,data_start,data_stride,datai,
                                   dval_box,dval_start,dval_stride,dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
               {
                  datap[datai] += values[dvali];
               }
               hypre_BoxLoop2End(datai, dvali);
            }
            else
            {
               hypre_BoxLoop2Begin(loop_size,
                                   data_box,data_start,data_stride,datai,
                                   dval_box,dval_start,dval_stride,dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
               {
                   datap[datai] = values[dvali];
               }
               hypre_BoxLoop2End(datai, dvali);
            }
         }
      }

      hypre_BoxDestroy(dval_box);
   }
   hypre_BoxArrayDestroy(box_array1);

   if (vol_offproc) /* nonzero only when adding values. */
   {
      data_space = hypre_StructVectorDataSpace(vector);
      hypre_SetIndex(data_stride, 1, 1, 1);
 
      dval_box = hypre_BoxDuplicate(value_box);
      hypre_SetIndex(dval_stride, 1, 1, 1);
 
      hypre_ForBoxI(i, data_space)
      {
         data_box  = hypre_BoxArrayBox(data_space, i);
         box_array2= hypre_BoxArrayArrayBoxArray(box_aarray, i);

         hypre_ForBoxI(j, box_array2)
         {
            box= hypre_BoxArrayBox(box_array2, j);
 
           /* if there was an intersection */
            if (box)
            {
               data_start = hypre_BoxIMin(box);
               hypre_CopyIndex(data_start, dval_start);
 
               datap = hypre_StructVectorBoxData(vector, i);
 
               hypre_BoxGetSize(box, loop_size);

               if (add_to) /* don't really need this conditional */
               {
                   hypre_BoxLoop2Begin(loop_size,
                                       data_box,data_start,data_stride,datai,
                                       dval_box,dval_start,dval_stride,dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
                   hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
                   {
                      datap[datai] += values[dvali];
                   }
                   hypre_BoxLoop2End(datai, dvali);
               }
           }   /* if (box) */
        }      /* hypre_ForBoxI(j, box_array2) */
     }         /* hypre_ForBoxI(i, data_space) */

     hypre_BoxDestroy(dval_box);
     hypre_BoxArrayArrayDestroy(box_aarray);
  }

  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorGetValues. OffProc values on the ghostlayer will be
 * extracted out, and hence, the values_ptr must contain ghostlayers.
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorGetValues( hypre_StructVector *vector,
                             hypre_Index         grid_index,
                             double             *values_ptr )
{
   int    ierr = 0;

   int                *add_num_ghost= hypre_StructVectorAddNumGhost(vector);
   double              values;

   hypre_BoxArray     *boxes;
   hypre_Box          *box, *orig_box;

   double             *vecp;

   int                 i, j, found;
   int                 true = 1;
   int                 false= 0;

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));

   /* search first to see if it is in the box. If not then check
      the ghostlayered boxes. */
   found= false;
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);

         if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
             (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
             (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
             (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
             (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
             (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
         {
            vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);
            values = *vecp;
            found= true;
         }
         if (found) break;
      }

   /* now search if on the ghostlayer */
   if (!found)
   {
      hypre_ForBoxI(i, boxes)
      {
         orig_box = hypre_BoxArrayBox(boxes, i);
         box      = hypre_BoxDuplicate(orig_box );
         for (j= 0; j< 3; j++)
         {
            hypre_BoxIMin(box)[j]-= add_num_ghost[2*j];
            hypre_BoxIMax(box)[j]+= add_num_ghost[2*j+1];
         }

         if ((hypre_IndexX(grid_index) >= hypre_BoxIMinX(box)) &&
             (hypre_IndexX(grid_index) <= hypre_BoxIMaxX(box)) &&
             (hypre_IndexY(grid_index) >= hypre_BoxIMinY(box)) &&
             (hypre_IndexY(grid_index) <= hypre_BoxIMaxY(box)) &&
             (hypre_IndexZ(grid_index) >= hypre_BoxIMinZ(box)) &&
             (hypre_IndexZ(grid_index) <= hypre_BoxIMaxZ(box))   )
         {
            vecp = hypre_StructVectorBoxDataValue(vector, i, grid_index);
            values= *vecp;
            found= true;
         }
         hypre_BoxDestroy(box);
         if (found) break;
      }
   }

  *values_ptr = values;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorGetBoxValues. Loop over ghostlayer also.
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorGetBoxValues( hypre_StructVector *vector,
                                hypre_Box          *value_box,
                                double             *values    )
{
   int    ierr = 0;

   int                *add_num_ghost= hypre_StructVectorAddNumGhost(vector);

   hypre_BoxArray     *grid_boxes;
   hypre_Box          *grid_box;
   hypre_BoxArray     *box_array1, *box_array2, *tmp_box_array;
   hypre_BoxArray     *value_boxarray;
   hypre_BoxArrayArray *box_aarray;

   hypre_Box          *box, *tmp_box, *orig_box, *vbox;

   hypre_BoxArray     *data_space;
   hypre_Box          *data_box;
   hypre_IndexRef      data_start;
   hypre_Index         data_stride;
   int                 datai;
   double             *datap;

   hypre_Box          *dval_box;
   hypre_Index         dval_start;
   hypre_Index         dval_stride;
   int                 dvali;

   hypre_Index         loop_size;

   int                 i, j, k, vol_vbox, vol_iboxes, vol_offproc;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set up `box_array' by intersecting `box' with the grid boxes 
    *-----------------------------------------------------------------------*/
   vol_vbox  = hypre_BoxVolume(value_box);
   vol_iboxes= 0;

   grid_boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   box_array1 = hypre_BoxArrayCreate(hypre_BoxArraySize(grid_boxes));
   box = hypre_BoxCreate();
   hypre_ForBoxI(i, grid_boxes)
   {
       grid_box = hypre_BoxArrayBox(grid_boxes, i);
       hypre_IntersectBoxes(value_box, grid_box, box);
       hypre_CopyBox(box, hypre_BoxArrayBox(box_array1, i));
       vol_iboxes+= hypre_BoxVolume(box);
   }

   /* Check if possible off_proc setting */
   vol_offproc= 0;
   if (vol_vbox > vol_iboxes)
   {
      box_aarray= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes));

      /* to prevent overlapping intersected boxes, we subtract the intersected
         boxes from value_box. This requires a box_array structure. */
      value_boxarray= hypre_BoxArrayCreate(0);
      hypre_AppendBox(value_box, value_boxarray);

      hypre_ForBoxI(i, grid_boxes)
      {
         tmp_box_array= hypre_BoxArrayCreate(0);

         /* get ghostlayer boxes */
         orig_box= hypre_BoxArrayBox(grid_boxes, i);
         tmp_box  = hypre_BoxDuplicate(orig_box );
         for (j= 0; j< 3; j++)
         {
            hypre_BoxIMin(tmp_box)[j]-= add_num_ghost[2*j];
            hypre_BoxIMax(tmp_box)[j]+= add_num_ghost[2*j+1];
         }
         hypre_SubtractBoxes(tmp_box, orig_box, tmp_box_array);
         hypre_BoxDestroy(tmp_box);

         box_array2= hypre_BoxArrayArrayBoxArray(box_aarray, i);
         /* intersect the value_box and the ghostlayer boxes */
         hypre_ForBoxI(j, tmp_box_array)
         {
            tmp_box= hypre_BoxArrayBox(tmp_box_array, j);
            hypre_ForBoxI(k, value_boxarray)
            {
               vbox= hypre_BoxArrayBox(value_boxarray, k);
               hypre_IntersectBoxes(vbox, tmp_box, box);
               hypre_AppendBox(box, box_array2);

               vol_offproc+= hypre_BoxVolume(box);
            }
         }

         /* eliminate intersected boxes so that we do not get overlapping */
         hypre_SubtractBoxArrays(value_boxarray, box_array2, tmp_box_array);
         hypre_BoxArrayDestroy(tmp_box_array);

      }  /* hypre_ForBoxI(i, grid_boxes) */
      /* if vol_offproc= 0, trying to set values away from ghostlayer */

      if (!vol_offproc)
      {
         hypre_BoxArrayArrayDestroy(box_aarray);
      }
      hypre_BoxArrayDestroy(value_boxarray);
   }
   hypre_BoxDestroy(box);

   /*-----------------------------------------------------------------------
    * Get the vector coefficients
    *-----------------------------------------------------------------------*/
   if (box_array1)
   {
      data_space = hypre_StructVectorDataSpace(vector);
      hypre_SetIndex(data_stride, 1, 1, 1);
 
      dval_box = hypre_BoxDuplicate(value_box);
      hypre_SetIndex(dval_stride, 1, 1, 1);
 
      hypre_ForBoxI(i, box_array1)
         {
            box      = hypre_BoxArrayBox(box_array1, i);
            data_box = hypre_BoxArrayBox(data_space, i);
 
            /* if there was an intersection */
            if (box)
            {
               data_start = hypre_BoxIMin(box);
               hypre_CopyIndex(data_start, dval_start);
 
               datap = hypre_StructVectorBoxData(vector, i);
 
               hypre_BoxGetSize(box, loop_size);

               hypre_BoxLoop2Begin(loop_size,
                                   data_box, data_start, data_stride, datai,
                                   dval_box, dval_start, dval_stride, dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
                  {
                     values[dvali] = datap[datai];
                  }
               hypre_BoxLoop2End(datai, dvali);
            }
         }

      hypre_BoxDestroy(dval_box);
   }
   hypre_BoxArrayDestroy(box_array1);

   if (vol_offproc) 
   {
      data_space = hypre_StructVectorDataSpace(vector);
      hypre_SetIndex(data_stride, 1, 1, 1);

      dval_box = hypre_BoxDuplicate(value_box);
      hypre_SetIndex(dval_stride, 1, 1, 1);

      hypre_ForBoxI(i, data_space)
      {
         data_box  = hypre_BoxArrayBox(data_space, i);
         box_array2= hypre_BoxArrayArrayBoxArray(box_aarray, i);

         hypre_ForBoxI(j, box_array2)
         {
            box= hypre_BoxArrayBox(box_array2, j);

           /* if there was an intersection */
            if (box)
            {
               data_start = hypre_BoxIMin(box);
               hypre_CopyIndex(data_start, dval_start);

               datap = hypre_StructVectorBoxData(vector, i);

               hypre_BoxGetSize(box, loop_size);
               hypre_BoxLoop2Begin(loop_size,
                                   data_box, data_start, data_stride, datai,
                                   dval_box, dval_start, dval_stride, dvali);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai,dvali
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop2For(loopi, loopj, loopk, datai, dvali)
               {
                     values[dvali] = datap[datai];
               }
               hypre_BoxLoop2End(datai, dvali);
            }   /* if (box) */
         }      /* hypre_ForBoxI(j, box_array2) */
     }          /* hypre_ForBoxI(i, data_space) */

     hypre_BoxDestroy(dval_box);
     hypre_BoxArrayArrayDestroy(box_aarray);
  }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
int
hypre_StructVectorSetNumGhost( hypre_StructVector *vector,
                               int                *num_ghost )
{
   int  ierr = 0;
   int  i;
 
   for (i = 0; i < 6; i++)
      hypre_StructVectorNumGhost(vector)[i] = num_ghost[i];

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_StructVectorAssemble
 * Before assembling the vector, all vector values added from 
 * off_procs are communicated to the correct proc. However, note that since
 * the comm_pkg is created from an "inverted" comm_info derived from the
 * vector, not all the communicated data is valid (i.e., we did not mark
 * which values are actually set off_proc). Because the communicated values
 * are added to existing values, the user is assumed to have set the
 * values correctly on or off the proc. 
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorAssemble( hypre_StructVector *vector )
{
   int  ierr = 0;

   int  sum_OffProcAdd;
   int  OffProcAdd= hypre_StructVectorOffProcAdd(vector);

   /* add_values may be off-proc. Communication needed, which is triggered
      if one of the OffProcAdd is 1 */
   sum_OffProcAdd= 0;
   MPI_Allreduce(&OffProcAdd, &sum_OffProcAdd, 1, MPI_INT, MPI_SUM,
                  hypre_StructVectorComm(vector));

   if (sum_OffProcAdd)
   {
      /* since the off_proc add_values are located on the ghostlayer, we
         need "inverse" communication. */

      hypre_CommInfo        *comm_info;
      hypre_CommInfo        *inv_comm_info;
      hypre_CommPkg         *comm_pkg;
      int                   *num_ghost   = hypre_StructVectorAddNumGhost(vector);

      hypre_BoxArrayArray   *send_boxes;
      hypre_BoxArrayArray   *recv_boxes;
      int                  **send_procs;
      int                  **recv_procs;
      int                  **send_rboxnums;
      int                  **recv_rboxnums; 
      hypre_BoxArrayArray   *send_rboxes;
      hypre_BoxArray        *box_array, *recv_array;
      
      double                *data;
      hypre_CommHandle      *comm_handle;
       
      hypre_Box             *data_box;
      hypre_Box             *box;

      double                *data_vec;
      double                *comm_data;
      hypre_Index            loop_size;
      hypre_IndexRef         start;
      hypre_Index            unit_stride;

      int                    i, j, xi, loopi, loopj, loopk;

      hypre_CreateCommInfoFromNumGhost(hypre_StructVectorGrid(vector),
                                       num_ghost, &comm_info);

      /* inverse CommInfo achieved by switching send & recv structures of
         CommInfo */
      send_boxes= hypre_BoxArrayArrayDuplicate(hypre_CommInfoRecvBoxes(comm_info));
      recv_boxes= hypre_BoxArrayArrayDuplicate(hypre_CommInfoSendBoxes(comm_info));
      send_rboxes= hypre_BoxArrayArrayDuplicate(send_boxes);

      send_procs= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_boxes));
      recv_procs= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(recv_boxes));
      send_rboxnums= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_boxes));
      recv_rboxnums= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(recv_boxes));

      hypre_ForBoxArrayI(i, send_boxes)
      {
         box_array= hypre_BoxArrayArrayBoxArray(send_boxes, i);
         send_procs[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
         memcpy(send_procs[i], hypre_CommInfoRecvProcesses(comm_info)[i],
                hypre_BoxArraySize(box_array)*sizeof(int));

         send_rboxnums[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
         memcpy(send_rboxnums[i], hypre_CommInfoRecvRBoxnums(comm_info)[i],
                hypre_BoxArraySize(box_array)*sizeof(int));
      }

      hypre_ForBoxArrayI(i, recv_boxes)
      {
         box_array= hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         recv_procs[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
         memcpy(recv_procs[i], hypre_CommInfoSendProcesses(comm_info)[i],
                hypre_BoxArraySize(box_array)*sizeof(int));

         recv_rboxnums[i]= hypre_CTAlloc(int, hypre_BoxArraySize(box_array));
         memcpy(recv_rboxnums[i], hypre_CommInfoSendRBoxnums(comm_info)[i],
                hypre_BoxArraySize(box_array)*sizeof(int));
      }

      hypre_CommInfoCreate(send_boxes, recv_boxes, send_procs, recv_procs,
                           send_rboxnums, recv_rboxnums, send_rboxes,
                           &inv_comm_info);

      hypre_CommPkgCreate(inv_comm_info,
                          hypre_StructVectorDataSpace(vector),
                          hypre_StructVectorDataSpace(vector),
                          1,
                          hypre_StructVectorComm(vector),
                          &comm_pkg);

      /* communicate the add value entries */
      data= hypre_CTAlloc(double, hypre_StructVectorDataSize(vector));
      hypre_InitializeCommunication(comm_pkg,
                                    hypre_StructVectorData(vector),
                                    data,
                                   &comm_handle);

      hypre_FinalizeCommunication(comm_handle);

      /* this proc will recved data in it's send_boxes of comm_info, or
         equivalently, in the recv_boxes of inv_comm_info. Since inv_comm_info
         has already been destroyed, we use the send_boxes of comm_info */ 
      hypre_SetIndex(unit_stride, 1, 1, 1);
      box_array= hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
      hypre_ForBoxI(i, box_array)
      {
         recv_array= 
             hypre_BoxArrayArrayBoxArray(hypre_CommInfoSendBoxes(comm_info), i);
                                                                                                                   
         data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         data_vec = hypre_StructVectorBoxData(vector, i);
         comm_data=(data + hypre_StructVectorDataIndices(vector)[i]);

         hypre_ForBoxI(j, recv_array)
         {
            box  = hypre_BoxArrayBox(recv_array, j);
            start= hypre_BoxIMin(box);
            hypre_BoxGetSize(box, loop_size);

           /* note that every proc adds since we don't track which
              ones should. */
            hypre_BoxLoop1Begin(loop_size,
                                data_box, start, unit_stride, xi)
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop1For(loopi, loopj, loopk, xi)
            {
                data_vec[xi] += comm_data[xi];
            }
            hypre_BoxLoop1End(xi);

         }  /* hypre_ForBoxI(j, recv_array) */
      }     /* hypre_ForBoxI(i, box_array) */

      hypre_TFree(data);
      hypre_CommInfoDestroy(comm_info);
      hypre_CommPkgDestroy(comm_pkg);

   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorCopy
 * copies data from x to y
 * y has its own data array, so this is a deep copy in that sense.
 * The grid and other size information are not copied - they are
 * assumed to have already been set up to be consistent.
 *--------------------------------------------------------------------------*/

int
hypre_StructVectorCopy( hypre_StructVector *x,
                        hypre_StructVector *y )
{


   int    ierr = 0;

   hypre_Box          *x_data_box;
                    
   int                 vi;
   double             *xp, *yp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int                 i;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
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

         hypre_BoxLoop1Begin(loop_size,
                             x_data_box, start, unit_stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, vi)
            {
               yp[vi] = xp[vi];
            }
         hypre_BoxLoop1End(vi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorSetConstantValues( hypre_StructVector *vector,
                                     double              values )
{
   int    ierr = 0;

   hypre_Box          *v_data_box;
                    
   int                 vi;
   double             *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int                 i;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
      {
         box      = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         vp = hypre_StructVectorBoxData(vector, i);
 
         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             v_data_box, start, unit_stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, vi)
            {
               vp[vi] = values;
            }
         hypre_BoxLoop1End(vi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorSetFunctionValues
 *
 * Takes a function pointer of the form:
 *
 *   double  f(i,j,k)
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorSetFunctionValues( hypre_StructVector *vector,
                                     double            (*fcn)() )
{
   int    ierr = 0;

   hypre_Box          *v_data_box;
                    
   int                 vi;
   double             *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int                 b, i, j, k;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(b, boxes)
      {
         box      = hypre_BoxArrayBox(boxes, b);
         start = hypre_BoxIMin(box);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), b);
         vp = hypre_StructVectorBoxData(vector, b);
 
         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             v_data_box, start, unit_stride, vi);
         i = hypre_IndexX(start);
         j = hypre_IndexY(start);
         k = hypre_IndexZ(start);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, vi)
            {
               vp[vi] = fcn(i, j, k);
               i++;
               j++;
               k++;
            }
         hypre_BoxLoop1End(vi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorClearGhostValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorClearGhostValues( hypre_StructVector *vector )
{
   int    ierr = 0;

   hypre_Box          *v_data_box;
                    
   int                 vi;
   double             *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_BoxArray     *diff_boxes;
   hypre_Box          *diff_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int                 i, j;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   diff_boxes = hypre_BoxArrayCreate(0);
   hypre_ForBoxI(i, boxes)
      {
         box        = hypre_BoxArrayBox(boxes, i);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         vp = hypre_StructVectorBoxData(vector, i);

         hypre_BoxArraySetSize(diff_boxes, 0);
         hypre_SubtractBoxes(v_data_box, box, diff_boxes);
         hypre_ForBoxI(j, diff_boxes)
            {
               diff_box = hypre_BoxArrayBox(diff_boxes, j);
               start = hypre_BoxIMin(diff_box);

               hypre_BoxGetSize(diff_box, loop_size);

               hypre_BoxLoop1Begin(loop_size,
                                   v_data_box, start, unit_stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop1For(loopi, loopj, loopk, vi)
                  {
                     vp[vi] = 0.0;
                  }
               hypre_BoxLoop1End(vi);
            }
      }
   hypre_BoxArrayDestroy(diff_boxes);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorClearBoundGhostValues
 * clears vector values on the physical boundaries
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorClearBoundGhostValues( hypre_StructVector *vector )
{
   int    ierr = 0;
   int                 vi;
   double             *vp;
   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Box          *v_data_box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         stride;
   hypre_Box *bbox;
   hypre_StructGrid   *grid;
   hypre_BoxArray     *boundary_boxes;
   hypre_BoxArray     *array_of_box;
   hypre_BoxArray     *work_boxarray;
      
   int                 i, i2;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   grid = hypre_StructVectorGrid(vector);
   boxes = hypre_StructGridBoxes(grid);
   hypre_SetIndex(stride, 1, 1, 1);

   hypre_ForBoxI(i, boxes)
      {
         box        = hypre_BoxArrayBox(boxes, i);
         boundary_boxes = hypre_BoxArrayCreate( 0 );
         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         ierr += hypre_BoxBoundaryG( v_data_box, grid, boundary_boxes );
         vp = hypre_StructVectorBoxData(vector, i);

         /* box is a grid box, no ghost zones.
            v_data_box is vector data box, may or may not have ghost zones
            To get only ghost zones, subtract box from boundary_boxes.   */
         work_boxarray = hypre_BoxArrayCreate( 0 );
         array_of_box = hypre_BoxArrayCreate( 1 );
         hypre_BoxArrayBoxes(array_of_box)[0] = *box;
         hypre_SubtractBoxArrays( boundary_boxes, array_of_box, work_boxarray );

         hypre_ForBoxI(i2, boundary_boxes)
            {
               bbox       = hypre_BoxArrayBox(boundary_boxes, i2);
               hypre_BoxGetSize(bbox, loop_size);
               start = hypre_BoxIMin(bbox);
               hypre_BoxLoop1Begin(loop_size,
                                   v_data_box, start, stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
               hypre_BoxLoop1For(loopi, loopj, loopk, vi)
                  {
                     vp[vi] = 0.0;
                  }
               hypre_BoxLoop1End(vi);
            }
         hypre_BoxArrayDestroy(boundary_boxes);
         hypre_BoxArrayDestroy(work_boxarray);
         hypre_BoxArrayDestroy(array_of_box);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorScaleValues( hypre_StructVector *vector, double factor )
{
   int               ierr = 0;

   int               datai;
   double           *data;

   hypre_Index       imin;
   hypre_Index       imax;
   hypre_Box        *box;
   hypre_Index       loop_size;

   int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   box = hypre_BoxCreate();
   hypre_SetIndex(imin, 1, 1, 1);
   hypre_SetIndex(imax, hypre_StructVectorDataSize(vector), 1, 1);
   hypre_BoxSetExtents(box, imin, imax);
   data = hypre_StructVectorData(vector);
   hypre_BoxGetSize(box, loop_size);

   hypre_BoxLoop1Begin(loop_size,
                       box, imin, imin, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
   hypre_BoxLoop1For(loopi, loopj, loopk, datai)
      {
         data[datai] *= factor;
      }
   hypre_BoxLoop1End(datai);

   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorClearAllValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorClearAllValues( hypre_StructVector *vector )
{
   int               ierr = 0;

   int               datai;
   double           *data;

   hypre_Index       imin;
   hypre_Index       imax;
   hypre_Box        *box;
   hypre_Index       loop_size;

   int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   box = hypre_BoxCreate();
   hypre_SetIndex(imin, 1, 1, 1);
   hypre_SetIndex(imax, hypre_StructVectorDataSize(vector), 1, 1);
   hypre_BoxSetExtents(box, imin, imax);
   data = hypre_StructVectorData(vector);
   hypre_BoxGetSize(box, loop_size);

   hypre_BoxLoop1Begin(loop_size,
                       box, imin, imin, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
   hypre_BoxLoop1For(loopi, loopj, loopk, datai)
      {
         data[datai] = 0.0;
      }
   hypre_BoxLoop1End(datai);

   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorGetMigrateCommPkg
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
                       hypre_StructVectorDataSpace(to_vector), 1,
                       hypre_StructVectorComm(from_vector), &comm_pkg);
   /* is this correct for periodic? */

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorMigrate
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorMigrate( hypre_CommPkg      *comm_pkg,
                           hypre_StructVector *from_vector,
                           hypre_StructVector *to_vector   )
{
   hypre_CommHandle      *comm_handle;

   int                    ierr = 0;

   /*-----------------------------------------------------------------------
    * Migrate the vector data
    *-----------------------------------------------------------------------*/
 
   hypre_InitializeCommunication(comm_pkg,
                                 hypre_StructVectorData(from_vector),
                                 hypre_StructVectorData(to_vector),
                                 &comm_handle);
   hypre_FinalizeCommunication(comm_handle);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorPrint
 *--------------------------------------------------------------------------*/

int
hypre_StructVectorPrint( const char         *filename,
                         hypre_StructVector *vector,
                         int                 all      )
{
   int                ierr = 0;

   FILE              *file;
   char               new_filename[255];

   hypre_StructGrid  *grid;
   hypre_BoxArray    *boxes;

   hypre_BoxArray    *data_space;

   int                myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   MPI_Comm_rank(hypre_StructVectorComm(vector), &myid );

   sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   fprintf(file, "StructVector\n");

   /* print grid info */
   fprintf(file, "\nGrid:\n");
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

   fprintf(file, "\nData:\n");
   hypre_PrintBoxArrayData(file, boxes, data_space, 1,
                           hypre_StructVectorData(vector));
 
   /*----------------------------------------
    * Close file
    *----------------------------------------*/
 
   fflush(file);
   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorRead
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_StructVectorRead( MPI_Comm    comm,
                        const char *filename,
                        int        *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];
                      
   hypre_StructVector   *vector;

   hypre_StructGrid     *grid;
   hypre_BoxArray       *boxes;

   hypre_BoxArray       *data_space;

   int                   myid;
 
   /*----------------------------------------
    * Open file
    *----------------------------------------*/

#ifdef HYPRE_USE_PTHREADS
#if MPI_Comm_rank == hypre_thread_MPI_Comm_rank
#undef MPI_Comm_rank
#endif
#endif

   MPI_Comm_rank(comm, &myid );

   sprintf(new_filename, "%s.%05d", filename, myid);
 
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   fscanf(file, "StructVector\n");

   /* read grid info */
   fscanf(file, "\nGrid:\n");
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
 
   fscanf(file, "\nData:\n");
   hypre_ReadBoxArrayData(file, boxes, data_space, 1,
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

int 
hypre_StructVectorMaxValue( hypre_StructVector *vector,
                            double *max_value, int *max_index,
                            hypre_Index max_xyz_index )
/* Input: vector, and pointers to where to put returned data.
   Return value: error flag, 0 means ok.
   Finds the maximum value in a vector, puts it in max_value.
   The corresponding index is put in max_index.
   A hypre_Index corresponding to max_index is put in max_xyz_index.
   We assume that there is only one box to deal with. */
{
   int               ierr = 0;

   int               datai;
   double           *data;

   hypre_Index       imin;
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_Index       unit_stride;

   int               loopi, loopj, loopk, i;
   double maxvalue;
   int maxindex;

   boxes = hypre_StructVectorDataSpace(vector);
   if ( hypre_BoxArraySize(boxes)!=1 ) {
      /* if more than one box, the return system max_xyz_index is too simple
         if needed, fix later */
      ierr = 1;
      return ierr;
   }
   hypre_SetIndex(unit_stride, 1, 1, 1);
   hypre_ForBoxI(i, boxes)
      {
         box  = hypre_BoxArrayBox(boxes, i);
         /*v_data_box =
           hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);*/
         data = hypre_StructVectorBoxData(vector, i);
         hypre_BoxGetSize(box, loop_size);
         hypre_CopyIndex( hypre_BoxIMin(box), imin );

         hypre_BoxLoop1Begin(loop_size,
                             box, imin, unit_stride, datai);
         maxindex = hypre_BoxIndexRank( box, imin );
         maxvalue = data[maxindex];
         hypre_CopyIndex( imin, max_xyz_index );
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, datai)
            {
               if ( data[datai] > maxvalue )
               {
                  maxvalue = data[datai];
                  maxindex = datai;
                  hypre_SetIndex(max_xyz_index, loopi+hypre_IndexX(imin),
                                 loopj+hypre_IndexY(imin),
                                 loopk+hypre_IndexZ(imin) );
               }
            }
         hypre_BoxLoop1End(datai);
      }

   *max_value = maxvalue;
   *max_index = maxindex;

   return ierr;
}

