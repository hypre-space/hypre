/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * SMG axpy routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SMGAxpy
 *--------------------------------------------------------------------------*/

int
hypre_SMGAxpy( double              alpha,
               hypre_StructVector *x,
               hypre_StructVector *y,
               hypre_Index         base_index,
               hypre_Index         base_stride )
{
   int ierr = 0;

   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;
                 
   int               xi;
   int               yi;
                    
   double           *xp;
   double           *yp;
                    
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_IndexRef    start;
                    
   int               i;
   int               loopi, loopj, loopk;

   box = hypre_BoxCreate();
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), box);
         hypre_ProjectBox(box, base_index, base_stride);
         start = hypre_BoxIMin(box);

         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxGetStrideSize(box, base_stride, loop_size);
         hypre_BoxLoop2Begin(loop_size,
                             x_data_box, start, base_stride, xi,
                             y_data_box, start, base_stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi,yi
#include "hypre_box_smp_forloop.h"
	 hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
            {
               yp[yi] += alpha * xp[xi];
            }
         hypre_BoxLoop2End(xi, yi);
      }
   hypre_BoxDestroy(box);

   return ierr;
}
