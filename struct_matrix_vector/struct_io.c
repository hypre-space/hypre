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
 * Functions for scanning and printing "box-dimensioned" data.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_PrintBoxArrayData
 *--------------------------------------------------------------------------*/

void
zzz_PrintBoxArrayData( FILE             *file,
                       zzz_BoxArray     *box_array,
                       zzz_BoxArray     *data_space,
                       int               num_values,
                       double           *data       )
{
   zzz_Box         *box;
   zzz_Box         *data_box;

   int              data_box_volume;
   int              datai;

   zzz_Index       *loop_index;
   zzz_Index       *loop_size;
   zzz_Index       *start;
   zzz_Index       *stride;

   int              i, j;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   loop_index = zzz_NewIndex();
   loop_size  = zzz_NewIndex();
   stride = zzz_NewIndex();
   zzz_SetIndex(stride, 1, 1, 1);

   zzz_ForBoxI(i, box_array)
   {
      box      = zzz_BoxArrayBox(box_array, i);
      data_box = zzz_BoxArrayBox(data_space, i);

      start = zzz_BoxIMin(box);
      data_box_volume = zzz_BoxVolume(data_box);

      zzz_GetBoxSize(box, loop_size);
      zzz_BoxLoop1(loop_index, loop_size,
                   data_box, start, stride, datai,
                   {
                      for (j = 0; j < num_values; j++)
                      {
                         fprintf(file, "%d: (%d, %d, %d; %d) %e\n", i,
                                 zzz_IndexX(start) + zzz_IndexX(loop_index),
                                 zzz_IndexY(start) + zzz_IndexY(loop_index),
                                 zzz_IndexZ(start) + zzz_IndexZ(loop_index),
                                 j, data[datai + j*data_box_volume]);
                      }
                   });

      data += num_values*data_box_volume;
   }

   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);
   zzz_FreeIndex(stride);
}

/*--------------------------------------------------------------------------
 * zzz_ReadBoxArrayData
 *--------------------------------------------------------------------------*/

void
zzz_ReadBoxArrayData( FILE             *file,
                      zzz_BoxArray     *box_array,
                      zzz_BoxArray     *data_space,
                      int               num_values,
                      double           *data       )
{
   zzz_Box         *box;
   zzz_Box         *data_box;

   int              data_box_volume;
   int              datai;

   zzz_Index       *loop_index;
   zzz_Index       *loop_size;
   zzz_Index       *start;
   zzz_Index       *stride;

   int              i, j, idummy;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   loop_index = zzz_NewIndex();
   loop_size  = zzz_NewIndex();
   stride = zzz_NewIndex();
   zzz_SetIndex(stride, 1, 1, 1);

   zzz_ForBoxI(i, box_array)
   {
      box      = zzz_BoxArrayBox(box_array, i);
      data_box = zzz_BoxArrayBox(data_space, i);

      start = zzz_BoxIMin(box);
      data_box_volume = zzz_BoxVolume(data_box);

      zzz_GetBoxSize(box, loop_size);
      zzz_BoxLoop1(loop_index, loop_size,
                   data_box, start, stride, datai,
                   {
                      for (j = 0; j < num_values; j++)
                      {
                         fscanf(file, "%d: (%d, %d, %d; %d) %le\n", &idummy,
                                &idummy,
                                &idummy,
                                &idummy,
                                &idummy, &data[datai + j*data_box_volume]);
                      }
                   });

      data += num_values*data_box_volume;
   }

   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);
   zzz_FreeIndex(stride);
}
