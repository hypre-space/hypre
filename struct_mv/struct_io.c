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

   zzz_Index        loop_size;
   zzz_IndexRef     start;
   zzz_Index        stride;

   int              i, j;
   int              loopi, loopj, loopk;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   zzz_SetIndex(stride, 1, 1, 1);

   zzz_ForBoxI(i, box_array)
   {
      box      = zzz_BoxArrayBox(box_array, i);
      data_box = zzz_BoxArrayBox(data_space, i);

      start = zzz_BoxIMin(box);
      data_box_volume = zzz_BoxVolume(data_box);

      zzz_GetBoxSize(box, loop_size);
      zzz_BoxLoop1(loopi, loopj, loopk, loop_size,
                   data_box, start, stride, datai,
                   {
                      for (j = 0; j < num_values; j++)
                      {
                         fprintf(file, "%d: (%d, %d, %d; %d) %e\n", i,
                                 zzz_IndexX(start) + loopi,
                                 zzz_IndexY(start) + loopj,
                                 zzz_IndexZ(start) + loopk,
                                 j, data[datai + j*data_box_volume]);
                      }
                   });

      data += num_values*data_box_volume;
   }
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

   zzz_Index        loop_size;
   zzz_IndexRef     start;
   zzz_Index        stride;

   int              i, j, idummy;
   int              loopi, loopj, loopk;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   zzz_SetIndex(stride, 1, 1, 1);

   zzz_ForBoxI(i, box_array)
   {
      box      = zzz_BoxArrayBox(box_array, i);
      data_box = zzz_BoxArrayBox(data_space, i);

      start = zzz_BoxIMin(box);
      data_box_volume = zzz_BoxVolume(data_box);

      zzz_GetBoxSize(box, loop_size);
      zzz_BoxLoop1(loopi, loopj, loopk, loop_size,
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
}
