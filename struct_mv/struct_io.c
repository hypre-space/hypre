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

   double          *data_ptr;
   int              data_box_volume;
   int              datai;

   zzz_Index       *index;
   zzz_Index       *stride;

   int              i, j, d;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   index = zzz_NewIndex();
   stride = zzz_NewIndex();
   for (d = 0; d < 3; d++)
      zzz_IndexD(stride, d) = 1;

   zzz_ForBoxI(i, box_array)
   {
      box      = zzz_BoxArrayBox(box_array, i);
      data_box = zzz_BoxArrayBox(data_space, i);

      data_box_volume = zzz_BoxVolume(data_box);

      zzz_BoxLoop1(box, index,
                   data_box, zzz_BoxIMin(box), stride, datai,
                   {
                      for (j = 0; j < num_values; j++)
                      {
                         fprintf(file, "%d: (%d, %d, %d; %d) %e\n", i,
                                 zzz_IndexX(index),
                                 zzz_IndexY(index),
                                 zzz_IndexZ(index),
                                 j, data[datai + j*data_box_volume]);
                      }
                   });

      data += num_values*data_box_volume;
   }

   zzz_FreeIndex(index);
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

   double          *data_ptr;
   int              data_box_volume;
   int              datai;

   zzz_Index       *index;
   zzz_Index       *stride;

   int              i, j, d, idummy;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   index = zzz_NewIndex();
   stride = zzz_NewIndex();
   for (d = 0; d < 3; d++)
      zzz_IndexD(stride, d) = 1;

   zzz_ForBoxI(i, box_array)
   {
      box      = zzz_BoxArrayBox(box_array, i);
      data_box = zzz_BoxArrayBox(data_space, i);

      data_box_volume = zzz_BoxVolume(data_box);

      zzz_BoxLoop1(box, index,
                   data_box, zzz_BoxIMin(box), stride, datai,
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

   zzz_FreeIndex(index);
   zzz_FreeIndex(stride);
}
