/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CopyBoxArrayData
 *  This function assumes only one box in box_array_in and
 *  that box_array_out consists of a sub_grid to that in box_array_in.
 *  This routine then copies data values from box_array_in to box_array_out.
 *  Author: pnb, 12-16-97
 *--------------------------------------------------------------------------*/

void
hypre_CopyBoxArrayData( hypre_BoxArray *box_array_in,
                        hypre_BoxArray *data_space_in,
                        int             num_values_in,
                        double         *data_in,
                        hypre_BoxArray *box_array_out,
                        hypre_BoxArray *data_space_out,
                        int             num_values_out,
                        double         *data_out       )
{
   hypre_Box    *box_in, *box_out;
   hypre_Box    *data_box_in, *data_box_out;
                
   int           data_box_volume_in, data_box_volume_out;
   int           datai_in, datai_out;
                
   hypre_Index   loop_size;
   hypre_Index   stride;
                
   int           j;
   int           loopi, loopj, loopk;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   box_in      = hypre_BoxArrayBox(box_array_in, 0);
   data_box_in = hypre_BoxArrayBox(data_space_in, 0);
   
   data_box_volume_in = hypre_BoxVolume(data_box_in);
   
   box_out      = hypre_BoxArrayBox(box_array_out, 0);
   data_box_out = hypre_BoxArrayBox(data_space_out, 0);
   
   data_box_volume_out = hypre_BoxVolume(data_box_out);

   hypre_GetBoxSize(box_out, loop_size);
   hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                  data_box_in, hypre_BoxIMin(box_out), stride, datai_in,
                  data_box_out, hypre_BoxIMin(box_out), stride, datai_out,
                  for (j = 0; j < num_values_out; j++)
                  {
                     data_out[datai_out + j*data_box_volume_out] =
                        data_in[datai_in + j*data_box_volume_in];
                  });
}

