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

#ifdef HYPRE_USE_PTHREADS
#undef HYPRE_USE_PTHREADS
#endif

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_PrintBoxArrayData
 *--------------------------------------------------------------------------*/

int
hypre_PrintBoxArrayData( FILE            *file,
                         hypre_BoxArray  *box_array,
                         hypre_BoxArray  *data_space,
                         int              num_values,
                         double          *data       )
{
   int              ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   int              data_box_volume;
   int              datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   int              i, j;
   int              loopi, loopj, loopk;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   hypre_ForBoxI(i, box_array)
      {
         box      = hypre_BoxArrayBox(box_array, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         start = hypre_BoxIMin(box);
         data_box_volume = hypre_BoxVolume(data_box);

         hypre_BoxGetSize(box, loop_size);

	 hypre_BoxLoop1Begin(loop_size,
                             data_box, start, stride, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
	 hypre_BoxLoop1For(loopi, loopj, loopk, datai)
            {
               for (j = 0; j < num_values; j++)
               {
		  fprintf(file, "%d: (%d, %d, %d; %d) %e\n",
                          i,
                          hypre_IndexX(start) + loopi,
                          hypre_IndexY(start) + loopj,
                          hypre_IndexZ(start) + loopk,
                          j,
                          data[datai + j*data_box_volume]);
               }
            }
         hypre_BoxLoop1End(datai);

         data += num_values*data_box_volume;
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ReadBoxArrayData
 *--------------------------------------------------------------------------*/

int
hypre_ReadBoxArrayData( FILE            *file,
                        hypre_BoxArray  *box_array,
                        hypre_BoxArray  *data_space,
                        int              num_values,
                        double          *data       )
{
   int              ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   int              data_box_volume;
   int              datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   int              i, j, idummy;
   int              loopi, loopj, loopk;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   hypre_ForBoxI(i, box_array)
      {
         box      = hypre_BoxArrayBox(box_array, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         start = hypre_BoxIMin(box);
         data_box_volume = hypre_BoxVolume(data_box);

         hypre_BoxGetSize(box, loop_size);

	 hypre_BoxLoop1Begin(loop_size,
                             data_box, start, stride, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
	 hypre_BoxLoop1For(loopi, loopj, loopk, datai)
            {
               for (j = 0; j < num_values; j++)
               {
                  fscanf(file, "%d: (%d, %d, %d; %d) %le\n",
                         &idummy,
                         &idummy,
                         &idummy,
                         &idummy,
                         &idummy,
                         &data[datai + j*data_box_volume]);
	       }
	   }
         hypre_BoxLoop1End(datai);

         data += num_values*data_box_volume;
      }

   return ierr;
}
