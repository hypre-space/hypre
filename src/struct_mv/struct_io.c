/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/



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

HYPRE_Int
hypre_PrintBoxArrayData( FILE            *file,
                         hypre_BoxArray  *box_array,
                         hypre_BoxArray  *data_space,
                         HYPRE_Int        num_values,
                         double          *data       )
{
   HYPRE_Int        ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   HYPRE_Int        data_box_volume;
   HYPRE_Int        datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   HYPRE_Int        i, j;
   HYPRE_Int        loopi, loopj, loopk;

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
	 hypre_BoxLoop1For(loopi, loopj, loopk, datai)
            {
               for (j = 0; j < num_values; j++)
               {
		  hypre_fprintf(file, "%d: (%d, %d, %d; %d) %.14e\n",
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
 * hypre_PrintCCVDBoxArrayData
 * Note that the the stencil loop (j) is _outside_ the space index loop (datai),
 * unlie hypre_PrintBoxArrayData (there is no j loop in hypre_PrintCCBoxArrayData)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PrintCCVDBoxArrayData( FILE            *file,
                             hypre_BoxArray  *box_array,
                             hypre_BoxArray  *data_space,
                             HYPRE_Int        num_values,
                             HYPRE_Int        center_rank,
                             HYPRE_Int        stencil_size,
                             HYPRE_Int       *symm_elements,
                             double          *data       )
{
   HYPRE_Int        ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   HYPRE_Int        data_box_volume, datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   HYPRE_Int        i, j;
   HYPRE_Int        loopi, loopj, loopk;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   /* First is the constant, off-diagonal, part of the matrix: */
   for (  j=0; j<stencil_size; ++j )
   {
      if (symm_elements[j] < 0 && j!=center_rank )
      {
         hypre_fprintf( file, "*: (*, *, *; %d) %.14e\n",
                  j, data[0] );
      }
      ++data;
   }
   

   /* Then each box has a variable, diagonal, part of the matrix: */
   hypre_ForBoxI(i, box_array)
      {
         box      = hypre_BoxArrayBox(box_array, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         start = hypre_BoxIMin(box);
         data_box_volume = hypre_BoxVolume(data_box);

         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             data_box, start, stride, datai);
         hypre_BoxLoop1For(loopi, loopj, loopk, datai)
            {
               hypre_fprintf(file, "%d: (%d, %d, %d; %d) %.14e\n",
                       i,
                       hypre_IndexX(start) + loopi,
                       hypre_IndexY(start) + loopj,
                       hypre_IndexZ(start) + loopk,
                       center_rank,
                       data[datai]);
            }
         hypre_BoxLoop1End(datai);
         data += data_box_volume;
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PrintCCBoxArrayData
 * same as hypre_PrintBoxArrayData but for constant coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PrintCCBoxArrayData( FILE            *file,
                         hypre_BoxArray  *box_array,
                         hypre_BoxArray  *data_space,
                         HYPRE_Int        num_values,
                         double          *data       )
{
   HYPRE_Int        ierr = 0;

   hypre_Box       *box;
                   
   HYPRE_Int        datai;
                   
   hypre_IndexRef   start;
                   
   HYPRE_Int        i, j;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   hypre_ForBoxI(i, box_array)
      {
         box      = hypre_BoxArrayBox(box_array, i);

         start = hypre_BoxIMin(box);

         datai = hypre_CCBoxIndexRank_noargs();

         for (j = 0; j < num_values; j++)
         {
         hypre_fprintf( file, "*: (*, *, *; %d) %.14e\n",
                  j, data[datai + j] );
         }

         data += num_values;
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ReadBoxArrayData  (for non-constant coefficients)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ReadBoxArrayData( FILE            *file,
                        hypre_BoxArray  *box_array,
                        hypre_BoxArray  *data_space,
                        HYPRE_Int        num_values,
                        double          *data       )
{
   HYPRE_Int        ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   HYPRE_Int        data_box_volume;
   HYPRE_Int        datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   HYPRE_Int        i, j, idummy;
   HYPRE_Int        loopi, loopj, loopk;

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
	 hypre_BoxLoop1For(loopi, loopj, loopk, datai)
            {
               for (j = 0; j < num_values; j++)
               {
                  hypre_fscanf(file, "%d: (%d, %d, %d; %d) %le\n",
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


/*--------------------------------------------------------------------------
 * hypre_ReadBoxArrayData_CC  (for when there are some constant coefficients)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ReadBoxArrayData_CC( FILE            *file,
                           hypre_BoxArray  *box_array,
                           hypre_BoxArray  *data_space,
                           HYPRE_Int        stencil_size,
                           HYPRE_Int        real_stencil_size,
                           HYPRE_Int        constant_coefficient,
                           double          *data       )
{
   HYPRE_Int        ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   HYPRE_Int        data_box_volume, constant_stencil_size;
   HYPRE_Int        datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   HYPRE_Int        i, j, idummy;
   HYPRE_Int        loopi, loopj, loopk;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   if ( constant_coefficient==1 ) constant_stencil_size = stencil_size;
   if ( constant_coefficient==2 ) constant_stencil_size = stencil_size - 1;

   hypre_SetIndex(stride, 1, 1, 1);

   hypre_ForBoxI(i, box_array)
      {
         box      = hypre_BoxArrayBox(box_array, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         start = hypre_BoxIMin(box);
         data_box_volume = hypre_BoxVolume(data_box);

         hypre_BoxGetSize(box, loop_size);

         /* First entries will be the constant part of the matrix.
            There is one entry for each constant stencil element,
            excluding ones which are redundant due to symmetry.*/
         for (j=0; j <constant_stencil_size; j++)
         {
            hypre_fscanf(file, "*: (*, *, *; %d) %le\n",
                   &idummy,
                   &data[j]);
         }

         /* Next entries, if any, will be for a variable diagonal: */
         data += real_stencil_size;

         if ( constant_coefficient==2 )
         {
            hypre_BoxLoop1Begin(loop_size,
                                data_box, start, stride, datai);
            hypre_BoxLoop1For(loopi, loopj, loopk, datai)
               {
                  hypre_fscanf(file, "%d: (%d, %d, %d; %d) %le\n",
                         &idummy,
                         &idummy,
                         &idummy,
                         &idummy,
                         &idummy,
                         &data[datai]);
               }
            hypre_BoxLoop1End(datai);
            data += data_box_volume;
         }

      }

   return ierr;
}

