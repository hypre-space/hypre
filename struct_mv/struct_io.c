/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
		  fprintf(file, "%d: (%d, %d, %d; %d) %.14e\n",
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

int
hypre_PrintCCVDBoxArrayData( FILE            *file,
                             hypre_BoxArray  *box_array,
                             hypre_BoxArray  *data_space,
                             int              num_values,
                             int              center_rank,
                             int              stencil_size,
                             int             *symm_elements,
                             double          *data       )
{
   int              ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   int              data_box_volume, datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   int              i, j;
   int              loopi, loopj, loopk;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   /* First is the constant, off-diagonal, part of the matrix: */
   for (  j=0; j<stencil_size; ++j )
   {
      if (symm_elements[j] < 0 && j!=center_rank )
      {
         fprintf( file, "*: (*, *, *; %d) %.14e\n",
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
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, datai)
            {
               fprintf(file, "%d: (%d, %d, %d; %d) %.14e\n",
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

int
hypre_PrintCCBoxArrayData( FILE            *file,
                         hypre_BoxArray  *box_array,
                         hypre_BoxArray  *data_space,
                         int              num_values,
                         double          *data       )
{
   int              ierr = 0;

   hypre_Box       *box;
                   
   int              datai;
                   
   hypre_IndexRef   start;
                   
   int              i, j;

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
         fprintf( file, "*: (*, *, *; %d) %.14e\n",
                  j, data[datai + j] );
         }

         data += num_values;
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ReadBoxArrayData  (for non-constant coefficients)
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


/*--------------------------------------------------------------------------
 * hypre_ReadBoxArrayData_CC  (for when there are some constant coefficients)
 *--------------------------------------------------------------------------*/

int
hypre_ReadBoxArrayData_CC( FILE            *file,
                           hypre_BoxArray  *box_array,
                           hypre_BoxArray  *data_space,
                           int              stencil_size,
                           int              real_stencil_size,
                           int              constant_coefficient,
                           double          *data       )
{
   int              ierr = 0;

   hypre_Box       *box;
   hypre_Box       *data_box;
                   
   int              data_box_volume, constant_stencil_size;
   int              datai;
                   
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
                   
   int              i, j, idummy;
   int              loopi, loopj, loopk;

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
            fscanf(file, "*: (*, *, *; %d) %le\n",
                   &idummy,
                   &data[j]);
         }

         /* Next entries, if any, will be for a variable diagonal: */
         data += real_stencil_size;

         if ( constant_coefficient==2 )
         {
            hypre_BoxLoop1Begin(loop_size,
                                data_box, start, stride, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop1For(loopi, loopj, loopk, datai)
               {
                  fscanf(file, "%d: (%d, %d, %d; %d) %le\n",
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

