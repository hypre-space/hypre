/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Functions for scanning and printing "box-dimensioned" data.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_PrintBoxArrayData
 *
 * Note: data array is expected to live on the host memory.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PrintBoxArrayData( FILE            *file,
                         hypre_BoxArray  *box_array,
                         hypre_BoxArray  *data_space,
                         HYPRE_Int        num_values,
                         HYPRE_Int       *value_ids,
                         HYPRE_Int        ndim,
                         HYPRE_Complex   *data       )
{
   hypre_Box       *box;
   hypre_Box       *data_box;

   HYPRE_Int        data_box_volume;

   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;
   hypre_Index      index;

   HYPRE_Int        i, j, d;
   HYPRE_Complex    value;

   /* Print data from the host */
   hypre_SetIndex(stride, 1);
   hypre_ForBoxI(i, box_array)
   {
      box      = hypre_BoxArrayBox(box_array, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      start = hypre_BoxIMin(box);
      data_box_volume = hypre_BoxVolume(data_box);

      hypre_BoxGetSize(box, loop_size);

      hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                data_box, start, stride, datai);
      {
         /* Print lines of the form: "%d: (%d, %d, %d; %d) %.14e\n" */
         zypre_BoxLoopGetIndex(index);
         for (j = 0; j < num_values; j++)
         {
            hypre_fprintf(file, "%d: (%d", i, hypre_IndexD(start, 0) + hypre_IndexD(index, 0));
            for (d = 1; d < ndim; d++)
            {
               hypre_fprintf(file, ", %d", hypre_IndexD(start, d) + hypre_IndexD(index, d));
            }
            value = data[datai + j * data_box_volume];

            /* Make zero values "positive" */
            if (value == 0.0)
            {
               value = 0.0;
            }
#ifdef HYPRE_COMPLEX
            hypre_fprintf(file, "; %d) %.14e , %.14e\n",
                          value_ids[j], hypre_creal(value), hypre_cimag(value));
#else
            hypre_fprintf(file, "; %d) %.14e\n", value_ids[j], value);
#endif
         }
      }
      hypre_SerialBoxLoop1End(datai);

      data += num_values * data_box_volume;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ReadBoxArrayData  (for non-constant coefficients)
 *
 * RDF TODO
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ReadBoxArrayData( FILE            *file,
                        hypre_BoxArray  *box_array,
                        hypre_BoxArray  *data_space,
                        HYPRE_Int        num_values,
                        HYPRE_Int        ndim,
                        HYPRE_Complex   *data       )
{
   hypre_Box       *box;
   hypre_Box       *data_box;

   HYPRE_Int        data_box_volume;

   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;

   HYPRE_Int        i, j, d, idummy;

   /* Read data on the host */
   hypre_SetIndex(stride, 1);
   hypre_ForBoxI(i, box_array)
   {
      box      = hypre_BoxArrayBox(box_array, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      start = hypre_BoxIMin(box);
      data_box_volume = hypre_BoxVolume(data_box);

      hypre_BoxGetSize(box, loop_size);

      hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                data_box, start, stride, datai);
      {
         /* Read lines of the form: "%d: (%d, %d, %d; %d) %le\n" */
         for (j = 0; j < num_values; j++)
         {
            hypre_fscanf(file, "%d: (%d", &idummy, &idummy);
            for (d = 1; d < ndim; d++)
            {
               hypre_fscanf(file, ", %d", &idummy);
            }
            hypre_fscanf(file, "; %d) %le\n",
                         &idummy, &data[datai + j * data_box_volume]);
         }
      }
      hypre_SerialBoxLoop1End(datai);

      data += num_values * data_box_volume;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ReadBoxArrayData_CC  (for when there are some constant coefficients)
 *
 * RDF TODO: Get rid of this routine entirely
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ReadBoxArrayData_CC( FILE            *file,
                           hypre_BoxArray  *box_array,
                           hypre_BoxArray  *data_space,
                           HYPRE_Int        stencil_size,
                           HYPRE_Int        real_stencil_size,
                           HYPRE_Int        constant_coefficient,
                           HYPRE_Int        ndim,
                           HYPRE_Complex   *data       )
{
   hypre_Box       *box;
   hypre_Box       *data_box;

   HYPRE_Int        data_box_volume;
   HYPRE_Int        constant_stencil_size;

   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;

   HYPRE_Int        i, j, d, idummy;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   switch (constant_coefficient)
   {
      case 1:
         constant_stencil_size = stencil_size;
         break;

      case 2:
         constant_stencil_size = stencil_size - 1;
         break;

      default:
         constant_stencil_size = 0;
         break;
   }

   hypre_SetIndex(stride, 1);
   hypre_ForBoxI(i, box_array)
   {
      box      = hypre_BoxArrayBox(box_array, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      start = hypre_BoxIMin(box);
      data_box_volume = hypre_BoxVolume(data_box);

      hypre_BoxGetSize(box, loop_size);

      /* First entries will be the constant part of the matrix.
         There is one value for each constant stencil entry,
         excluding ones which are redundant due to symmetry.*/
      for (j = 0; j < constant_stencil_size; j++)
      {
         hypre_fscanf(file, "*: (*, *, *; %d) %le\n", &idummy, &data[j]);
      }

      /* Next entries, if any, will be for a variable diagonal: */
      data += real_stencil_size;

      if (constant_coefficient == 2)
      {
         hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                   data_box, start, stride, datai);
         {
            /* Read line of the form: "%d: (%d, %d, %d; %d) %.14e\n" */
            hypre_fscanf(file, "%d: (%d", &idummy, &idummy);
            for (d = 1; d < ndim; d++)
            {
               hypre_fscanf(file, ", %d", &idummy);
            }
            hypre_fscanf(file, "; %d) %le\n", &idummy, &data[datai]);
         }
         hypre_SerialBoxLoop1End(datai);
         data += data_box_volume;
      }
   }

   return hypre_error_flag;
}
