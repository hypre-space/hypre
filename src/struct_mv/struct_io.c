/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
   HYPRE_Complex   *data_host;
   HYPRE_Complex   *data_host_saved = NULL;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/
   if (hypre_GetActualMemLocation(HYPRE_MEMORY_DEVICE) != hypre_MEMORY_HOST)
   {
      HYPRE_Int tot_size = 0;
      hypre_ForBoxI(i, data_space)
      {
         data_box = hypre_BoxArrayBox(data_space, i);
         data_box_volume = hypre_BoxVolume(data_box);
         tot_size += num_values * data_box_volume;
      }
      data_host = hypre_CTAlloc(HYPRE_Complex, tot_size, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(data_host, data, HYPRE_Complex, tot_size, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      data_host_saved = data_host;
   }
   else
   {
      data_host = data;
   }

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
         hypre_BoxLoopGetIndex(index);
         hypre_AddIndexes(index, start, ndim, index); /* shift by start */
         for (j = 0; j < num_values; j++)
         {
            hypre_fprintf(file, "%d: (%d", i, hypre_IndexD(index, 0));
            for (d = 1; d < ndim; d++)
            {
               hypre_fprintf(file, ", %d", hypre_IndexD(index, d));
            }
            value = data_host[datai + j*data_box_volume];
#ifdef HYPRE_COMPLEX
            hypre_fprintf(file, "; %d) %.14e , %.14e\n",
                          value_ids[j], hypre_creal(value), hypre_cimag(value));
#else
            hypre_fprintf(file, "; %d) %.14e\n", value_ids[j], value);
#endif
         }
      }
      hypre_SerialBoxLoop1End(datai);

      data_host += num_values*data_box_volume;
   }

   hypre_TFree(data_host_saved, HYPRE_MEMORY_HOST);

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

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

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
                         &idummy, &data[datai + j*data_box_volume]);
         }
      }
      hypre_SerialBoxLoop1End(datai);

      data += num_values*data_box_volume;
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

   HYPRE_Int        data_box_volume, constant_stencil_size;

   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;

   HYPRE_Int        i, j, d, idummy;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   if ( constant_coefficient==1 ) constant_stencil_size = stencil_size;
   if ( constant_coefficient==2 ) constant_stencil_size = stencil_size - 1;

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
      for (j=0; j <constant_stencil_size; j++)
      {
         hypre_fscanf(file, "*: (*, *, *; %d) %le\n", &idummy, &data[j]);
      }

      /* Next entries, if any, will be for a variable diagonal: */
      data += real_stencil_size;

      if ( constant_coefficient==2 )
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

