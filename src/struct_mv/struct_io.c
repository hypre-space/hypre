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
                         HYPRE_Int        ndim,
                         hypre_BoxArray  *box_array,
                         hypre_BoxArray  *data_space,
                         HYPRE_Int        num_values,
                         HYPRE_Int       *value_ids,
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
   hypre_fprintf(file, "%d\n", num_values);
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
 * hypre_ReadBoxArrayData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ReadBoxArrayData( FILE             *file,
                        HYPRE_Int         ndim,
                        hypre_BoxArray   *box_array,
                        HYPRE_Int        *num_values_ptr,
                        HYPRE_Int       **value_ids_ptr,
                        HYPRE_Complex   **values_ptr     )
{
   HYPRE_Int        num_values;
   HYPRE_Int       *value_ids;
   HYPRE_Complex   *values;

   hypre_Box       *box;

   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      stride;

   HYPRE_Complex    value;
#ifdef HYPRE_COMPLEX
   HYPRE_Complex    rvalue, ivalue;
#endif
   HYPRE_Int        i, j, vi, d, idummy;

   /* Read data on the host */
   hypre_fscanf(file, "%d\n", &num_values);
   hypre_SetIndex(stride, 1);
   value_ids = hypre_TAlloc(HYPRE_Int, num_values, HYPRE_MEMORY_HOST);
   values    = hypre_TAlloc(HYPRE_Complex, num_values * hypre_BoxArrayVolume(box_array),
                            HYPRE_MEMORY_HOST);
   vi = 0;
   hypre_ForBoxI(i, box_array)
   {
      box = hypre_BoxArrayBox(box_array, i);
      start = hypre_BoxIMin(box);
      hypre_BoxGetSize(box, loop_size);
      hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                box, start, stride, bi);
      {
         /* Read lines of the form: "%d: (%d, %d, %d; %d) %le\n" */
         for (j = 0; j < num_values; j++)
         {
            hypre_fscanf(file, "%d: (%d", &idummy, &idummy);
            for (d = 1; d < ndim; d++)
            {
               hypre_fscanf(file, ", %d", &idummy);
            }
#ifdef HYPRE_COMPLEX
            hypre_fscanf(file, "; %d) %le , %le\n", &value_ids[j], &rvalue, &ivalue);
            value = rvalue + I*ivalue;
#else
            hypre_fscanf(file, "; %d) %le\n", &value_ids[j], &value);
#endif
            values[vi + bi] = value;
         }
      }
      hypre_SerialBoxLoop1End(bi);

      vi += num_values * hypre_BoxVolume(box);
   }

   *num_values_ptr = num_values;
   *value_ids_ptr  = value_ids;
   *values_ptr     = values;

   return hypre_error_flag;
}

