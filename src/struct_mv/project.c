/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * Snap 'index' in the positive direction to the nearest point in the strided
 * index space that contains index 'origin' and has stride 'stride'.
 *
 * If 'origin' is NULL, a zero origin is used.
 *
 * This is equivalent to:
 *    index = ceil( (index - origin) / stride ) * stride
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SnapIndexPos( hypre_Index    index,
                    hypre_IndexRef origin,
                    hypre_Index    stride,
                    HYPRE_Int      ndim )
{
   HYPRE_Int  d, s;

   for (d = 0; d < ndim; d++)
   {
      if (origin != NULL)
      {
         s = (index[d] - origin[d]) % stride[d];
      }
      else
      {
         s = index[d] % stride[d];
      }
      if (s > 0)
      {
         index[d] += -s + stride[d];
      }
      else if (s < 0)
      {
         index[d] += -s;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Snap 'index' in the negative direction to the nearest point in the strided
 * index space that contains index 'origin' and has stride 'stride'.
 *
 * If 'origin' is NULL, a zero origin is used.
 *
 * This is equivalent to:
 *    index = floor( (index - origin) / stride ) * stride
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SnapIndexNeg( hypre_Index    index,
                    hypre_IndexRef origin,
                    hypre_Index    stride,
                    HYPRE_Int      ndim )
{
   HYPRE_Int  d, s;

   for (d = 0; d < ndim; d++)
   {
      if (origin != NULL)
      {
         s = (index[d] - origin[d]) % stride[d];
      }
      else
      {
         s = index[d] % stride[d];
      }
      if (s > 0)
      {
         index[d] += -s;
      }
      else if (s < 0)
      {
         index[d] += -s - stride[d];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Convert 'index' to a canonical index in the interval [0, 'stride').
 *
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ConvertToCanonicalIndex( hypre_Index    index,
                               hypre_Index    stride,
                               HYPRE_Int      ndim )
{
   HYPRE_Int  d;

   for (d = 0; d < ndim; d++)
   {
      index[d] %= stride[d];

      if (index[d] < 0)
      {
         index[d] += stride[d];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Projects a box onto a strided index space that contains the index 'origin'
 * and has stride 'stride'.
 *
 * An "empty" projection is represented by a box with volume 0.
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ProjectBox( hypre_Box      *box,
                  hypre_IndexRef  origin,
                  hypre_Index     stride )
{
   HYPRE_Int  ndim = hypre_BoxNDim(box);

   hypre_SnapIndexPos(hypre_BoxIMin(box), origin, stride, ndim);
   hypre_SnapIndexNeg(hypre_BoxIMax(box), origin, stride, ndim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * The dimensions of the modified box array are not changed.
 * It is possible to have boxes with volume 0.
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ProjectBoxArray( hypre_BoxArray  *box_array,
                       hypre_IndexRef   origin,
                       hypre_Index      stride )
{
   hypre_Box  *box;
   HYPRE_Int   i;

   hypre_ForBoxI(i, box_array)
   {
      box = hypre_BoxArrayBox(box_array, i);
      hypre_ProjectBox(box, origin, stride);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * The dimensions of the modified box array-array are not changed.
 * It is possible to have boxes with volume 0.
 * If 'origin' is NULL, a zero origin is used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ProjectBoxArrayArray( hypre_BoxArrayArray  *box_array_array,
                            hypre_IndexRef        origin,
                            hypre_Index           stride )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;
   HYPRE_Int        i, j;

   hypre_ForBoxArrayI(i, box_array_array)
   {
      box_array = hypre_BoxArrayArrayBoxArray(box_array_array, i);
      hypre_ForBoxI(j, box_array)
      {
         box = hypre_BoxArrayBox(box_array, j);
         hypre_ProjectBox(box, origin, stride);
      }
   }

   return hypre_error_flag;
}

