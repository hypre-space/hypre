/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_StructMatrix class.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * hypre_StructMatrixExtractPointerByIndex
 *    Returns pointer to data for stencil entry coresponding to
 *    `index' in `matrix'. If the index does not exist in the matrix's
 *    stencil, the NULL pointer is returned.
 *--------------------------------------------------------------------------*/

HYPRE_Complex *
hypre_StructMatrixExtractPointerByIndex( hypre_StructMatrix *matrix,
                                         HYPRE_Int           b,
                                         hypre_Index         index  )
{
   hypre_StructStencil   *stencil;
   HYPRE_Int              rank;

   stencil = hypre_StructMatrixStencil(matrix);
   rank = hypre_StructStencilElementRank( stencil, index );

   if ( rank >= 0 )
   {
      return hypre_StructMatrixBoxData(matrix, b, rank);
   }
   else
   {
      return NULL;  /* error - invalid index */
   }
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixCreate( MPI_Comm             comm,
                          hypre_StructGrid    *grid,
                          hypre_StructStencil *user_stencil )
{
   HYPRE_Int            ndim = hypre_StructGridNDim(grid);
   hypre_StructMatrix  *matrix;
   HYPRE_Int            i;

   matrix = hypre_CTAlloc(hypre_StructMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_StructMatrixComm(matrix)        = comm;
   hypre_StructGridRef(grid, &hypre_StructMatrixGrid(matrix));
   hypre_StructMatrixUserStencil(matrix) = hypre_StructStencilRef(user_stencil);
   hypre_StructMatrixDataAlloced(matrix) = 1;
   hypre_StructMatrixRefCount(matrix)    = 1;

   /* set defaults */
   hypre_StructMatrixSymmetric(matrix) = 0;
   hypre_StructMatrixConstantCoefficient(matrix) = 0;
   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructMatrixNumGhost(matrix)[i] = hypre_StructGridNumGhost(grid)[i];
   }

   hypre_StructMatrixMemoryLocation(matrix) = hypre_HandleMemoryLocation(hypre_handle());

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixRef
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixRef( hypre_StructMatrix *matrix )
{
   hypre_StructMatrixRefCount(matrix) ++;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixDestroy( hypre_StructMatrix *matrix )
{
   if (matrix)
   {
      hypre_StructMatrixRefCount(matrix) --;
      if (hypre_StructMatrixRefCount(matrix) == 0)
      {
         if (hypre_StructMatrixDataAlloced(matrix))
         {
            hypre_TFree(hypre_StructMatrixData(matrix), hypre_StructMatrixMemoryLocation(matrix));
            hypre_TFree(hypre_StructMatrixDataConst(matrix), HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_StructMatrixStencilData(matrix), HYPRE_MEMORY_HOST);
         hypre_CommPkgDestroy(hypre_StructMatrixCommPkg(matrix));
         if (hypre_BoxArraySize(hypre_StructMatrixDataSpace(matrix)) > 0)
         {
            hypre_TFree(hypre_StructMatrixDataIndices(matrix)[0], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_StructMatrixDataIndices(matrix), HYPRE_MEMORY_HOST);
         hypre_BoxArrayDestroy(hypre_StructMatrixDataSpace(matrix));
         hypre_TFree(hypre_StructMatrixSymmElements(matrix), HYPRE_MEMORY_HOST);
         hypre_StructStencilDestroy(hypre_StructMatrixUserStencil(matrix));
         hypre_StructStencilDestroy(hypre_StructMatrixStencil(matrix));
         hypre_StructGridDestroy(hypre_StructMatrixGrid(matrix));
         hypre_TFree(matrix, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixInitializeShell
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixInitializeShell( hypre_StructMatrix *matrix )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(matrix);
   hypre_StructGrid     *grid = hypre_StructMatrixGrid(matrix);

   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size;
   HYPRE_Complex       **stencil_data;
   HYPRE_Int             num_values;
   HYPRE_Int            *symm_elements;
   HYPRE_Int             constant_coefficient;

   HYPRE_Int            *num_ghost;
   HYPRE_Int             extra_ghost[2 * HYPRE_MAXDIM];

   hypre_BoxArray       *data_space;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Box            *data_box;

   HYPRE_Int           **data_indices;
   HYPRE_Int             data_size;
   HYPRE_Int             data_const_size;
   HYPRE_Int             data_box_volume;

   HYPRE_Int             i, j, d;

   /*-----------------------------------------------------------------------
    * Set up stencil and num_values:
    *
    * If the matrix is symmetric, then the stencil is a "symmetrized"
    * version of the user's stencil.  If the matrix is not symmetric,
    * then the stencil is the same as the user's stencil.
    *
    * The `symm_elements' array is used to determine what data is
    * explicitely stored (symm_elements[i] < 0) and what data does is
    * not explicitely stored (symm_elements[i] >= 0), but is instead
    * stored as the transpose coefficient at a neighboring grid point.
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixStencil(matrix) == NULL)
   {
      user_stencil = hypre_StructMatrixUserStencil(matrix);

      if (hypre_StructMatrixSymmetric(matrix))
      {
         /* store only symmetric stencil entry data */
         hypre_StructStencilSymmetrize(user_stencil, &stencil, &symm_elements);
         num_values = ( hypre_StructStencilSize(stencil) + 1 ) / 2;
      }
      else
      {
         /* store all stencil entry data */
         stencil = hypre_StructStencilRef(user_stencil);
         num_values = hypre_StructStencilSize(stencil);
         symm_elements = hypre_TAlloc(HYPRE_Int,  num_values, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_values; i++)
         {
            symm_elements[i] = -1;
         }
      }

      hypre_StructMatrixStencil(matrix)      = stencil;
      hypre_StructMatrixSymmElements(matrix) = symm_elements;
      hypre_StructMatrixNumValues(matrix)    = num_values;
   }

   /*-----------------------------------------------------------------------
    * Set ghost-layer size for symmetric storage
    *   - All stencil coeffs are to be available at each point in the
    *     grid, as well as in the user-specified ghost layer.
    *-----------------------------------------------------------------------*/

   num_ghost     = hypre_StructMatrixNumGhost(matrix);
   stencil       = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   symm_elements = hypre_StructMatrixSymmElements(matrix);

   stencil_data  = hypre_TAlloc(HYPRE_Complex*, stencil_size, HYPRE_MEMORY_HOST);
   hypre_StructMatrixStencilData(matrix) = stencil_data;

   for (d = 0; d < 2 * ndim; d++)
   {
      extra_ghost[d] = 0;
   }

   for (i = 0; i < stencil_size; i++)
   {
      if (symm_elements[i] >= 0)
      {
         for (d = 0; d < ndim; d++)
         {
            extra_ghost[2 * d]     = hypre_max( extra_ghost[2 * d],
                                                -hypre_IndexD(stencil_shape[i], d) );
            extra_ghost[2 * d + 1] = hypre_max( extra_ghost[2 * d + 1],
                                                hypre_IndexD(stencil_shape[i], d) );
         }
      }
   }

   for (d = 0; d < ndim; d++)
   {
      num_ghost[2 * d]     += extra_ghost[2 * d];
      num_ghost[2 * d + 1] += extra_ghost[2 * d + 1];
   }

   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixDataSpace(matrix) == NULL)
   {
      boxes = hypre_StructGridBoxes(grid);
      data_space = hypre_BoxArrayCreate(hypre_BoxArraySize(boxes), ndim);

      hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         hypre_CopyBox(box, data_box);
         for (d = 0; d < ndim; d++)
         {
            hypre_BoxIMinD(data_box, d) -= num_ghost[2 * d];
            hypre_BoxIMaxD(data_box, d) += num_ghost[2 * d + 1];
         }
      }

      hypre_StructMatrixDataSpace(matrix) = data_space;
   }

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data-size
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixDataIndices(matrix) == NULL)
   {
      data_space = hypre_StructMatrixDataSpace(matrix);
      data_indices = hypre_TAlloc(HYPRE_Int *, hypre_BoxArraySize(data_space),
                                  HYPRE_MEMORY_HOST);
      if (hypre_BoxArraySize(data_space) > 0)
      {
         data_indices[0] = hypre_TAlloc(HYPRE_Int, stencil_size * hypre_BoxArraySize(data_space),
                                        HYPRE_MEMORY_HOST);
      }
      constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

      data_size = 0;
      data_const_size = 0;
      if ( constant_coefficient == 0 )
      {
         hypre_ForBoxI(i, data_space)
         {
            data_box = hypre_BoxArrayBox(data_space, i);
            data_box_volume  = hypre_BoxVolume(data_box);

            data_indices[i] = data_indices[0] + stencil_size * i;

            /* set pointers for "stored" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] < 0)
               {
                  data_indices[i][j] = data_size;
                  data_size += data_box_volume;
               }
            }

            /* set pointers for "symmetric" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] >= 0)
               {
                  data_indices[i][j] = data_indices[i][symm_elements[j]] +
                                       hypre_BoxOffsetDistance(data_box, stencil_shape[j]);
               }
            }
         }
      }
      else if ( constant_coefficient == 1 )
      {
         hypre_ForBoxI(i, data_space)
         {
            data_box = hypre_BoxArrayBox(data_space, i);
            data_box_volume  = hypre_BoxVolume(data_box);

            data_indices[i] = data_indices[0] + stencil_size * i;
            /* set pointers for "stored" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] < 0)
               {
                  data_indices[i][j] = data_const_size;
                  ++data_const_size;
               }
            }

            /* set pointers for "symmetric" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] >= 0)
               {
                  data_indices[i][j] = data_indices[i][symm_elements[j]];
               }
            }
         }
      }
      else
      {
         hypre_assert( constant_coefficient == 2 );
         data_const_size += stencil_size;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         if (hypre_StructGridDataLocation(grid) == HYPRE_MEMORY_HOST)
         {
            /* in this case, "data" is put on host using the space of
             * "data_const". so, "data" need to be shifted by the size of
             * const coeff */
            data_size += stencil_size;/* all constant coeffs at the beginning */
         }
#endif
         /* ... this allocates a little more space than is absolutely necessary */
         hypre_ForBoxI(i, data_space)
         {
            data_box = hypre_BoxArrayBox(data_space, i);
            data_box_volume  = hypre_BoxVolume(data_box);

            data_indices[i] = data_indices[0] + stencil_size * i;
            /* set pointers for "stored" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] < 0)
               {
                  /* diagonal, variable coefficient */
                  if (hypre_IndexEqual(stencil_shape[j], 0, ndim))
                  {
                     data_indices[i][j] = data_size;
                     data_size += data_box_volume;
                  }
                  /* off-diagonal, constant coefficient */
                  else
                  {
                     data_indices[i][j] = j;
                  }
               }
            }

            /* set pointers for "symmetric" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] >= 0)
               {
                  /* diagonal, variable coefficient */
                  if (hypre_IndexEqual(stencil_shape[j], 0, ndim))
                  {
                     data_indices[i][j] = data_indices[i][symm_elements[j]] +
                                          hypre_BoxOffsetDistance(data_box, stencil_shape[j]);
                  }
                  /* off-diagonal, constant coefficient */
                  else
                  {
                     data_indices[i][j] = data_indices[i][symm_elements[j]];
                  }
               }
            }
         }
      }

      hypre_StructMatrixDataIndices(matrix) = data_indices;

      /*-----------------------------------------------------------------------
       * if data location has not been set outside, set up the data location
       * based on the total number of
       *-----------------------------------------------------------------------*/
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (hypre_StructGridDataLocation(grid) == HYPRE_MEMORY_HOST)
      {
         data_const_size = data_size + data_const_size;
         data_size       = 0;
      }
#endif
      hypre_StructMatrixDataSize(matrix)      = data_size;
      hypre_StructMatrixDataConstSize(matrix) = data_const_size;

      /*
      if (hypre_BoxArraySize(data_space) > 0)
      {
      hypre_StructMatrixDataDeviceIndices(matrix) = data_indices[0];
      }
      */
   }

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    * For constant coefficients, this is unrelated to the amount of data
    * actually stored.
    *-----------------------------------------------------------------------*/

   hypre_StructMatrixGlobalSize(matrix) = hypre_StructGridGlobalSize(grid) * stencil_size;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixInitializeData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixInitializeData( hypre_StructMatrix *matrix,
                                  HYPRE_Complex      *data,
                                  HYPRE_Complex      *data_const)
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(matrix);
   HYPRE_Int constant_coefficient;
   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Complex       **stencil_data;
   HYPRE_Int stencil_size, i;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_StructGrid     *grid = hypre_StructMatrixGrid(matrix);
#endif
   hypre_StructMatrixData(matrix) = data;
   hypre_StructMatrixDataConst(matrix) = data_const;
   hypre_StructMatrixDataAlloced(matrix) = 0;

   stencil       = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   stencil_data  = hypre_StructMatrixStencilData(matrix);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

   if (constant_coefficient == 0)
   {
      for (i = 0; i < stencil_size; i++)
      {
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         if (hypre_StructGridDataLocation(grid) != HYPRE_MEMORY_HOST)
         {
            stencil_data[i] = hypre_StructMatrixData(matrix);
         }
         else
         {
            stencil_data[i] = hypre_StructMatrixDataConst(matrix);
         }
#else
         stencil_data[i] = hypre_StructMatrixData(matrix);
#endif
      }
   }
   else if (constant_coefficient == 1)
   {
      for (i = 0; i < stencil_size; i++)
      {
         stencil_data[i] = hypre_StructMatrixDataConst(matrix);
      }
   }
   else
   {
      for (i = 0; i < stencil_size; i++)
      {
         /* diagonal, variable coefficient */
         if (hypre_IndexEqual(stencil_shape[i], 0, ndim))
         {
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
            if (hypre_StructGridDataLocation(grid) != HYPRE_MEMORY_HOST)
            {
               stencil_data[i] = hypre_StructMatrixData(matrix);
            }
            else
            {
               stencil_data[i] = hypre_StructMatrixDataConst(matrix);
            }
#else
            stencil_data[i] = hypre_StructMatrixData(matrix);
#endif
         }
         /* off-diagonal, constant coefficient */
         else
         {
            stencil_data[i] = hypre_StructMatrixDataConst(matrix);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixInitialize
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatrixInitialize( hypre_StructMatrix *matrix )
{
   HYPRE_Complex *data;
   HYPRE_Complex *data_const;

   hypre_StructMatrixInitializeShell(matrix);

   data = hypre_CTAlloc(HYPRE_Complex, hypre_StructMatrixDataSize(matrix),
                        hypre_StructMatrixMemoryLocation(matrix));
   data_const = hypre_CTAlloc(HYPRE_Complex, hypre_StructMatrixDataConstSize(matrix),
                              HYPRE_MEMORY_HOST);


   hypre_StructMatrixInitializeData(matrix, data, data_const);
   hypre_StructMatrixDataAlloced(matrix) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * should not be called to set a constant-coefficient part of the matrix,
 *   call hypre_StructMatrixSetConstantValues instead
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetValues( hypre_StructMatrix *matrix,
                             hypre_Index         grid_index,
                             HYPRE_Int           num_stencil_indices,
                             HYPRE_Int          *stencil_indices,
                             HYPRE_Complex      *values,
                             HYPRE_Int           action,
                             HYPRE_Int           boxnum,
                             HYPRE_Int           outside )
{
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   hypre_Index          center_index;
   hypre_StructStencil *stencil;
   HYPRE_Int            center_rank;
   HYPRE_Int           *symm_elements;
   HYPRE_Int            constant_coefficient;
   HYPRE_Complex       *matp;
   HYPRE_Int            i, s, istart, istop;
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation memory_location = hypre_StructMatrixMemoryLocation(matrix);
#endif

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);
   symm_elements        = hypre_StructMatrixSymmElements(matrix);

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   center_rank = 0;
   if ( constant_coefficient == 2 )
   {
      hypre_SetIndex(center_index, 0);
      stencil = hypre_StructMatrixStencil(matrix);
      center_rank = hypre_StructStencilElementRank( stencil, center_index );
   }

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);

      if (hypre_IndexInBox(grid_index, grid_box))
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            /* only set stored stencil values */
            if (symm_elements[stencil_indices[s]] < 0)
            {
               if ( (constant_coefficient == 1) ||
                    (constant_coefficient == 2 && stencil_indices[s] != center_rank) )
               {
                  /* call SetConstantValues instead */
                  hypre_error(HYPRE_ERROR_GENERIC);
                  matp = hypre_StructMatrixBoxData(matrix, i, stencil_indices[s]);
               }
               else /* variable coefficient, constant_coefficient=0 */
               {
                  matp = hypre_StructMatrixBoxDataValue(matrix, i, stencil_indices[s], grid_index);
               }

#if defined(HYPRE_USING_GPU)
               if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
               {
                  if (action > 0)
                  {
#define DEVICE_VAR is_device_ptr(matp,values)
                     hypre_LoopBegin(1, k)
                     {
                        *matp += values[s];
                     }
                     hypre_LoopEnd()
#undef DEVICE_VAR
                  }
                  else if (action > -1)
                  {
                     hypre_TMemcpy(matp, values + s, HYPRE_Complex, 1, memory_location, memory_location);
                  }
                  else /* action < 0 */
                  {
                     hypre_TMemcpy(values + s, matp, HYPRE_Complex, 1, memory_location, memory_location);
                  }
               }
               else
#endif
               {
                  if (action > 0)
                  {
                     *matp += values[s];
                  }
                  else if (action > -1)
                  {
                     *matp = values[s];
                  }
                  else /* action < 0 */
                  {
                     values[s] = *matp;
                  }
               }
            }
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *
 * should not be called to set a constant-coefficient part of the matrix,
 *   call hypre_StructMatrixSetConstantValues instead
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetBoxValues( hypre_StructMatrix *matrix,
                                hypre_Box          *set_box,
                                hypre_Box          *value_box,
                                HYPRE_Int           num_stencil_indices,
                                HYPRE_Int          *stencil_indices,
                                HYPRE_Complex      *values,
                                HYPRE_Int           action,
                                HYPRE_Int           boxnum,
                                HYPRE_Int           outside )
{
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   hypre_Box           *int_box;
   hypre_Index          center_index;
   hypre_StructStencil *stencil;
   HYPRE_Int            center_rank = 0;

   HYPRE_Int           *symm_elements;
   hypre_BoxArray      *data_space;
   hypre_Box           *data_box;
   hypre_IndexRef       data_start;
   hypre_Index          data_stride;
   HYPRE_Int            datai;
   HYPRE_Complex       *datap;
   HYPRE_Int            constant_coefficient;

   hypre_Box           *dval_box;
   hypre_Index          dval_start;
   hypre_Index          dval_stride;
   HYPRE_Int            dvali;

   hypre_Index          loop_size;

   HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);
   symm_elements        = hypre_StructMatrixSymmElements(matrix);

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }
   data_space = hypre_StructMatrixDataSpace(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(data_stride, 1);

   int_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));
   dval_box = hypre_BoxDuplicate(value_box);
   hypre_BoxIMinD(dval_box, 0) *= num_stencil_indices;
   hypre_BoxIMaxD(dval_box, 0) *= num_stencil_indices;
   hypre_BoxIMaxD(dval_box, 0) += num_stencil_indices - 1;
   hypre_SetIndex(dval_stride, 1);
   hypre_IndexD(dval_stride, 0) = num_stencil_indices;

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_IntersectBoxes(set_box, grid_box, int_box);

      /* if there was an intersection */
      if (hypre_BoxVolume(int_box))
      {
         data_start = hypre_BoxIMin(int_box);
         hypre_CopyIndex(data_start, dval_start);
         hypre_IndexD(dval_start, 0) *= num_stencil_indices;

         if (constant_coefficient == 2)
         {
            hypre_SetIndex(center_index, 0);
            stencil = hypre_StructMatrixStencil(matrix);
            center_rank = hypre_StructStencilElementRank(stencil, center_index);
         }

         for (s = 0; s < num_stencil_indices; s++)
         {
            /* only set stored stencil values */
            if (symm_elements[stencil_indices[s]] < 0)
            {
               datap = hypre_StructMatrixBoxData(matrix, i, stencil_indices[s]);

               if ( (constant_coefficient == 1) ||
                    (constant_coefficient == 2 && stencil_indices[s] != center_rank ))
                  /* datap has only one data point for a given i and s */
               {
                  /* should have called SetConstantValues */
                  hypre_error(HYPRE_ERROR_GENERIC);
                  hypre_BoxGetSize(int_box, loop_size);

                  if (action > 0)
                  {
                     datai = hypre_CCBoxIndexRank(data_box, data_start);
                     dvali = hypre_BoxIndexRank(dval_box, dval_start);
                     datap[datai] += values[dvali];
                  }
                  else if (action > -1)
                  {
                     datai = hypre_CCBoxIndexRank(data_box, data_start);
                     dvali = hypre_BoxIndexRank(dval_box, dval_start);
                     datap[datai] = values[dvali];
                  }
                  else
                  {
                     datai = hypre_CCBoxIndexRank(data_box, data_start);
                     dvali = hypre_BoxIndexRank(dval_box, dval_start);
                     values[dvali] = datap[datai];
                     if (action == -2)
                     {
                        datap[datai] = 0;
                     }
                  }

               }
               else   /* variable coefficient: constant_coefficient==0
                         or diagonal with constant_coefficient==2   */
               {
#define DEVICE_VAR is_device_ptr(datap,values)
                  hypre_BoxGetSize(int_box, loop_size);

                  if (action > 0)
                  {
                     hypre_BoxLoop2Begin(hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        datap[datai] += values[dvali];
                     }
                     hypre_BoxLoop2End(datai, dvali);
                  }
                  else if (action > -1)
                  {
                     hypre_BoxLoop2Begin(hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        datap[datai] = values[dvali];
                     }
                     hypre_BoxLoop2End(datai, dvali);
                  }
                  else if (action == -2)
                  {
                     hypre_BoxLoop2Begin(hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        values[dvali] = datap[datai];
                        datap[datai] = 0;
                     }
                     hypre_BoxLoop2End(datai, dvali);
                  }
                  else
                  {
                     hypre_BoxLoop2Begin(hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        values[dvali] = datap[datai];
                     }
                     hypre_BoxLoop2End(datai, dvali);
                  }
#undef DEVICE_VAR
               }
            } /* end if (symm_elements) */

            hypre_IndexD(dval_start, 0) ++;
         }
      }
   }

   hypre_BoxDestroy(int_box);
   hypre_BoxDestroy(dval_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out (not implemented, just gets values)
 * should be called to set a constant-coefficient part of the matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetConstantValues( hypre_StructMatrix *matrix,
                                     HYPRE_Int       num_stencil_indices,
                                     HYPRE_Int      *stencil_indices,
                                     HYPRE_Complex  *values,
                                     HYPRE_Int       action )
{
   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index        center_index;
   hypre_StructStencil  *stencil;
   HYPRE_Int          center_rank;
   HYPRE_Int          constant_coefficient;

   HYPRE_Complex      *matp;

   HYPRE_Int           i, s;

   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   constant_coefficient = hypre_StructMatrixConstantCoefficient(matrix);

   if ( constant_coefficient == 1 )
   {
      hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         if (action > 0)
         {
            for (s = 0; s < num_stencil_indices; s++)
            {
               matp = hypre_StructMatrixBoxData(matrix, i,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
         else if (action > -1)
         {
            for (s = 0; s < num_stencil_indices; s++)
            {
               matp = hypre_StructMatrixBoxData(matrix, i,
                                                stencil_indices[s]);
               *matp = values[s];
            }
         }
         else  /* action < 0 */
         {
            for (s = 0; s < num_stencil_indices; s++)
            {
               matp = hypre_StructMatrixBoxData(matrix, i,
                                                stencil_indices[s]);
               values[s] = *matp;
            }
         }
      }
   }
   else if ( constant_coefficient == 2 )
   {
      hypre_SetIndex(center_index, 0);
      stencil = hypre_StructMatrixStencil(matrix);
      center_rank = hypre_StructStencilElementRank( stencil, center_index );
      if ( action > 0 )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {
               /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               hypre_error(HYPRE_ERROR_GENERIC);
               hypre_ForBoxI(i, boxes)
               {
                  box = hypre_BoxArrayBox(boxes, i);
                  hypre_StructMatrixSetBoxValues( matrix, box, box,
                                                  num_stencil_indices,
                                                  stencil_indices,
                                                  values, action, -1, 0 );
               }
            }
            else
            {
               /* non-center, like constant_coefficient==1 */
               matp = hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
      }
      else if ( action > -1 )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {
               /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               hypre_error(HYPRE_ERROR_GENERIC);
               hypre_ForBoxI(i, boxes)
               {
                  box = hypre_BoxArrayBox(boxes, i);
                  hypre_StructMatrixSetBoxValues( matrix, box, box,
                                                  num_stencil_indices,
                                                  stencil_indices,
                                                  values, action, -1, 0 );
               }
            }
            else
            {
               /* non-center, like constant_coefficient==1 */
               matp = hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
      }
      else  /* action<0 */
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {
               /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               hypre_error(HYPRE_ERROR_GENERIC);
               hypre_ForBoxI(i, boxes)
               {
                  box = hypre_BoxArrayBox(boxes, i);
                  hypre_StructMatrixSetBoxValues( matrix, box, box,
                                                  num_stencil_indices,
                                                  stencil_indices,
                                                  values, -1, -1, 0 );
               }
            }
            else
            {
               /* non-center, like constant_coefficient==1 */
               matp = hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               values[s] = *matp;
            }
         }
      }
   }
   else /* constant_coefficient==0 */
   {
      /* We consider this an error, but do the best we can. */
      hypre_error(HYPRE_ERROR_GENERIC);
      hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         hypre_StructMatrixSetBoxValues( matrix, box, box,
                                         num_stencil_indices, stencil_indices,
                                         values, action, -1, 0 );
      }
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearValues( hypre_StructMatrix *matrix,
                               hypre_Index         grid_index,
                               HYPRE_Int           num_stencil_indices,
                               HYPRE_Int          *stencil_indices,
                               HYPRE_Int           boxnum,
                               HYPRE_Int           outside )
{
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;

   HYPRE_Complex       *matp;

   HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Clear the matrix coefficients
    *-----------------------------------------------------------------------*/

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);

      if (hypre_IndexInBox(grid_index, grid_box))
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            matp = hypre_StructMatrixBoxDataValue(matrix, i, stencil_indices[s],
                                                  grid_index);
            *matp = 0.0;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearBoxValues( hypre_StructMatrix *matrix,
                                  hypre_Box          *clear_box,
                                  HYPRE_Int           num_stencil_indices,
                                  HYPRE_Int          *stencil_indices,
                                  HYPRE_Int           boxnum,
                                  HYPRE_Int           outside )
{
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   hypre_Box           *int_box;

   HYPRE_Int           *symm_elements;
   hypre_BoxArray      *data_space;
   hypre_Box           *data_box;
   hypre_IndexRef       data_start;
   hypre_Index          data_stride;
   HYPRE_Complex       *datap;

   hypre_Index          loop_size;

   HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }
   data_space = hypre_StructMatrixDataSpace(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Clear the matrix coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(data_stride, 1);

   symm_elements = hypre_StructMatrixSymmElements(matrix);

   int_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_IntersectBoxes(clear_box, grid_box, int_box);

      /* if there was an intersection */
      if (hypre_BoxVolume(int_box))
      {
         data_start = hypre_BoxIMin(int_box);

         for (s = 0; s < num_stencil_indices; s++)
         {
            /* only clear stencil entries that are explicitly stored */
            if (symm_elements[stencil_indices[s]] < 0)
            {
               datap = hypre_StructMatrixBoxData(matrix, i,
                                                 stencil_indices[s]);

               hypre_BoxGetSize(int_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap)
               hypre_BoxLoop1Begin(hypre_StructMatrixNDim(matrix), loop_size,
                                   data_box, data_start, data_stride, datai);
               {
                  datap[datai] = 0.0;
               }
               hypre_BoxLoop1End(datai);
#undef DEVICE_VAR
            }
         }
      }
   }

   hypre_BoxDestroy(int_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixAssemble( hypre_StructMatrix *matrix )
{
   HYPRE_Int              ndim = hypre_StructMatrixNDim(matrix);
   HYPRE_Int             *num_ghost = hypre_StructMatrixNumGhost(matrix);

   HYPRE_Int              comm_num_values, mat_num_values, constant_coefficient;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int              stencil_size;
   hypre_StructStencil   *stencil;
#endif
   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;

   hypre_CommHandle      *comm_handle;

   HYPRE_Complex         *matrix_data = hypre_StructMatrixData(matrix);

   HYPRE_Complex         *matrix_data_comm = matrix_data;

   /* BEGIN - variables for ghost layer identity code below */
   hypre_StructGrid      *grid;
   hypre_BoxManager      *boxman;
   hypre_BoxArray        *data_space;
   hypre_BoxArrayArray   *boundary_boxes;
   hypre_BoxArray        *boundary_box_a;
   hypre_BoxArray        *entry_box_a;
   hypre_BoxArray        *tmp_box_a;
   hypre_Box             *data_box;
   hypre_Box             *boundary_box;
   hypre_Box             *entry_box;
   hypre_BoxManEntry    **entries;
   hypre_Index            loop_size;
   hypre_Index            index;
   hypre_IndexRef         start;
   hypre_Index            stride;
   HYPRE_Complex         *datap;
   HYPRE_Int              i, j, ei;
   HYPRE_Int              num_entries;
   /* End - variables for ghost layer identity code below */

   constant_coefficient = hypre_StructMatrixConstantCoefficient( matrix );

   /*-----------------------------------------------------------------------
    * Set ghost zones along the domain boundary to the identity to enable code
    * simplifications elsewhere in hypre (e.g., CyclicReduction).
    *
    * Intersect each data box with the BoxMan to get neighbors, then subtract
    * the neighbors from the box to get the boundary boxes.
    *-----------------------------------------------------------------------*/

   if ( constant_coefficient != 1 )
   {
      data_space = hypre_StructMatrixDataSpace(matrix);
      grid       = hypre_StructMatrixGrid(matrix);
      boxman     = hypre_StructGridBoxMan(grid);

      boundary_boxes = hypre_BoxArrayArrayCreate(
                          hypre_BoxArraySize(data_space), ndim);
      entry_box_a    = hypre_BoxArrayCreate(0, ndim);
      tmp_box_a      = hypre_BoxArrayCreate(0, ndim);
      hypre_ForBoxI(i, data_space)
      {
         /* copy data box to boundary_box_a */
         boundary_box_a = hypre_BoxArrayArrayBoxArray(boundary_boxes, i);
         hypre_BoxArraySetSize(boundary_box_a, 1);
         boundary_box = hypre_BoxArrayBox(boundary_box_a, 0);
         hypre_CopyBox(hypre_BoxArrayBox(data_space, i), boundary_box);

         hypre_BoxManIntersect(boxman,
                               hypre_BoxIMin(boundary_box),
                               hypre_BoxIMax(boundary_box),
                               &entries, &num_entries);

         /* put neighbor boxes into entry_box_a */
         hypre_BoxArraySetSize(entry_box_a, num_entries);
         for (ei = 0; ei < num_entries; ei++)
         {
            entry_box = hypre_BoxArrayBox(entry_box_a, ei);
            hypre_BoxManEntryGetExtents(entries[ei],
                                        hypre_BoxIMin(entry_box),
                                        hypre_BoxIMax(entry_box));
         }
         hypre_TFree(entries, HYPRE_MEMORY_HOST);

         /* subtract neighbor boxes (entry_box_a) from data box (boundary_box_a) */
         hypre_SubtractBoxArrays(boundary_box_a, entry_box_a, tmp_box_a);
      }
      hypre_BoxArrayDestroy(entry_box_a);
      hypre_BoxArrayDestroy(tmp_box_a);

      /* set boundary ghost zones to the identity equation */

      hypre_SetIndex(index, 0);
      hypre_SetIndex(stride, 1);
      data_space = hypre_StructMatrixDataSpace(matrix);
      hypre_ForBoxI(i, data_space)
      {
         datap = hypre_StructMatrixExtractPointerByIndex(matrix, i, index);

         if (datap)
         {
            data_box = hypre_BoxArrayBox(data_space, i);
            boundary_box_a = hypre_BoxArrayArrayBoxArray(boundary_boxes, i);
            hypre_ForBoxI(j, boundary_box_a)
            {
               boundary_box = hypre_BoxArrayBox(boundary_box_a, j);
               start = hypre_BoxIMin(boundary_box);

               hypre_BoxGetSize(boundary_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap)
               hypre_BoxLoop1Begin(hypre_StructMatrixNDim(matrix), loop_size,
                                   data_box, start, stride, datai);
               {
                  datap[datai] = 1.0;
               }
               hypre_BoxLoop1End(datai);
#undef DEVICE_VAR
            }
         }
      }

      hypre_BoxArrayArrayDestroy(boundary_boxes);
   }

   /*-----------------------------------------------------------------------
    * If the CommPkg has not been set up, set it up
    *
    * The matrix data array is assumed to have two segments - an initial
    * segment of data constant over all space, followed by a segment with
    * comm_num_values matrix entries for each mesh element.  The mesh-dependent
    * data is, of course, the only part relevent to communications.
    * For constant_coefficient==0, all the data is mesh-dependent.
    * For constant_coefficient==1, all  data is constant.
    * For constant_coefficient==2, both segments are non-null.
    *-----------------------------------------------------------------------*/

   mat_num_values = hypre_StructMatrixNumValues(matrix);

   if ( constant_coefficient == 0 )
   {
      comm_num_values = mat_num_values;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (hypre_StructGridDataLocation(grid) == HYPRE_MEMORY_HOST)
      {
         matrix_data_comm = hypre_StructMatrixDataConst(matrix);
      }
#endif
   }
   else if ( constant_coefficient == 1 )
   {
      comm_num_values = 0;
   }
   else /* constant_coefficient==2 */
   {
      comm_num_values = 1;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (hypre_StructGridDataLocation(grid) == HYPRE_MEMORY_HOST)
      {
         stencil = hypre_StructMatrixStencil(matrix);
         stencil_size  = hypre_StructStencilSize(stencil);
         matrix_data_comm = hypre_StructMatrixDataConst(matrix) + stencil_size;
      }
#endif
   }

   comm_pkg = hypre_StructMatrixCommPkg(matrix);

   if (!comm_pkg)
   {
      hypre_CreateCommInfoFromNumGhost(hypre_StructMatrixGrid(matrix),
                                       num_ghost, &comm_info);
      hypre_CommPkgCreate(comm_info,
                          hypre_StructMatrixDataSpace(matrix),
                          hypre_StructMatrixDataSpace(matrix),
                          comm_num_values, NULL, 0,
                          hypre_StructMatrixComm(matrix), &comm_pkg);
      hypre_CommInfoDestroy(comm_info);

      hypre_StructMatrixCommPkg(matrix) = comm_pkg;
   }

   /*-----------------------------------------------------------------------
    * Update the ghost data
    * This takes care of the communication needs of all known functions
    * referencing the matrix.
    *
    * At present this is the only place where matrix data gets communicated.
    * However, comm_pkg is kept as long as the matrix is, in case some
    * future version hypre has a use for it - e.g. if the user replaces
    * a matrix with a very similar one, we may not want to recompute comm_pkg.
    *-----------------------------------------------------------------------*/

   if ( constant_coefficient != 1 )
   {
      hypre_InitializeCommunication( comm_pkg,
                                     matrix_data_comm,
                                     matrix_data_comm, 0, 0,
                                     &comm_handle );
      hypre_FinalizeCommunication( comm_handle );
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetNumGhost( hypre_StructMatrix *matrix,
                               HYPRE_Int          *num_ghost )
{
   HYPRE_Int  d, ndim = hypre_StructMatrixNDim(matrix);

   for (d = 0; d < ndim; d++)
   {
      hypre_StructMatrixNumGhost(matrix)[2 * d]     = num_ghost[2 * d];
      hypre_StructMatrixNumGhost(matrix)[2 * d + 1] = num_ghost[2 * d + 1];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetConstantCoefficient
 * deprecated in user interface, in favor of SetConstantEntries.
 * left here for internal use
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetConstantCoefficient( hypre_StructMatrix *matrix,
                                          HYPRE_Int          constant_coefficient )
{
   hypre_StructMatrixConstantCoefficient(matrix) = constant_coefficient;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixSetConstantEntries
 * - nentries is the number of array entries
 * - Each HYPRE_Int entries[i] is an index into the shape array of the stencil
 *   of the matrix
 * In the present version, only three possibilites are recognized:
 * - no entries constant                 (constant_coefficient==0)
 * - all entries constant                (constant_coefficient==1)
 * - all but the diagonal entry constant (constant_coefficient==2)
 * If something else is attempted, this function will return a nonzero error.
 * In the present version, if this function is called more than once, only
 * the last call will take effect.
 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_StructMatrixSetConstantEntries( hypre_StructMatrix *matrix,
                                                 HYPRE_Int           nentries,
                                                 HYPRE_Int          *entries )
{
   /* We make an array offdconst corresponding to the stencil's shape array,
      and use "entries" to fill it with flags - 1 for constant, 0 otherwise.
      By counting the nonzeros in offdconst, and by checking whether its
      diagonal entry is nonzero, we can distinguish among the three
      presently legal values of constant_coefficient, and detect input errors.
      We do not need to treat duplicates in "entries" as an error condition.
   */
   hypre_StructStencil *stencil = hypre_StructMatrixUserStencil(matrix);
   /* ... Stencil doesn't exist yet */
   HYPRE_Int stencil_size  = hypre_StructStencilSize(stencil);
   HYPRE_Int *offdconst = hypre_CTAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
   /* ... note: CTAlloc initializes to 0 (normally it works by calling calloc) */
   HYPRE_Int nconst = 0;
   HYPRE_Int constant_coefficient, diag_rank;
   hypre_Index diag_index;
   HYPRE_Int i, j;

   for ( i = 0; i < nentries; ++i )
   {
      offdconst[ entries[i] ] = 1;
   }

   for ( j = 0; j < stencil_size; ++j )
   {
      nconst += offdconst[j];
   }

   if ( nconst <= 0 )
   {
      constant_coefficient = 0;
   }
   else if ( nconst >= stencil_size )
   {
      constant_coefficient = 1;
   }
   else
   {
      hypre_SetIndex(diag_index, 0);
      diag_rank = hypre_StructStencilElementRank( stencil, diag_index );
      if ( offdconst[diag_rank] == 0 )
      {
         constant_coefficient = 2;
         if ( nconst != (stencil_size - 1) )
         {
            hypre_error(HYPRE_ERROR_GENERIC);
         }
      }
      else
      {
         constant_coefficient = 0;
         hypre_error(HYPRE_ERROR_GENERIC);
      }
   }

   hypre_StructMatrixSetConstantCoefficient( matrix, constant_coefficient );

   hypre_TFree(offdconst, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearGhostValues( hypre_StructMatrix *matrix )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(matrix);
   hypre_Box            *m_data_box;

   HYPRE_Complex        *mp;

   hypre_StructStencil  *stencil;
   HYPRE_Int            *symm_elements;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_BoxArray       *diff_boxes;
   hypre_Box            *diff_box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           unit_stride;

   HYPRE_Int             i, j, s;

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(unit_stride, 1);

   stencil = hypre_StructMatrixStencil(matrix);
   symm_elements = hypre_StructMatrixSymmElements(matrix);
   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   diff_boxes = hypre_BoxArrayCreate(0, ndim);
   hypre_ForBoxI(i, boxes)
   {
      box        = hypre_BoxArrayBox(boxes, i);
      m_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), i);
      hypre_BoxArraySetSize(diff_boxes, 0);
      hypre_SubtractBoxes(m_data_box, box, diff_boxes);

      for (s = 0; s < hypre_StructStencilSize(stencil); s++)
      {
         /* only clear stencil entries that are explicitly stored */
         if (symm_elements[s] < 0)
         {
            mp = hypre_StructMatrixBoxData(matrix, i, s);
            hypre_ForBoxI(j, diff_boxes)
            {
               diff_box = hypre_BoxArrayBox(diff_boxes, j);
               start = hypre_BoxIMin(diff_box);

               hypre_BoxGetSize(diff_box, loop_size);

#define DEVICE_VAR is_device_ptr(mp)
               hypre_BoxLoop1Begin(hypre_StructMatrixNDim(matrix), loop_size,
                                   m_data_box, start, unit_stride, mi);
               {
                  mp[mi] = 0.0;
               }
               hypre_BoxLoop1End(mi);
#undef DEVICE_VAR
            }
         }
      }
   }
   hypre_BoxArrayDestroy(diff_boxes);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixPrintData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixPrintData( FILE               *file,
                             hypre_StructMatrix *matrix,
                             HYPRE_Int           all )
{
   HYPRE_Int             ndim            = hypre_StructMatrixNDim(matrix);
   HYPRE_Int             num_values      = hypre_StructMatrixNumValues(matrix);
   HYPRE_Int             ctecoef         = hypre_StructMatrixConstantCoefficient(matrix);
   hypre_StructGrid     *grid            = hypre_StructMatrixGrid(matrix);
   hypre_StructStencil  *stencil         = hypre_StructMatrixStencil(matrix);
   HYPRE_Int             stencil_size    = hypre_StructStencilSize(stencil);
   HYPRE_Int            *symm_elements   = hypre_StructMatrixSymmElements(matrix);
   hypre_BoxArray       *data_space      = hypre_StructMatrixDataSpace(matrix);
   HYPRE_Int             data_size       = hypre_StructMatrixDataSize(matrix);
   hypre_BoxArray       *grid_boxes      = hypre_StructGridBoxes(grid);
   HYPRE_Complex        *data            = hypre_StructMatrixData(matrix);
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(matrix);
   hypre_BoxArray       *boxes;
   hypre_Index           center_index;
   HYPRE_Int             center_rank;
   HYPRE_Complex        *h_data;

   /* Allocate/Point to data on the host memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      h_data = hypre_CTAlloc(HYPRE_Complex, data_size, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(h_data, data, HYPRE_Complex, data_size,
                    HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_data = data;
   }

   /* Print ghost data (all) also or only real data? */
   boxes = (all) ? data_space : grid_boxes;

   /* Print data to file */
   if (ctecoef == 1)
   {
      hypre_PrintCCBoxArrayData(file, boxes, data_space, num_values, h_data);
   }
   else if (ctecoef == 2)
   {
      hypre_SetIndex(center_index, 0);
      center_rank = hypre_StructStencilElementRank(stencil, center_index);

      hypre_PrintCCVDBoxArrayData(file, boxes, data_space, num_values,
                                  center_rank, stencil_size, symm_elements,
                                  ndim, h_data);
   }
   else
   {
      hypre_PrintBoxArrayData(file, boxes, data_space, num_values,
                              ndim, h_data);
   }

   /* Free memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixReadData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixReadData( FILE               *file,
                            hypre_StructMatrix *matrix )
{
   HYPRE_Int             ndim            = hypre_StructMatrixNDim(matrix);
   HYPRE_Int             num_values      = hypre_StructMatrixNumValues(matrix);
   HYPRE_Int             ctecoef         = hypre_StructMatrixConstantCoefficient(matrix);
   hypre_StructGrid     *grid            = hypre_StructMatrixGrid(matrix);
   hypre_StructStencil  *stencil         = hypre_StructMatrixStencil(matrix);
   HYPRE_Int             stencil_size    = hypre_StructStencilSize(stencil);
   HYPRE_Int             symmetric       = hypre_StructMatrixSymmetric(matrix);
   hypre_BoxArray       *data_space      = hypre_StructMatrixDataSpace(matrix);
   hypre_BoxArray       *boxes           = hypre_StructGridBoxes(grid);
   HYPRE_Complex        *data            = hypre_StructMatrixData(matrix);
   HYPRE_Int             data_size       = hypre_StructMatrixDataSize(matrix);
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(matrix);
   HYPRE_Complex        *h_data;
   HYPRE_Int             real_stencil_size;

   /* Allocate/Point to data on the host memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      h_data = hypre_CTAlloc(HYPRE_Complex, data_size, HYPRE_MEMORY_HOST);
   }
   else
   {
      h_data = data;
   }

   /* real_stencil_size is the stencil size of the matrix after it's fixed up
      by the call (if any) of hypre_StructStencilSymmetrize from
      hypre_StructMatrixInitializeShell.*/
   if (symmetric)
   {
      real_stencil_size = 2 * stencil_size - 1;
   }
   else
   {
      real_stencil_size = stencil_size;
   }

   /* Read data from file */
   if (ctecoef == 0)
   {
      hypre_ReadBoxArrayData(file, boxes, data_space,
                             num_values, ndim, h_data);
   }
   else
   {
      hypre_assert(ctecoef <= 2);
      hypre_ReadBoxArrayData_CC(file, boxes, data_space,
                                stencil_size, real_stencil_size,
                                ctecoef, ndim, h_data);
   }

   /* Move data to the device memory if necessary and free host data */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      hypre_TMemcpy(data, h_data, HYPRE_Complex, data_size,
                    memory_location, HYPRE_MEMORY_HOST);
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixPrint( const char         *filename,
                         hypre_StructMatrix *matrix,
                         HYPRE_Int           all      )
{
   FILE                 *file;
   char                  new_filename[255];

   hypre_StructGrid     *grid;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size;

   HYPRE_Int             ndim, num_values;

   HYPRE_Int            *symm_elements;

   HYPRE_Int             i, j, d;
   HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(hypre_StructMatrixComm(matrix), &myid);

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   hypre_fprintf(file, "StructMatrix\n");

   hypre_fprintf(file, "\nSymmetric: %d\n", hypre_StructMatrixSymmetric(matrix));
   hypre_fprintf(file, "\nConstantCoefficient: %d\n",
                 hypre_StructMatrixConstantCoefficient(matrix));

   /* print grid info */
   hypre_fprintf(file, "\nGrid:\n");
   grid = hypre_StructMatrixGrid(matrix);
   hypre_StructGridPrint(file, grid);

   /* print stencil info */
   hypre_fprintf(file, "\nStencil:\n");
   stencil = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);

   ndim = hypre_StructMatrixNDim(matrix);
   num_values = hypre_StructMatrixNumValues(matrix);
   symm_elements = hypre_StructMatrixSymmElements(matrix);
   hypre_fprintf(file, "%d\n", num_values);
   stencil_size = hypre_StructStencilSize(stencil);
   j = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (symm_elements[i] < 0)
      {
         /* Print line of the form: "%d: %d %d %d\n" */
         hypre_fprintf(file, "%d:", j++);
         for (d = 0; d < ndim; d++)
         {
            hypre_fprintf(file, " %d", hypre_IndexD(stencil_shape[i], d));
         }
         hypre_fprintf(file, "\n");
      }
   }

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   hypre_fprintf(file, "\nData:\n");
   hypre_StructMatrixPrintData(file, matrix, all);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fflush(file);
   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixRead
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixRead( MPI_Comm    comm,
                        const char *filename,
                        HYPRE_Int  *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];

   hypre_StructMatrix   *matrix;

   hypre_StructGrid     *grid;
   HYPRE_Int             ndim;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size;
   HYPRE_Int             symmetric;
   HYPRE_Int             constant_coefficient;

   HYPRE_Int             i, d, idummy;

   HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid );

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   hypre_fscanf(file, "StructMatrix\n");

   hypre_fscanf(file, "\nSymmetric: %d\n", &symmetric);
   hypre_fscanf(file, "\nConstantCoefficient: %d\n", &constant_coefficient);

   /* read grid info */
   hypre_fscanf(file, "\nGrid:\n");
   hypre_StructGridRead(comm, file, &grid);

   /* read stencil info */
   hypre_fscanf(file, "\nStencil:\n");
   ndim = hypre_StructGridNDim(grid);
   hypre_fscanf(file, "%d\n", &stencil_size);
   stencil_shape = hypre_CTAlloc(hypre_Index,  stencil_size, HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      /* Read line of the form: "%d: %d %d %d\n" */
      hypre_fscanf(file, "%d:", &idummy);
      for (d = 0; d < ndim; d++)
      {
         hypre_fscanf(file, " %d", &hypre_IndexD(stencil_shape[i], d));
      }
      hypre_fscanf(file, "\n");
   }
   stencil = hypre_StructStencilCreate(ndim, stencil_size, stencil_shape);

   /*----------------------------------------
    * Initialize the matrix
    *----------------------------------------*/

   matrix = hypre_StructMatrixCreate(comm, grid, stencil);
   hypre_StructMatrixSymmetric(matrix) = symmetric;
   hypre_StructMatrixConstantCoefficient(matrix) = constant_coefficient;
   hypre_StructMatrixSetNumGhost(matrix, num_ghost);
   hypre_StructMatrixInitialize(matrix);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   hypre_fscanf(file, "\nData:\n");
   hypre_StructMatrixReadData(file, matrix);

   /*----------------------------------------
    * Assemble the matrix
    *----------------------------------------*/

   hypre_StructMatrixAssemble(matrix);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fclose(file);

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixMigrate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMigrate( hypre_StructMatrix *from_matrix,
                           hypre_StructMatrix *to_matrix   )
{
   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;
   hypre_CommHandle      *comm_handle;

   HYPRE_Int              constant_coefficient, comm_num_values;
   HYPRE_Int              stencil_size, mat_num_values;
   hypre_StructStencil   *stencil;

   HYPRE_Complex         *matrix_data_from = hypre_StructMatrixData(from_matrix);
   HYPRE_Complex         *matrix_data_to = hypre_StructMatrixData(to_matrix);
   HYPRE_Complex         *matrix_data_comm_from = matrix_data_from;
   HYPRE_Complex         *matrix_data_comm_to = matrix_data_to;

   /*------------------------------------------------------
    * Set up hypre_CommPkg
    *------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient( from_matrix );
   hypre_assert( constant_coefficient == hypre_StructMatrixConstantCoefficient( to_matrix ) );

   mat_num_values = hypre_StructMatrixNumValues(from_matrix);
   hypre_assert( mat_num_values == hypre_StructMatrixNumValues(to_matrix) );

   if ( constant_coefficient == 0 )
   {
      comm_num_values = mat_num_values;
   }
   else if ( constant_coefficient == 1 )
   {
      comm_num_values = 0;
   }
   else /* constant_coefficient==2 */
   {
      comm_num_values = 1;
      stencil = hypre_StructMatrixStencil(from_matrix);
      stencil_size = hypre_StructStencilSize(stencil);
      hypre_assert(stencil_size ==
                   hypre_StructStencilSize( hypre_StructMatrixStencil(to_matrix) ) );
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (hypre_StructGridDataLocation(hypre_StructMatrixGrid(from_matrix)) == HYPRE_MEMORY_HOST)
      {
         stencil = hypre_StructMatrixStencil(from_matrix);
         stencil_size  = hypre_StructStencilSize(stencil);
         matrix_data_comm_from = hypre_StructMatrixDataConst(from_matrix) + stencil_size;
         stencil = hypre_StructMatrixStencil(to_matrix);
         stencil_size  = hypre_StructStencilSize(stencil);
         matrix_data_comm_to = hypre_StructMatrixDataConst(to_matrix) + stencil_size;
      }
#endif
   }

   hypre_CreateCommInfoFromGrids(hypre_StructMatrixGrid(from_matrix),
                                 hypre_StructMatrixGrid(to_matrix),
                                 &comm_info);
   hypre_CommPkgCreate(comm_info,
                       hypre_StructMatrixDataSpace(from_matrix),
                       hypre_StructMatrixDataSpace(to_matrix),
                       comm_num_values, NULL, 0,
                       hypre_StructMatrixComm(from_matrix), &comm_pkg);
   hypre_CommInfoDestroy(comm_info);
   /* is this correct for periodic? */

   /*-----------------------------------------------------------------------
    * Migrate the matrix data
    *-----------------------------------------------------------------------*/

   if ( constant_coefficient != 1 )
   {
      hypre_InitializeCommunication( comm_pkg,
                                     matrix_data_comm_from,
                                     matrix_data_comm_to, 0, 0,
                                     &comm_handle );
      hypre_FinalizeCommunication( comm_handle );
   }
   hypre_CommPkgDestroy(comm_pkg);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * clears matrix stencil coefficients reaching outside of the physical boundaries
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearBoundary( hypre_StructMatrix *matrix)
{
   HYPRE_Int            ndim = hypre_StructMatrixNDim(matrix);
   HYPRE_Complex       *data;
   hypre_BoxArray      *grid_boxes;
   hypre_BoxArray      *data_space;
   /*hypre_Box           *box;*/
   hypre_Box           *grid_box;
   hypre_Box           *data_box;
   hypre_Box           *tmp_box;
   hypre_Index         *shape;
   hypre_Index          stencil_element;
   hypre_Index          loop_size;
   hypre_IndexRef       start;
   hypre_Index          stride;
   hypre_StructGrid    *grid;
   hypre_StructStencil *stencil;
   hypre_BoxArray      *boundary;

   HYPRE_Int           i, i2, j;

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   grid = hypre_StructMatrixGrid(matrix);
   stencil = hypre_StructMatrixStencil(matrix);
   grid_boxes = hypre_StructGridBoxes(grid);
   ndim = hypre_StructStencilNDim(stencil);
   data_space = hypre_StructMatrixDataSpace(matrix);
   hypre_SetIndex(stride, 1);
   shape = hypre_StructStencilShape(stencil);

   for (j = 0; j < hypre_StructStencilSize(stencil); j++)
   {
      hypre_CopyIndex(shape[j], stencil_element);
      if (!hypre_IndexEqual(stencil_element, 0, ndim))
      {
         hypre_ForBoxI(i, grid_boxes)
         {
            grid_box = hypre_BoxArrayBox(grid_boxes, i);
            data_box = hypre_BoxArrayBox(data_space, i);
            boundary = hypre_BoxArrayCreate( 0, ndim );
            hypre_GeneralBoxBoundaryIntersect(grid_box, grid, stencil_element,
                                              boundary);
            data = hypre_StructMatrixBoxData(matrix, i, j);
            hypre_ForBoxI(i2, boundary)
            {
               tmp_box = hypre_BoxArrayBox(boundary, i2);
               hypre_BoxGetSize(tmp_box, loop_size);
               start = hypre_BoxIMin(tmp_box);
#define DEVICE_VAR is_device_ptr(data)
               hypre_BoxLoop1Begin(ndim, loop_size, data_box, start, stride, ixyz);
               {
                  data[ixyz] = 0.0;
               }
               hypre_BoxLoop1End(ixyz);
#undef DEVICE_VAR
            }
            hypre_BoxArrayDestroy(boundary);
         }
      }
   }

   return hypre_error_flag;
}
