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

/*--------------------------------------------------------------------------
 * hypre_StructMatrixCreateMask
 *    This routine returns the matrix, `mask', containing pointers to
 *    some of the data in the input matrix `matrix'.  This can be useful,
 *    for example, to construct "splittings" of a matrix for use in
 *    iterative methods.  The key note here is that the matrix `mask' does
 *    NOT contain a copy of the data in `matrix', but it can be used as
 *    if it were a normal StructMatrix object.
 *
 *    Notes:
 *    (1) Only the stencil, data_indices, and global_size components of the
 *        StructMatrix structure are modified.
 *    (2) PrintStructMatrix will not correctly print the stencil-to-data
 *        correspondence.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixCreateMask( hypre_StructMatrix *matrix,
                              HYPRE_Int           num_stencil_indices,
                              HYPRE_Int          *stencil_indices     )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(matrix);
   hypre_StructMatrix   *mask;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size;
   HYPRE_Complex       **stencil_data;
   hypre_Index          *mask_stencil_shape;
   HYPRE_Int             mask_stencil_size;
   HYPRE_Complex       **mask_stencil_data;

   hypre_BoxArray       *data_space;
   HYPRE_Int           **data_indices;
   HYPRE_Int           **mask_data_indices;

   HYPRE_Int             i, j;

   stencil       = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   stencil_data  = hypre_StructMatrixStencilData(matrix);

   mask = hypre_CTAlloc(hypre_StructMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_StructMatrixComm(mask) = hypre_StructMatrixComm(matrix);

   hypre_StructGridRef(hypre_StructMatrixGrid(matrix),
                       &hypre_StructMatrixGrid(mask));

   hypre_StructMatrixUserStencil(mask) =
      hypre_StructStencilRef(hypre_StructMatrixUserStencil(matrix));

   mask_stencil_size  = num_stencil_indices;
   mask_stencil_shape = hypre_CTAlloc(hypre_Index, num_stencil_indices, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_stencil_indices; i++)
   {
      hypre_CopyIndex(stencil_shape[stencil_indices[i]],
                      mask_stencil_shape[i]);
   }
   hypre_StructMatrixStencil(mask) =
      hypre_StructStencilCreate(hypre_StructStencilNDim(stencil),
                                mask_stencil_size,
                                mask_stencil_shape);

   hypre_StructMatrixNumValues(mask) = hypre_StructMatrixNumValues(matrix);

   hypre_StructMatrixDataSpace(mask) =
      hypre_BoxArrayDuplicate(hypre_StructMatrixDataSpace(matrix));

   hypre_StructMatrixMemoryLocation(mask) = hypre_StructMatrixMemoryLocation(matrix);

   hypre_StructMatrixData(mask) = hypre_StructMatrixData(matrix);
   hypre_StructMatrixDataConst(mask) = hypre_StructMatrixDataConst(matrix);

   hypre_StructMatrixDataAlloced(mask) = 0;
   hypre_StructMatrixDataSize(mask) = hypre_StructMatrixDataSize(matrix);
   hypre_StructMatrixDataConstSize(mask) = hypre_StructMatrixDataConstSize(matrix);
   data_space   = hypre_StructMatrixDataSpace(matrix);
   data_indices = hypre_StructMatrixDataIndices(matrix);
   mask_data_indices = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(data_space), HYPRE_MEMORY_HOST);
   mask_stencil_data  = hypre_TAlloc(HYPRE_Complex*, mask_stencil_size, HYPRE_MEMORY_HOST);
   if (hypre_BoxArraySize(data_space) > 0)
   {
      mask_data_indices[0] = hypre_TAlloc(HYPRE_Int,
                                          num_stencil_indices * hypre_BoxArraySize(data_space),
                                          HYPRE_MEMORY_HOST);
   }

   hypre_ForBoxI(i, data_space)
   {
      mask_data_indices[i] = mask_data_indices[0] + num_stencil_indices * i;
      for (j = 0; j < num_stencil_indices; j++)
      {
         mask_data_indices[i][j] = data_indices[i][stencil_indices[j]];
      }
   }
   for (i = 0; i < mask_stencil_size; i++)
   {
      mask_stencil_data[i] = stencil_data[stencil_indices[i]];
   }
   hypre_StructMatrixStencilData(mask) = mask_stencil_data;

   hypre_StructMatrixDataIndices(mask) = mask_data_indices;

   hypre_StructMatrixSymmetric(mask) = hypre_StructMatrixSymmetric(matrix);

   hypre_StructMatrixSymmElements(mask) = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_StructMatrixSymmElements(mask)[i] =
         hypre_StructMatrixSymmElements(matrix)[i];
   }

   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructMatrixNumGhost(mask)[i] =
         hypre_StructMatrixNumGhost(matrix)[i];
   }

   hypre_StructMatrixGlobalSize(mask) =
      hypre_StructGridGlobalSize(hypre_StructMatrixGrid(mask)) *
      mask_stencil_size;

   hypre_StructMatrixCommPkg(mask) = NULL;

   hypre_StructMatrixRefCount(mask) = 1;

   return mask;
}

