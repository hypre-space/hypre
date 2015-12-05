/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Member functions for hypre_StructMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

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
   hypre_StructMatrix   *mask;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size;
   hypre_Index          *mask_stencil_shape;
   HYPRE_Int             mask_stencil_size;

   hypre_BoxArray       *data_space;
   HYPRE_Int           **data_indices;
   HYPRE_Int           **mask_data_indices;

   HYPRE_Int             i, j;

   stencil       = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   mask = hypre_CTAlloc(hypre_StructMatrix, 1);

   hypre_StructMatrixComm(mask) = hypre_StructMatrixComm(matrix);

   hypre_StructGridRef(hypre_StructMatrixGrid(matrix),
                       &hypre_StructMatrixGrid(mask));

   hypre_StructMatrixUserStencil(mask) =
      hypre_StructStencilRef(hypre_StructMatrixUserStencil(matrix));

   mask_stencil_size  = num_stencil_indices;
   mask_stencil_shape = hypre_CTAlloc(hypre_Index, num_stencil_indices);
   for (i = 0; i < num_stencil_indices; i++)
   {
      hypre_CopyIndex(stencil_shape[stencil_indices[i]],
                      mask_stencil_shape[i]);
   }
   hypre_StructMatrixStencil(mask) =
      hypre_StructStencilCreate(hypre_StructStencilDim(stencil),
                                mask_stencil_size,
                                mask_stencil_shape);

   hypre_StructMatrixNumValues(mask) = hypre_StructMatrixNumValues(matrix);

   hypre_StructMatrixDataSpace(mask) =
      hypre_BoxArrayDuplicate(hypre_StructMatrixDataSpace(matrix));

   hypre_StructMatrixData(mask) = hypre_StructMatrixData(matrix);
   hypre_StructMatrixDataAlloced(mask) = 0;
   hypre_StructMatrixDataSize(mask) = hypre_StructMatrixDataSize(matrix);
   data_space   = hypre_StructMatrixDataSpace(matrix);
   data_indices = hypre_StructMatrixDataIndices(matrix);
   mask_data_indices = hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(data_space));
   hypre_ForBoxI(i, data_space)
      {
         mask_data_indices[i] = hypre_TAlloc(HYPRE_Int, num_stencil_indices);
         for (j = 0; j < num_stencil_indices; j++)
         {
            mask_data_indices[i][j] = data_indices[i][stencil_indices[j]];
         }
      }
   hypre_StructMatrixDataIndices(mask) = mask_data_indices;

   hypre_StructMatrixSymmetric(mask) = hypre_StructMatrixSymmetric(matrix);

   hypre_StructMatrixSymmElements(mask) = hypre_TAlloc(HYPRE_Int, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_StructMatrixSymmElements(mask)[i] =
         hypre_StructMatrixSymmElements(matrix)[i];
   }

   for (i = 0; i < 6; i++)
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

