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
 * Member functions for zzz_StructMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructMatrixMask
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

zzz_StructMatrix *
zzz_NewStructMatrixMask( zzz_StructMatrix *matrix,
                         int               num_stencil_indices,
                         int              *stencil_indices     )
{
   zzz_StructMatrix   *mask;

   zzz_StructStencil  *stencil;
   zzz_Index         **stencil_shape;
   zzz_StructStencil  *mask_stencil;
   zzz_Index         **mask_stencil_shape;
   int                 mask_stencil_size;

   zzz_BoxArray       *data_space;
   int               **data_indices;
   int               **mask_data_indices;

   int                 i, j;

   mask = zzz_CTAlloc(zzz_StructMatrix, 1);

   zzz_StructMatrixComm(mask)        = zzz_StructMatrixComm(matrix);
   zzz_StructMatrixGrid(mask)        = zzz_StructMatrixGrid(matrix);
   zzz_StructMatrixUserStencil(mask) = zzz_StructMatrixUserStencil(matrix);
   zzz_StructMatrixNumValues(mask)   = zzz_StructMatrixNumValues(matrix);
   zzz_StructMatrixDataSpace(mask)   = zzz_StructMatrixDataSpace(matrix);
   zzz_StructMatrixData(mask)        = zzz_StructMatrixData(matrix);
   zzz_StructMatrixDataSize(mask)    = zzz_StructMatrixDataSize(matrix);
   zzz_StructMatrixSymmetric(mask)   = zzz_StructMatrixSymmetric(matrix);
   zzz_StructMatrixSymmCoeff(mask)   = zzz_StructMatrixSymmCoeff(matrix);
   for (i = 0; i < 6; i++)
      zzz_StructMatrixNumGhost(mask)[i] = zzz_StructMatrixNumGhost(matrix)[i];
   zzz_StructMatrixCommPkg(mask)     = zzz_StructMatrixCommPkg(matrix);

   /* create mask_stencil */
   stencil       = zzz_StructMatrixStencil(matrix);
   stencil_shape = zzz_StructStencilShape(stencil);
   mask_stencil_size  = num_stencil_indices;
   mask_stencil_shape = zzz_CTAlloc(zzz_Index *, num_stencil_indices);
   for (i = 0; i < num_stencil_indices; i++)
   {
      mask_stencil_shape[i] = zzz_NewIndex();
      zzz_CopyIndex(stencil_shape[stencil_indices[i]], mask_stencil_shape[i]);
   }
   mask_stencil = zzz_NewStructStencil(zzz_StructStencilDim(stencil),
                                       mask_stencil_size, mask_stencil_shape);

   /* create a new data_indices array */
   data_space   = zzz_StructMatrixDataSpace(matrix);
   data_indices = zzz_StructMatrixDataIndices(matrix);
   mask_data_indices = zzz_CTAlloc(int *, zzz_BoxArraySize(data_space));
   zzz_ForBoxI(i, data_space)
   {
      mask_data_indices[i] = zzz_TAlloc(int, num_stencil_indices);
      for (j = 0; j < num_stencil_indices; j++)
      {
         mask_data_indices[i][j] = data_indices[i][stencil_indices[j]];
      }
   }

   zzz_StructMatrixStencil(mask)     = mask_stencil;
   zzz_StructMatrixDataIndices(mask) = mask_data_indices;
   zzz_StructMatrixGlobalSize(mask) =
      zzz_StructGridGlobalSize(zzz_StructMatrixGrid(mask)) * mask_stencil_size;

   return mask;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructMatrixMask
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructMatrixMask( zzz_StructMatrix *mask )
{
   int  ierr;

   int  i;

   if (mask)
   {
      zzz_ForBoxI(i, zzz_StructMatrixDataSpace(mask))
         zzz_TFree(zzz_StructMatrixDataIndices(mask)[i]);
      zzz_TFree(zzz_StructMatrixDataIndices(mask));

      zzz_FreeStructStencil(zzz_StructMatrixStencil(mask));

      zzz_TFree(mask);
   }

   return ierr;
}

