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
 * zzz_NewStructMatrix
 *--------------------------------------------------------------------------*/

zzz_StructMatrix *
zzz_NewStructMatrix( MPI_Comm     context,
		      zzz_StructGrid    *grid,
		      zzz_StructStencil *stencil )
{
   zzz_StructMatrix    *matrix;

   matrix = talloc(zzz_StructMatrix, 1);

   zzz_StructMatrixContext(matrix) = context;
   zzz_StructMatrixStructGrid(matrix)    = grid;
   zzz_StructMatrixStructStencil(matrix) = stencil;

   zzz_StructMatrixTranslator(matrix) = NULL;
   zzz_StructMatrixStorageType(matrix) = 0;
   zzz_StructMatrixData(matrix) = NULL;

   /* set defaults */
   zzz_SetStructMatrixStorageType(matrix, ZZZ_PETSC_MATRIX);

   return matrix;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructMatrix( zzz_StructMatrix *matrix )
{

   if ( zzz_StructMatrixStorageType(matrix) == ZZZ_PETSC_MATRIX )
      zzz_FreeStructMatrixPETSc( matrix );
   else
      return(-1);

   tfree(matrix);

   return(0);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixCoeffs
 *   
 *   Set elements in a Struct Matrix interface.
 *   Coefficients are referred to in stencil format; grid points are
 *   identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructMatrixCoeffs( zzz_StructMatrix *matrix,
			    zzz_Index         *grid_index,
			    double            *coeffs     )
{
   int    ierr;

   if ( zzz_StructMatrixStorageType(matrix) == ZZZ_PETSC_MATRIX )
      return( zzz_SetStructMatrixPETScCoeffs( matrix, grid_index, coeffs ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructMatrix
 *   User-level routine for assembling zzz_StructMatrix.
 *--------------------------------------------------------------------------*/

int 
zzz_AssembleStructMatrix( zzz_StructMatrix *matrix )
{
   if ( zzz_StructMatrixStorageType(matrix) == ZZZ_PETSC_MATRIX )
      return( zzz_AssembleStructMatrixPETSc( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_PrintStructMatrix
 *   
 *--------------------------------------------------------------------------*/

int 
zzz_PrintStructMatrix( zzz_StructMatrix *matrix )
{
   if ( zzz_StructMatrixStorageType(matrix) == ZZZ_PETSC_MATRIX )
      return( zzz_PrintStructMatrixPETSc( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixStorageType
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructMatrixStorageType( zzz_StructMatrix *matrix,
				 int                type   )
{
   zzz_StructMatrixStorageType(matrix) = type;

   return(0);
}

/*--------------------------------------------------------------------------
 * zzz_FindBoxNeighborhood:
 *
 *   Finds boxes in the `all_boxes' zzz_BoxArray that form a neighborhood
 *   of the `boxes' zzz_BoxArray.  This neighborhood is determined by the
 *   shape of the stencil passed in and represents the minimum number
 *   of boxes touched by the stencil elements.
 *
 *   The routine returns an integer array of boolean flags indicating
 *   which boxes in `all_boxes' are in the neighborhood and which are not.
 *   The reason for doing this instead of returning another zzz_BoxArray
 *   is so that additional information that may be associated with
 *   the boxes in `all_boxes' (e.g. process number) can be extracted.
 *   Also, the size of the returned array is given by the size of the
 *  `all_boxes' zzz_BoxArray, so that this info does not have to be returned.
 *--------------------------------------------------------------------------*/

int *
zzz_FindBoxNeighborhood( zzz_BoxArray *boxes,
			 zzz_BoxArray *all_boxes,
			 zzz_StructStencil  *stencil   )
{
   int         *neighborhood_flags;

   zzz_Box     *box;
   zzz_Box     *shift_box;
   zzz_Box     *all_box;
   zzz_Box     *tmp_box;

   int          i, j, d, s;

   zzz_StructStencilElt  *stencil_shape = zzz_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Determine `neighborhood_flags'
    *-----------------------------------------------------------------------*/

   neighborhood_flags = ctalloc(int, zzz_BoxArraySize(all_boxes));

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);
      shift_box = zzz_DuplicateBox(box);

      for (s = 0; s < zzz_StructStencilSize(stencil); s++)
      {
	 for (d = 0; d < 3; d++)
	 {
	    zzz_BoxIMinD(shift_box, d) =
	       zzz_BoxIMinD(box, d) + stencil_shape[s][d];
	    zzz_BoxIMaxD(shift_box, d) =
	       zzz_BoxIMaxD(box, d) + stencil_shape[s][d];
	 }

	 zzz_ForBoxI(j, all_boxes)
	 {
	    all_box = zzz_BoxArrayBox(all_boxes, j);

	    tmp_box = zzz_IntersectBoxes(shift_box, all_box);
	    if (tmp_box)
	    {
	       neighborhood_flags[j] = 1;
	       zzz_FreeBox(tmp_box);
	    }
	 }
      }

      zzz_FreeBox(shift_box);
   }

   return neighborhood_flags;
}

/*--------------------------------------------------------------------------
 * zzz_FindBoxApproxNeighborhood:
 *
 *   Finds boxes in the `all_boxes' zzz_BoxArray that form an approximate
 *   neighborhood of the `boxes' zzz_BoxArray.  This neighborhood is
 *   determined by the min and max shape offsets of the stencil passed in.
 *   It contains the neighborhood computed by zzz_FindBoxNeighborhood.
 *
 *   The routine returns an integer array of boolean flags indicating
 *   which boxes in `all_boxes' are in the neighborhood and which are not.
 *   The reason for doing this instead of returning another BoxArray
 *   is so that additional information that may be associated with
 *   the boxes in `all_boxes' (e.g. process number) can be extracted.
 *   Also, the size of the returned array is given by the size of the
 *  `all_boxes' zzz_BoxArray, so that this info does not have to be returned.
 *--------------------------------------------------------------------------*/

int *
zzz_FindBoxApproxNeighborhood( zzz_BoxArray   *boxes,
			       zzz_BoxArray   *all_boxes,
			       zzz_StructStencil    *stencil   )
{
   int         *neighborhood_flags;

   zzz_Box     *box;
   zzz_Box     *grow_box;
   zzz_Box     *all_box;
   zzz_Box     *tmp_box;

   int          min_offset[3], max_offset[3];

   int          i, j, d, s;

   zzz_StructStencilElt  *stencil_shape = zzz_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Compute min and max stencil offsets
    *-----------------------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      min_offset[d] = 0;
      max_offset[d] = 0;
   }

   for (s = 0; s < zzz_StructStencilSize(stencil); s++)
   {
      for (d = 0; d < 3; d++)
      {
	 min_offset[d] = min(min_offset[d], stencil_shape[s][d]);
	 max_offset[d] = max(max_offset[d], stencil_shape[s][d]);
      }
   }

   /*-----------------------------------------------------------------------
    * Determine `neighborhood_flags'
    *-----------------------------------------------------------------------*/

   neighborhood_flags = ctalloc(int, zzz_BoxArraySize(all_boxes));

   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);

      /* grow the box */
      grow_box = zzz_DuplicateBox(box);
      for (d = 0; d < 3; d++)
      {
	 zzz_BoxIMinD(grow_box, d) += min_offset[d];
	 zzz_BoxIMaxD(grow_box, d) += max_offset[d];
      }

      zzz_ForBoxI(j, all_boxes)
      {
	 all_box = zzz_BoxArrayBox(all_boxes, j);

	 tmp_box = zzz_IntersectBoxes(grow_box, all_box);
	 if (tmp_box)
	 {
	    neighborhood_flags[j] = 1;
	    zzz_FreeBox(tmp_box);
	 }
      }

      zzz_FreeBox(grow_box);
   }

   return neighborhood_flags;
}
