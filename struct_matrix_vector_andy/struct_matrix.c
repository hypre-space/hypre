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
 * Member functions for hypre_StructMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructMatrix
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_NewStructMatrix( MPI_Comm     context,
		      hypre_StructGrid    *grid,
		      hypre_StructStencil *stencil )
{
   hypre_StructMatrix    *matrix;

   matrix = hypre_CTAlloc(hypre_StructMatrix, 1);

   hypre_StructMatrixContext(matrix) = context;
   hypre_StructMatrixStructGrid(matrix)    = grid;
   hypre_StructMatrixStructStencil(matrix) = stencil;

   hypre_StructMatrixTranslator(matrix) = NULL;
   hypre_StructMatrixStorageType(matrix) = 0;
   hypre_StructMatrixData(matrix) = NULL;

   /* set defaults */
   hypre_SetStructMatrixStorageType(matrix, HYPRE_PETSC_MATRIX);

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructMatrix( hypre_StructMatrix *matrix )
{

   if ( hypre_StructMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      hypre_FreeStructMatrixPETSc( matrix );
   else
      return(-1);

   hypre_TFree(matrix);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructMatrixCoeffs
 *   
 *   Set elements in a Struct Matrix interface.
 *   Coefficients are referred to in stencil format; grid points are
 *   identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructMatrixCoeffs( hypre_StructMatrix *matrix,
			    hypre_Index         *grid_index,
			    double            *coeffs     )
{
   int    ierr;

   if ( hypre_StructMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_SetStructMatrixPETScCoeffs( matrix, grid_index, coeffs ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructMatrix
 *   User-level routine for assembling hypre_StructMatrix.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructMatrix( hypre_StructMatrix *matrix )
{
   if ( hypre_StructMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_AssembleStructMatrixPETSc( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructMatrix
 *   
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructMatrix( hypre_StructMatrix *matrix )
{
   if ( hypre_StructMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_PrintStructMatrixPETSc( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructMatrixStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructMatrixStorageType( hypre_StructMatrix *matrix,
				 int                type   )
{
   hypre_StructMatrixStorageType(matrix) = type;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_FindBoxNeighborhood:
 *
 *   Finds boxes in the `all_boxes' hypre_BoxArray that form a neighborhood
 *   of the `boxes' hypre_BoxArray.  This neighborhood is determined by the
 *   shape of the stencil passed in and represents the minimum number
 *   of boxes touched by the stencil elements.
 *
 *   The routine returns an integer array of boolean flags indicating
 *   which boxes in `all_boxes' are in the neighborhood and which are not.
 *   The reason for doing this instead of returning another hypre_BoxArray
 *   is so that additional information that may be associated with
 *   the boxes in `all_boxes' (e.g. process number) can be extracted.
 *   Also, the size of the returned array is given by the size of the
 *  `all_boxes' hypre_BoxArray, so that this info does not have to be returned.
 *--------------------------------------------------------------------------*/

int *
hypre_FindBoxNeighborhood( hypre_BoxArray *boxes,
			 hypre_BoxArray *all_boxes,
			 hypre_StructStencil  *stencil   )
{
   int         *neighborhood_flags;

   hypre_Box     *box;
   hypre_Box     *shift_box;
   hypre_Box     *all_box;
   hypre_Box     *tmp_box;

   int          i, j, d, s;

   hypre_StructStencilElt  *stencil_shape = hypre_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Determine `neighborhood_flags'
    *-----------------------------------------------------------------------*/

   neighborhood_flags = hypre_CTAlloc(int, hypre_BoxArraySize(all_boxes));

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      shift_box = hypre_DuplicateBox(box);

      for (s = 0; s < hypre_StructStencilSize(stencil); s++)
      {
	 for (d = 0; d < 3; d++)
	 {
	    hypre_BoxIMinD(shift_box, d) =
	       hypre_BoxIMinD(box, d) + stencil_shape[s][d];
	    hypre_BoxIMaxD(shift_box, d) =
	       hypre_BoxIMaxD(box, d) + stencil_shape[s][d];
	 }

	 hypre_ForBoxI(j, all_boxes)
	 {
	    all_box = hypre_BoxArrayBox(all_boxes, j);

	    tmp_box = hypre_IntersectBoxes(shift_box, all_box);
	    if (tmp_box)
	    {
	       neighborhood_flags[j] = 1;
	       hypre_FreeBox(tmp_box);
	    }
	 }
      }

      hypre_FreeBox(shift_box);
   }

   return neighborhood_flags;
}

/*--------------------------------------------------------------------------
 * hypre_FindBoxApproxNeighborhood:
 *
 *   Finds boxes in the `all_boxes' hypre_BoxArray that form an approximate
 *   neighborhood of the `boxes' hypre_BoxArray.  This neighborhood is
 *   determined by the min and max shape offsets of the stencil passed in.
 *   It contains the neighborhood computed by hypre_FindBoxNeighborhood.
 *
 *   The routine returns an integer array of boolean flags indicating
 *   which boxes in `all_boxes' are in the neighborhood and which are not.
 *   The reason for doing this instead of returning another BoxArray
 *   is so that additional information that may be associated with
 *   the boxes in `all_boxes' (e.g. process number) can be extracted.
 *   Also, the size of the returned array is given by the size of the
 *  `all_boxes' hypre_BoxArray, so that this info does not have to be returned.
 *--------------------------------------------------------------------------*/

int *
hypre_FindBoxApproxNeighborhood( hypre_BoxArray   *boxes,
			       hypre_BoxArray   *all_boxes,
			       hypre_StructStencil    *stencil   )
{
   int         *neighborhood_flags;

   hypre_Box     *box;
   hypre_Box     *grow_box;
   hypre_Box     *all_box;
   hypre_Box     *tmp_box;

   int          min_offset[3], max_offset[3];

   int          i, j, d, s;

   hypre_StructStencilElt  *stencil_shape = hypre_StructStencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Compute min and max stencil offsets
    *-----------------------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      min_offset[d] = 0;
      max_offset[d] = 0;
   }

   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
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

   neighborhood_flags = hypre_CTAlloc(int, hypre_BoxArraySize(all_boxes));

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      /* grow the box */
      grow_box = hypre_DuplicateBox(box);
      for (d = 0; d < 3; d++)
      {
	 hypre_BoxIMinD(grow_box, d) += min_offset[d];
	 hypre_BoxIMaxD(grow_box, d) += max_offset[d];
      }

      hypre_ForBoxI(j, all_boxes)
      {
	 all_box = hypre_BoxArrayBox(all_boxes, j);

	 tmp_box = hypre_IntersectBoxes(grow_box, all_box);
	 if (tmp_box)
	 {
	    neighborhood_flags[j] = 1;
	    hypre_FreeBox(tmp_box);
	 }
      }

      hypre_FreeBox(grow_box);
   }

   return neighborhood_flags;
}
