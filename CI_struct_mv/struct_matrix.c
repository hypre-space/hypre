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
 * Member functions for hypre_StructInterfaceMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructInterfaceMatrix
 *--------------------------------------------------------------------------*/

hypre_StructInterfaceMatrix *
hypre_NewStructInterfaceMatrix( MPI_Comm     context,
		      hypre_StructGrid    *grid,
		      hypre_StructStencil *stencil )
{
   hypre_StructInterfaceMatrix    *matrix;

   matrix = hypre_CTAlloc(hypre_StructInterfaceMatrix, 1);

   hypre_StructInterfaceMatrixContext(matrix) = context;
   hypre_StructInterfaceMatrixStructGrid(matrix)    = grid;
   hypre_StructInterfaceMatrixStructStencil(matrix) = stencil;

   /* set defaults */
   hypre_StructInterfaceMatrixStorageType(matrix) = 0;
   hypre_StructInterfaceMatrixSymmetric(matrix) = 0;
   hypre_StructInterfaceMatrixTranslator(matrix) = NULL;
   hypre_StructInterfaceMatrixData(matrix) = NULL;

   hypre_SetStructInterfaceMatrixStorageType(matrix, HYPRE_PETSC_MATRIX);

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructInterfaceMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructInterfaceMatrix( hypre_StructInterfaceMatrix *matrix )
{

   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      hypre_FreeStructInterfaceMatrixPETSc( matrix );
   else
   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PARCSR_MATRIX )
      hypre_FreeStructInterfaceMatrixParCSR( matrix );
   else
      return(-1);

   hypre_TFree(matrix);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceMatrixCoeffs
 *   
 *   Set elements in a Struct Matrix interface.
 *   Coefficients are referred to in stencil format; grid points are
 *   identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceMatrixCoeffs( hypre_StructInterfaceMatrix *matrix,
			    hypre_Index         *grid_index,
			    double            *coeffs     )
{
   int    ierr;

   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_SetStructInterfaceMatrixPETScCoeffs( matrix, grid_index, coeffs ) );
   else
   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PARCSR_MATRIX )
      return( hypre_SetStructInterfaceMatrixParCSRCoeffs( matrix, grid_index, coeffs ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceMatrixBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceMatrixBoxValues( hypre_StructInterfaceMatrix *matrix,
			    hypre_Index         *lower_grid_index,
			    hypre_Index         *upper_grid_index,
                            int                 num_stencil_indices,
                            int                 *stencil_indices,
			    double            *coeffs      )
     /* 
        USES: hypre_SetStructInterfaceMatrixCoeffs
        ASSUMES: that hypre_SetStructInterfaceMatrixCoeffs is used in an
                 ADD mode, that is, coefficients sent to this function are added
                 to any (if any) previously entered values for the same coefficient  
     */
{
   hypre_Index *loop_index;
   hypre_StructStencil *stencil;
   int         ierr=0;
   int         stencil_size;
   int         i, j, k, l, coeffs_index;
   double      *coeffs_buffer;

   /* Allocate loop_index */
   loop_index = hypre_CTAlloc( hypre_Index, 3); 

   /* Get stencil object out of matrix object */
   stencil = hypre_StructInterfaceMatrixStructStencil( matrix );

   /* Get size of stencil */
   stencil_size = hypre_StructStencilSize( stencil );

   /* Allocate coeffs_buffer to be size of stencil and zero it out */
   coeffs_buffer = hypre_CTAlloc( double, stencil_size );

   /* Insert coefficients one grid point at a time */
   for (k = hypre_IndexZ(lower_grid_index), coeffs_index = 0; k <= hypre_IndexZ(upper_grid_index); k++)
      for (j = hypre_IndexY(lower_grid_index); j <= hypre_IndexY(upper_grid_index); j++)
         for (i = hypre_IndexX(lower_grid_index); i <= hypre_IndexX(upper_grid_index); i++, coeffs_index += num_stencil_indices)
         /* Loop over grid dimensions specified in input arguments */
         {
            hypre_SetIndex(loop_index, i, j, k);

            /* Get non-zero coefficients out of coeffs and into form for call
               to hypre_SetStructInterfaceMatrixCoeffs */

            for ( l=0; l < num_stencil_indices; l++ )
            /* Loop over stencil_indices */
            {
               /* Copy coefficient from coeffs to coeffs_buffer */
               coeffs_buffer[ stencil_indices[ l ] ] = coeffs[ coeffs_index + l ];

            }
            /* End Loop over stencil_indices */

            /* Insert coefficients in coeffs_buffer */
            ierr = hypre_SetStructInterfaceMatrixCoeffs( 
                            matrix,
			    loop_index,
			    coeffs_buffer     );

         }
   /* End Loop from lower_grid_index to upper_grid index */


   hypre_TFree( loop_index );
   hypre_TFree( coeffs_buffer );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructInterfaceMatrix
 *   User-level routine for assembling hypre_StructInterfaceMatrix.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructInterfaceMatrix( hypre_StructInterfaceMatrix *matrix )
{
   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_AssembleStructInterfaceMatrixPETSc( matrix ) );
   else
   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PARCSR_MATRIX )
      return( hypre_AssembleStructInterfaceMatrixParCSR( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructInterfaceMatrix
 *   
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructInterfaceMatrix( hypre_StructInterfaceMatrix *matrix )
{
   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PETSC_MATRIX )
      return( hypre_PrintStructInterfaceMatrixPETSc( matrix ) );
   else
   if ( hypre_StructInterfaceMatrixStorageType(matrix) == HYPRE_PARCSR_MATRIX )
      return( hypre_PrintStructInterfaceMatrixParCSR( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceMatrixStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceMatrixStorageType( hypre_StructInterfaceMatrix *matrix,
				 int                type   )
{
   hypre_StructInterfaceMatrixStorageType(matrix) = type;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceMatrixSymmetric
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceMatrixSymmetric( hypre_StructInterfaceMatrix *matrix,
				 int                type   )
{
   hypre_StructInterfaceMatrixSymmetric(matrix) = type;

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
	 min_offset[d] = hypre_min(min_offset[d], stencil_shape[s][d]);
	 max_offset[d] = hypre_max(max_offset[d], stencil_shape[s][d]);
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
