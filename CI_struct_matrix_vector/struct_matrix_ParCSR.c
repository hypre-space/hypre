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
 * Member functions for hypre_StructInterfaceMatrix class for ParCSR storage scheme.
 *
 *****************************************************************************/

#include "headers.h"

/*
pound include "HYPRE_parcsr_mv.h"
*/

#include "parcsr_matrix_vector.h"


#define hypre_AddStructStencilEltToStructGridpoint( stencil_shape, index, new_index ) \
hypre_IndexD(new_index, 0) = hypre_IndexD(index, 0) + stencil_shape[0];\
hypre_IndexD(new_index, 1) = hypre_IndexD(index, 1) + stencil_shape[1];\
hypre_IndexD(new_index, 2) = hypre_IndexD(index, 2) + stencil_shape[2];

/*--------------------------------------------------------------------------
 * hypre_FreeStructInterfaceMatrixParCSR
 *   Internal routine for freeing a matrix stored in ParCSR form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructInterfaceMatrixParCSR( hypre_StructInterfaceMatrix *struct_matrix )
{
   HYPRE_ParCSRMatrix ParCSR_matrix = (HYPRE_ParCSRMatrix) hypre_StructInterfaceMatrixData(struct_matrix);


   printf("unimplemented function\n");
   return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceMatrixParCSRCoeffs
 *   
 *   Set elements in a StructStencil Matrix interface into ParCSR storage format. 
 *   Coefficients are referred to in stencil
 *   format; grid points are identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceMatrixParCSRCoeffs( hypre_StructInterfaceMatrix *struct_matrix, 
				 hypre_Index         *index,
				 double            *coeffs )
{
   int                         ierr;
   int                         i;

   /* variables meaningful to the interface */
   hypre_StructGrid                   *grid;
   hypre_StructStencil                *stencil;
   hypre_Index                  *new_index;

   /* variables meaningful to the storage format and translator */
   int                         row_coord, col_coord;
   hypre_StructGridToCoordTable       *grid_to_coord_table;
   hypre_StructGridToCoordTableEntry  *grid_to_coord_table_entry;


   printf("unimplemented function\n");
   ierr = -1;

#if 0
   new_index = hypre_NewIndex();

   grid    = hypre_StructInterfaceMatrixStructGrid(struct_matrix);
   stencil = hypre_StructInterfaceMatrixStructStencil(struct_matrix);

   /* If first coefficient, allocate data and build translator */
   if ( hypre_StructInterfaceMatrixData(struct_matrix) == NULL )
   {
      grid_to_coord_table = hypre_NewStructGridToCoordTable(grid, stencil);

      ierr = MatCreateMPIAIJ( hypre_StructInterfaceMatrixContext(struct_matrix), 
			      hypre_StructGridLocalSize(grid),
			      hypre_StructGridLocalSize(grid), 
			      hypre_StructGridGlobalSize(grid),
			      hypre_StructGridGlobalSize(grid), 
			      hypre_StructStencilSize(stencil), PETSC_NULL,
			      0, PETSC_NULL,
			      &ParCSR_matrix );

      hypre_StructInterfaceMatrixTranslator(struct_matrix) =
	 (void *) grid_to_coord_table;
      hypre_StructInterfaceMatrixData(struct_matrix) =
	 (void *) ParCSR_matrix;
   }
   else
   {
      grid_to_coord_table =
	 (hypre_StructGridToCoordTable *) hypre_StructInterfaceMatrixTranslator(struct_matrix);
      ParCSR_matrix =
	 (Mat) hypre_StructInterfaceMatrixData(struct_matrix);
   }

   grid_to_coord_table_entry =
      hypre_FindStructGridToCoordTableEntry( index, grid_to_coord_table );

   if (grid_to_coord_table_entry==NULL)
   {
      printf("Warning: Attempt to set coeffs for point not in grid\n");
      printf("SetCoeffs call aborted for grid point %d, %d, %d\n",
              hypre_IndexD(index, 0), hypre_IndexD(index,1), hypre_IndexD(index,2) );
      return(0);
   }

   row_coord = hypre_MapStructGridToCoord( index, grid_to_coord_table_entry );
        
   for (i = 0; i < hypre_StructStencilSize(stencil); i++)
   {
      if ( coeffs[i] != (double) 0.0 )
      {
	 hypre_AddStructStencilEltToStructGridpoint( hypre_StructStencilShape(stencil)[i],
				       index, new_index );

	 grid_to_coord_table_entry =
	    hypre_FindStructGridToCoordTableEntry( new_index, grid_to_coord_table );

	 if ( grid_to_coord_table_entry != NULL )
	 {
	    col_coord = hypre_MapStructGridToCoord( new_index,
					    grid_to_coord_table_entry );

	    ierr = MatSetValues ( ParCSR_matrix, 1, &row_coord, 1, &col_coord,
				  &(coeffs[i]), ADD_VALUES );   
 
            /*The following line is for symmetric matrices   */
            if ( hypre_StructInterfaceMatrixSymmetric(struct_matrix) &&
                (row_coord != col_coord) )
	       ierr = MatSetValues ( ParCSR_matrix, 1, &col_coord, 1, &row_coord,
				  &(coeffs[i]), ADD_VALUES );    
	 }

      } /*End of "if coeff not equal to zero" */

   } /* End of "for" loop over the elements of the stencil */

   hypre_FreeIndex(new_index);

#endif
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructInterfaceMatrixParCSR
 *   Internal routine for printing a matrix stored in ParCSR form.
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructInterfaceMatrixParCSR( hypre_StructInterfaceMatrix *struct_matrix )
{
   int  ierr=0;

   printf("unimplemented function\n");
   ierr = -1;
   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructInterfaceMatrixParCSR
 *   Internal routine for assembling a matrix stored in ParCSR form.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructInterfaceMatrixParCSR( hypre_StructInterfaceMatrix *struct_matrix )
{
   int  ierr=0;

   printf("unimplemented function\n");
   ierr = -1;
   return( ierr );
}

