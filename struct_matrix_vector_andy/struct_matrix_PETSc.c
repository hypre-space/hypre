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
 * Member functions for hypre_StructMatrix class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "headers.h"
#include "sles.h"


#define hypre_AddStructStencilEltToStructGridpoint( stencil_shape, index, new_index ) \
hypre_IndexD(new_index, 0) = hypre_IndexD(index, 0) + stencil_shape[0];\
hypre_IndexD(new_index, 1) = hypre_IndexD(index, 1) + stencil_shape[1];\
hypre_IndexD(new_index, 2) = hypre_IndexD(index, 2) + stencil_shape[2];

/*--------------------------------------------------------------------------
 * hypre_FreeStructMatrixPETSc
 *   Internal routine for freeing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructMatrixPETSc( hypre_StructMatrix *struct_matrix )
{
   Mat PETSc_matrix = (Mat) hypre_StructMatrixData(struct_matrix);;

   hypre_FreeStructGridToCoordTable( (hypre_StructGridToCoordTable *)
			     hypre_StructMatrixTranslator(struct_matrix) );

   MatDestroy( PETSc_matrix );

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructMatrixPETScCoeffs
 *   
 *   Set elements in a StructStencil Matrix interface into PETSc storage format. 
 *   Coefficients are referred to in stencil
 *   format; grid points are identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructMatrixPETScCoeffs( hypre_StructMatrix *struct_matrix, 
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

   Mat                        PETSc_matrix;

   new_index = hypre_NewIndex();

   grid    = hypre_StructMatrixStructGrid(struct_matrix);
   stencil = hypre_StructMatrixStructStencil(struct_matrix);

   /* If first coefficient, allocate data and build translator */
   if ( hypre_StructMatrixData(struct_matrix) == NULL )
   {
      grid_to_coord_table = hypre_NewStructGridToCoordTable(grid, stencil);

      ierr = OptionsSetValue( "-mat_aij_oneindex", PETSC_NULL );
      ierr = MatCreateMPIAIJ( hypre_StructMatrixContext(struct_matrix), 
			      hypre_StructGridLocalSize(grid),
			      hypre_StructGridLocalSize(grid), 
			      hypre_StructGridGlobalSize(grid),
			      hypre_StructGridGlobalSize(grid), 
			      hypre_StructStencilSize(stencil), PETSC_NULL,
			      hypre_StructStencilSize(stencil), PETSC_NULL,
			      &PETSc_matrix );

      hypre_StructMatrixTranslator(struct_matrix) =
	 (void *) grid_to_coord_table;
      hypre_StructMatrixData(struct_matrix) =
	 (void *) PETSc_matrix;
   }
   else
   {
      grid_to_coord_table =
	 (hypre_StructGridToCoordTable *) hypre_StructMatrixTranslator(struct_matrix);
      PETSc_matrix =
	 (Mat) hypre_StructMatrixData(struct_matrix);
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

	    ierr = MatSetValues ( PETSc_matrix, 1, &row_coord, 1, &col_coord,
				  &(coeffs[i]), INSERT_VALUES );    
            /* ADD THE FOLLOWING LINE FOR SYMMETRIC MATRICES...     
	    ierr = MatSetValues ( *PETSc_matrix, 1, &col_coord, 1, &row_coord,
				  &(coeffs[i]), INSERT_VALUES );    */      
	 }

      } /*End of "if coeff not equal to zero" */

   } /* End of "for" loop over the elements of the stencil */

   hypre_FreeIndex(new_index);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructMatrixPETSc
 *   Internal routine for printing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructMatrixPETSc( hypre_StructMatrix *struct_matrix )
{
   Mat PETSc_matrix = (Mat ) hypre_StructMatrixData(struct_matrix);
   int  ierr;

   return( MatView( PETSc_matrix, VIEWER_STDOUT_WORLD ) );
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructMatrixPETSc
 *   Internal routine for assembling a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructMatrixPETSc( hypre_StructMatrix *struct_matrix )
{
   Mat PETSc_matrix = (Mat) hypre_StructMatrixData(struct_matrix);
   int  ierr;

   ierr = MatAssemblyBegin( PETSc_matrix, MAT_FINAL_ASSEMBLY );
   if (ierr)
      return( ierr );

   return( MatAssemblyEnd( PETSc_matrix, MAT_FINAL_ASSEMBLY ) );
}

