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
 * Member functions for hypre_StructInterfaceMatrix class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "headers.h"
#include "sles.h"


#define hypre_AddStructStencilEltToStructGridpoint( stencil_shape, index, new_index ) \
hypre_IndexD(new_index, 0) = hypre_IndexD(index, 0) + stencil_shape[0];\
hypre_IndexD(new_index, 1) = hypre_IndexD(index, 1) + stencil_shape[1];\
hypre_IndexD(new_index, 2) = hypre_IndexD(index, 2) + stencil_shape[2];

/*--------------------------------------------------------------------------
 * hypre_FreeStructInterfaceMatrixPETSc
 *   Internal routine for freeing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructInterfaceMatrixPETSc( hypre_StructInterfaceMatrix *struct_matrix )
{
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_StructInterfaceMatrixData(struct_matrix);

   MatDestroy( PETSc_matrix );
#endif
   hypre_FreeStructGridToCoordTable( (hypre_StructGridToCoordTable *)
			     hypre_StructInterfaceMatrixTranslator(struct_matrix) );

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceMatrixPETScCoeffs
 *   
 *   Set elements in a StructStencil Matrix interface into PETSc storage format. 
 *   Coefficients are referred to in stencil
 *   format; grid points are identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceMatrixPETScCoeffs( hypre_StructInterfaceMatrix *struct_matrix, 
				 hypre_Index         *index,
				 double            *coeffs )
{
#ifdef PETSC_AVAILABLE
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
			      &PETSc_matrix );

      hypre_StructInterfaceMatrixTranslator(struct_matrix) =
	 (void *) grid_to_coord_table;
      hypre_StructInterfaceMatrixData(struct_matrix) =
	 (void *) PETSc_matrix;
   }
   else
   {
      grid_to_coord_table =
	 (hypre_StructGridToCoordTable *) hypre_StructInterfaceMatrixTranslator(struct_matrix);
      PETSc_matrix =
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

	    ierr = MatSetValues ( PETSc_matrix, 1, &row_coord, 1, &col_coord,
				  &(coeffs[i]), ADD_VALUES );   
 
            /*The following line is for symmetric matrices   */
            if ( hypre_StructInterfaceMatrixSymmetric(struct_matrix) &&
                (row_coord != col_coord) )
	       ierr = MatSetValues ( PETSc_matrix, 1, &col_coord, 1, &row_coord,
				  &(coeffs[i]), ADD_VALUES );    
	 }

      } /*End of "if coeff not equal to zero" */

   } /* End of "for" loop over the elements of the stencil */

   hypre_FreeIndex(new_index);

#endif
   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructInterfaceMatrixPETSc
 *   Internal routine for printing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructInterfaceMatrixPETSc( hypre_StructInterfaceMatrix *struct_matrix )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat ) hypre_StructInterfaceMatrixData(struct_matrix);

   ierr = MatView( PETSc_matrix, VIEWER_STDOUT_WORLD );
#endif
   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructInterfaceMatrixPETSc
 *   Internal routine for assembling a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructInterfaceMatrixPETSc( hypre_StructInterfaceMatrix *struct_matrix )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_StructInterfaceMatrixData(struct_matrix);

   ierr = MatAssemblyBegin( PETSc_matrix, MAT_FINAL_ASSEMBLY );
   if (ierr)
      return( ierr );

   ierr = MatAssemblyEnd( PETSc_matrix, MAT_FINAL_ASSEMBLY );
#endif
   return( ierr );
}

