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
 * Member functions for zzz_StructMatrix class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "headers.h"
#include "sles.h"


#define zzz_AddStructStencilEltToStructGridpoint( stencil_shape, index, new_index ) \
zzz_IndexD(new_index, 0) = zzz_IndexD(index, 0) + stencil_shape[0];\
zzz_IndexD(new_index, 1) = zzz_IndexD(index, 1) + stencil_shape[1];\
zzz_IndexD(new_index, 2) = zzz_IndexD(index, 2) + stencil_shape[2];

/*--------------------------------------------------------------------------
 * zzz_FreeStructMatrixPETSc
 *   Internal routine for freeing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructMatrixPETSc( zzz_StructMatrix *struct_matrix )
{
   Mat *PETSc_matrix = (Mat *) zzz_StructMatrixData(struct_matrix);;

   zzz_FreeStructGridToCoordTable( (zzz_StructGridToCoordTable *)
			     zzz_StructMatrixTranslator(struct_matrix) );

   MatDestroy( *PETSc_matrix );

   tfree( PETSc_matrix );

   return(0);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructMatrixPETScCoeffs
 *   
 *   Set elements in a StructStencil Matrix interface into PETSc storage format. 
 *   Coefficients are referred to in stencil
 *   format; grid points are identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructMatrixPETScCoeffs( zzz_StructMatrix *struct_matrix, 
				 zzz_Index         *index,
				 double            *coeffs )
{
   int                         ierr;
   int                         i;

   /* variables meaningful to the interface */
   zzz_StructGrid                   *grid;
   zzz_StructStencil                *stencil;
   zzz_Index                  *new_index;

   /* variables meaningful to the storage format and translator */
   int                         row_coord, col_coord;
   zzz_StructGridToCoordTable       *grid_to_coord_table;
   zzz_StructGridToCoordTableEntry  *grid_to_coord_table_entry;

   Mat                        *PETSc_matrix;

   new_index = zzz_NewIndex();

   grid    = zzz_StructMatrixStructGrid(struct_matrix);
   stencil = zzz_StructMatrixStructStencil(struct_matrix);

   /* If first coefficient, allocate data and build translator */
   if ( zzz_StructMatrixData(struct_matrix) == NULL )
   {
      grid_to_coord_table = zzz_NewStructGridToCoordTable(grid, stencil);
      PETSc_matrix = talloc(Mat, 1);
      ierr = OptionsSetValue( "-mat_aij_oneindex", PETSC_NULL );
      ierr = MatCreateMPIAIJ( zzz_StructMatrixContext(struct_matrix), 
			      zzz_StructGridLocalSize(grid),
			      zzz_StructGridLocalSize(grid), 
			      zzz_StructGridGlobalSize(grid),
			      zzz_StructGridGlobalSize(grid), 
			      zzz_StructStencilSize(stencil), PETSC_NULL,
			      zzz_StructStencilSize(stencil), PETSC_NULL,
			      PETSc_matrix );

      zzz_StructMatrixTranslator(struct_matrix) =
	 (void *) grid_to_coord_table;
      zzz_StructMatrixData(struct_matrix) =
	 (void *) PETSc_matrix;
   }
   else
   {
      grid_to_coord_table =
	 (zzz_StructGridToCoordTable *) zzz_StructMatrixTranslator(struct_matrix);
      PETSc_matrix =
	 (Mat *) zzz_StructMatrixData(struct_matrix);
   }

   grid_to_coord_table_entry =
      zzz_FindStructGridToCoordTableEntry( index, grid_to_coord_table );

   if (grid_to_coord_table_entry==NULL)
   {
      printf("Warning: Attempt to set coeffs for point not in grid\n");
      printf("SetCoeffs call aborted for grid point %d, %d, %d\n",
              zzz_IndexD(index, 0), zzz_IndexD(index,1), zzz_IndexD(index,2) );
      return(0);
   }

   row_coord = zzz_MapStructGridToCoord( index, grid_to_coord_table_entry );
        
   for (i = 0; i < zzz_StructStencilSize(stencil); i++)
   {
      if ( coeffs[i] != (double) 0.0 )
      {
	 zzz_AddStructStencilEltToStructGridpoint( zzz_StructStencilShape(stencil)[i],
				       index, new_index );

	 grid_to_coord_table_entry =
	    zzz_FindStructGridToCoordTableEntry( new_index, grid_to_coord_table );

	 if ( grid_to_coord_table_entry != NULL )
	 {
	    col_coord = zzz_MapStructGridToCoord( new_index,
					    grid_to_coord_table_entry );

	    ierr = MatSetValues ( *PETSc_matrix, 1, &row_coord, 1, &col_coord,
				  &(coeffs[i]), INSERT_VALUES );    
            /* ADD THE FOLLOWING LINE FOR SYMMETRIC MATRICES...     
	    ierr = MatSetValues ( *PETSc_matrix, 1, &col_coord, 1, &row_coord,
				  &(coeffs[i]), INSERT_VALUES );    */      
	 }

      } /*End of "if coeff not equal to zero" */

   } /* End of "for" loop over the elements of the stencil */

   zzz_FreeIndex(new_index);

   return(0);
}

/*--------------------------------------------------------------------------
 * zzz_PrintStructMatrixPETSc
 *   Internal routine for printing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_PrintStructMatrixPETSc( zzz_StructMatrix *struct_matrix )
{
   Mat *PETSc_matrix = (Mat *) zzz_StructMatrixData(struct_matrix);
   int  ierr;

   return( MatView( *PETSc_matrix, VIEWER_STDOUT_WORLD ) );
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructMatrixPETSc
 *   Internal routine for assembling a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_AssembleStructMatrixPETSc( zzz_StructMatrix *struct_matrix )
{
   Mat *PETSc_matrix = (Mat *) zzz_StructMatrixData(struct_matrix);
   int  ierr;

   ierr = MatAssemblyBegin( *PETSc_matrix, MAT_FINAL_ASSEMBLY );
   if (ierr)
      return( ierr );

   return( MatAssemblyEnd( *PETSc_matrix, MAT_FINAL_ASSEMBLY ) );
}

