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
 * Member functions for hypre_StructVector class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "headers.h"
#include "sles.h"


/*--------------------------------------------------------------------------
 * hypre_FreeStructVectorPETSc
 *   Internal routine for freeing a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructVectorPETSc( hypre_StructVector *struct_vector )
{
   Vec PETSc_vector = (Vec) hypre_StructVectorData(struct_vector);

   int  ierr;

   hypre_FreeStructGridToCoordTable( (hypre_StructGridToCoordTable *)
			     hypre_StructVectorTranslator(struct_vector) );

   VecDestroy(PETSc_vector );

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_SetStructVectorPETScCoeffs
 *   
 *   Set elements in a Struct Vector interface into PETSc storage format. 
 *   StructGrid points are identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorPETScCoeffs( hypre_StructVector *struct_vector, 
				 hypre_Index         *index,
				 double            *coeffs )
{
   int                         ierr;

   /* variables meaningful to the interface */
   hypre_StructGrid                   *grid;
   hypre_StructStencil                *stencil;
   hypre_Index                  *new_index;

   /* variables meaningful to the storage format and translator */
   int                         row_coord, col_coord;
   hypre_StructGridToCoordTable       *grid_to_coord_table;
   hypre_StructGridToCoordTableEntry  *grid_to_coord_table_entry;

   Vec                        PETSc_vector;


   new_index = hypre_NewIndex();

   grid    = hypre_StructVectorStructGrid(struct_vector);
   stencil = hypre_StructVectorStructStencil(struct_vector);

   /* If first coefficient, allocate data and build translator */
   if ( hypre_StructVectorData(struct_vector) == NULL )
   {
      grid_to_coord_table = hypre_NewStructGridToCoordTable(grid, stencil);

      ierr = VecCreateMPI( hypre_StructVectorContext(struct_vector), 
			      hypre_StructGridLocalSize(grid),
			      hypre_StructGridGlobalSize(grid), 
			      &PETSc_vector );
      if (ierr) return(ierr);

      hypre_StructVectorTranslator(struct_vector) =
	 (void *) grid_to_coord_table;
      hypre_StructVectorData(struct_vector) =
	 (void *) PETSc_vector;
   }
   else
   {
      grid_to_coord_table =
	 (hypre_StructGridToCoordTable *) hypre_StructVectorTranslator(struct_vector);
      PETSc_vector =
	 (Vec) hypre_StructVectorData(struct_vector);
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
        
   ierr = VecSetValues ( PETSc_vector, 1, &row_coord, 
				  &(coeffs[0]), INSERT_VALUES );         

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorPETSc
 *   Internal routine for setting a vector stored in PETSc form to a value.
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorPETSc( hypre_StructVector *struct_vector, 
                           double            *val )
{
   Vec                        PETSc_vector;
   hypre_StructGridToCoordTable       *grid_to_coord_table;
   int                         ierr;

   /* If first coefficient, allocate data and build translator */
   if ( hypre_StructVectorData(struct_vector) == NULL )
   {
      grid_to_coord_table = hypre_NewStructGridToCoordTable(
                            hypre_StructVectorStructGrid(struct_vector), 
                            hypre_StructVectorStructStencil(struct_vector));

      ierr = VecCreateMPI( hypre_StructVectorContext(struct_vector), 
			      hypre_StructGridLocalSize(hypre_StructVectorStructGrid(struct_vector)),
			      hypre_StructGridGlobalSize(hypre_StructVectorStructGrid(struct_vector)), 
			      &PETSc_vector );
      if (ierr) return(ierr);

      hypre_StructVectorTranslator(struct_vector) =
	 (void *) grid_to_coord_table;
      hypre_StructVectorData(struct_vector) =
	 (void *) PETSc_vector;
   }

   return( VecSet( val, PETSc_vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructVectorPETSc
 *   Internal routine for assembling a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructVectorPETSc( hypre_StructVector *struct_vector )
{
   Vec PETSc_vector = (Vec) hypre_StructVectorData(struct_vector);

   int  ierr;

   ierr = VecAssemblyBegin( PETSc_vector );
   if (ierr)
      return( ierr );

   return( VecAssemblyEnd( PETSc_vector ) );

}

/*--------------------------------------------------------------------------
 * hypre_PrintStructVectorPETSc
 *   Internal routine for printing a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructVectorPETSc( hypre_StructVector *struct_vector )
{
   Vec PETSc_vector = (Vec) hypre_StructVectorData(struct_vector);

   int  ierr;

   ierr = VecView( PETSc_vector, VIEWER_STDOUT_WORLD );

   return( ierr );

}

