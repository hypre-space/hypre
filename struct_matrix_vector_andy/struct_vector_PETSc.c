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
 * Member functions for hypre_StructInterfaceVector class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "headers.h"
#include "sles.h"


/*--------------------------------------------------------------------------
 * hypre_FreeStructInterfaceVectorPETSc
 *   Internal routine for freeing a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructInterfaceVectorPETSc( hypre_StructInterfaceVector *struct_vector )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Vec PETSc_vector = (Vec) hypre_StructInterfaceVectorData(struct_vector);


   hypre_FreeStructGridToCoordTable( (hypre_StructGridToCoordTable *)
			     hypre_StructInterfaceVectorTranslator(struct_vector) );

   VecDestroy(PETSc_vector );
#endif
   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceVectorPETScCoeffs
 *   
 *   Set elements in a Struct Vector interface into PETSc storage format. 
 *   StructGrid points are identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceVectorPETScCoeffs( hypre_StructInterfaceVector *struct_vector, 
				 hypre_Index         *index,
				 double            *coeffs )
{
   int                         ierr=0;
#ifdef PETSC_AVAILABLE

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

   grid    = hypre_StructInterfaceVectorStructGrid(struct_vector);
   stencil = hypre_StructInterfaceVectorStructStencil(struct_vector);

   /* If first coefficient, allocate data and build translator */
   if ( hypre_StructInterfaceVectorData(struct_vector) == NULL )
   {
      grid_to_coord_table = hypre_NewStructGridToCoordTable(grid, stencil);

      ierr = VecCreateMPI( hypre_StructInterfaceVectorContext(struct_vector), 
			      hypre_StructGridLocalSize(grid),
			      hypre_StructGridGlobalSize(grid), 
			      &PETSc_vector );
      if (ierr) return(ierr);

      hypre_StructInterfaceVectorTranslator(struct_vector) =
	 (void *) grid_to_coord_table;
      hypre_StructInterfaceVectorData(struct_vector) =
	 (void *) PETSc_vector;
   }
   else
   {
      grid_to_coord_table =
	 (hypre_StructGridToCoordTable *) hypre_StructInterfaceVectorTranslator(struct_vector);
      PETSc_vector =
	 (Vec) hypre_StructInterfaceVectorData(struct_vector);
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

#endif
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceVectorPETSc
 *   Internal routine for setting a vector stored in PETSc form to a value.
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceVectorPETSc( hypre_StructInterfaceVector *struct_vector, 
                           double            *val )
{
   int                         ierr=0;
#ifdef PETSC_AVAILABLE
   Vec                        PETSc_vector;
   hypre_StructGridToCoordTable       *grid_to_coord_table;

   /* If first coefficient, allocate data and build translator */
   if ( hypre_StructInterfaceVectorData(struct_vector) == NULL )
   {
      grid_to_coord_table = hypre_NewStructGridToCoordTable(
                            hypre_StructInterfaceVectorStructGrid(struct_vector), 
                            hypre_StructInterfaceVectorStructStencil(struct_vector));

      ierr = VecCreateMPI( hypre_StructInterfaceVectorContext(struct_vector), 
			      hypre_StructGridLocalSize(hypre_StructInterfaceVectorStructGrid(struct_vector)),
			      hypre_StructGridGlobalSize(hypre_StructInterfaceVectorStructGrid(struct_vector)), 
			      &PETSc_vector );
      if (ierr) return(ierr);

      hypre_StructInterfaceVectorTranslator(struct_vector) =
	 (void *) grid_to_coord_table;
      hypre_StructInterfaceVectorData(struct_vector) =
	 (void *) PETSc_vector;
   }

   ierr = VecSet( val, PETSc_vector );
#endif
   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructInterfaceVectorPETSc
 *   Internal routine for assembling a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructInterfaceVectorPETSc( hypre_StructInterfaceVector *struct_vector )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Vec PETSc_vector = (Vec) hypre_StructInterfaceVectorData(struct_vector);

   ierr = VecAssemblyBegin( PETSc_vector );
   if (ierr)
      return( ierr );

   ierr = VecAssemblyEnd( PETSc_vector );
#endif
   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructInterfaceVectorPETSc
 *   Internal routine for printing a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructInterfaceVectorPETSc( hypre_StructInterfaceVector *struct_vector )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Vec PETSc_vector = (Vec) hypre_StructInterfaceVectorData(struct_vector);


   ierr = VecView( PETSc_vector, VIEWER_STDOUT_WORLD );

#endif
   return( ierr );

}

