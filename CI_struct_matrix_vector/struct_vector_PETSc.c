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

#ifdef PETSC_AVAILABLE
#include "sles.h"
#endif


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
   return ierr;
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
				 hypre_Index         index,
				 double            *coeffs )
{
   int                         ierr=0;
#ifdef PETSC_AVAILABLE

   /* variables meaningful to the interface */
   hypre_StructGrid                   *grid;
   hypre_StructStencil                *stencil;
   hypre_Index                  new_index;

   /* variables meaningful to the storage format and translator */
   int                         row_coord, col_coord;
   hypre_StructGridToCoordTable       *grid_to_coord_table;
   hypre_StructGridToCoordTableEntry  *grid_to_coord_table_entry;

   Vec                        PETSc_vector;


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

/*--------------------------------------------------------------------------
 * hypre_RetrievalOnStructInterfaceVectorPETSc
 *--------------------------------------------------------------------------*/

int 
hypre_RetrievalOnStructInterfaceVectorPETSc( hypre_StructInterfaceVector *vector )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Vec PETSc_vector = (Vec) hypre_StructInterfaceVectorData(vector);

   /* Allocate Auxialiary data space to hold VecArray */
   hypre_StructInterfaceVectorAuxData(vector) =
      hypre_CTAlloc( hypre_StructInterfaceVectorPETScAD, 1);

   ierr = VecGetArray( PETSc_vector, &(hypre_StructInterfaceVectorVecArray(vector)) );

#endif
   return( ierr );

}

/*--------------------------------------------------------------------------
 * hypre_RetrievalOffStructInterfaceVectorPETSc
 *--------------------------------------------------------------------------*/

int 
hypre_RetrievalOffStructInterfaceVectorPETSc( hypre_StructInterfaceVector *vector )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Vec PETSc_vector = (Vec) hypre_StructInterfaceVectorData(vector);

   ierr = VecRestoreArray( PETSc_vector, &(hypre_StructInterfaceVectorVecArray(vector)) );

   /* DeAllocate Auxialiary data space to hold VecArray */
   hypre_TFree( hypre_StructInterfaceVectorAuxData(vector) );

#endif
   return( ierr );

}

/*--------------------------------------------------------------------------
 * hypre_GetStructInterfaceVectorPETScValue
 *--------------------------------------------------------------------------*/

int 
hypre_GetStructInterfaceVectorPETScValue( 
       hypre_StructInterfaceVector *vector, hypre_Index index, double *value )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Vec PETSc_vector ;

   /* variables meaningful to the interface */
   hypre_StructGrid                   *grid;
   hypre_StructStencil                *stencil;

   /* variables meaningful to the storage format and translator */
   int                         row_coord, col_coord;
   hypre_StructGridToCoordTable       *grid_to_coord_table;
   hypre_StructGridToCoordTableEntry  *grid_to_coord_table_entry;

   int low, high;


   grid    = hypre_StructInterfaceVectorStructGrid(vector);
   stencil = hypre_StructInterfaceVectorStructStencil(vector);

   grid_to_coord_table =
	 (hypre_StructGridToCoordTable *) hypre_StructInterfaceVectorTranslator(vector);
   PETSc_vector =
	 (Vec) hypre_StructInterfaceVectorData(vector);

   grid_to_coord_table_entry =
      hypre_FindStructGridToCoordTableEntry( index, grid_to_coord_table );

   row_coord = hypre_MapStructGridToCoord( index, grid_to_coord_table_entry );
        
   ierr = VecGetOwnershipRange ( PETSc_vector, &low, &high );         

   *value = hypre_StructInterfaceVectorVecArray(vector)[row_coord-low];
#endif
   return( ierr );

}

