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
 * Member functions for zzz_StructVector class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "headers.h"
#include "sles.h"


/*--------------------------------------------------------------------------
 * zzz_FreeStructVectorPETSc
 *   Internal routine for freeing a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructVectorPETSc( zzz_StructVector *struct_vector )
{
   Vec *PETSc_vector = (Vec *) zzz_StructVectorData(struct_vector);

   int  ierr;

   zzz_FreeStructGridToCoordTable( (zzz_StructGridToCoordTable *)
			     zzz_StructVectorTranslator(struct_vector) );

   VecDestroy( *PETSc_vector );

   tfree( PETSc_vector );

   return(0);
}


/*--------------------------------------------------------------------------
 * zzz_SetStructVectorPETScCoeffs
 *   
 *   Set elements in a Struct Vector interface into PETSc storage format. 
 *   StructGrid points are identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructVectorPETScCoeffs( zzz_StructVector *struct_vector, 
				 zzz_Index         *index,
				 double            *coeffs )
{
   int                         ierr;

   /* variables meaningful to the interface */
   zzz_StructGrid                   *grid;
   zzz_StructStencil                *stencil;
   zzz_Index                  *new_index;

   /* variables meaningful to the storage format and translator */
   int                         row_coord, col_coord;
   zzz_StructGridToCoordTable       *grid_to_coord_table;
   zzz_StructGridToCoordTableEntry  *grid_to_coord_table_entry;

   Vec                        *PETSc_vector;


   new_index = zzz_NewIndex();

   grid    = zzz_StructVectorStructGrid(struct_vector);
   stencil = zzz_StructVectorStructStencil(struct_vector);

   /* If first coefficient, allocate data and build translator */
   if ( zzz_StructVectorData(struct_vector) == NULL )
   {
      grid_to_coord_table = zzz_NewStructGridToCoordTable(grid, stencil);
      PETSc_vector = talloc(Vec, 1);
      ierr = VecCreateMPI( zzz_StructVectorContext(struct_vector), 
			      zzz_StructGridLocalSize(grid),
			      zzz_StructGridGlobalSize(grid), 
			      PETSc_vector );
      if (ierr) return(ierr);

      zzz_StructVectorTranslator(struct_vector) =
	 (void *) grid_to_coord_table;
      zzz_StructVectorData(struct_vector) =
	 (void *) PETSc_vector;
   }
   else
   {
      grid_to_coord_table =
	 (zzz_StructGridToCoordTable *) zzz_StructVectorTranslator(struct_vector);
      PETSc_vector =
	 (Vec *) zzz_StructVectorData(struct_vector);
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
        
   ierr = VecSetValues ( *PETSc_vector, 1, &row_coord, 
				  &(coeffs[0]), INSERT_VALUES );         

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructVectorPETSc
 *   Internal routine for setting a vector stored in PETSc form to a value.
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructVectorPETSc( zzz_StructVector *struct_vector, 
                           double            *val )
{
   Vec                        *PETSc_vector;
   zzz_StructGridToCoordTable       *grid_to_coord_table;
   int                         ierr;

   /* If first coefficient, allocate data and build translator */
   if ( zzz_StructVectorData(struct_vector) == NULL )
   {
      grid_to_coord_table = zzz_NewStructGridToCoordTable(
                            zzz_StructVectorStructGrid(struct_vector), 
                            zzz_StructVectorStructStencil(struct_vector));
      PETSc_vector = talloc(Vec, 1);
      ierr = VecCreateMPI( zzz_StructVectorContext(struct_vector), 
			      zzz_StructGridLocalSize(zzz_StructVectorStructGrid(struct_vector)),
			      zzz_StructGridGlobalSize(zzz_StructVectorStructGrid(struct_vector)), 
			      PETSc_vector );
      if (ierr) return(ierr);

      zzz_StructVectorTranslator(struct_vector) =
	 (void *) grid_to_coord_table;
      zzz_StructVectorData(struct_vector) =
	 (void *) PETSc_vector;
   }

   return( VecSet( val, *PETSc_vector ) );
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructVectorPETSc
 *   Internal routine for assembling a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_AssembleStructVectorPETSc( zzz_StructVector *struct_vector )
{
   Vec *PETSc_vector = (Vec *) zzz_StructVectorData(struct_vector);

   int  ierr;

   ierr = VecAssemblyBegin( *PETSc_vector );
   if (ierr)
      return( ierr );

   return( VecAssemblyEnd( *PETSc_vector ) );

}

/*--------------------------------------------------------------------------
 * zzz_PrintStructVectorPETSc
 *   Internal routine for printing a vector stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_PrintStructVectorPETSc( zzz_StructVector *struct_vector )
{
   Vec *PETSc_vector = (Vec *) zzz_StructVectorData(struct_vector);

   int  ierr;

   ierr = VecView( *PETSc_vector, VIEWER_STDOUT_WORLD );

   return( ierr );

}

