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
 * HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewStructVector
 *--------------------------------------------------------------------------*/

HYPRE_StructVector
HYPRE_NewStructVector( MPI_Comm             comm,
                       HYPRE_StructGrid     grid,
                       HYPRE_StructStencil  stencil )
{
   return ( (HYPRE_StructVector)
            hypre_NewStructVector( comm,
                                   (hypre_StructGrid *) grid ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeStructVector( HYPRE_StructVector struct_vector )
{
   return( hypre_FreeStructVector( (hypre_StructVector *) struct_vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeStructVector
 *--------------------------------------------------------------------------*/

int
HYPRE_InitializeStructVector( HYPRE_StructVector vector )
{
   return ( hypre_InitializeStructVector( (hypre_StructVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructVectorValues( HYPRE_StructVector  vector,
                             int                *grid_index,
                             double              values     )
{
   hypre_StructVector *new_vector = (hypre_StructVector *) vector;
   hypre_Index         new_grid_index;

   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0;
        d < hypre_StructGridDim(hypre_StructVectorGrid(new_vector));
        d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_SetStructVectorValues( new_vector, new_grid_index, values );

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_GetStructVectorValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_GetStructVectorValues( HYPRE_StructVector  vector,
                             int                *grid_index,
                             double             *values_ptr )
{
   hypre_StructVector *new_vector = (hypre_StructVector *) vector;
   hypre_Index         new_grid_index;

   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0;
        d < hypre_StructGridDim(hypre_StructVectorGrid(new_vector));
        d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_GetStructVectorValues( new_vector, new_grid_index,
                                       values_ptr );

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructVectorBoxValues( HYPRE_StructVector  vector,
                                int                *ilower,
                                int                *iupper,
                                double             *values              )
{
   hypre_StructVector *new_vector = (hypre_StructVector *) vector;
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
                    
   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0;
        d < hypre_StructGridDim(hypre_StructVectorGrid(new_vector));
        d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_NewBox(new_ilower, new_iupper);

   ierr = hypre_SetStructVectorBoxValues( new_vector, new_value_box, values );

   hypre_FreeBox(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_GetStructVectorBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_GetStructVectorBoxValues( HYPRE_StructVector  vector,
                                int                *ilower,
                                int                *iupper,
                                double            **values_ptr          )
{
   hypre_StructVector *new_vector = (hypre_StructVector *) vector;
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
                    
   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0;
        d < hypre_StructGridDim(hypre_StructVectorGrid(new_vector));
        d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_NewBox(new_ilower, new_iupper);

   ierr = hypre_GetStructVectorBoxValues( new_vector, new_value_box,
                                          values_ptr );

   hypre_FreeBox(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleStructVector( HYPRE_StructVector vector )
{
   return( hypre_AssembleStructVector( (hypre_StructVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintStructVector
 *--------------------------------------------------------------------------*/

void
HYPRE_PrintStructVector( char               *filename,
                         HYPRE_StructVector  vector,
                         int                 all )
{
   hypre_PrintStructVector( filename, (hypre_StructVector *) vector, all );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorNumGhost
 *--------------------------------------------------------------------------*/
 
void
HYPRE_SetStructVectorNumGhost( HYPRE_StructVector  vector,
                               int                *num_ghost )
{
   hypre_SetStructVectorNumGhost( (hypre_StructVector *) vector, num_ghost);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVectorConstantValues
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SetStructVectorConstantValues( HYPRE_StructVector  vector,
                                     double              values )
{
   return( hypre_SetStructVectorConstantValues( (hypre_StructVector *) vector,
                                                values) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetMigrateStructVectorCommPkg
 *--------------------------------------------------------------------------*/

HYPRE_CommPkg
HYPRE_GetMigrateStructVectorCommPkg( HYPRE_StructVector from_vector,
				     HYPRE_StructVector to_vector   )
{
   return( (HYPRE_CommPkg) 
	   hypre_GetMigrateStructVectorCommPkg( 
                		(hypre_StructVector *)from_vector, 
				(hypre_StructVector *)to_vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MigrateStructVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_MigrateStructVector( HYPRE_CommPkg      comm_pkg,
			   HYPRE_StructVector from_vector,
                           HYPRE_StructVector to_vector   )
{
  return( hypre_MigrateStructVector( (hypre_CommPkg *)comm_pkg,
				     (hypre_StructVector *)from_vector,
				     (hypre_StructVector *)to_vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeCommPkg
 *--------------------------------------------------------------------------*/

void 
HYPRE_FreeCommPkg( HYPRE_CommPkg comm_pkg )
{
  hypre_FreeCommPkg( (hypre_CommPkg *)comm_pkg );
}


