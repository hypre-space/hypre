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
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewStructVector
 *--------------------------------------------------------------------------*/

HYPRE_StructVector
HYPRE_NewStructVector( MPI_Comm     context,
		      HYPRE_StructGrid     grid,
		      HYPRE_StructStencil  stencil )
{
   return ( (HYPRE_StructVector)
	    hypre_NewStructVector( context,
				  (hypre_StructGrid *) grid,
				  (hypre_StructStencil *) stencil ) );
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
 * HYPRE_SetStructVectorCoeffs
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructVectorCoeffs( HYPRE_StructVector  vector,
			    int               *grid_index,
			    double            *coeffs      )
{
   hypre_StructVector *new_vector;
   hypre_Index         *new_grid_index;

   int                d;
   int                ierr;

   new_vector = (hypre_StructVector *) vector;
   new_grid_index = hypre_NewIndex();
   for (d = 0;
	d < hypre_StructGridDim(hypre_StructVectorStructGrid(new_vector));
	d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_SetStructVectorCoeffs( new_vector, new_grid_index, coeffs );

   hypre_FreeIndex(new_grid_index);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructVector( HYPRE_StructVector  vector,
			    double      *val      )
{
   hypre_StructVector *new_vector;

   int                ierr;

   new_vector = (hypre_StructVector *) vector;

   ierr = hypre_SetStructVector( new_vector, val );

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
 * HYPRE_SetStructVectorStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_SetStructVectorStorageType( HYPRE_StructVector  struct_vector,
				 int                type           )
{
   return( hypre_SetStructVectorStorageType(
      (hypre_StructVector *) struct_vector, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetData
 *--------------------------------------------------------------------------*/

void *
HYPRE_StructVectorGetData( HYPRE_StructVector vector )
{
   return( hypre_StructVectorData( (hypre_StructVector *) vector ) );
}

/* OUTPUT */
/*--------------------------------------------------------------------------
 * HYPRE_PrintStructVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_PrintStructVector( HYPRE_StructVector vector )
{
   return( hypre_PrintStructVector( (hypre_StructVector *) vector ) );
}
