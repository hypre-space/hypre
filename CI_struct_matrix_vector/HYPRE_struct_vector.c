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
 * HYPRE_StructInterfaceVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructInterfaceVectorCreate( MPI_Comm             context,
		                   HYPRE_StructGrid     grid,
		                   HYPRE_StructStencil  stencil,
                                   HYPRE_StructInterfaceVector  *vector)
{
   *vector = (HYPRE_StructInterfaceVector)
	      hypre_NewStructInterfaceVector( context,
				  (hypre_StructGrid *) grid,
				  (hypre_StructStencil *) stencil );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorDestroy( HYPRE_StructInterfaceVector struct_vector )
{
   return( hypre_FreeStructInterfaceVector( (hypre_StructInterfaceVector *) struct_vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceVectorCoeffs
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructInterfaceVectorCoeffs( HYPRE_StructInterfaceVector  vector,
			    int               *grid_index,
			    double            *coeffs      )
{
   hypre_StructInterfaceVector *new_vector;
   hypre_Index         new_grid_index;

   int                d;
   int                ierr;

   new_vector = (hypre_StructInterfaceVector *) vector;
   for (d = 0;
	d < hypre_StructGridDim(hypre_StructInterfaceVectorStructGrid(new_vector));
	d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_SetStructInterfaceVectorCoeffs( new_vector, new_grid_index, coeffs );

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorSetValues( HYPRE_StructInterfaceVector  vector,
			    int               *grid_index,
			    double            *coeffs      )
{
   return ( 0 );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorGetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorGetValues( HYPRE_StructInterfaceVector  vector,
			    int               *grid_index,
			    double            *values_ptr     )
{
   return ( 0 );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorSetBoxValues( HYPRE_StructInterfaceVector  vector,
			    int               *lower_grid_index,
			    int               *upper_grid_index,
			    double            *coeffs      )
{
   hypre_StructInterfaceVector *new_vector;
   hypre_Index         new_lower_grid_index;
   hypre_Index         new_upper_grid_index;

   int                d;
   int                ierr;

   new_vector = (hypre_StructInterfaceVector *) vector;
   for (d = 0;
	d < hypre_StructGridDim(hypre_StructInterfaceVectorStructGrid(new_vector));
	d++)
   {
      hypre_IndexD(new_lower_grid_index, d) = lower_grid_index[d];
      hypre_IndexD(new_upper_grid_index, d) = upper_grid_index[d];
   }

   ierr = hypre_SetStructInterfaceVectorBoxValues( new_vector, new_lower_grid_index, new_upper_grid_index, coeffs );

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorGetBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorGetBoxValues( HYPRE_StructInterfaceVector  vector,
			    int               *lower_grid_index,
			    int               *upper_grid_index,
			    double            *values_ptr     )
{
   return ( 0 );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceVector
 *--------------------------------------------------------------------------*/


int 
HYPRE_SetStructInterfaceVector( HYPRE_StructInterfaceVector  vector,
			    double      *val      )
{
   hypre_StructInterfaceVector *new_vector;

   int                ierr;

   new_vector = (hypre_StructInterfaceVector *) vector;

   ierr = hypre_SetStructInterfaceVector( new_vector, val );

   return (ierr);
}


/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorInitialize( HYPRE_StructInterfaceVector vector )
{
  /* Currently a no-op */
   return( 0 );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorAssemble( HYPRE_StructInterfaceVector vector )
{
   return( hypre_AssembleStructInterfaceVector( (hypre_StructInterfaceVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceVectorStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_SetStructInterfaceVectorStorageType( HYPRE_StructInterfaceVector  struct_vector,
				 int                type           )
{
   return( hypre_SetStructInterfaceVectorStorageType(
      (hypre_StructInterfaceVector *) struct_vector, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorGetData
 *--------------------------------------------------------------------------*/

void *
HYPRE_StructInterfaceVectorGetData( HYPRE_StructInterfaceVector vector )
{
   return( hypre_StructInterfaceVectorData( (hypre_StructInterfaceVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorPrint
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceVectorPrint( HYPRE_StructInterfaceVector vector )
{
   return( hypre_PrintStructInterfaceVector( (hypre_StructInterfaceVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_RetrievalOnStructInterfaceVector
 *--------------------------------------------------------------------------*/


int 
HYPRE_RetrievalOnStructInterfaceVector( HYPRE_StructInterfaceVector vector )
{
   return( hypre_RetrievalOnStructInterfaceVector( (hypre_StructInterfaceVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_RetrievalOffStructInterfaceVector
 *--------------------------------------------------------------------------*/

int 
HYPRE_RetrievalOffStructInterfaceVector( HYPRE_StructInterfaceVector vector )
{
   return( hypre_RetrievalOffStructInterfaceVector( (hypre_StructInterfaceVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetStructInterfaceVectorValue
 *--------------------------------------------------------------------------*/

int 
HYPRE_GetStructInterfaceVectorValue( HYPRE_StructInterfaceVector vector, 
       int *index, double *value )
{
   int ierr=0;
   int d;
   hypre_Index         new_grid_index;

   for (d = 0;
	d < hypre_StructGridDim(hypre_StructInterfaceVectorStructGrid
           ((hypre_StructInterfaceVector *) vector));
	d++)
   {
      hypre_IndexD(new_grid_index, d) = index[d];
   }

   ierr = hypre_GetStructInterfaceVectorValue
          ( (hypre_StructInterfaceVector *) vector, new_grid_index, value );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceVectorSetNumGhost
 *--------------------------------------------------------------------------*/

int
HYPRE_StructInterfaceVectorSetNumGhost( HYPRE_StructInterfaceVector vector,
				        int *num_ghost )
{
   return( 0 );
}
