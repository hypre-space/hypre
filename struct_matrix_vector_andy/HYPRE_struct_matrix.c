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
 * HYPRE_NewStructMatrix
 *--------------------------------------------------------------------------*/

HYPRE_StructMatrix 
HYPRE_NewStructMatrix( MPI_Comm    context,
		      HYPRE_StructGrid    grid,
		      HYPRE_StructStencil stencil )
{
   return ( (HYPRE_StructMatrix)
	    hypre_NewStructMatrix( context,
				  (hypre_StructGrid *) grid,
				  (hypre_StructStencil *) stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeStructMatrix( HYPRE_StructMatrix struct_matrix )
{
   return( hypre_FreeStructMatrix( (hypre_StructMatrix *) struct_matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixCoeffs
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructMatrixCoeffs( HYPRE_StructMatrix  matrix,
			    int               *grid_index,
			    double            *coeffs      )
{
   hypre_StructMatrix *new_matrix;
   hypre_Index         *new_grid_index;

   int                d;
   int                ierr;

   new_matrix = (hypre_StructMatrix *) matrix;
   new_grid_index = hypre_NewIndex();
   for (d = 0;
	d < hypre_StructGridDim(hypre_StructMatrixStructGrid(new_matrix));
	d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_SetStructMatrixCoeffs( new_matrix, new_grid_index, coeffs );

   hypre_FreeIndex(new_grid_index);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleStructMatrix( HYPRE_StructMatrix matrix )
{
   return( hypre_AssembleStructMatrix( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetData
 *--------------------------------------------------------------------------*/

void *
HYPRE_StructMatrixGetData( HYPRE_StructMatrix matrix )
{
   return( hypre_StructMatrixData( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintStructMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_PrintStructMatrix( HYPRE_StructMatrix matrix )
{
   return( hypre_PrintStructMatrix( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_SetStructMatrixStorageType( HYPRE_StructMatrix struct_matrix,
				 int               type           )
{
   return( hypre_SetStructMatrixStorageType(
      (hypre_StructMatrix *) struct_matrix, type ) );
}

