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
 * HYPRE_NewStructInterfaceMatrix
 *--------------------------------------------------------------------------*/

HYPRE_StructInterfaceMatrix 
HYPRE_NewStructInterfaceMatrix( MPI_Comm    context,
		      HYPRE_StructGrid    grid,
		      HYPRE_StructStencil stencil )
{
   return ( (HYPRE_StructInterfaceMatrix)
	    hypre_NewStructInterfaceMatrix( context,
				  (hypre_StructGrid *) grid,
				  (hypre_StructStencil *) stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructInterfaceMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeStructInterfaceMatrix( HYPRE_StructInterfaceMatrix struct_matrix )
{
   return( hypre_FreeStructInterfaceMatrix( (hypre_StructInterfaceMatrix *) struct_matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceMatrixCoeffs
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructInterfaceMatrixCoeffs( HYPRE_StructInterfaceMatrix  matrix,
			    int               *grid_index,
			    double            *coeffs      )
{
   hypre_StructInterfaceMatrix *new_matrix;
   hypre_Index         *new_grid_index;

   int                d;
   int                ierr;

   new_matrix = (hypre_StructInterfaceMatrix *) matrix;
   new_grid_index = hypre_NewIndex();
   for (d = 0;
	d < hypre_StructGridDim(hypre_StructInterfaceMatrixStructGrid(new_matrix));
	d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_SetStructInterfaceMatrixCoeffs( new_matrix, new_grid_index, coeffs );

   hypre_FreeIndex(new_grid_index);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceMatrixBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructInterfaceMatrixBoxValues( HYPRE_StructInterfaceMatrix  matrix,
			    int               *lower_grid_index,
			    int               *upper_grid_index,
                            int                num_stencil_indices,
                            int               *stencil_indices,
			    double            *coeffs      )

{
   hypre_StructInterfaceMatrix *new_matrix;
   hypre_Index         *new_lower_grid_index;
   hypre_Index         *new_upper_grid_index;

   int                d;
   int                ierr;

   new_matrix = (hypre_StructInterfaceMatrix *) matrix;
   new_lower_grid_index = hypre_NewIndex();
   new_upper_grid_index = hypre_NewIndex();
   for (d = 0;
	d < hypre_StructGridDim(hypre_StructInterfaceMatrixStructGrid(new_matrix));
	d++)
   {
      hypre_IndexD(new_lower_grid_index, d) = lower_grid_index[d];
      hypre_IndexD(new_upper_grid_index, d) = upper_grid_index[d];
   }

   ierr = hypre_SetStructInterfaceMatrixBoxValues( (hypre_StructInterfaceMatrix *) matrix ,
                              new_lower_grid_index,
                              new_upper_grid_index,
                              num_stencil_indices,
                              stencil_indices,
                              coeffs );


   hypre_FreeIndex(new_lower_grid_index);
   hypre_FreeIndex(new_upper_grid_index);

   return (ierr);

}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeStructInterfaceMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_InitializeStructInterfaceMatrix( HYPRE_StructInterfaceMatrix matrix )
     /* No-op */
{
   return( 0 );
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructInterfaceMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleStructInterfaceMatrix( HYPRE_StructInterfaceMatrix matrix )
{
   return( hypre_AssembleStructInterfaceMatrix( (hypre_StructInterfaceMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceMatrixGetData
 *--------------------------------------------------------------------------*/

void *
HYPRE_StructInterfaceMatrixGetData( HYPRE_StructInterfaceMatrix matrix )
{
   return( hypre_StructInterfaceMatrixData( (hypre_StructInterfaceMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintStructInterfaceMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_PrintStructInterfaceMatrix( HYPRE_StructInterfaceMatrix matrix )
{
   return( hypre_PrintStructInterfaceMatrix( (hypre_StructInterfaceMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceMatrixStorageType
 *--------------------------------------------------------------------------*/

int
HYPRE_SetStructInterfaceMatrixStorageType( HYPRE_StructInterfaceMatrix struct_matrix,
				 int               type           )
{
   return( hypre_SetStructInterfaceMatrixStorageType(
      (hypre_StructInterfaceMatrix *) struct_matrix, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceMatrixSymmetric
 *--------------------------------------------------------------------------*/

int
HYPRE_SetStructInterfaceMatrixSymmetric( HYPRE_StructInterfaceMatrix struct_matrix,
				 int               type           )
{
   return( hypre_SetStructInterfaceMatrixSymmetric(
      (hypre_StructInterfaceMatrix *) struct_matrix, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructInterfaceMatrixNumGhost
 *--------------------------------------------------------------------------*/

int
HYPRE_SetStructInterfaceMatrixNumGhost( HYPRE_StructInterfaceMatrix struct_matrix,
				 int               num_ghost         )
     /* This does not mean anything for a PETSc matrix under the hood,
        so this is a no-op for now */
{
   return( 0 );
}

