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
 * ZZZ_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * ZZZ_NewStructMatrix
 *--------------------------------------------------------------------------*/

ZZZ_StructMatrix 
ZZZ_NewStructMatrix( MPI_Comm    context,
		      ZZZ_StructGrid    grid,
		      ZZZ_StructStencil stencil )
{
   return ( (ZZZ_StructMatrix)
	    zzz_NewStructMatrix( context,
				  (zzz_StructGrid *) grid,
				  (zzz_StructStencil *) stencil ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
ZZZ_FreeStructMatrix( ZZZ_StructMatrix struct_matrix )
{
   return( zzz_FreeStructMatrix( (zzz_StructMatrix *) struct_matrix ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructMatrixCoeffs
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructMatrixCoeffs( ZZZ_StructMatrix  matrix,
			    int               *grid_index,
			    double            *coeffs      )
{
   zzz_StructMatrix *new_matrix;
   zzz_Index         *new_grid_index;

   int                d;
   int                ierr;

   new_matrix = (zzz_StructMatrix *) matrix;
   new_grid_index = zzz_NewIndex();
   for (d = 0;
	d < zzz_StructGridDim(zzz_StructMatrixStructGrid(new_matrix));
	d++)
   {
      zzz_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = zzz_SetStructMatrixCoeffs( new_matrix, new_grid_index, coeffs );

   zzz_FreeIndex(new_grid_index);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_AssembleStructMatrix
 *--------------------------------------------------------------------------*/

int 
ZZZ_AssembleStructMatrix( ZZZ_StructMatrix matrix )
{
   return( zzz_AssembleStructMatrix( (zzz_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_PrintStructMatrix
 *--------------------------------------------------------------------------*/

int 
ZZZ_PrintStructMatrix( ZZZ_StructMatrix matrix )
{
   return( zzz_PrintStructMatrix( (zzz_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructMatrixStorageType
 *--------------------------------------------------------------------------*/

int
ZZZ_SetStructMatrixStorageType( ZZZ_StructMatrix struct_matrix,
				 int               type           )
{
   return( zzz_SetStructMatrixStorageType(
      (zzz_StructMatrix *) struct_matrix, type ) );
}

