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
 * ZZZ_NewStructVector
 *--------------------------------------------------------------------------*/

ZZZ_StructVector
ZZZ_NewStructVector( MPI_Comm     context,
		      ZZZ_StructGrid     grid,
		      ZZZ_StructStencil  stencil )
{
   return ( (ZZZ_StructVector)
	    zzz_NewStructVector( context,
				  (zzz_StructGrid *) grid,
				  (zzz_StructStencil *) stencil ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
ZZZ_FreeStructVector( ZZZ_StructVector struct_vector )
{
   return( zzz_FreeStructVector( (zzz_StructVector *) struct_vector ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructVectorCoeffs
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructVectorCoeffs( ZZZ_StructVector  vector,
			    int               *grid_index,
			    double            *coeffs      )
{
   zzz_StructVector *new_vector;
   zzz_Index         *new_grid_index;

   int                d;
   int                ierr;

   new_vector = (zzz_StructVector *) vector;
   new_grid_index = zzz_NewIndex();
   for (d = 0;
	d < zzz_StructGridDim(zzz_StructVectorStructGrid(new_vector));
	d++)
   {
      zzz_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = zzz_SetStructVectorCoeffs( new_vector, new_grid_index, coeffs );

   zzz_FreeIndex(new_grid_index);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructVector
 *--------------------------------------------------------------------------*/

int 
ZZZ_SetStructVector( ZZZ_StructVector  vector,
			    double      *val      )
{
   zzz_StructVector *new_vector;

   int                ierr;

   new_vector = (zzz_StructVector *) vector;

   ierr = zzz_SetStructVector( new_vector, val );

   return (ierr);
}

/*--------------------------------------------------------------------------
 * ZZZ_AssembleStructVector
 *--------------------------------------------------------------------------*/

int 
ZZZ_AssembleStructVector( ZZZ_StructVector vector )
{
   return( zzz_AssembleStructVector( (zzz_StructVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructVectorStorageType
 *--------------------------------------------------------------------------*/

int
ZZZ_SetStructVectorStorageType( ZZZ_StructVector  struct_vector,
				 int                type           )
{
   return( zzz_SetStructVectorStorageType(
      (zzz_StructVector *) struct_vector, type ) );
}

/* OUTPUT */
/*--------------------------------------------------------------------------
 * ZZZ_PrintStructVector
 *--------------------------------------------------------------------------*/

int 
ZZZ_PrintStructVector( ZZZ_StructVector vector )
{
   return( zzz_PrintStructVector( (zzz_StructVector *) vector ) );
}
