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
 * Member functions for zzz_StructVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructVector
 *--------------------------------------------------------------------------*/

zzz_StructVector *
zzz_NewStructVector( MPI_Comm     context,
		      zzz_StructGrid    *grid,
		      zzz_StructStencil *stencil )
{
   zzz_StructVector    *vector;

   vector = talloc(zzz_StructVector, 1);

   zzz_StructVectorContext(vector) = context;
   zzz_StructVectorStructGrid(vector)    = grid;
   zzz_StructVectorStructStencil(vector) = stencil;

   zzz_StructVectorTranslator(vector) = NULL;
   zzz_StructVectorStorageType(vector) = 0;
   zzz_StructVectorData(vector) = NULL;

   /* set defaults */
   zzz_SetStructVectorStorageType(vector, ZZZ_PETSC_VECTOR);

   return vector;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructVector( zzz_StructVector *vector )
{

   if ( zzz_StructVectorStorageType(vector) == ZZZ_PETSC_VECTOR )
      zzz_FreeStructVectorPETSc( vector );
   else
      return(-1);

   tfree(vector);

   return(0);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructVectorCoeffs
 *   
 *   Set elements in a Struct Vector interface.
 *   Coefficients are referred to in stencil format; grid points are
 *   identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructVectorCoeffs( zzz_StructVector *vector,
			    zzz_Index         *grid_index,
			    double            *coeffs     )
{
   int    ierr;

   if ( zzz_StructVectorStorageType(vector) == ZZZ_PETSC_VECTOR )
      return( zzz_SetStructVectorPETScCoeffs( vector, grid_index, coeffs ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructVector
 *   Storage independent routine for setting a vector to a value.
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructVector( zzz_StructVector *vector, double *val )
{
   if ( zzz_StructVectorStorageType(vector) == ZZZ_PETSC_VECTOR )
      return( zzz_SetStructVectorPETSc( vector, val ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructVector
 *   User-level routine for assembling zzz_StructVector.
 *--------------------------------------------------------------------------*/

int 
zzz_AssembleStructVector( zzz_StructVector *vector )
{
   if ( zzz_StructVectorStorageType(vector) == ZZZ_PETSC_VECTOR )
      return( zzz_AssembleStructVectorPETSc( vector ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructVectorStorageType
 *--------------------------------------------------------------------------*/

int 
zzz_SetStructVectorStorageType( zzz_StructVector *vector,
				 int                type   )
{
   zzz_StructVectorStorageType(vector) = type;

   return(0);
}


/*--------------------------------------------------------------------------
 * zzz_PrintStructVector
 *   Internal routine for printing a vector
 *--------------------------------------------------------------------------*/

int 
zzz_PrintStructVector( zzz_StructVector *vector )
{
   if ( zzz_StructVectorStorageType(vector) == ZZZ_PETSC_VECTOR )
      return( zzz_PrintStructVectorPETSc( vector ) );
   else
      return(-1);
}

