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
 * Member functions for hypre_StructVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructVector
 *--------------------------------------------------------------------------*/

hypre_StructVector *
hypre_NewStructVector( MPI_Comm     context,
		      hypre_StructGrid    *grid,
		      hypre_StructStencil *stencil )
{
   hypre_StructVector    *vector;

   vector = hypre_CTAlloc(hypre_StructVector, 1);

   hypre_StructVectorContext(vector) = context;
   hypre_StructVectorStructGrid(vector)    = grid;
   hypre_StructVectorStructStencil(vector) = stencil;

   hypre_StructVectorTranslator(vector) = NULL;
   hypre_StructVectorStorageType(vector) = 0;
   hypre_StructVectorData(vector) = NULL;

   /* set defaults */
   hypre_SetStructVectorStorageType(vector, HYPRE_PETSC_VECTOR);

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructVector
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructVector( hypre_StructVector *vector )
{

   if ( hypre_StructVectorStorageType(vector) == HYPRE_PETSC_VECTOR )
      hypre_FreeStructVectorPETSc( vector );
   else
      return(-1);

   hypre_TFree(vector);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorCoeffs
 *   
 *   Set elements in a Struct Vector interface.
 *   Coefficients are referred to in stencil format; grid points are
 *   identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorCoeffs( hypre_StructVector *vector,
			    hypre_Index         *grid_index,
			    double            *coeffs     )
{
   int    ierr;

   if ( hypre_StructVectorStorageType(vector) == HYPRE_PETSC_VECTOR )
      return( hypre_SetStructVectorPETScCoeffs( vector, grid_index, coeffs ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVector
 *   Storage independent routine for setting a vector to a value.
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVector( hypre_StructVector *vector, double *val )
{
   if ( hypre_StructVectorStorageType(vector) == HYPRE_PETSC_VECTOR )
      return( hypre_SetStructVectorPETSc( vector, val ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructVector
 *   User-level routine for assembling hypre_StructVector.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructVector( hypre_StructVector *vector )
{
   if ( hypre_StructVectorStorageType(vector) == HYPRE_PETSC_VECTOR )
      return( hypre_AssembleStructVectorPETSc( vector ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructVectorStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructVectorStorageType( hypre_StructVector *vector,
				 int                type   )
{
   hypre_StructVectorStorageType(vector) = type;

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_PrintStructVector
 *   Internal routine for printing a vector
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructVector( hypre_StructVector *vector )
{
   if ( hypre_StructVectorStorageType(vector) == HYPRE_PETSC_VECTOR )
      return( hypre_PrintStructVectorPETSc( vector ) );
   else
      return(-1);
}

