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
 * Member functions for hypre_StructInterfaceVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructInterfaceVector
 *--------------------------------------------------------------------------*/

hypre_StructInterfaceVector *
hypre_NewStructInterfaceVector( MPI_Comm     context,
		      hypre_StructGrid    *grid,
		      hypre_StructStencil *stencil )
{
   hypre_StructInterfaceVector    *vector;

   vector = hypre_CTAlloc(hypre_StructInterfaceVector, 1);

   hypre_StructInterfaceVectorContext(vector) = context;
   hypre_StructInterfaceVectorStructGrid(vector)    = grid;
   hypre_StructInterfaceVectorStructStencil(vector) = stencil;
   hypre_StructInterfaceVectorRetrievalOn(vector) = 0;

   hypre_StructInterfaceVectorTranslator(vector) = NULL;
   hypre_StructInterfaceVectorStorageType(vector) = 0;
   hypre_StructInterfaceVectorData(vector) = NULL;
   hypre_StructInterfaceVectorAuxData(vector) = NULL;

   /* set defaults */
   hypre_SetStructInterfaceVectorStorageType(vector, HYPRE_PETSC);

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructInterfaceVector
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructInterfaceVector( hypre_StructInterfaceVector *vector )
{

   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      hypre_FreeStructInterfaceVectorPETSc( vector );
   else
      return(-1);

   hypre_TFree(vector);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceVectorCoeffs
 *   
 *   Set elements in a Struct Vector interface.
 *   Coefficients are referred to in stencil format; grid points are
 *   identified by their coordinates in the
 *   *global* grid. -AC
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceVectorCoeffs( hypre_StructInterfaceVector *vector,
			    hypre_Index         grid_index,
			    double            *coeffs     )
{
   int    ierr;

   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      return( hypre_SetStructInterfaceVectorPETScCoeffs( vector, grid_index, coeffs ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceVectorBoxValues
 *--------------------------------------------------------------------------*/
int 
hypre_SetStructInterfaceVectorBoxValues( hypre_StructInterfaceVector *vector,
			    hypre_Index         lower_grid_index,
			    hypre_Index         upper_grid_index,
			    double            *coeffs     )
{
   hypre_Index loop_index;
   int         ierr=0;
   int         i, j, k, coeffs_index;

   /* Insert coefficients one grid point at a time */
   for (k = hypre_IndexZ(lower_grid_index), coeffs_index = 0; k <= hypre_IndexZ(upper_grid_index); k++)
      for (j = hypre_IndexY(lower_grid_index); j <= hypre_IndexY(upper_grid_index); j++)
         for (i = hypre_IndexX(lower_grid_index); i <= hypre_IndexX(upper_grid_index); i++, coeffs_index ++)
         /* Loop over grid dimensions specified in input arguments */
         {
            hypre_SetIndex(loop_index, i, j, k);

            /* Insert coefficients in coeffs_buffer */
            ierr = hypre_SetStructInterfaceVectorCoeffs( 
                            vector,
			    loop_index,
			    &(coeffs[ coeffs_index ])  );

         }
   /* End Loop from lower_grid_index to upper_grid index */

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceVector
 *   Storage independent routine for setting a vector to a value.
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceVector( hypre_StructInterfaceVector *vector, double *val )
{
   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      return( hypre_SetStructInterfaceVectorPETSc( vector, val ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructInterfaceVector
 *   User-level routine for assembling hypre_StructInterfaceVector.
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructInterfaceVector( hypre_StructInterfaceVector *vector )
{
   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      return( hypre_AssembleStructInterfaceVectorPETSc( vector ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructInterfaceVectorStorageType
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructInterfaceVectorStorageType( hypre_StructInterfaceVector *vector,
				 int                type   )
{
   hypre_StructInterfaceVectorStorageType(vector) = type;

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_PrintStructInterfaceVector
 *   Internal routine for printing a vector
 *--------------------------------------------------------------------------*/

int 
hypre_PrintStructInterfaceVector( hypre_StructInterfaceVector *vector )
{
   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      return( hypre_PrintStructInterfaceVectorPETSc( vector ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_RetrievalOnStructInterfaceVector
 *--------------------------------------------------------------------------*/

int 
hypre_RetrievalOnStructInterfaceVector( hypre_StructInterfaceVector *vector )
{
   int ierr = 0;

   hypre_StructInterfaceVectorRetrievalOn(vector) = 1;

   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      return( hypre_RetrievalOnStructInterfaceVectorPETSc( vector ) );
   else
      return(-1);

}

/*--------------------------------------------------------------------------
 * hypre_RetrievalOffStructInterfaceVector
 *--------------------------------------------------------------------------*/

int 
hypre_RetrievalOffStructInterfaceVector( hypre_StructInterfaceVector *vector )
{
   int ierr = 0;

   hypre_StructInterfaceVectorRetrievalOn(vector) = 0;

   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      return( hypre_RetrievalOffStructInterfaceVectorPETSc( vector ) );
   else
      return(-1);

}

/*--------------------------------------------------------------------------
 * hypre_GetStructInterfaceVectorValue
 *--------------------------------------------------------------------------*/

int 
hypre_GetStructInterfaceVectorValue( hypre_StructInterfaceVector *vector, 
      hypre_Index index, double *value )
{
   int ierr = 0;

   if ( hypre_StructInterfaceVectorRetrievalOn(vector) == 0 )
   {
     ierr = -1;
     printf("GetStructInterfaceVectorValue ERROR\n");
     return(ierr);
   }

   if ( hypre_StructInterfaceVectorStorageType(vector) == HYPRE_PETSC )
      return( hypre_GetStructInterfaceVectorPETScValue( vector, index, value ) );
   else
      return(-1);

}

