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
 * HYPRE_MappedMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewMappedMatrix
 *--------------------------------------------------------------------------*/

HYPRE_MappedMatrix 
HYPRE_NewMappedMatrix( )
{
   return ( (HYPRE_MappedMatrix)
            hypre_NewMappedMatrix(  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeMappedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeMappedMatrix( HYPRE_MappedMatrix matrix )
{
   return( hypre_FreeMappedMatrix( (hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LimitedFreeMappedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_LimitedFreeMappedMatrix( HYPRE_MappedMatrix matrix )
{
   return( hypre_LimitedFreeMappedMatrix( (hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeMappedMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_InitializeMappedMatrix( HYPRE_MappedMatrix matrix )
{
   return ( hypre_InitializeMappedMatrix( (hypre_MappedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_AssembleMappedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleMappedMatrix( HYPRE_MappedMatrix matrix )
{
   return( hypre_AssembleMappedMatrix( (hypre_MappedMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_PrintMappedMatrix
 *--------------------------------------------------------------------------*/

void 
HYPRE_PrintMappedMatrix( HYPRE_MappedMatrix matrix )
{
   hypre_PrintMappedMatrix( (hypre_MappedMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_GetMappedMatrixColIndex
 *--------------------------------------------------------------------------*/

int
HYPRE_GetMappedMatrixColIndex( HYPRE_MappedMatrix matrix, int j )
{
   return( hypre_GetMappedMatrixColIndex( (hypre_MappedMatrix *) matrix, j ));
}

/*--------------------------------------------------------------------------
 * HYPRE_GetMappedMatrixMatrix
 *--------------------------------------------------------------------------*/

void *
HYPRE_GetMappedMatrixMatrix( HYPRE_MappedMatrix matrix )
{
   return( hypre_GetMappedMatrixMatrix( (hypre_MappedMatrix *) matrix ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SetMappedMatrixMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_SetMappedMatrixMatrix( HYPRE_MappedMatrix matrix, void *matrix_data )
{
   return( hypre_SetMappedMatrixColMap( (hypre_MappedMatrix *) matrix, matrix_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetMappedMatrixColMap
 *--------------------------------------------------------------------------*/

int
HYPRE_SetMappedMatrixColMap( HYPRE_MappedMatrix matrix, int (*ColMap)(int, void *) )
{
   return( hypre_SetMappedMatrixColMap( (hypre_MappedMatrix *) matrix, ColMap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetMappedMatrixMapData
 *--------------------------------------------------------------------------*/

int
HYPRE_SetMappedMatrixMapData( HYPRE_MappedMatrix matrix, void *MapData )
{
   return( hypre_SetMappedMatrixMapData( (hypre_MappedMatrix *) matrix, MapData ) );
}
