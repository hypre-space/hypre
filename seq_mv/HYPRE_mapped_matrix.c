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
 * HYPRE_MappedMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_MappedMatrix 
HYPRE_MappedMatrixCreate( )
{
   return ( (HYPRE_MappedMatrix)
            hypre_MappedMatrixCreate(  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_MappedMatrixDestroy( HYPRE_MappedMatrix matrix )
{
   return( hypre_MappedMatrixDestroy( (hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_MappedMatrixLimitedDestroy( HYPRE_MappedMatrix matrix )
{
   return( hypre_MappedMatrixLimitedDestroy( (hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_MappedMatrixInitialize( HYPRE_MappedMatrix matrix )
{
   return ( hypre_MappedMatrixInitialize( (hypre_MappedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_MappedMatrixAssemble( HYPRE_MappedMatrix matrix )
{
   return( hypre_MappedMatrixAssemble( (hypre_MappedMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixPrint
 *--------------------------------------------------------------------------*/

void 
HYPRE_MappedMatrixPrint( HYPRE_MappedMatrix matrix )
{
   hypre_MappedMatrixPrint( (hypre_MappedMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixGetColIndex
 *--------------------------------------------------------------------------*/

int
HYPRE_MappedMatrixGetColIndex( HYPRE_MappedMatrix matrix, int j )
{
   return( hypre_MappedMatrixGetColIndex( (hypre_MappedMatrix *) matrix, j ));
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixGetMatrix
 *--------------------------------------------------------------------------*/

void *
HYPRE_MappedMatrixGetMatrix( HYPRE_MappedMatrix matrix )
{
   return( hypre_MappedMatrixGetMatrix( (hypre_MappedMatrix *) matrix ));
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixSetMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_MappedMatrixSetMatrix( HYPRE_MappedMatrix matrix, void *matrix_data )
{
   return( hypre_MappedMatrixSetMatrix( (hypre_MappedMatrix *) matrix, matrix_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

int
HYPRE_MappedMatrixSetColMap( HYPRE_MappedMatrix matrix, int (*ColMap)(int, void *) )
{
   return( hypre_MappedMatrixSetColMap( (hypre_MappedMatrix *) matrix, ColMap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

int
HYPRE_MappedMatrixSetMapData( HYPRE_MappedMatrix matrix, void *MapData )
{
   return( hypre_MappedMatrixSetMapData( (hypre_MappedMatrix *) matrix, MapData ) );
}
