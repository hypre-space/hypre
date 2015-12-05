/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




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

HYPRE_Int 
HYPRE_MappedMatrixDestroy( HYPRE_MappedMatrix matrix )
{
   return( hypre_MappedMatrixDestroy( (hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_MappedMatrixLimitedDestroy( HYPRE_MappedMatrix matrix )
{
   return( hypre_MappedMatrixLimitedDestroy( (hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MappedMatrixInitialize( HYPRE_MappedMatrix matrix )
{
   return ( hypre_MappedMatrixInitialize( (hypre_MappedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int 
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

HYPRE_Int
HYPRE_MappedMatrixGetColIndex( HYPRE_MappedMatrix matrix, HYPRE_Int j )
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

HYPRE_Int
HYPRE_MappedMatrixSetMatrix( HYPRE_MappedMatrix matrix, void *matrix_data )
{
   return( hypre_MappedMatrixSetMatrix( (hypre_MappedMatrix *) matrix, matrix_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MappedMatrixSetColMap( HYPRE_MappedMatrix matrix, HYPRE_Int (*ColMap)(HYPRE_Int, void *) )
{
   return( hypre_MappedMatrixSetColMap( (hypre_MappedMatrix *) matrix, ColMap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MappedMatrixSetMapData( HYPRE_MappedMatrix matrix, void *MapData )
{
   return( hypre_MappedMatrixSetMapData( (hypre_MappedMatrix *) matrix, MapData ) );
}
