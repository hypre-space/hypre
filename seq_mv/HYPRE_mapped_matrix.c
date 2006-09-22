/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
