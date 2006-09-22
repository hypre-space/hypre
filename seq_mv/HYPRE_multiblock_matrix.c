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
 * HYPRE_MultiblockMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_MultiblockMatrix 
HYPRE_MultiblockMatrixCreate( )
{
   return ( (HYPRE_MultiblockMatrix)
            hypre_MultiblockMatrixCreate(  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixDestroy( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_MultiblockMatrixDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixLimitedDestroy( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_MultiblockMatrixLimitedDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_MultiblockMatrixInitialize( HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixInitialize( (hypre_MultiblockMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixAssemble( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_MultiblockMatrixAssemble( (hypre_MultiblockMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void 
HYPRE_MultiblockMatrixPrint( HYPRE_MultiblockMatrix matrix )
{
   hypre_MultiblockMatrixPrint( (hypre_MultiblockMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixSetNumSubmatrices
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixSetNumSubmatrices( HYPRE_MultiblockMatrix matrix, int n )
{
   return( hypre_MultiblockMatrixSetNumSubmatrices( 
             (hypre_MultiblockMatrix *) matrix, n ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixSetSubmatrixType( HYPRE_MultiblockMatrix matrix, 
                                      int j,
                                      int type )
{
   return( hypre_MultiblockMatrixSetSubmatrixType( 
             (hypre_MultiblockMatrix *) matrix, j, type ) );
}
