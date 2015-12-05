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
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * HYPRE_CSRMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix 
HYPRE_CSRMatrixCreate( int  num_rows,
                       int  num_cols,
                       int *row_sizes )
{
   hypre_CSRMatrix *matrix;
   int             *matrix_i;
   int              i;

   matrix_i = hypre_CTAlloc(int, num_rows + 1);
   matrix_i[0] = 0;
   for (i = 0; i < num_rows; i++)
   {
      matrix_i[i+1] = matrix_i[i] + row_sizes[i];
   }

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;

   return ( (HYPRE_CSRMatrix) matrix );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRMatrixDestroy( HYPRE_CSRMatrix matrix )
{
   return( hypre_CSRMatrixDestroy( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRMatrixInitialize( HYPRE_CSRMatrix matrix )
{
   return ( hypre_CSRMatrixInitialize( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixRead
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix 
HYPRE_CSRMatrixRead( char            *file_name )
{
   return ( (HYPRE_CSRMatrix) hypre_CSRMatrixRead( file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

void 
HYPRE_CSRMatrixPrint( HYPRE_CSRMatrix  matrix,
                      char            *file_name )
{
   hypre_CSRMatrixPrint( (hypre_CSRMatrix *) matrix,
                         file_name );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixGetNumRows
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRMatrixGetNumRows( HYPRE_CSRMatrix matrix, int *num_rows )
{
   hypre_CSRMatrix *csr_matrix = (hypre_CSRMatrix *) matrix;

   *num_rows =  hypre_CSRMatrixNumRows( csr_matrix );

   return 0;
}


