/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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




/* Include AMG Headers */
#include "BlockJacobiAmgPcKsp.h"


int CsrGen_to_CsrDiagFirst( Matrix *A, int **diag_loc_ret )
     /* Converts CSR matrix, using AMG structure definition,
        from general to one in which the diagonal is first in each row.
        Information is stored for later restoral. */
{
   int i, j, k, itemp;
   int *diag_loc;
   double dtemp;

   /* allocate space for storing information for recovery */
   diag_loc = ctalloc( int, MatrixSize( A ) );

   /* Variable i loops over the rows */
   for (i=1; i <= MatrixSize( A ); i++) {
      /* Loop until the diagonal element is found or end of row reached */
      for ( k=MatrixIA( A )[ i-1 ], j=MatrixJA( A )[ k-1 ]; 
            (j != i) && (k <= MatrixIA( A )[i]);
            k++, j=MatrixJA( A )[ k-1 ] );

      /* If diagonal was found, interchange it with first element */
      if( j <= MatrixIA( A )[ i ] ) {
         dtemp = MatrixData( A )[ MatrixIA( A )[ i-1 ]-1 ];
         MatrixData( A )[ MatrixIA( A )[ i-1 ]-1 ] = 
            MatrixData( A )[ k-1 ];
         MatrixData( A )[ k-1 ] = dtemp;

         itemp = MatrixJA( A )[ MatrixIA( A )[ i-1 ]-1 ];
         MatrixJA( A )[ MatrixIA( A )[ i-1 ]-1 ] =
            MatrixJA( A )[ k-1 ];
         MatrixJA( A )[ k-1 ] = itemp;

         diag_loc[ i-1 ] = k;
      } else {
        /* Set marker indicating diagonal element was zero */
         diag_loc[ i-1 ] = -1;
         return( i );
      }
   }

   *diag_loc_ret = diag_loc;

   return(0);

}


int CsrDiagFirst_backto_CsrGen( Matrix *A, int *diag_loc )
     /* Converts CSR matrix, using AMG structure definition,
        from general to one in which the diagonal is first in each row.
        Information is stored for later restoral. */
{
   int i, j, itemp;
   double dtemp;

   /* Variable i loops over the rows */
   for (i=1; i <= MatrixSize( A ); i++) {

     if ( diag_loc[ i-1 ] >= 0 ) {
      /* Interchange first element with former first element */
      dtemp = MatrixData( A )[ MatrixIA( A )[ i-1 ]-1 ];
      MatrixData( A )[ MatrixIA( A )[ i-1 ]-1 ] = 
         MatrixData( A )[ diag_loc[ i-1 ]-1 ];
      MatrixData( A )[ diag_loc[ i-1 ]-1 ] = dtemp;

      itemp = MatrixJA( A )[ MatrixIA( A )[ i-1 ]-1 ];
      MatrixJA( A )[ MatrixIA( A )[ i-1 ]-1 ] = 
         MatrixJA( A )[ diag_loc[ i-1 ]-1 ];
      MatrixJA( A )[ diag_loc[ i-1 ]-1 ] = itemp;
     }
   }

   tfree( diag_loc );

   return(0);
}
