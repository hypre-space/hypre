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
