/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"


void *BlockJacobiINCFACTPcKspInitialize( void *in_ptr )
     /* Initializes solver structure */
{
   SLES       *sles;
   PC          pc;
   KSP         ksp;

   BJData     *BJ_data;
   int         i, ierr, flg, size, first_row, last_row;



   /* Allocate structure for holding solver data */
   BJ_data = (BJData *) ctalloc( BJData, 1);

   /* Initialize components of BJ_data */
   BJDataA_is_true(BJ_data) = 0;
   BJDataSles_ptr(BJ_data)  = NULL;
   BJDataSlesOwner(BJ_data) = NULL;
   BJDataSystemMatrixPtr(BJ_data)   = NULL;
   BJDataPreconditionerMatrixPtr(BJ_data)   = NULL;
  
   /* Initial setup of incfact_data structure */
   BJDataLsData(BJ_data) = incfact_initialize( (void *) NULL );

   /* Return created BJ structure to calling routine */
   return( BJ_data );

}

int BlockJacobiINCFACTPcKspFinalize (void *data )
{
  BJData      *BJ_data = data;

  incfact_free(BJDataLsData(BJ_data));
  FreeMatrixLimited(BJDataA(BJ_data));
  if( BJDataSlesOwner( BJ_data) == BJLibrary )
  {  
     SLESDestroy(*(BJDataSles_ptr(BJ_data)));
     tfree( BJDataSles_ptr(BJ_data) );
  }
  tfree(BJ_data);

}
