
/******************************************************
 *
 *  File:  Hypre_MapStructVector.c
 *
 *********************************************************/

#include "Hypre_MapStructVector_Skel.h" 
#include "Hypre_MapStructVector_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_MapStructVector_constructor(Hypre_MapStructVector babel_this) {
   babel_this->Hypre_MapStructVector_data =
      (struct Hypre_MapStructVector_private_type *)
      malloc( sizeof( struct Hypre_MapStructVector_private_type ) );
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_MapStructVector_destructor(Hypre_MapStructVector babel_this) {
   free(babel_this->Hypre_MapStructVector_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_MapStructVector_GetNumGhost
 **********************************************************/
int  impl_Hypre_MapStructVector_GetNumGhost(Hypre_MapStructVector babel_this, array1int* values) {
   int i;
   int * num_ghost = babel_this->Hypre_MapStructVector_data->num_ghost;
   assert( num_ghost!= NULL );
   assert( values->lower[0] == 0 );
   assert( values->upper[0] == 6 );
   for ( i=0; i<6; ++i ) values->data[i] = num_ghost[i];
   return 0;
} /* end impl_Hypre_MapStructVector_GetNumGhost */

/* ********************************************************
 * impl_Hypre_MapStructVector_GetGrid
 **********************************************************/
int  impl_Hypre_MapStructVector_GetGrid
( Hypre_MapStructVector babel_this, Hypre_StructuredGrid* grid ) {
   Hypre_StructGrid mygrid = babel_this->Hypre_MapStructVector_data->grid;
   assert( grid != NULL );
   *grid = (Hypre_StructuredGrid) Hypre_StructGrid_castTo( mygrid, "Hypre_StructuredGrid" );
   return 0;
} /* end impl_Hypre_MapStructVector_GetGrid */

