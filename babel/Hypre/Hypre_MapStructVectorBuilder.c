
/******************************************************
 *
 *  File:  Hypre_MapStructVectorBuilder.c
 *
 *********************************************************/

#include "Hypre_MapStructVectorBuilder_Skel.h" 
#include "Hypre_MapStructVectorBuilder_Data.h" 
#include "Hypre_MapStructVector_Skel.h" 
#include "Hypre_MapStructVector_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_MapStructVectorBuilder_constructor( Hypre_MapStructVectorBuilder babel_this ) {
   babel_this->Hypre_MapStructVectorBuilder_data =
      (struct Hypre_MapStructVectorBuilder_private_type *)
      malloc( sizeof( struct Hypre_MapStructVectorBuilder_private_type ) );
   babel_this->Hypre_MapStructVectorBuilder_data->newmap = NULL;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_MapStructVectorBuilder_destructor( Hypre_MapStructVectorBuilder babel_this ) {
   if ( babel_this->Hypre_MapStructVectorBuilder_data->newmap != NULL ) {
      Hypre_MapStructVector_deleteReference(
         babel_this->Hypre_MapStructVectorBuilder_data->newmap );
      /* ... will delete newmap if there are no other references to it */
   };
   free(babel_this->Hypre_MapStructVectorBuilder_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_MapStructVectorBuilder_Start
 **********************************************************/
int  impl_Hypre_MapStructVectorBuilder_Start
( Hypre_MapStructVectorBuilder babel_this, array1int num_ghost,
  Hypre_StructuredGrid grid ) {
   struct Hypre_MapStructVector_private_type * MSVp;

   /* delete the old newmap and make a new one ... */
   if ( babel_this->Hypre_MapStructVectorBuilder_data->newmap != NULL ) {
      Hypre_MapStructVector_deleteReference(
         babel_this->Hypre_MapStructVectorBuilder_data->newmap );
   };
   babel_this->Hypre_MapStructVectorBuilder_data->newmap = Hypre_MapStructVector_New();
   Hypre_MapStructVector_addReference(
      babel_this->Hypre_MapStructVectorBuilder_data->newmap );

   MSVp = babel_this->Hypre_MapStructVectorBuilder_data->newmap->
      Hypre_MapStructVector_data;
/* not done: check that input arguments are valid, the cast is non-null, etc. */
   MSVp->grid = (Hypre_StructGrid) Hypre_StructuredGrid_castTo( grid, "Hypre.StructGrid" );
   MSVp->num_ghost = num_ghost.data;

   return 0;
} /* end impl_Hypre_MapStructVectorBuilder_Start */

/* ********************************************************
 * impl_Hypre_MapStructVectorBuilder_SetNumGhost
 **********************************************************/
int  impl_Hypre_MapStructVectorBuilder_SetNumGhost
( Hypre_MapStructVectorBuilder babel_this, array1int values ) {
   Hypre_MapStructVector newmap = babel_this->Hypre_MapStructVectorBuilder_data->newmap;
   struct Hypre_MapStructVector_private_type * MSVp;
   if ( newmap == NULL ) return 1;
   MSVp = newmap->Hypre_MapStructVector_data;
   MSVp->num_ghost = values.data;
   return 0;
} /* end impl_Hypre_MapStructVectorBuilder_SetNumGhost */

/* ********************************************************
 * impl_Hypre_MapStructVectorBuilder_Setup
 **********************************************************/
int  impl_Hypre_MapStructVectorBuilder_Setup
( Hypre_MapStructVectorBuilder babel_this ) {
   return 0;
} /* end impl_Hypre_MapStructVectorBuilder_Setup */

/* ********************************************************
 * impl_Hypre_MapStructVectorBuilder_GetConstructedObject
 **********************************************************/
int  impl_Hypre_MapStructVectorBuilder_GetConstructedObject
(Hypre_MapStructVectorBuilder babel_this, Hypre_MapStructuredVector* obj ) {
   Hypre_MapStructVector newmap = babel_this->Hypre_MapStructVectorBuilder_data->newmap;
   if ( newmap == NULL ) return 1;
   *obj = (Hypre_MapStructuredVector)
      Hypre_MapStructVector_castTo( newmap, "Hypre.MapStructuredVector" );
   return 0;
} /* end impl_Hypre_MapStructVectorBuilder_GetConstructedObject */

