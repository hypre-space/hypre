
/******************************************************
 *
 *  File:  Hypre_StructVector.c
 *
 *********************************************************/

#include "Hypre_StructVector_Skel.h" 
#include "Hypre_StructVector_Data.h" 

            /*gkk: added...*/
#include "Hypre_Box_Skel.h"
#include "Hypre_Box_Data.h"
#include "Hypre_StructuredGrid_Skel.h"
#include "Hypre_StructuredGrid_Data.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructVector_constructor(Hypre_StructVector this) {
   this->d_table = (struct Hypre_StructVector_private_type *)
      malloc( sizeof( struct Hypre_StructVector_private_type ) );

   this->d_table->hsvec = (HYPRE_StructVector *)
      malloc( sizeof( HYPRE_StructVector ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructVector_destructor(Hypre_StructVector this) {
   struct Hypre_StructVector_private_type *SVp = this->d_table;
   HYPRE_StructVector *V = SVp->hsvec;

   HYPRE_StructVectorDestroy( *V );

   free(this->d_table);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructVectorprint
 **********************************************************/
void  impl_Hypre_StructVector_print(Hypre_StructVector this) {
   int boxarray_size;
   FILE * file;

   struct Hypre_StructVector_private_type *SVp = this->d_table;
   HYPRE_StructVector *V = SVp->hsvec;
   hypre_StructVector *v = (hypre_StructVector *) *V;

   if ( v->data_space==NULL )
      boxarray_size = -1;
   else
      boxarray_size = v->data_space->size;

   printf( "StructVector, data size =%i, BoxArray size=%i\n",
           v->data_size, boxarray_size );

   file = fopen( "testuv.out", "a" );
   fprintf( file, "\nVector Data:\n");
   hypre_PrintBoxArrayData(
      file, hypre_StructVectorDataSpace(v),
      hypre_StructVectorDataSpace(v), 1,
      hypre_StructVectorData(v) );
   fflush(file);
   fclose(file);
} /* end impl_Hypre_StructVectorprint */
