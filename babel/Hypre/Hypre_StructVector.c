
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

/* ********************************************************
 * impl_Hypre_StructVectorSetValues
 *    Note that Setup needs to be called afterwards.
 **********************************************************/
int  impl_Hypre_StructVector_SetValues
(Hypre_StructVector this, Hypre_Box box, array1double values) {

   int i, ssize, lower[3], upper[3];

   struct Hypre_StructVector_private_type *SVp = this->d_table;
   HYPRE_StructVector *V = SVp->hsvec;

   struct Hypre_Box_private_type *Bp = box->d_table;
   hypre_Box *B = Bp->hbox;

   for ( i=0; i<Bp->dimension; ++i ) {
      lower[i] = B->imin[i];
      upper[i] = B->imax[i];
   };

   HYPRE_StructVectorSetBoxValues( *V, lower, upper,
                                   &(values.data[*(values.lower)]) );

} /* end impl_Hypre_StructVectorSetValues */

/* ********************************************************
 * impl_Hypre_StructVectorGetConstructedObject
 **********************************************************/
Hypre_StructVector
impl_Hypre_StructVector_GetConstructedObject(Hypre_StructVector this) {
   return this;
} /* end impl_Hypre_StructVectorGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructVectorNew
 *     Note that Setup also must be called.
 **********************************************************/
void  impl_Hypre_StructVector_New
(Hypre_StructVector this, Hypre_StructuredGrid grid) {
   struct Hypre_StructVector_private_type *SVp = this->d_table;
   HYPRE_StructVector *V = SVp->hsvec;

   struct Hypre_StructuredGrid_private_type *Gp = grid->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   MPI_Comm comm = hypre_StructGridComm( g );

/*    HYPRE_StructVectorCreate( comm, *G, *SS, V );
      ... This function doesn't use the stencil.  Here we reproduce
      its internals so as not to have to suppy it ... */
   *V = (HYPRE_StructVector) hypre_StructVectorCreate( comm, g ) ;

   HYPRE_StructVectorInitialize( *V );

} /* end impl_Hypre_StructVectorNew */

/* ********************************************************
 * impl_Hypre_StructVectorConstructor
 **********************************************************/
Hypre_StructVector  impl_Hypre_StructVector_Constructor(Hypre_StructuredGrid grid) {
   /* declared static; just combines the new and New functions */
   Hypre_StructVector SV = Hypre_StructVector_new();
   Hypre_StructVector_New( SV, grid );
   return SV;
} /* end impl_Hypre_StructVectorConstructor */

/* ********************************************************
 * impl_Hypre_StructVectorSetup
 **********************************************************/
int  impl_Hypre_StructVector_Setup(Hypre_StructVector this) {
   struct Hypre_StructVector_private_type *SVp = this->d_table;
   HYPRE_StructVector *V = SVp->hsvec;
   HYPRE_StructVectorAssemble( *V );
} /* end impl_Hypre_StructVectorSetup */

