
/******************************************************
 *
 *  File:  Hypre_StructVectorBldr.c
 *
 *********************************************************/

#include "Hypre_StructVectorBldr_Skel.h" 
#include "Hypre_StructVectorBldr_Data.h" 

#include "Hypre_StructVector_Skel.h" 
#include "Hypre_StructVector_Data.h" 
#include "HYPRE_mv.h"
#include "struct_matrix_vector.h"
#include "Hypre_Box_Skel.h"
#include "Hypre_StructuredGrid_Skel.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructVectorBldr_constructor(Hypre_StructVectorBldr this) {
   this->d_table = (struct Hypre_StructVectorBldr_private_type *)
      malloc( sizeof( struct Hypre_StructVectorBldr_private_type ) );
   this->d_table->newvec = NULL;
   this->d_table->vecgood = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructVectorBldr_destructor(Hypre_StructVectorBldr this) {
   if ( this->d_table->newvec != NULL ) {
      Hypre_StructVector_deleteReference( this->d_table->newvec );
      /* ... will delete newvec if there are no other references to it */
   };
   free(this->d_table);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructVectorBldrprint
 **********************************************************/
void  impl_Hypre_StructVectorBldr_print(Hypre_StructVectorBldr this) {
   printf( "StructVectorBldr\n" );
} /* end impl_Hypre_StructVectorBldrprint */

/* ********************************************************
 * impl_Hypre_StructVectorBldrSetValues
 **********************************************************/
int  impl_Hypre_StructVectorBldr_SetValues
(Hypre_StructVectorBldr this, Hypre_Box box, array1double values) {
   int i, ssize, lower[3], upper[3];
   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;
   struct Hypre_Box_private_type * Bp;
   hypre_Box * B;

   Hypre_StructVector SV = this->d_table->newvec;
   if ( SV == NULL ) return -1;

   SVp = SV->d_table;
   V = SVp->hsvec;
   Bp = box->d_table;
   B = Bp->hbox;

   for ( i=0; i<Bp->dimension; ++i ) {
      lower[i] = B->imin[i];
      upper[i] = B->imax[i];
   };

   return HYPRE_StructVectorSetBoxValues( *V, lower, upper,
                                          &(values.data[*(values.lower)]) );
} /* end impl_Hypre_StructVectorBldrSetValues */

/* ********************************************************
 * impl_Hypre_StructVectorBldrSetNumGhost
 **********************************************************/
int  impl_Hypre_StructVectorBldr_SetNumGhost
( Hypre_StructVectorBldr this, array1int values )
{
   HYPRE_StructVector  vector;
   int * num_ghost = &(values.data[*(values.lower)]);
   HYPRE_StructVector * V;
   Hypre_StructVector SV = this->d_table->newvec;
   if ( SV == NULL ) return -1;
   V = SV->d_table->hsvec;

   return HYPRE_StructVectorSetNumGhost( *V, num_ghost );

} /* end impl_Hypre_StructVectorBldrSetNumGhost */


/* ********************************************************
 * impl_Hypre_StructVectorBldrGetConstructedObject
 **********************************************************/
Hypre_Vector
impl_Hypre_StructVectorBldr_GetConstructedObject(Hypre_StructVectorBldr this) {
   Hypre_StructVector newvec = this->d_table->newvec;
   if ( newvec==NULL || this->d_table->vecgood==0 ) {
      printf( "Hypre_StructVectorBldr: object not constructed yet\n");
      return (Hypre_Vector) NULL;
   };
   return (Hypre_Vector) Hypre_StructVector_castTo( newvec, "Hypre_Vector" );
} /* end impl_Hypre_StructVectorBldrGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructVectorBldrNew
 **********************************************************/
int  impl_Hypre_StructVectorBldr_New
(Hypre_StructVectorBldr this, Hypre_StructuredGrid grid) {

   struct Hypre_StructuredGrid_private_type *Gp = grid->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   MPI_Comm comm = hypre_StructGridComm( g );

   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;
   if ( this->d_table->newvec != NULL )
      Hypre_StructVector_deleteReference( this->d_table->newvec );
   this->d_table->newvec = Hypre_StructVector_new();
   this->d_table->vecgood = 0;
   Hypre_StructVector_addReference( this->d_table->newvec );

   SVp = this->d_table->newvec->d_table;
   SVp->grid = grid;
   V = SVp->hsvec;

/*    HYPRE_StructVectorCreate( comm, *G, *SS, V );
      ... This function doesn't use the stencil.  Here we reproduce
      its internals so as not to have to suppy it ... */
   *V = (HYPRE_StructVector) hypre_StructVectorCreate( comm, g ) ;

   return HYPRE_StructVectorInitialize( *V );

} /* end impl_Hypre_StructVectorBldrNew */

/* ********************************************************
 * impl_Hypre_StructVectorBldrConstructor
 *
 * The argument is ignored; it really belongs in the New function
 * which is separately called.
 * However, the argument must be in the interface because if a matrix
 * class be its own builder, then the Constructor will call New directly,
 * and it needs the argument for that call.
 **********************************************************/
Hypre_StructVectorBldr
impl_Hypre_StructVectorBldr_Constructor(Hypre_StructuredGrid grid) {
   return Hypre_StructVectorBldr_new();
} /* end impl_Hypre_StructVectorBldrConstructor */

/* ********************************************************
 * impl_Hypre_StructVectorBldrSetup
 **********************************************************/
int  impl_Hypre_StructVectorBldr_Setup(Hypre_StructVectorBldr this) {
   int ierr;
   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;

   Hypre_StructVector SV = this->d_table->newvec;
   if ( SV == NULL ) return -1;
   
   SVp = SV->d_table;
   V = SVp->hsvec;

   ierr = HYPRE_StructVectorAssemble( *V );

   if ( ierr==0 ) this->d_table->vecgood = 1;
   return ierr;
} /* end impl_Hypre_StructVectorBldrSetup */

