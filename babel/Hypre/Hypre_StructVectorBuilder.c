
/******************************************************
 *
 *  File:  Hypre_StructVectorBuilder.c
 *
 *********************************************************/

#include "Hypre_StructVectorBuilder_Skel.h" 
#include "Hypre_StructVectorBuilder_Data.h" 


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
void Hypre_StructVectorBuilder_constructor(Hypre_StructVectorBuilder this) {
   this->d_table = (struct Hypre_StructVectorBuilder_private_type *)
      malloc( sizeof( struct Hypre_StructVectorBuilder_private_type ) );
   this->d_table->newvec = NULL;
   this->d_table->vecgood = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructVectorBuilder_destructor(Hypre_StructVectorBuilder this) {
   if ( this->d_table->newvec != NULL ) {
      Hypre_StructVector_deleteReference( this->d_table->newvec );
      /* ... will delete newvec if there are no other references to it */
   };
   free(this->d_table);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderprint
 **********************************************************/
void  impl_Hypre_StructVectorBuilder_print(Hypre_StructVectorBuilder this) {
   printf( "StructVectorBuilder\n" );
} /* end impl_Hypre_StructVectorBuilderprint */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderSetValue
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_SetValue
(Hypre_StructVectorBuilder this, array1int where, double value) {
   printf("not implemented as of 5/5/2000.");
   return 1;
} /* end impl_Hypre_StructVectorBuilderSetValue */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderSetBoxValues
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_SetBoxValues
(Hypre_StructVectorBuilder this, Hypre_Box box, array1double values) {
   int i, ssize, lower[3], upper[3];
   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;
   struct Hypre_Box_private_type * Bp;
   hypre_Box * B;

   Hypre_StructVector SV = this->d_table->newvec;
   if ( SV == NULL ) return 1;

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

} /* end impl_Hypre_StructVectorBuilderSetBoxValues */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderSetNumGhost
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_SetNumGhost
(Hypre_StructVectorBuilder this, array1int values) {
   HYPRE_StructVector  vector;
   int * num_ghost = &(values.data[*(values.lower)]);
   HYPRE_StructVector * V;
   Hypre_StructVector SV = this->d_table->newvec;
   if ( SV == NULL ) return 1;
   V = SV->d_table->hsvec;

   return HYPRE_StructVectorSetNumGhost( *V, num_ghost );
} /* end impl_Hypre_StructVectorBuilderSetNumGhost */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderGetConstructedObject
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_GetConstructedObject
(Hypre_StructVectorBuilder this, Hypre_Vector* obj) {
   Hypre_StructVector newvec = this->d_table->newvec;
   if ( newvec==NULL || this->d_table->vecgood==0 ) {
      printf( "Hypre_StructVectorBuilder: object not constructed yet\n");
      return (Hypre_Vector) NULL;
   };
   return (Hypre_Vector) Hypre_StructVector_castTo( newvec, "Hypre_Vector" );
} /* end impl_Hypre_StructVectorBuilderGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderNew
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_New
(Hypre_StructVectorBuilder this, Hypre_StructGrid grid) {

   struct Hypre_StructGrid_private_type *Gp = grid->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   Hypre_MPI_Com com = Gp->comm;
   /*   MPI_Comm comm = hypre_StructGridComm( g ); ... this is ok, but the
        following requires less knowledge of what's in a hypre_StructGrid ...
    */
   MPI_Comm comm = *(com->d_table->hcom);

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

} /* end impl_Hypre_StructVectorBuilderNew */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderConstructor
 * The argument is ignored; it really belongs in the New function
 * which is separately called.
 * However, the argument must be in the interface because if a vector
 * class be its own builder, then the Constructor will call New directly,
 * and it needs the argument for that call.
 **********************************************************/
Hypre_StructVectorBuilder  impl_Hypre_StructVectorBuilder_Constructor
(Hypre_StructGrid grid) {
   return Hypre_StructVectorBuilder_new();
} /* end impl_Hypre_StructVectorBuilderConstructor */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderSetup
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_Setup(Hypre_StructVectorBuilder this) {
   int ierr;
   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;

   Hypre_StructVector SV = this->d_table->newvec;
   if ( SV == NULL ) return 1;
   
   SVp = SV->d_table;
   V = SVp->hsvec;

   ierr = HYPRE_StructVectorAssemble( *V );

   if ( ierr==0 ) this->d_table->vecgood = 1;
   return ierr;
} /* end impl_Hypre_StructVectorBuilderSetup */

