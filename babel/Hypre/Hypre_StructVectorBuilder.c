
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
#include "Hypre_StructGrid_Skel.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructVectorBuilder_constructor(Hypre_StructVectorBuilder this) {
   this->Hypre_StructVectorBuilder_data = (struct Hypre_StructVectorBuilder_private_type *)
      malloc( sizeof( struct Hypre_StructVectorBuilder_private_type ) );
   this->Hypre_StructVectorBuilder_data->newvec = NULL;
   this->Hypre_StructVectorBuilder_data->vecgood = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructVectorBuilder_destructor(Hypre_StructVectorBuilder this) {
   if ( this->Hypre_StructVectorBuilder_data->newvec != NULL ) {
      Hypre_StructVector_deleteReference( this->Hypre_StructVectorBuilder_data->newvec );
      /* ... will delete newvec if there are no other references to it */
   };
   free(this->Hypre_StructVectorBuilder_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_print
 **********************************************************/
void  impl_Hypre_StructVectorBuilder_print(Hypre_StructVectorBuilder this) {
   printf( "StructVectorBuilder\n" );
} /* end impl_Hypre_StructVectorBuilder_print */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_SetValue
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_SetValue
(Hypre_StructVectorBuilder this, array1int where, double value) {
   printf("not implemented as of 5/5/2000.");/* TO DO: implement this */
   return 1;
} /* end impl_Hypre_StructVectorBuilder_SetValue */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_SetBoxValues
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_SetBoxValues
(Hypre_StructVectorBuilder this, Hypre_Box box, array1double values) {
   int i, ssize, lower[3], upper[3];
   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;
   struct Hypre_Box_private_type * Bp;
   hypre_Box * B;

   Hypre_StructVector SV = this->Hypre_StructVectorBuilder_data->newvec;
   if ( SV == NULL ) return 1;

   SVp = SV->Hypre_StructVector_data;
   V = SVp->hsvec;
   Bp = box->Hypre_Box_data;
   B = Bp->hbox;

   for ( i=0; i<Bp->dimension; ++i ) {
      lower[i] = B->imin[i];
      upper[i] = B->imax[i];
   };

   return HYPRE_StructVectorSetBoxValues( *V, lower, upper,
                                          &(values.data[*(values.lower)]) );

} /* end impl_Hypre_StructVectorBuilder_SetBoxValues */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_SetNumGhost
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_SetNumGhost
(Hypre_StructVectorBuilder this, array1int values) {
   HYPRE_StructVector  vector;
   int * num_ghost = &(values.data[*(values.lower)]);
   HYPRE_StructVector * V;
   Hypre_StructVector SV = this->Hypre_StructVectorBuilder_data->newvec;
   if ( SV == NULL ) return 1;
   V = SV->Hypre_StructVector_data->hsvec;

   return HYPRE_StructVectorSetNumGhost( *V, num_ghost );
} /* end impl_Hypre_StructVectorBuilder_SetNumGhost */


/* ********************************************************
 * impl_Hypre_StructVectorBuilder_SetMap
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_SetMap(Hypre_StructVectorBuilder this, Hypre_Map map) {
   printf("Hypre_StructVectorBuilder_SetMap doesn't work. TO DO: implement this\n");
   return 1;
} /* end impl_Hypre_StructVectorBuilder_SetMap */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_GetMap
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_GetMap(Hypre_StructVectorBuilder this, Hypre_Map* map) {
   printf("Hypre_StructVectorBuilder_GetMap doesn't work. TO DO: implement this\n");
   return 1;
} /* end impl_Hypre_StructVectorBuilder_GetMap */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_GetConstructedObject
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_GetConstructedObject
(Hypre_StructVectorBuilder this, Hypre_Vector* obj) {
   Hypre_StructVector newvec = this->Hypre_StructVectorBuilder_data->newvec;
   if ( newvec==NULL || this->Hypre_StructVectorBuilder_data->vecgood==0 ) {
      printf( "Hypre_StructVectorBuilder: object not constructed yet\n");
      *obj = (Hypre_Vector) NULL;
      return 1;
   };
   *obj = (Hypre_Vector) Hypre_StructVector_castTo( newvec, "Hypre.Vector" );
   return 0;
} /* end impl_Hypre_StructVectorBuilder_GetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_Start
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_Start
(Hypre_StructVectorBuilder this, Hypre_StructGrid grid) {

   struct Hypre_StructGrid_private_type *Gp = grid->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   Hypre_MPI_Com com = Gp->comm;
   /*   MPI_Comm comm = hypre_StructGridComm( g ); ... this is ok, but the
        following requires less knowledge of what's in a hypre_StructGrid ...
    */
   MPI_Comm comm = *(com->Hypre_MPI_Com_data->hcom);

   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;
   if ( this->Hypre_StructVectorBuilder_data->newvec != NULL )
      Hypre_StructVector_deleteReference( this->Hypre_StructVectorBuilder_data->newvec );
   this->Hypre_StructVectorBuilder_data->newvec = Hypre_StructVector_New();
   this->Hypre_StructVectorBuilder_data->vecgood = 0;
   Hypre_StructVector_addReference( this->Hypre_StructVectorBuilder_data->newvec );

   SVp = this->Hypre_StructVectorBuilder_data->newvec->Hypre_StructVector_data;
   SVp->grid = grid;
   V = SVp->hsvec;

/*    HYPRE_StructVectorCreate( comm, *G, *SS, V );
      ... This function doesn't use the stencil.  Here we reproduce
      its internals so as not to have to suppy it ... */
   *V = (HYPRE_StructVector) hypre_StructVectorCreate( comm, g ) ;

   return HYPRE_StructVectorInitialize( *V );

} /* end impl_Hypre_StructVectorBuilder_Start */

/* ********************************************************
 * impl_Hypre_StructVectorBuilderConstructor
 * The argument is ignored; it really belongs in the Start function
 * which is separately called.
 * However, the argument must be in the interface because if a vector
 * class be its own builder, then the Constructor will call Start directly,
 * and it needs the argument for that call.
 **********************************************************/
Hypre_StructVectorBuilder  impl_Hypre_StructVectorBuilder_Constructor
(Hypre_StructGrid grid) {
   return Hypre_StructVectorBuilder_New();
} /* end impl_Hypre_StructVectorBuilderConstructor */

/* ********************************************************
 * impl_Hypre_StructVectorBuilder_Setup
 **********************************************************/
int  impl_Hypre_StructVectorBuilder_Setup(Hypre_StructVectorBuilder this) {
   int ierr;
   struct Hypre_StructVector_private_type * SVp;
   HYPRE_StructVector * V;

   Hypre_StructVector SV = this->Hypre_StructVectorBuilder_data->newvec;
   if ( SV == NULL ) return 1;
   
   SVp = SV->Hypre_StructVector_data;
   V = SVp->hsvec;

   ierr = HYPRE_StructVectorAssemble( *V );

   if ( ierr==0 ) this->Hypre_StructVectorBuilder_data->vecgood = 1;
   return ierr;
} /* end impl_Hypre_StructVectorBuilder_Setup */

