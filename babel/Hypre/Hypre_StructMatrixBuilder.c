/* TO DO: this file has unimplemented functions (marked as such).  Implement some of them */
/******************************************************
 *
 *  File:  Hypre_StructMatrixBuilder.c
 *
 *********************************************************/

#include "Hypre_StructMatrixBuilder_Skel.h" 
#include "Hypre_StructMatrixBuilder_Data.h" 

/* This builder makes a StructMatrix.  It can only make one at a time;
   any matrix under construction will be abandoned if a new one is begun.
   Matrices are reference-counted by Hypre_StructMatrix_addReference and
   Hypre_StructMatrix_deleteReference; they will be deleted when the reference
   count drops to zero.
*/

#include "HYPRE_mv.h"
#include "struct_matrix_vector.h"
#include "Hypre_Box_Skel.h"
#include "Hypre_StructGrid_Skel.h"
#include "Hypre_StructStencil_Skel.h"
#include "Hypre_StructMatrix_Skel.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructMatrixBuilder_constructor(Hypre_StructMatrixBuilder this) {
   this->Hypre_StructMatrixBuilder_data = (struct Hypre_StructMatrixBuilder_private_type *)
      malloc( sizeof( struct Hypre_StructMatrixBuilder_private_type ) );
   this->Hypre_StructMatrixBuilder_data->newmat = NULL;
   this->Hypre_StructMatrixBuilder_data->matgood = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructMatrixBuilder_destructor(Hypre_StructMatrixBuilder this) {
   if ( this->Hypre_StructMatrixBuilder_data->newmat != NULL ) {
      Hypre_StructMatrix_deleteReference( this->Hypre_StructMatrixBuilder_data->newmat );
      /* ... will delete newmat if there are no other references to it */
      };
   free(this->Hypre_StructMatrixBuilder_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderSetStencil
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_SetStencil
(Hypre_StructMatrixBuilder this, Hypre_StructStencil stencil) {
/* not implemented; this functionality isn't in Hypre (though doesn't
   look too hard to put in)
   */
   printf( "unimplemented function, Hypre_StructMatrixBuilder_SetStencil, was called\n" );
   return 1;
} /* end impl_Hypre_StructMatrixBuilderSetStencil */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderSetValue
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_SetValue
(Hypre_StructMatrixBuilder this, array1int where, double value) {
   printf( "unimplemented function, Hypre_StructMatrixBuilder_SetValue, was called\n" );
   return 1;
} /* end impl_Hypre_StructMatrixBuilderSetValue */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderSetBoxValues
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_SetBoxValues
(Hypre_StructMatrixBuilder this, Hypre_Box box,
 array1int stencil_indices, array1double values) {
   int i, ssize, lower[3], upper[3];
   struct Hypre_StructMatrix_private_type * SMp;
   HYPRE_StructMatrix * M;
   hypre_StructMatrix * m;
   struct Hypre_Box_ior BO;
   struct Hypre_Box_private_type * Bp;
   hypre_Box * B;

   Hypre_StructMatrix SM = this->Hypre_StructMatrixBuilder_data->newmat;
   if ( SM == NULL ) return 1;
   
   SMp = SM->Hypre_StructMatrix_data;
   M = SMp->hsmat;
   m = (hypre_StructMatrix *) *M;

   BO = *box;
   Bp = BO.Hypre_Box_data;
   B = Bp->hbox;

   for ( i=0; i<Bp->dimension; ++i ) {
      lower[i] = B->imin[i];
      upper[i] = B->imax[i];
   };

   ssize = stencil_indices.upper[0] - stencil_indices.lower[0];

   return HYPRE_StructMatrixSetBoxValues(
      *M, lower, upper, ssize,
      &(stencil_indices.data[*(stencil_indices.lower)]),
      &(values.data[*(values.lower)]) );

} /* end impl_Hypre_StructMatrixBuilderSetBoxValues */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderGetDims
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_GetDims
(Hypre_StructMatrixBuilder this, int* m, int* n) {
   printf( "unimplemented function, Hypre_StructMatrixBuilderGetdims, was called\n");
   return 1;
} /* end impl_Hypre_StructMatrixBuilderGetDims */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilder_SetMap
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_SetMap(Hypre_StructMatrixBuilder this, Hypre_Map map) {
   printf("Hypre_StructMatrixBuilder_SetMap doesn't work. TO DO: implement this\n" );
   return 1;
} /* end impl_Hypre_StructMatrixBuilder_SetMap */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilder_GetMap
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_GetMap(Hypre_StructMatrixBuilder this, Hypre_Map* map) {
   printf("Hypre_StructMatrixBuilder_GetMap doesn't work. TO DO: implement this\n" );
   return 1;
} /* end impl_Hypre_StructMatrixBuilder_GetMap */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderGetConstructedObject
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_GetConstructedObject
(Hypre_StructMatrixBuilder this, Hypre_LinearOperator* obj) {
   Hypre_StructMatrix newmat = this->Hypre_StructMatrixBuilder_data->newmat;
   if ( newmat==NULL  ||  this->Hypre_StructMatrixBuilder_data->matgood==0 ) {
      *obj = (Hypre_LinearOperator) NULL;
      return 1;
   };
   *obj = (Hypre_LinearOperator)
      Hypre_StructMatrix_castTo( newmat, "Hypre.LinearOperator" );
   return 0;
} /* end impl_Hypre_StructMatrixBuilderGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilder_Start
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_Start
(Hypre_StructMatrixBuilder this, Hypre_StructGrid grid,
 Hypre_StructStencil stencil, int symmetric, array1int num_ghost) {

   struct Hypre_StructGrid_private_type *Gp = grid->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   struct Hypre_StructStencil_private_type *SSp = stencil->Hypre_StructStencil_data;
   HYPRE_StructStencil *SS = SSp->hsstencil;

   MPI_Comm comm = hypre_StructGridComm( g );

   struct Hypre_StructMatrix_private_type * SMp;
   HYPRE_StructMatrix * M;
   int ierr = 0;

   if ( this->Hypre_StructMatrixBuilder_data->newmat != NULL )
      Hypre_StructMatrix_deleteReference( this->Hypre_StructMatrixBuilder_data->newmat );
   this->Hypre_StructMatrixBuilder_data->newmat = Hypre_StructMatrix_New();
   this->Hypre_StructMatrixBuilder_data->matgood = 0;
   Hypre_StructMatrix_addReference( this->Hypre_StructMatrixBuilder_data->newmat );

   SMp = this->Hypre_StructMatrixBuilder_data->newmat->Hypre_StructMatrix_data;
   M = SMp->hsmat;

   ierr += HYPRE_StructMatrixCreate( comm, *G, *SS, M );

   ierr += HYPRE_StructMatrixSetSymmetric( *M, symmetric );

   ierr += HYPRE_StructMatrixSetNumGhost( *M, num_ghost.data );
   ierr += HYPRE_StructMatrixInitialize( *M );

   return ierr;
} /* end impl_Hypre_StructMatrixBuilder_Start */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderConstructor
 *
 * The arguments are ignored; they really belong in the Start function
 * which is separately called.
 * However, the arguments must be in the interface because if a matrix
 * class be its own builder, then the Constructor will call Start directly,
 * and it needs the arguments for that call.
 **********************************************************/
Hypre_StructMatrixBuilder  impl_Hypre_StructMatrixBuilder_Constructor
(Hypre_StructGrid grid, Hypre_StructStencil stencil,
 int symmetric, array1int num_ghost) {
   return Hypre_StructMatrixBuilder_New();
} /* end impl_Hypre_StructMatrixBuilderConstructor */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderSetup
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_Setup(Hypre_StructMatrixBuilder this) {
   int ierr;
   struct Hypre_StructMatrix_private_type * SMp;
   HYPRE_StructMatrix * M;

   Hypre_StructMatrix SM = this->Hypre_StructMatrixBuilder_data->newmat;
   if ( SM == NULL ) return 1;

   SMp = SM->Hypre_StructMatrix_data;
   M = SMp->hsmat;

   ierr = HYPRE_StructMatrixAssemble( *M );

   if ( ierr==0 ) this->Hypre_StructMatrixBuilder_data->matgood = 1;
   return ierr;

} /* end impl_Hypre_StructMatrixBuilderSetup */

