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
   this->d_table = (struct Hypre_StructMatrixBuilder_private_type *)
      malloc( sizeof( struct Hypre_StructMatrixBuilder_private_type ) );
   this->d_table->newmat = NULL;
   this->d_table->matgood = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructMatrixBuilder_destructor(Hypre_StructMatrixBuilder this) {
   if ( this->d_table->newmat != NULL ) {
      Hypre_StructMatrix_deleteReference( this->d_table->newmat );
      /* ... will delete newmat if there are no other references to it */
      };
   free(this->d_table);
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
   struct Hypre_Box_object_ BO;
   struct Hypre_Box_private_type * Bp;
   hypre_Box * B;

   Hypre_StructMatrix SM = this->d_table->newmat;
   if ( SM == NULL ) return 1;
   
   SMp = SM->d_table;
   M = SMp->hsmat;
   m = (hypre_StructMatrix *) *M;

   BO = *box;
   Bp = BO.d_table;
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
 * impl_Hypre_StructMatrixBuilderGetConstructedObject
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_GetConstructedObject
(Hypre_StructMatrixBuilder this, Hypre_LinearOperator* obj) {
   Hypre_StructMatrix newmat = this->d_table->newmat;
   if ( newmat==NULL  ||  this->d_table->matgood==0 ) {
      *obj = (Hypre_LinearOperator) NULL;
      return 1;
   };
   *obj = (Hypre_LinearOperator)
      Hypre_StructMatrix_castTo( newmat, "Hypre_LinearOperator" );
   return 0;
} /* end impl_Hypre_StructMatrixBuilderGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderNew
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_New
(Hypre_StructMatrixBuilder this, Hypre_StructGrid grid,
 Hypre_StructStencil stencil, int symmetric, array1int num_ghost) {

   struct Hypre_StructGrid_private_type *Gp = grid->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   struct Hypre_StructStencil_private_type *SSp = stencil->d_table;
   HYPRE_StructStencil *SS = SSp->hsstencil;

   MPI_Comm comm = hypre_StructGridComm( g );

   struct Hypre_StructMatrix_private_type * SMp;
   HYPRE_StructMatrix * M;
   int ierr = 0;

   if ( this->d_table->newmat != NULL )
      Hypre_StructMatrix_deleteReference( this->d_table->newmat );
   this->d_table->newmat = Hypre_StructMatrix_new();
   this->d_table->matgood = 0;
   Hypre_StructMatrix_addReference( this->d_table->newmat );

   SMp = this->d_table->newmat->d_table;
   M = SMp->hsmat;

   ierr += HYPRE_StructMatrixCreate( comm, *G, *SS, M );

   ierr += HYPRE_StructMatrixSetSymmetric( *M, symmetric );

   ierr += HYPRE_StructMatrixSetNumGhost( *M, num_ghost.data );
   ierr += HYPRE_StructMatrixInitialize( *M );

   return ierr;
} /* end impl_Hypre_StructMatrixBuilderNew */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderConstructor
 *
 * The arguments are ignored; they really belong in the New function
 * which is separately called.
 * However, the arguments must be in the interface because if a matrix
 * class be its own builder, then the Constructor will call New directly,
 * and it needs the arguments for that call.
 **********************************************************/
Hypre_StructMatrixBuilder  impl_Hypre_StructMatrixBuilder_Constructor
(Hypre_StructGrid grid, Hypre_StructStencil stencil,
 int symmetric, array1int num_ghost) {
   return Hypre_StructMatrixBuilder_new();
} /* end impl_Hypre_StructMatrixBuilderConstructor */

/* ********************************************************
 * impl_Hypre_StructMatrixBuilderSetup
 **********************************************************/
int  impl_Hypre_StructMatrixBuilder_Setup(Hypre_StructMatrixBuilder this) {
   int ierr;
   struct Hypre_StructMatrix_private_type * SMp;
   HYPRE_StructMatrix * M;

   Hypre_StructMatrix SM = this->d_table->newmat;
   if ( SM == NULL ) return 1;

   SMp = SM->d_table;
   M = SMp->hsmat;

   ierr = HYPRE_StructMatrixAssemble( *M );

   if ( ierr==0 ) this->d_table->matgood = 1;
   return ierr;

} /* end impl_Hypre_StructMatrixBuilderSetup */

