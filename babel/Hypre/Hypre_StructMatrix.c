
/******************************************************
 *
 *  File:  Hypre_StructMatrix.c
 *
 *********************************************************/

#include "Hypre_StructMatrix_Skel.h" 
#include "Hypre_StructMatrix_Data.h" 

 /* gkk: added... */
#include "Hypre_StructGrid_Skel.h"
#include "Hypre_StructGrid_Data.h"
#include "Hypre_StructStencil_Skel.h"
#include "Hypre_StructStencil_Data.h"
#include "Hypre_StructVector_Skel.h"
#include "Hypre_StructVector_Data.h"
#include "Hypre_Box_Skel.h"
#include "Hypre_Box_Data.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructMatrix_constructor(Hypre_StructMatrix this) {
   this->d_table = (struct Hypre_StructMatrix_private_type *)
      malloc( sizeof( struct Hypre_StructMatrix_private_type ) );

   this->d_table->hsmat = (HYPRE_StructMatrix *)
      malloc( sizeof( HYPRE_StructMatrix ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructMatrix_destructor(Hypre_StructMatrix this) {
   struct Hypre_StructMatrix_private_type *SMp = this->d_table;
   HYPRE_StructMatrix *M = SMp->hsmat;

   HYPRE_StructMatrixDestroy( *M );
   free(this->d_table);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructMatrixprint
 **********************************************************/
void  impl_Hypre_StructMatrix_print(Hypre_StructMatrix this) {

   int boxarray_size;
   FILE * file;

   struct Hypre_StructMatrix_private_type *SMp = this->d_table;
   HYPRE_StructMatrix *M = SMp->hsmat;
   hypre_StructMatrix *m = (hypre_StructMatrix *) *M;

   if ( m->data_space==NULL )
      boxarray_size = -1;
   else
      boxarray_size = m->data_space->size;

   printf( "StructMatrix, data size =%i, BoxArray size=%i, symmetric=%i\n",
           m->data_size, boxarray_size, m->symmetric );

   /* c.f. struct_matrix.c, line 717... */
   file = fopen( "testum.out", "w" );
   fprintf( file, "\nMatrix Data:\n");
   hypre_PrintBoxArrayData(
      file, hypre_StructMatrixDataSpace(m),
      hypre_StructMatrixDataSpace(m), m->num_values,
      hypre_StructMatrixData(m) );
   fflush(file);
   fclose(file);
} /* end impl_Hypre_StructMatrixprint */


/* ********************************************************
 * impl_Hypre_StructMatrixApply
 **********************************************************/
int impl_Hypre_StructMatrix_Apply
(Hypre_StructMatrix this, Hypre_Vector b, Hypre_Vector* x) {
   /* x = A * b   where this = A  */
/* was...
(Hypre_StructMatrix this, Hypre_StructVector b, Hypre_StructVector* x) */

   struct Hypre_StructMatrix_private_type *SMp = this->d_table;
   HYPRE_StructMatrix *M = SMp->hsmat;
   hypre_StructMatrix *hA = (hypre_StructMatrix *) *M;

   Hypre_StructVector SVb, SVx;
   struct Hypre_StructVector_private_type * SVbp;
   HYPRE_StructVector * Vb;
   hypre_StructVector * hb;
   struct Hypre_StructVector_private_type * SVxp;
   HYPRE_StructVector * Vx;
   hypre_StructVector * hx;

   SVb = (Hypre_StructVector) Hypre_Vector_castTo( b, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;
   SVx = (Hypre_StructVector) Hypre_Vector_castTo( *x, "Hypre_StructVector" );
   if ( SVb==NULL ) return 1;

   SVxp = SVx->d_table;
   Vx = SVxp->hsvec;
   hx = (hypre_StructVector *) *Vx;

   SVbp = SVb->d_table;
   Vb = SVbp->hsvec;
   hb = (hypre_StructVector *) *Vb;

   return hypre_StructMatvec( 1.0, hA, hb, 0.0, hx );  /* x = A*b */

} /* end impl_Hypre_StructMatrixApply */

