
/******************************************************
 *
 *  File:  Hypre_StructVector.c
 *
 *********************************************************/

#include "Hypre_StructVector_Skel.h" 
#include "Hypre_StructVector_Data.h" 

#include "Hypre_StructVectorBuilder_Skel.h" 
#include "Hypre_StructVectorBuilder_Data.h" 

#include "Hypre_Box_Skel.h"
#include "Hypre_Box_Data.h"
#include "Hypre_StructGrid_Skel.h"
#include "Hypre_StructGrid_Data.h"

/* A Hypre_StructVector points and interfaces to a hypre_StructVector.
   The hypre_StructVector is reference-counted.
   The Hypre_StructVector is also reference-counted.
   Both reference counting systems are independent - they don't know about
   each other. */

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructVector_constructor(Hypre_StructVector this) {
   this->Hypre_StructVector_data = (struct Hypre_StructVector_private_type *)
      malloc( sizeof( struct Hypre_StructVector_private_type ) );

   this->Hypre_StructVector_data->hsvec = (HYPRE_StructVector *)
      malloc( sizeof( HYPRE_StructVector ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructVector_destructor(Hypre_StructVector this) {
   struct Hypre_StructVector_private_type *SVp = this->Hypre_StructVector_data;
   HYPRE_StructVector *V = SVp->hsvec;

   HYPRE_StructVectorDestroy( *V );

   free(this->Hypre_StructVector_data);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructVector_GetNumGhost
 **********************************************************/
int  impl_Hypre_StructVector_GetNumGhost
( Hypre_StructVector this, array1int values )
{
   int  i;
   int * num_ghost = &(values.data[*(values.lower)]);
   HYPRE_StructVector *V = this->Hypre_StructVector_data->hsvec;
   hypre_StructVector * vector = (hypre_StructVector *) (*V);

   for (i = 0; i < 6; i++)
      num_ghost[i] = hypre_StructVectorNumGhost(vector)[i];

   return 0;

} /* end impl_Hypre_StructVector_GetNumGhost */

/* ********************************************************
 * impl_Hypre_StructVector_print
 **********************************************************/
void  impl_Hypre_StructVector_print(Hypre_StructVector this) {
   int boxarray_size;
   FILE * file;

   struct Hypre_StructVector_private_type *SVp = this->Hypre_StructVector_data;
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
} /* end impl_Hypre_StructVector_print */

/* ********************************************************
 * impl_Hypre_StructVector_Clear
 *    int Clear ();                          // y <- 0 (where y=self)
 **********************************************************/
int  impl_Hypre_StructVector_Clear(Hypre_StructVector this) {

   struct Hypre_StructVector_private_type *SVp = this->Hypre_StructVector_data;
   HYPRE_StructVector *V = SVp->hsvec;
   hypre_StructVector *v = (hypre_StructVector *) *V;

   return hypre_StructVectorClearAllValues( v );

} /* end impl_Hypre_StructVector_Clear */

/* ********************************************************
 * impl_Hypre_StructVector_Copy
 *    int Copy (in Vector x);                // y <- x 
 **********************************************************/
int  impl_Hypre_StructVector_Copy(Hypre_StructVector this, Hypre_Vector x) {
   struct Hypre_StructVector_private_type *SVyp = this->Hypre_StructVector_data;
   HYPRE_StructVector *Vy = SVyp->hsvec;
   hypre_StructVector *vy = (hypre_StructVector *) *Vy;

   Hypre_StructVector SVx;
   struct Hypre_StructVector_private_type * SVxp;
   HYPRE_StructVector * Vx;
   hypre_StructVector * vx;

   SVx = (Hypre_StructVector) Hypre_Vector_castTo( x, "Hypre.StructVector" );
   if ( SVx==NULL ) return 1;
   SVxp = SVx->Hypre_StructVector_data;
   Vx = SVxp->hsvec;
   vx = (hypre_StructVector *) *Vx;

   return hypre_StructCopy( vx, vy );

} /* end impl_Hypre_StructVector_Copy */

/* ********************************************************
 * impl_Hypre_StructVector_Clone
 *    int Clone (out Vector x);              // create an x compatible with y
 **********************************************************/
int  impl_Hypre_StructVector_Clone(Hypre_StructVector this, Hypre_Vector* x) {
   int ierr = 0;
   int numghost_data[6] = {0, 0, 0, 0, 0, 0};
   array1int num_ghost;
   int dim;
   struct Hypre_StructVector_private_type *SVyp = this->Hypre_StructVector_data;
   HYPRE_StructVector *Vy = SVyp->hsvec;
   hypre_StructVector *vy = (hypre_StructVector *) *Vy;
   Hypre_StructGrid G = SVyp->grid;
   Hypre_StructVectorBuilder SVB = Hypre_StructVectorBuilder_Constructor( G );
   /* ... SVB ignores G */

   ierr += Hypre_StructGrid_GetParameterInt( G, "dim", &dim );
   Hypre_StructVectorBuilder_Start( SVB, G );

   num_ghost.lower[0] = 0;
   num_ghost.upper[0] = 2*dim;
   num_ghost.data = numghost_data;
   Hypre_StructVector_GetNumGhost( this, num_ghost );
   Hypre_StructVectorBuilder_SetNumGhost( SVB, num_ghost );

   Hypre_StructVectorBuilder_Setup( SVB );
   ierr += Hypre_StructVectorBuilder_GetConstructedObject( SVB, x );
   /* ... *x is really a Hypre_StructVector */

   return ierr;

/* TO DO: change get/set numghost to get/set parameter. */

} /* end impl_Hypre_StructVector_Clone */



/*--------------------------------------------------------------------------
 * hypre_StructVectorScaleAllValues
 * This function really belongs in struct_vector.c
 * It is the same as ClearAllValues, except for the arg list and the line
 * inside the loop.
 *--------------------------------------------------------------------------*/

int 
hypre_StructVectorScaleAllValues( hypre_StructVector *vector, double a )
{
   int               ierr = 0;

   int               datai;
   double           *data;

   hypre_Index       imin;
   hypre_Index       imax;
   hypre_Box        *box;
   hypre_Index       loop_size;

   int               loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   box = hypre_BoxCreate();
   hypre_SetIndex(imin, 1, 1, 1);
   hypre_SetIndex(imax, hypre_StructVectorDataSize(vector), 1, 1);
   hypre_BoxSetExtents(box, imin, imax);
   data = hypre_StructVectorData(vector);
   hypre_BoxGetSize(box, loop_size);

   hypre_BoxLoop1Begin(loop_size,
                       box, imin, imin, datai);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,datai
#include "hypre_box_smp_forloop.h"
   hypre_BoxLoop1For(loopi, loopj, loopk, datai)
      {
         data[datai] = a * data[datai];
      }
   hypre_BoxLoop1End(datai);

   hypre_BoxDestroy(box);

   return ierr;
}

/* ********************************************************
 * impl_Hypre_StructVector_Scale
 *    int Scale (in double a);               // y <- a*y 
 **********************************************************/
int  impl_Hypre_StructVector_Scale(Hypre_StructVector this, double a) {
   struct Hypre_StructVector_private_type *SVyp = this->Hypre_StructVector_data;
   HYPRE_StructVector *Vy = SVyp->hsvec;
   hypre_StructVector *vy = (hypre_StructVector *) *Vy;

   return hypre_StructVectorScaleAllValues( vy, a );

} /* end impl_Hypre_StructVector_Scale */

/* ********************************************************
 * impl_Hypre_StructVector_Dot
 *    int Dot (in Vector x, out double d);   // d <- (y,x)
 **********************************************************/
int  impl_Hypre_StructVector_Dot(Hypre_StructVector this, Hypre_Vector x, double* d) {
   struct Hypre_StructVector_private_type *SVyp = this->Hypre_StructVector_data;
   HYPRE_StructVector *Vy = SVyp->hsvec;
   hypre_StructVector *vy = (hypre_StructVector *) *Vy;

   Hypre_StructVector SVx;
   struct Hypre_StructVector_private_type * SVxp;
   HYPRE_StructVector * Vx;
   hypre_StructVector * vx;

   SVx = (Hypre_StructVector) Hypre_Vector_castTo( x, "Hypre.StructVector" );
   if ( SVx==NULL ) return 1;
   SVxp = SVx->Hypre_StructVector_data;
   Vx = SVxp->hsvec;
   vx = (hypre_StructVector *) *Vx;

   *d = hypre_StructInnerProd(  vx, vy );
   return 0;

} /* end impl_Hypre_StructVector_Dot */

/* ********************************************************
 * impl_Hypre_StructVector_Axpy
 *    int Axpy (in double a, in Vector x);   // y <- a*x + y
 **********************************************************/
int  impl_Hypre_StructVector_Axpy(Hypre_StructVector this, double a, Hypre_Vector x) {
   struct Hypre_StructVector_private_type *SVyp = this->Hypre_StructVector_data;
   HYPRE_StructVector *Vy = SVyp->hsvec;
   hypre_StructVector *vy = (hypre_StructVector *) *Vy;

   Hypre_StructVector SVx;
   struct Hypre_StructVector_private_type * SVxp;
   HYPRE_StructVector * Vx;
   hypre_StructVector * vx;

   SVx = (Hypre_StructVector) Hypre_Vector_castTo( x, "Hypre.StructVector" );
   if ( SVx==NULL ) return 1;
   SVxp = SVx->Hypre_StructVector_data;
   Vx = SVxp->hsvec;
   vx = (hypre_StructVector *) *Vx;

   return hypre_StructAxpy( a, vx, vy );

} /* end impl_Hypre_StructVector_Axpy */


/* ********************************************************
 * impl_Hypre_StructVector_GetGlobalSize
 **********************************************************/
int  impl_Hypre_StructVector_GetGlobalSize(Hypre_StructVector this, int* size) {
   struct Hypre_StructVector_private_type *SVyp = this->Hypre_StructVector_data;
   HYPRE_StructVector *Vy = SVyp->hsvec;
   hypre_StructVector *vy = (hypre_StructVector *) *Vy;
   *size = hypre_StructVectorGlobalSize(vy);
   return 0;
} /* end impl_Hypre_StructVector_GetGlobalSize */

/* ********************************************************
 * impl_Hypre_StructVector_GetMap
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_StructVector_GetMap(Hypre_StructVector this, Hypre_Map* map) {
   printf("Hypre_StructVector_GetMap doesn't work. TO DO: implement this\n");
   return 1;
} /* end impl_Hypre_StructVector_GetMap */

