/*#*****************************************************
#
#	File:  Hypre_StructVector.c
#
#********************************************************/

#include "Hypre_StructVector_Skel.h" 


/*#************************************************
#	Constructor
#**************************************************/

void Hypre_StructVector_constructor(Hypre_StructVector this) {

/* JFP: Allocates Memory */
   struct Hypre_StructVector_private * HSVp;
   HSVp = (struct Hypre_StructVector_private *)
      malloc( sizeof( struct Hypre_StructVector_private ) );
   this->d_table = (Hypre_StructVector_Private) HSVp;

   this->d_table->hsvec = (HYPRE_StructVector *)
      malloc( sizeof( HYPRE_StructVector ) );
}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_StructVector_destructor(Hypre_StructVector this) {

   /* JFP: Deallocates memory. */

   Hypre_StructVector_Private SVP = this->d_table;
   struct Hypre_StructVector_private *SVp = SVP;
   HYPRE_StructVector *V = SVp->hsvec;

   HYPRE_StructVectorDestroy( *V );

   free(this->d_table);
}

Hypre_StructVector  impl__Hypre_StructVector_NewVector(
   Hypre_StructVector this, Hypre_StructuredGrid grid ) {

   Hypre_StructVector_Private SVP = this->d_table;
   struct Hypre_StructVector_private *SVp = SVP;
   HYPRE_StructVector *V = SVp->hsvec;

   Hypre_StructuredGrid_Private GP = grid->d_table;
   struct Hypre_StructuredGrid_private *Gp = GP;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   MPI_Comm comm = hypre_StructGridComm( g );


/*    HYPRE_StructVectorCreate( comm, *G, *SS, V );
      ... This function doesn't use the stencil.  Here we reproduce
      its internals so as not to have to suppy it ... */
   *V = (HYPRE_StructVector) hypre_StructVectorCreate( comm, g ) ;

   HYPRE_StructVectorInitialize( *V );

   /* I don't want to put this in the interface (makes it too unnatural or
      complicated for a user), so I'm trying to call it multiple times.
      This may not work. (JfP 130100) */
   HYPRE_StructVectorAssemble( *V );
}

void  impl__Hypre_StructVector_print(Hypre_StructVector this) {

   int boxarray_size;
   FILE * file;

   Hypre_StructVector_Private SVP = this->d_table;
   struct Hypre_StructVector_private *SVp = SVP;
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

}

int  impl__Hypre_StructVector_SetGrid(Hypre_StructVector this, Hypre_StructuredGrid grid) {

/* not implemented; this functionality isn't in Hypre (though doesn't
   look too hard to put in)
   */
   printf( "unimplemented function, Hypre_StructVector_SetGrid, was called" );
}

int  impl__Hypre_StructVector_SetStencil(Hypre_StructVector this, Hypre_StructStencil stencil) {

/* This doesn't make sense for a Vector (it makes sense for a Matrix,
   which has the same interface)
   */
   printf( "silly function, Hypre_StructVector_SetStencil, was called" );

}

int  impl__Hypre_StructVector_SetValues(
   Hypre_StructVector this, Hypre_Box box,
   array1int stencil_indices, array1double values) {

   int i, ssize, lower[3], upper[3];

   Hypre_StructVector_Private SVP = this->d_table;
   struct Hypre_StructVector_private *SVp = SVP;
   HYPRE_StructVector *V = SVp->hsvec;

   struct Hypre_Box_object__ BO = *box;
   Hypre_Box_Private BP = BO.d_table;
   struct Hypre_Box_private *Bp = BP;
   hypre_Box *B = Bp->hbox;

   for ( i=0; i<Bp->dimension; ++i ) {
      lower[i] = B->imin[i];
      upper[i] = B->imax[i];
   };

   HYPRE_StructVectorSetBoxValues( *V, lower, upper,
                                   &(values.data[*(values.lower)]) );

   /* I don't want to put this in the interface (makes it too unnatural or
      complicated for a user), so I'm trying to call it multiple times.
      This may not work. (JfP 130100) */
   HYPRE_StructVectorAssemble( *V );
}

int  impl__Hypre_StructVector_Setup(
   Hypre_StructVector this, Hypre_StructuredGrid grid,
   Hypre_StructStencil stencil, int symmetric) {

   impl__Hypre_StructVector_NewVector( this, grid );
   return 0;
}

void  impl__Hypre_StructVector_Apply
(Hypre_StructVector this, Hypre_StructVector x, Hypre_StructVector* b) {

/*
  There is nothing reasonable for this function to do.
  See my comment in Interfaces.idl.
  */
   printf( "I don't know what you think Hypre_StructVector_Apply should do!\n");

}

Hypre_StructMatrix  impl__Hypre_StructVector_GetConstructedObject(Hypre_StructVector this) {

/* Next Babel run will have this return a Hypre_StructVector 
   At that time, uncomment the following line: */
/* return this; */

}


