/*#*****************************************************
#
#	File:  Hypre_StructMatrix.c
#
#********************************************************/

#include "Hypre_StructMatrix_Skel.h" 


/*#************************************************
#	Constructor
#**************************************************/

void Hypre_StructMatrix_constructor(Hypre_StructMatrix this) {

/* JFP: Allocates Memory */
   struct Hypre_StructMatrix_private * HSMp;
   HSMp = (struct Hypre_StructMatrix_private *)
      malloc( sizeof( struct Hypre_StructMatrix_private ) );
   this->d_table = (Hypre_StructMatrix_Private) HSMp;

   this->d_table->hsmat = (HYPRE_StructMatrix *)
      malloc( sizeof( HYPRE_StructMatrix ) );

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_StructMatrix_destructor(Hypre_StructMatrix this) {

   /* JFP: Deallocates memory. */

   Hypre_StructMatrix_Private SMP = this->d_table;
   struct Hypre_StructMatrix_private *SMp = SMP;
   HYPRE_StructMatrix *M = SMp->hsmat;

   HYPRE_StructMatrixDestroy( *M );

   free(this->d_table);

}

Hypre_StructMatrix  impl__Hypre_StructMatrix_NewMatrix(
   Hypre_StructMatrix this, Hypre_StructuredGrid grid,
   Hypre_StructStencil stencil, int symmetric ) {

   Hypre_StructMatrix_Private SMP = this->d_table;
   struct Hypre_StructMatrix_private *SMp = SMP;
   HYPRE_StructMatrix *M = SMp->hsmat;

   Hypre_StructuredGrid_Private GP = grid->d_table;
   struct Hypre_StructuredGrid_private *Gp = GP;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   Hypre_StructStencil_Private SSP = stencil->d_table;
   struct Hypre_StructStencil_private *SSp = SSP;
   HYPRE_StructStencil *SS = SSp->hsstencil;

   MPI_Comm comm = hypre_StructGridComm( g );

   HYPRE_StructMatrixCreate( comm, *G, *SS, M );

   HYPRE_StructMatrixSetSymmetric( *M, symmetric );

   HYPRE_StructMatrixInitialize( *M );

   /* I don't want to put this in the interface (makes it too unnatural or
      complicated for a user), so I'm trying to call it multiple times.
      This may not work. (JfP 130100) */
   HYPRE_StructMatrixAssemble( *M );

   return this;
}

void  impl__Hypre_StructMatrix_print(Hypre_StructMatrix this) {

   int boxarray_size;
   FILE * file;

   Hypre_StructMatrix_Private SMP = this->d_table;
   struct Hypre_StructMatrix_private *SMp = SMP;
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

}

int  impl__Hypre_StructMatrix_SetGrid(Hypre_StructMatrix this, Hypre_StructuredGrid grid) {

/* not implemented; this functionality isn't in Hypre (though doesn't
   look too hard to put in)
   */
   printf( "unimplemented function, Hypre_StructMatrix_SetGrid, was called" );

}

int  impl__Hypre_StructMatrix_SetStencil(Hypre_StructMatrix this, Hypre_StructStencil stencil) {

/* not implemented; this functionality isn't in Hypre (though doesn't
   look too hard to put in)
   */
   printf( "unimplemented function, Hypre_StructMatrix_SetStencil, was called" );

}

int  impl__Hypre_StructMatrix_SetValues(
   Hypre_StructMatrix this, Hypre_Box box,
   array1int stencil_indices, array1double values) {

   int i, ssize, lower[3], upper[3];

   Hypre_StructMatrix_Private SMP = this->d_table;
   struct Hypre_StructMatrix_private *SMp = SMP;
   HYPRE_StructMatrix *M = SMp->hsmat;
   hypre_StructMatrix *m = (hypre_StructMatrix *) *M;

   struct Hypre_Box_object__ BO = *box;
   Hypre_Box_Private BP = BO.d_table;
   struct Hypre_Box_private *Bp = BP;
   hypre_Box *B = Bp->hbox;

   for ( i=0; i<Bp->dimension; ++i ) {
      lower[i] = B->imin[i];
      upper[i] = B->imax[i];
   };

   ssize = stencil_indices.upper[0] - stencil_indices.lower[0];
   
   /*   printf(
      "lower[0]=%i, upper[0]=%i, stencil size=%i, first stencil data=%i, first matrix value=%f\n",
      lower[0], upper[0], ssize, stencil_indices.data[*(stencil_indices.lower)],
      values.data[*(values.lower)]  );
   */

   HYPRE_StructMatrixSetBoxValues(
      *M, lower, upper, ssize,
      &(stencil_indices.data[*(stencil_indices.lower)]),
      &(values.data[*(values.lower)]) );

   /* I don't want to put this in the interface (makes it too unnatural or
      complicated for a user), so I'm trying to call it multiple times.
      This may not work. (JfP 130100) */
   HYPRE_StructMatrixAssemble( *M );
}

int  impl__Hypre_StructMatrix_Setup(
   Hypre_StructMatrix this, Hypre_StructuredGrid grid,
   Hypre_StructStencil stencil, int symmetric) {

   impl__Hypre_StructMatrix_NewMatrix( this, grid, stencil, symmetric );
   return 0;
}

void  impl__Hypre_StructMatrix_Apply(Hypre_StructMatrix this, Hypre_StructVector x, Hypre_StructVector* b) {

   /* b = A * x   where this = A  */

   Hypre_StructMatrix_Private SMP = this->d_table;
   struct Hypre_StructMatrix_private *SMp = SMP;
   HYPRE_StructMatrix *M = SMp->hsmat;
   hypre_StructMatrix *hA = (hypre_StructMatrix *) *M;

   Hypre_StructVector_Private SVxP = x->d_table;
   struct Hypre_StructVector_private *SVxp = SVxP;
   HYPRE_StructVector *Vx = SVxp->hsvec;
   hypre_StructVector *hx = (hypre_StructVector *) *Vx;

   Hypre_StructVector_Private SVyP = (*b)->d_table;
   struct Hypre_StructVector_private *SVyp = SVyP;
   HYPRE_StructVector *Vy = SVyp->hsvec;
   hypre_StructVector *hy = (hypre_StructVector *) *Vy;

   hypre_StructMatvec( 1.0, hA, hx, 1.0, hy );  /* y = A*x */

}

Hypre_StructMatrix  impl__Hypre_StructMatrix_GetConstructedObject(Hypre_StructMatrix this) {

   return this;
}


