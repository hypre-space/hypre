/*#*****************************************************
#
#	File:  Hypre_StructuredGrid.c
#
#********************************************************/

#include "Hypre_StructuredGrid_Skel.h" 


/*#************************************************
#	Constructor
#**************************************************/

void Hypre_StructuredGrid_constructor(Hypre_StructuredGrid this) {

/* JFP: Allocates Memory */
   struct Hypre_StructuredGrid_private * HSGp;
   HSGp = (struct Hypre_StructuredGrid_private *)
      malloc( sizeof( struct Hypre_StructuredGrid_private ) );
   this->d_table = (Hypre_StructuredGrid_Private) HSGp;

   this->d_table->hsgrid = (HYPRE_StructGrid *)
      malloc( sizeof( HYPRE_StructGrid ) );

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_StructuredGrid_destructor(Hypre_StructuredGrid this) {

   /* JFP: Deallocates memory. */

   Hypre_StructuredGrid_Private GP = this->d_table;
   struct Hypre_StructuredGrid_private *Gp = GP;
   HYPRE_StructGrid *G = Gp->hsgrid;

   HYPRE_StructGridDestroy( *G );

   free(this->d_table);
}

Hypre_StructuredGrid  impl__Hypre_StructuredGrid_NewGrid(
   Hypre_StructuredGrid this, Hypre_MPI_Com com, int dimension) {

   Hypre_StructuredGrid_Private GP = this->d_table;
   struct Hypre_StructuredGrid_private *Gp = GP;
   HYPRE_StructGrid *G = Gp->hsgrid;

   Hypre_MPI_Com_Private CP = com->d_table;
   struct Hypre_MPI_Com_private *Cp = CP;
   MPI_Comm *C = CP->hcom;

   HYPRE_StructGridCreate( *C, dimension, G );

   return this;

}

void  impl__Hypre_StructuredGrid_print(Hypre_StructuredGrid this) {

   int i, d ;

   hypre_BoxArray *boxes;
   hypre_Box box;

   Hypre_StructuredGrid_Private GP = this->d_table;
   struct Hypre_StructuredGrid_private *Gp = GP;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;
   
   boxes = g->boxes;

   printf( "StructuredGrid, dimension=%i, global_size=%i, ",
           g->dim, g->global_size );
   printf( "%i boxes with imin,imax::\n",boxes->size );
   for ( i=0; i<boxes->size; ++i ) {
      box = (boxes->boxes)[i];
      for ( d=0; d<g->dim; ++d )
         printf( " %i,%i  ", box.imin[d], box.imax[d] );
      printf( "\n" );
   };

}

int  impl__Hypre_StructuredGrid_SetGridExtents(
   Hypre_StructuredGrid this, Hypre_Box box) {

   Hypre_StructuredGrid_Private GP = this->d_table;
   struct Hypre_StructuredGrid_private *Gp = GP;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   struct Hypre_Box_object__ BO = *box;
   Hypre_Box_Private BP = BO.d_table;
   struct Hypre_Box_private *Bp = BP;
   hypre_Box *B = Bp->hbox;

   return HYPRE_StructGridSetExtents( *G, B->imin, B->imax );
}

int  impl__Hypre_StructuredGrid_Assemble(Hypre_StructuredGrid this) {
   /* I don't know what this is for, but it's probably only for
   multiprocessing use - the actual code in struct_grid.c looks like that.
   I can't build a working code which uses this in sequential mode.  (JfP)
   JfP 130100: it works now, I don't know why */
/*#ifndef HYPRE_SEQUENTIAL*/

   Hypre_StructuredGrid_Private GP = this->d_table;
   struct Hypre_StructuredGrid_private *Gp = GP;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   hypre_StructGridAssemble( g );
/*#endif*/
}

Hypre_StructuredGrid  impl__Hypre_StructuredGrid_GetConstructedObject(Hypre_StructuredGrid this) {

	/*#*******************************************************
	#
	#	Put Library code here!!!!!!
	#
	#*********************************************************/

}


