/*#*****************************************************
#
#	File:  Hypre_StructuredGrid.c
#
#********************************************************/

#include "Hypre_StructuredGrid_Skel.h" 
#include "Hypre_StructuredGrid_Data.h"  /*gkk: added*/
#include "Hypre_MPI_Com_Skel.h"         /*gkk: added*/
#include "Hypre_MPI_Com_Data.h"         /*gkk: added*/
#include "Hypre_Box_Skel.h"             /*gkk: added*/
#include "Hypre_Box_Data.h"             /*gkk: added*/

/*#************************************************
#	Constructor
#**************************************************/

void Hypre_StructuredGrid_constructor(Hypre_StructuredGrid this) {

/* JFP: Allocates Memory */
   this->d_table = (struct Hypre_StructuredGrid_private_type *)
      malloc( sizeof( struct Hypre_StructuredGrid_private_type ) );

   this->d_table->hsgrid = (HYPRE_StructGrid *)
      malloc( sizeof( HYPRE_StructGrid ) );

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_StructuredGrid_destructor(Hypre_StructuredGrid this) {

   /* JFP: Deallocates memory. */

   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;

   HYPRE_StructGridDestroy( *G );

   free(this->d_table);
}

Hypre_StructuredGrid  impl_Hypre_StructuredGrid_NewGrid(
   Hypre_StructuredGrid this, Hypre_MPI_Com com, int dimension) {

   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;

   struct Hypre_MPI_Com_private_type *Cp = com->d_table;
   MPI_Comm *C = Cp->hcom; /*gkk: ??? was CP->hcom */

   HYPRE_StructGridCreate( *C, dimension, G );

   return this;

}

void  impl_Hypre_StructuredGrid_print(Hypre_StructuredGrid this) {

   int i, d ;

   hypre_BoxArray *boxes;
   hypre_Box box;

   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
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

int  impl_Hypre_StructuredGrid_SetGridExtents(
   Hypre_StructuredGrid this, Hypre_Box box) {

   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   struct Hypre_Box_private_type *Bp = box->d_table;
   hypre_Box *B = Bp->hbox;

   return HYPRE_StructGridSetExtents( *G, B->imin, B->imax );
}

int  impl_Hypre_StructuredGrid_Assemble(Hypre_StructuredGrid this) {
   /* I don't know what this is for, but it's probably only for
   multiprocessing use - the actual code in struct_grid.c looks like that.
   I can't build a working code which uses this in sequential mode.  (JfP)
   JfP 130100: it works now, I don't know why */
/*#ifndef HYPRE_SEQUENTIAL*/

   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   hypre_StructGridAssemble( g );
/*#endif*/
}

Hypre_StructuredGrid  impl_Hypre_StructuredGrid_GetConstructedObject(Hypre_StructuredGrid this) {
   return this;
}


