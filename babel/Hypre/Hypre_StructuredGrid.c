
/******************************************************
 *
 *  File:  Hypre_StructuredGrid.c
 *
 *********************************************************/

#include "Hypre_StructuredGrid_Skel.h" 
#include "Hypre_StructuredGrid_Data.h" 

         /*gkk: added...*/
#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"
#include "Hypre_Box_Skel.h"
#include "Hypre_Box_Data.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructuredGrid_constructor(Hypre_StructuredGrid this) {
   this->d_table = (struct Hypre_StructuredGrid_private_type *)
      malloc( sizeof( struct Hypre_StructuredGrid_private_type ) );

   this->d_table->hsgrid = (HYPRE_StructGrid *)
      malloc( sizeof( HYPRE_StructGrid ) );
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructuredGrid_destructor(Hypre_StructuredGrid this) {
   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;

   HYPRE_StructGridDestroy( *G );

   free(this->d_table);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructuredGridprint
 **********************************************************/
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

} /* end impl_Hypre_StructuredGridprint */

/* ********************************************************
 * impl_Hypre_StructuredGridSetGridExtents
 **********************************************************/
int  impl_Hypre_StructuredGrid_SetGridExtents
(Hypre_StructuredGrid this, Hypre_Box box) {

   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   struct Hypre_Box_private_type *Bp = box->d_table;
   hypre_Box *B = Bp->hbox;

   return HYPRE_StructGridSetExtents( *G, B->imin, B->imax );

} /* end impl_Hypre_StructuredGridSetGridExtents */

/* ********************************************************
 * impl_Hypre_StructuredGridSetDoubleParameter
 **********************************************************/
int  impl_Hypre_StructuredGrid_SetDoubleParameter
(Hypre_StructuredGrid this, char* name, double value) {
   printf( "Hypre_StructuredGrid_SetDoubleParameter does not recognize name ~s\n",
           name );
   return -1;
} /* end impl_Hypre_StructuredGridSetDoubleParameter */

/* ********************************************************
 * impl_Hypre_StructuredGridSetIntParameter
 **********************************************************/
int  impl_Hypre_StructuredGrid_SetIntParameter
(Hypre_StructuredGrid this, char* name, int value) {

   printf( "Hypre_StructuredGrid_SetIntParameter does not recognize name ~s\n", name );

   return -1;
} /* end impl_Hypre_StructuredGridSetIntParameter */

/* ********************************************************
 * impl_Hypre_StructuredGridGetIntParameter
 **********************************************************/
int  impl_Hypre_StructuredGrid_GetIntParameter
(Hypre_StructuredGrid this, char* name ) {
   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *grid = (hypre_StructGrid *) *G;

   if ( !strcmp(name,"dim") || !strcmp(name,"dimension") ) {
      return hypre_StructGridDim(grid);
   }
   else  {
      printf( "Hypre_StructuredGrid_GetIntParameter does not recognize name ~s\n",
              name );
      return -1;
   }

   return -1;
} /* end impl_Hypre_StructuredGridGetIntParameter */

/* ********************************************************
 * impl_Hypre_StructuredGridSetIntArrayParameter
 **********************************************************/
int  impl_Hypre_StructuredGrid_SetIntArrayParameter
(Hypre_StructuredGrid this, char* name, array1int value) {
   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;

   if ( !strcmp(name,"periodic") ) {
      HYPRE_StructGridSetPeriodic( *G, value.data );
      return 0;
   }
   else  {
      printf( "Hypre_StructuredGrid_SetIntArrayParameter does not recognize name ~s\n",
              name );
      return -1;
   }
} /* end impl_Hypre_StructuredGridSetIntArrayParameter */

/* ********************************************************
 * impl_Hypre_StructuredGridGetConstructedObject
 **********************************************************/
Hypre_StructuredGrid
impl_Hypre_StructuredGrid_GetConstructedObject(Hypre_StructuredGrid this) {
   return this;
} /* end impl_Hypre_StructuredGridGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructuredGridNew
 **********************************************************/
int  impl_Hypre_StructuredGrid_New
(Hypre_StructuredGrid this, Hypre_MPI_Com com, int dimension) {
   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;

   struct Hypre_MPI_Com_private_type *Cp = com->d_table;
   MPI_Comm *C = Cp->hcom; /*gkk: ??? was CP->hcom */

   return HYPRE_StructGridCreate( *C, dimension, G );

} /* end impl_Hypre_StructuredGridNew */

/* ********************************************************
 * impl_Hypre_StructuredGridConstructor
 **********************************************************/
Hypre_StructuredGrid  impl_Hypre_StructuredGrid_Constructor
(Hypre_MPI_Com com, int dimension) {
   /* declared static; just combines the new and New functions */
   Hypre_StructuredGrid SG = Hypre_StructuredGrid_new();
   Hypre_StructuredGrid_New( SG, com, dimension );
   return SG;
} /* end impl_Hypre_StructuredGridConstructor */

/* ********************************************************
 * impl_Hypre_StructuredGridSetup
 **********************************************************/
int  impl_Hypre_StructuredGrid_Setup(Hypre_StructuredGrid this) {
   struct Hypre_StructuredGrid_private_type *Gp = this->d_table;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   hypre_StructGridAssemble( g );
} /* end impl_Hypre_StructuredGridSetup */

