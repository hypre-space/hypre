
/******************************************************
 *
 *  File:  Hypre_StructGrid.c
 *
 *********************************************************/

#include "Hypre_StructGrid_Skel.h" 
#include "Hypre_StructGrid_Data.h" 

#include "Hypre_MPI_Com_Skel.h"
#include "Hypre_MPI_Com_Data.h"
#include "Hypre_Box_Skel.h"
#include "Hypre_Box_Data.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructGrid_constructor(Hypre_StructGrid this) {
   this->Hypre_StructGrid_data = (struct Hypre_StructGrid_private_type *)
      malloc( sizeof( struct Hypre_StructGrid_private_type ) );

   this->Hypre_StructGrid_data->hsgrid = (HYPRE_StructGrid *)
      malloc( sizeof( HYPRE_StructGrid ) );
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructGrid_destructor(Hypre_StructGrid this) {
   struct Hypre_StructGrid_private_type *Gp = this->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;

   HYPRE_StructGridDestroy( *G );

   free(this->Hypre_StructGrid_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructGridprint
 **********************************************************/
void  impl_Hypre_StructGrid_print(Hypre_StructGrid this) {

   int i, d ;

   hypre_BoxArray *boxes;
   hypre_Box box;

   struct Hypre_StructGrid_private_type *Gp = this->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;
   
   boxes = g->boxes;

   printf( "StructGrid, dimension=%i, global_size=%i, ",
           g->dim, g->global_size );
   printf( "%i boxes with imin,imax::\n",boxes->size );
   for ( i=0; i<boxes->size; ++i ) {
      box = (boxes->boxes)[i];
      for ( d=0; d<g->dim; ++d )
         printf( " %i,%i  ", box.imin[d], box.imax[d] );
      printf( "\n" );
   };

} /* end impl_Hypre_StructGridprint */

/* ********************************************************
 * impl_Hypre_StructGridSetGridExtents
 **********************************************************/
int  impl_Hypre_StructGrid_SetGridExtents
(Hypre_StructGrid this, Hypre_Box box) {

   struct Hypre_StructGrid_private_type *Gp = this->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   struct Hypre_Box_private_type *Bp = box->Hypre_Box_data;
   hypre_Box *B = Bp->hbox;

   return HYPRE_StructGridSetExtents( *G, B->imin, B->imax );

} /* end impl_Hypre_StructGridSetGridExtents */

/* ********************************************************
 * impl_Hypre_StructGridSetParameterDouble
 **********************************************************/
int  impl_Hypre_StructGrid_SetParameterDouble
(Hypre_StructGrid this, char* name, double value) {
   printf( "Hypre_StructGrid_SetParameterDouble does not recognize name %s\n",
           name );
   return 1;
} /* end impl_Hypre_StructGridSetParameterDouble */

/* ********************************************************
 * impl_Hypre_StructGridSetParameterInt
 **********************************************************/
int  impl_Hypre_StructGrid_SetParameterInt
(Hypre_StructGrid this, char* name, int value) {

   printf( "Hypre_StructGrid_SetParameterInt does not recognize name %s\n", name );

   return 1;
} /* end impl_Hypre_StructGridSetParameterInt */

/* ********************************************************
 * impl_Hypre_StructGridSetParameterIntArray
 **********************************************************/
int  impl_Hypre_StructGrid_SetParameterIntArray
(Hypre_StructGrid this, char* name, array1int value) {
   struct Hypre_StructGrid_private_type *Gp = this->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;

   if ( !strcmp(name,"periodic") ) {
      HYPRE_StructGridSetPeriodic( *G, value.data );
      return 0;
   }
   else  {
      printf( "Hypre_StructGrid_SetParameterIntArray does not recognize name %s\n",
              name );
      return 1;
   }
} /* end impl_Hypre_StructGridSetParameterIntArray */

/* ********************************************************
 * impl_Hypre_StructGridGetParameterInt
 **********************************************************/
int  impl_Hypre_StructGrid_GetParameterInt
(Hypre_StructGrid this, char* name, int* value) {
   struct Hypre_StructGrid_private_type *Gp = this->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *grid = (hypre_StructGrid *) *G;

   if ( !strcmp(name,"dim") || !strcmp(name,"dimension") ) {
      *value = hypre_StructGridDim(grid);
      return 0;
   }
   else  {
      printf( "Hypre_StructGrid_GetParameterInt does not recognize name %s\n",
              name );
      *value = -123456;
      return 1;
   }

   return 1;
} /* end impl_Hypre_StructGridGetParameterInt */

/* ********************************************************
 * impl_Hypre_StructGridGetConstructedObject
 **********************************************************/
int  impl_Hypre_StructGrid_GetConstructedObject
(Hypre_StructGrid this, Hypre_StructuredGrid* obj) {
   *obj = (Hypre_StructuredGrid) Hypre_StructGrid_castTo
      ( this, "Hypre_StructuredGrid" );
   return 0;
} /* end impl_Hypre_StructGridGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructGrid_Start
 **********************************************************/
int  impl_Hypre_StructGrid_Start
(Hypre_StructGrid this, Hypre_MPI_Com com, int dimension) {
   struct Hypre_StructGrid_private_type *Gp = this->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;

   struct Hypre_MPI_Com_private_type *Cp = com->Hypre_MPI_Com_data;
   MPI_Comm *C = Cp->hcom; /*gkk: ??? was CP->hcom */
   Gp->comm = com;

   return HYPRE_StructGridCreate( *C, dimension, G );
} /* end impl_Hypre_StructGrid_Start */

/* ********************************************************
 * impl_Hypre_StructGridConstructor
 **********************************************************/
Hypre_StructGrid  impl_Hypre_StructGrid_Constructor
(Hypre_MPI_Com com, int dimension) {
   /* declared static; just combines the New and Start functions */
   Hypre_StructGrid SG = Hypre_StructGrid_New();
   Hypre_StructGrid_Start( SG, com, dimension );
   return SG;
} /* end impl_Hypre_StructGridConstructor */

/* ********************************************************
 * impl_Hypre_StructGridSetup
 **********************************************************/
int  impl_Hypre_StructGrid_Setup(Hypre_StructGrid this) {
   struct Hypre_StructGrid_private_type *Gp = this->Hypre_StructGrid_data;
   HYPRE_StructGrid *G = Gp->hsgrid;
   hypre_StructGrid *g = (hypre_StructGrid *) *G;

   hypre_StructGridAssemble( g );
   return 0;
} /* end impl_Hypre_StructGridSetup */

