
/******************************************************
 *
 *  File:  Hypre_StructStencil.c
 *
 *********************************************************/

#include "Hypre_StructStencil_Skel.h" 
#include "Hypre_StructStencil_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructStencil_constructor(Hypre_StructStencil this) {

   this->Hypre_StructStencil_data = (struct Hypre_StructStencil_private_type *)
     malloc(sizeof(struct Hypre_StructStencil_private_type));
   
   this->Hypre_StructStencil_data->hsstencil = (HYPRE_StructStencil *)
     (malloc(sizeof(HYPRE_StructStencil)));

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructStencil_destructor(Hypre_StructStencil this) {
   struct Hypre_StructStencil_private_type *SSp = this->Hypre_StructStencil_data;
   HYPRE_StructStencil *SS = SSp->hsstencil;

   HYPRE_StructStencilDestroy( *SS );

   free(this->Hypre_StructStencil_data);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructStencilprint
 **********************************************************/
void  impl_Hypre_StructStencil_print(Hypre_StructStencil this) {
/* This function prints, to stdout, data about the stencil. */

   int i, j;

   struct Hypre_StructStencil_private_type *SSp = this->Hypre_StructStencil_data;
   HYPRE_StructStencil *SS = SSp->hsstencil;
   hypre_StructStencil *ss = (hypre_StructStencil *) *SS;

   printf( "Stencil dim=%i, size=%i; elements:\n  ", ss->dim, ss->size );
   for ( i=0; i<ss->size; ++i )
   {
      for ( j=0; j<ss->dim; ++j )
         printf( "%i,", hypre_StructStencilShape(ss)[i][j] );
      printf( "  " );
   }
   ;
   printf( "\n" );

} /* end impl_Hypre_StructStencilprint */

/* ********************************************************
 * impl_Hypre_StructStencilSetElement
 **********************************************************/
int  impl_Hypre_StructStencil_SetElement
(Hypre_StructStencil this, int element_index, array1int element_offset) {
/* This function sets a stencil element (element_offset is an array of
   three numbers */

   int ierr;

   struct Hypre_StructStencil_private_type *SSp = this->Hypre_StructStencil_data;
   HYPRE_StructStencil *SS = SSp->hsstencil;
   hypre_StructStencil *ss = (hypre_StructStencil *) *SS;
   
   ierr = HYPRE_StructStencilSetElement(
      *SS, element_index, element_offset.data );
   
   return ierr;

} /* end impl_Hypre_StructStencilSetElement */

/* ********************************************************
 * impl_Hypre_StructStencil_Start
 **********************************************************/
int  impl_Hypre_StructStencil_Start
(Hypre_StructStencil this, int dimension, int size) {

   struct Hypre_StructStencil_private_type *SSp = this->Hypre_StructStencil_data;
   HYPRE_StructStencil *SS = SSp->hsstencil;

   return HYPRE_StructStencilCreate( dimension, size, SS );

} /* end impl_Hypre_StructStencil_Start */

/* ********************************************************
 * impl_Hypre_StructStencilConstructor
 **********************************************************/
Hypre_StructStencil  impl_Hypre_StructStencil_Constructor(int dimension, int size) {
   /* declared static; just combines the New and Start functions */
   Hypre_StructStencil SS = Hypre_StructStencil_New();
   Hypre_StructStencil_Start( SS, dimension, size );
   return SS;

} /* end impl_Hypre_StructStencilConstructor */

/* ********************************************************
 * impl_Hypre_StructStencilSetup
 **********************************************************/
int  impl_Hypre_StructStencil_Setup(Hypre_StructStencil this) {
   /* nothing needs doing */
   return 0;
} /* end impl_Hypre_StructStencilSetup */

