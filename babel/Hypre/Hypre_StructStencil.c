/*#*****************************************************
#
#	File:  Hypre_StructStencil.c
#
#********************************************************/

#include "Hypre_StructStencil_Skel.h" 


/*#************************************************
#	Constructor
#**************************************************/

void Hypre_StructStencil_constructor(Hypre_StructStencil this) {

/* JFP: Allocates Memory */

   struct Hypre_StructStencil_private * HSSp;
   HSSp = (struct Hypre_StructStencil_private *)
      ( malloc(sizeof(struct Hypre_StructStencil_private)) );
   this->d_table = (Hypre_StructStencil_Private) HSSp;
   
   this->d_table->hsstencil = (HYPRE_StructStencil *)
      (malloc(sizeof(HYPRE_StructStencil)));

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_StructStencil_destructor(Hypre_StructStencil this) {

   /* JFP: Deallocates memory.
      Delete the Hypre object this object refers to, then delete
      this object's data table. */

   Hypre_StructStencil_Private SSP = this->d_table;
   struct Hypre_StructStencil_private *SSp = SSP;
   HYPRE_StructStencil *SS = SSp->hsstencil;

   HYPRE_StructStencilDestroy( *SS );

   free(this->d_table);
}

Hypre_StructStencil  impl__Hypre_StructStencil_NewStencil(Hypre_StructStencil this, int dimension, int size ) {

   struct Hypre_StructStencil_object__ SSO = *this;
   Hypre_StructStencil_Private SSP = SSO.d_table;
   struct Hypre_StructStencil_private *SSp = SSP;
   HYPRE_StructStencil *SS = SSp->hsstencil;
   hypre_StructStencil *ss2;

   HYPRE_StructStencilCreate( dimension, size, SS );

   return this;
}

void  impl__Hypre_StructStencil_print(Hypre_StructStencil this) {
/* JFP: This function prints, to stdout, data about the stencil. */

   int i, j;

   struct Hypre_StructStencil_object__ SSO = *this;
   Hypre_StructStencil_Private SSP = SSO.d_table;
   struct Hypre_StructStencil_private *SSp = SSP;
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
}

int  impl__Hypre_StructStencil_SetElement(
   Hypre_StructStencil this, int element_index, array1int* element_offset) {
/* JFP: This function sets a stencil element (element_offset is an array of
   three numbers */

   int ierr;

   struct Hypre_StructStencil_object__ SSO = *this;
   Hypre_StructStencil_Private SSP = SSO.d_table;
   struct Hypre_StructStencil_private *SSp = SSP;
   HYPRE_StructStencil *SS = SSp->hsstencil;
   hypre_StructStencil *ss = (hypre_StructStencil *) *SS;
   
   ierr = HYPRE_StructStencilSetElement(
      *SS, element_index, element_offset->data );
   
   return ierr;
}


