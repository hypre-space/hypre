
/******************************************************
 *
 *  File:  Hypre_StructMatrixBldr.c
 *
 *********************************************************/

#include "Hypre_StructMatrixBldr_Skel.h" 
#include "Hypre_StructMatrixBldr_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_StructMatrixBldr_constructor(Hypre_StructMatrixBldr this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_StructMatrixBldr_destructor(Hypre_StructMatrixBldr this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrprint
 *       insert the library code below
 **********************************************************/
void  impl_Hypre_StructMatrixBldr_print(Hypre_StructMatrixBldr this) {
} /* end impl_Hypre_StructMatrixBldrprint */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrSetStencil
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_StructMatrixBldr_SetStencil(Hypre_StructMatrixBldr this, Hypre_StructStencil stencil) {
} /* end impl_Hypre_StructMatrixBldrSetStencil */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrSetValues
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_StructMatrixBldr_SetValues(Hypre_StructMatrixBldr this, Hypre_Box box, array1int stencil_indices, array1double values) {
} /* end impl_Hypre_StructMatrixBldrSetValues */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrApply
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_StructMatrixBldr_Apply(Hypre_StructMatrixBldr this, Hypre_StructVector b, Hypre_StructVector* x) {
} /* end impl_Hypre_StructMatrixBldrApply */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_LinearOperator  impl_Hypre_StructMatrixBldr_GetConstructedObject(Hypre_StructMatrixBldr this) {
} /* end impl_Hypre_StructMatrixBldrGetConstructedObject */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrNew
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_StructMatrixBldr_New(Hypre_StructMatrixBldr this, Hypre_StructuredGrid grid, Hypre_StructStencil stencil, int symmetric, array1int num_ghost) {
} /* end impl_Hypre_StructMatrixBldrNew */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrConstructor
 *       insert the library code below
 **********************************************************/
Hypre_StructMatrixBldr  impl_Hypre_StructMatrixBldr_Constructor(Hypre_StructuredGrid grid, Hypre_StructStencil stencil, int symmetric, array1int num_ghost) {
} /* end impl_Hypre_StructMatrixBldrConstructor */

/* ********************************************************
 * impl_Hypre_StructMatrixBldrSetup
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_StructMatrixBldr_Setup(Hypre_StructMatrixBldr this) {
} /* end impl_Hypre_StructMatrixBldrSetup */

