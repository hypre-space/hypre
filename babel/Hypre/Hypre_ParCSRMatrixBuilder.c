
/******************************************************
 *
 *  File:  Hypre_ParCSRMatrixBuilder.c
 *
 *********************************************************/

#include "Hypre_ParCSRMatrixBuilder_Skel.h" 
#include "Hypre_ParCSRMatrixBuilder_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParCSRMatrixBuilder_constructor(Hypre_ParCSRMatrixBuilder this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParCSRMatrixBuilder_destructor(Hypre_ParCSRMatrixBuilder this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderConstructor
 *       insert the library code below
 **********************************************************/
Hypre_ParCSRMatrixBuilder  impl_Hypre_ParCSRMatrixBuilder_Constructor(Hypre_MPI_Com com, int global_m, int global_n) {
} /* end impl_Hypre_ParCSRMatrixBuilderConstructor */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderNew
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_New(Hypre_ParCSRMatrixBuilder this, Hypre_MPI_Com com, int global_m, int global_n) {
} /* end impl_Hypre_ParCSRMatrixBuilderNew */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetLocalSize
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetLocalSize(Hypre_ParCSRMatrixBuilder this, int local_m, int local_n) {
} /* end impl_Hypre_ParCSRMatrixBuilderSetLocalSize */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetRowSizes
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetRowSizes(Hypre_ParCSRMatrixBuilder this, array1int sizes) {
} /* end impl_Hypre_ParCSRMatrixBuilderSetRowSizes */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetDiagRowSizes
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetDiagRowSizes(Hypre_ParCSRMatrixBuilder this, array1int sizes) {
} /* end impl_Hypre_ParCSRMatrixBuilderSetDiagRowSizes */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetOffDiagRowSizes
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetOffDiagRowSizes(Hypre_ParCSRMatrixBuilder this, array1int sizes) {
} /* end impl_Hypre_ParCSRMatrixBuilderSetOffDiagRowSizes */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderInsertRow
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_InsertRow(Hypre_ParCSRMatrixBuilder this, int n, int row, array1int cols, array1double values) {
} /* end impl_Hypre_ParCSRMatrixBuilderInsertRow */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderAddToRow
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_AddToRow(Hypre_ParCSRMatrixBuilder this, int n, int row, array1int cols, array1double values) {
} /* end impl_Hypre_ParCSRMatrixBuilderAddToRow */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderInsertBlock
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_InsertBlock(Hypre_ParCSRMatrixBuilder this, int m, int n, array1int rows, array1int cols, array1double values) {
} /* end impl_Hypre_ParCSRMatrixBuilderInsertBlock */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderAddtoBlock
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_AddtoBlock(Hypre_ParCSRMatrixBuilder this, int m, int n, array1int rows, array1int cols, array1double values) {
} /* end impl_Hypre_ParCSRMatrixBuilderAddtoBlock */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderGetRowPartitioning
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_GetRowPartitioning(Hypre_ParCSRMatrixBuilder this, array1int* partitioning) {
} /* end impl_Hypre_ParCSRMatrixBuilderGetRowPartitioning */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetup
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_Setup(Hypre_ParCSRMatrixBuilder this) {
} /* end impl_Hypre_ParCSRMatrixBuilderSetup */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_LinearOperator  impl_Hypre_ParCSRMatrixBuilder_GetConstructedObject(Hypre_ParCSRMatrixBuilder this) {
} /* end impl_Hypre_ParCSRMatrixBuilderGetConstructedObject */

