
/******************************************************
 *
 *  File:  Hypre_ParCSRVectorBuilder.c
 *
 *********************************************************/

#include "Hypre_ParCSRVectorBuilder_Skel.h" 
#include "Hypre_ParCSRVectorBuilder_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParCSRVectorBuilder_constructor(Hypre_ParCSRVectorBuilder this) {

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParCSRVectorBuilder_destructor(Hypre_ParCSRVectorBuilder this) {

} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderConstructor
 *       insert the library code below
 **********************************************************/
Hypre_ParCSRVectorBuilder  impl_Hypre_ParCSRVectorBuilder_Constructor(Hypre_MPI_Com com, int global_n) {
} /* end impl_Hypre_ParCSRVectorBuilderConstructor */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderNew
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_New(Hypre_ParCSRVectorBuilder this, Hypre_MPI_Com com, int global_n) {
} /* end impl_Hypre_ParCSRVectorBuilderNew */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetPartitioning
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_SetPartitioning(Hypre_ParCSRVectorBuilder this, array1int partitioning) {
} /* end impl_Hypre_ParCSRVectorBuilderSetPartitioning */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetLocalComponents
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_SetLocalComponents(Hypre_ParCSRVectorBuilder this, int num_values, array1int glob_vec_indices, array1int value_indices, array1double values) {
} /* end impl_Hypre_ParCSRVectorBuilderSetLocalComponents */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderAddtoLocalComponents
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_AddtoLocalComponents(Hypre_ParCSRVectorBuilder this, int num_values, array1int glob_vec_indices, array1int value_indices, array1double values) {
} /* end impl_Hypre_ParCSRVectorBuilderAddtoLocalComponents */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetLocalComponentsInBlock
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_SetLocalComponentsInBlock(Hypre_ParCSRVectorBuilder this, int glob_vec_index_start, int glob_vec_index_stop, array1int value_indices, array1double values) {
} /* end impl_Hypre_ParCSRVectorBuilderSetLocalComponentsInBlock */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderAddToLocalComponentsInBlock
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_AddToLocalComponentsInBlock(Hypre_ParCSRVectorBuilder this, int glob_vec_index_start, int glob_vec_index_stop, array1int value_indices, array1double values) {
} /* end impl_Hypre_ParCSRVectorBuilderAddToLocalComponentsInBlock */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetup
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_Setup(Hypre_ParCSRVectorBuilder this) {
} /* end impl_Hypre_ParCSRVectorBuilderSetup */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderGetConstructedObject
 *       insert the library code below
 **********************************************************/
Hypre_Vector  impl_Hypre_ParCSRVectorBuilder_GetConstructedObject(Hypre_ParCSRVectorBuilder this) {
} /* end impl_Hypre_ParCSRVectorBuilderGetConstructedObject */

