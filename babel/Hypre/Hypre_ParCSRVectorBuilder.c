
/******************************************************
 *
 *  File:  Hypre_ParCSRVectorBuilder.c
 *
 *********************************************************/

#include "Hypre_ParCSRVectorBuilder_Skel.h" 
#include "Hypre_ParCSRVectorBuilder_Data.h" 

#include "HYPRE.h"
#include "Hypre_ParCSRVector_Skel.h" 
#include "Hypre_ParCSRVector_Data.h" 
#include "Hypre_MPI_Com_Skel.h" 
#include "Hypre_MPI_Com_Data.h" 
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParCSRVectorBuilder_constructor(Hypre_ParCSRVectorBuilder this) {
   this->d_table = (struct Hypre_ParCSRVectorBuilder_private_type *)
      malloc( sizeof( struct Hypre_ParCSRVectorBuilder_private_type ) );
   this->d_table->newvec = NULL;
   this->d_table->vecgood = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParCSRVectorBuilder_destructor(Hypre_ParCSRVectorBuilder this) {
   if ( this->d_table->newvec != NULL ) {
      Hypre_ParCSRVector_deleteReference( this->d_table->newvec );
      /* ... will delete newvec if there are no other references to it */
   };
   free(this->d_table);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderConstructor
 **********************************************************/
Hypre_ParCSRVectorBuilder  impl_Hypre_ParCSRVectorBuilder_Constructor
(Hypre_MPI_Com com, int global_n) {
   return Hypre_ParCSRVectorBuilder_new();
} /* end impl_Hypre_ParCSRVectorBuilderConstructor */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderNew
 **********************************************************/
int impl_Hypre_ParCSRVectorBuilder_New
( Hypre_ParCSRVectorBuilder this, Hypre_MPI_Com com, int global_n) {

   int ierr = 0;

   struct Hypre_MPI_Com_private_type * HMCp = com->d_table;
   MPI_Comm * comm = HMCp->hcom;

   struct Hypre_ParCSRVector_private_type * Vp;
   HYPRE_IJVector * V;
   if ( this->d_table->newvec != NULL )
      Hypre_ParCSRVector_deleteReference( this->d_table->newvec );
   this->d_table->newvec = Hypre_ParCSRVector_new();
   this->d_table->vecgood = 0;
   Hypre_ParCSRVector_addReference( this->d_table->newvec );

   Vp = this->d_table->newvec->d_table;
   V = Vp->Hvec;
   Vp->comm = com;

   ierr += HYPRE_IJVectorCreate( *comm, V, global_n );
   ierr += HYPRE_IJVectorSetLocalStorageType( *V, HYPRE_PARCSR );

   return ierr;

} /* end impl_Hypre_ParCSRVectorBuilderNew */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetPartitioning
 * This just calls HYPRE_IJVectorSetPartitioning.
 *
 * I (JfP) am not confident that it will be correct to call this (and do
 * nothing else) in all circumstances.  If the partitioning has already been
 * set and storage allocated, this just resets the partitioning vector.
 * But that could invalidate the already-set local storage and indices.
 * For some hints about this issue, look at
 * IJ_matrix_vector/hypre_IJVector_parcsr.c:hypre_IJVectorSetPartitioningPar
 * and parcsr_matrix_vector/par_vector.c:hypre_ParVectorCreate .
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_SetPartitioning
( Hypre_ParCSRVectorBuilder this, array1int partitioning ) {
   Hypre_ParCSRVector vec = this->d_table->newvec;
   HYPRE_IJVector * Hvec = vec->d_table->Hvec;
   int * partition_data = &(partitioning.data[*(partitioning.lower)]);

   if (this->d_table->vecgood==1) {
      /* ... error to set partitioning on a fully built vector */
      /* This check would better be done by a design-by-contract style "Require"
         There are many such cases in this code. */
      return 1;
   }
   else {
      return HYPRE_IJVectorSetPartitioning( *Hvec, partition_data );
   }
} /* end impl_Hypre_ParCSRVectorBuilderSetPartitioning */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderGetPartitioning
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_GetPartitioning
(Hypre_ParCSRVectorBuilder this, array1int* partitioning) {
   Hypre_ParCSRVector vec = this->d_table->newvec;
   return Hypre_ParCSRVector_GetPartitioning( vec, partitioning );
} /* end impl_Hypre_ParCSRVectorBuilderGetPartitioning */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetLocalComponents
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_SetLocalComponents
( Hypre_ParCSRVectorBuilder this, int num_values,
  array1int glob_vec_indices, array1int value_indices, array1double values) {

   Hypre_ParCSRVector vec = this->d_table->newvec;
   HYPRE_IJVector * Hvec = vec->d_table->Hvec;
   int * glob_vec_indices_data = &(glob_vec_indices.data[*(glob_vec_indices.lower)]);
   int * value_indices_data = &(value_indices.data[*(value_indices.lower)]);
   double * values_data = &(values.data[*(values.lower)]);

   return HYPRE_IJVectorSetLocalComponents(
      *Hvec, num_values, glob_vec_indices_data,
      value_indices_data, values_data );
} /* end impl_Hypre_ParCSRVectorBuilderSetLocalComponents */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderAddtoLocalComponents
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_AddtoLocalComponents
( Hypre_ParCSRVectorBuilder this, int num_values,
  array1int glob_vec_indices, array1int value_indices, array1double values) {

   Hypre_ParCSRVector vec = this->d_table->newvec;
   HYPRE_IJVector * Hvec = vec->d_table->Hvec;
   int * glob_vec_indices_data = &(glob_vec_indices.data[*(glob_vec_indices.lower)]);
   int * value_indices_data = &(value_indices.data[*(value_indices.lower)]);
   double * values_data = &(values.data[*(values.lower)]);

   return HYPRE_IJVectorAddToLocalComponents(
      *Hvec, num_values, glob_vec_indices_data,
      value_indices_data, values_data );
} /* end impl_Hypre_ParCSRVectorBuilderAddtoLocalComponents */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetLocalComponentsInBlock
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_SetLocalComponentsInBlock
( Hypre_ParCSRVectorBuilder this,
  int glob_vec_index_start, int glob_vec_index_stop,
  array1int value_indices, array1double values) {

   Hypre_ParCSRVector vec = this->d_table->newvec;
   HYPRE_IJVector * Hvec = vec->d_table->Hvec;
   int * value_indices_data = &(value_indices.data[*(value_indices.lower)]);
   double * values_data = &(values.data[*(values.lower)]);

   return HYPRE_IJVectorSetLocalComponentsInBlock(
      *Hvec, glob_vec_index_start, glob_vec_index_stop,
      value_indices_data, values_data );

} /* end impl_Hypre_ParCSRVectorBuilderSetLocalComponentsInBlock */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderAddToLocalComponentsInBlock
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_AddToLocalComponentsInBlock
(Hypre_ParCSRVectorBuilder this,
 int glob_vec_index_start, int glob_vec_index_stop,
 array1int value_indices, array1double values) {

   Hypre_ParCSRVector vec = this->d_table->newvec;
   HYPRE_IJVector * Hvec = vec->d_table->Hvec;
   int * value_indices_data = &(value_indices.data[*(value_indices.lower)]);
   double * values_data = &(values.data[*(values.lower)]);

   return HYPRE_IJVectorAddToLocalComponentsInBlock(
      *Hvec, glob_vec_index_start, glob_vec_index_stop,
      value_indices_data, values_data );

} /* end impl_Hypre_ParCSRVectorBuilderAddToLocalComponentsInBlock */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderSetup
 **********************************************************/
int  impl_Hypre_ParCSRVectorBuilder_Setup(Hypre_ParCSRVectorBuilder this) {
   int ierr = 0;
   struct Hypre_ParCSRVector_private_type * Vp = this->d_table->newvec->d_table;
   HYPRE_IJVector * V = Vp->Hvec;

   ierr += HYPRE_IJVectorInitialize( *V );
   ierr += HYPRE_IJVectorAssemble( *V );
   /* ... the one sample code using HYPRE_IJVector's which I looked at did not
      call Assemble.  But it's needed when more than one processor is running. */
   if ( ierr==0 ) this->d_table->vecgood = 1;
   return ierr;
} /* end impl_Hypre_ParCSRVectorBuilderSetup */

/* ********************************************************
 * impl_Hypre_ParCSRVectorBuilderGetConstructedObject
 **********************************************************/
Hypre_Vector  impl_Hypre_ParCSRVectorBuilder_GetConstructedObject
(Hypre_ParCSRVectorBuilder this) {
   Hypre_ParCSRVector newvec = this->d_table->newvec;
   if ( newvec==NULL || this->d_table->vecgood==0 ) {
      printf( "Hypre_ParCSRVectorBuilder: object not constructed yet\n");
      return (Hypre_Vector) NULL;
   };
   return (Hypre_Vector) Hypre_ParCSRVector_castTo( newvec, "Hypre_Vector" );

} /* end impl_Hypre_ParCSRVectorBuilderGetConstructedObject */

/* ********************************************************
 * ********************************************************
 *
 * The following functions are not declared in the SIDL file.
 *
 * ********************************************************
 * ********************************************************
 */


/* ********************************************************
 * Hypre_ParCSRVectorBuilder_New_fromHYPRE
 *
 * Input: V, a pointer to an already-constructed HYPRE_IJVector.
 * At a minimum, V represents a call of HYPRE_IJVectorCreate.
 * This function builds a Hypre_ParCSRVector which points to it.
 * The new Hypre_ParCSRVector is available by calling
 * GetConstructedObject.
 * There is no need to call Setup or Set functions if that has
 * already been done directly to the HYPRE_IJMatrix.
 **********************************************************/
int Hypre_ParCSRVectorBuilder_New_fromHYPRE
( Hypre_ParCSRVectorBuilder this, Hypre_MPI_Com com, HYPRE_IJVector * V ) {

   int ierr = 0;

   struct Hypre_ParCSRVector_private_type * Vp;
   if ( this->d_table->newvec != NULL )
      Hypre_ParCSRVector_deleteReference( this->d_table->newvec );
   this->d_table->newvec = Hypre_ParCSRVector_new();
   this->d_table->vecgood = 0;
   Hypre_ParCSRVector_addReference( this->d_table->newvec );

   Vp = this->d_table->newvec->d_table;
   Vp->Hvec = V;
   Vp->comm = com;
   this->d_table->vecgood = 1;

   return ierr;

} /* end impl_Hypre_ParCSRVectorBuilderNew */
