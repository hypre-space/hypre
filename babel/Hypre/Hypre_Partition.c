
/******************************************************
 *
 *  File:  Hypre_Partition.c
 *
 *********************************************************/

#include "Hypre_Partition_Skel.h" 
#include "Hypre_Partition_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_Partition_constructor(Hypre_Partition this) {
   this->Hypre_Partition_data = (struct Hypre_Partition_private_type *)
      malloc( sizeof( struct Hypre_Partition_private_type ) );
/* the builder will allocate space for int *partitioning */
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_Partition_destructor(Hypre_Partition this) {
   free(this->Hypre_Partition_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_Partition_GetPartitioning
 **********************************************************/
int  impl_Hypre_Partition_GetPartitioning
(Hypre_Partition this, array1int* partitioning) {

   int *my_partitioning = this->Hypre_Partition_data->partition;
   int lower = this->Hypre_Partition_data->lower;
   int upper = this->Hypre_Partition_data->upper;

   (*partitioning).lower[0] = lower;
   (*partitioning).upper[0] = upper;
   (*partitioning).data = my_partitioning;

   return 0;

} /* end impl_Hypre_Partition_GetPartitioning */

