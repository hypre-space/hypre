
/******************************************************
 *
 *  File:  Hypre_PartitionBuilder.c
 *
 *********************************************************/

#include "Hypre_PartitionBuilder_Skel.h" 
#include "Hypre_PartitionBuilder_Data.h" 


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_PartitionBuilder_constructor(Hypre_PartitionBuilder this) {
   this->Hypre_PartitionBuilder_data =
      (struct Hypre_PartitionBuilder_private_type *)
      malloc( sizeof( struct Hypre_PartitionBuilder_private_type ) );
   this->Hypre_PartitionBuilder_data->partition = NULL;
   this->Hypre_PartitionBuilder_data->partition_good = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_PartitionBuilder_destructor(Hypre_PartitionBuilder this) {
   Hypre_Partition_deleteReference(
     this->Hypre_PartitionBuilder_data->partition );
   /* ... will delete partition if there are no other references to it */
   free(this->Hypre_PartitionBuilder_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_PartitionBuilder_Start
 * WARNING: the provided array becomes part of the newly built
 * Partition object.  It should not be deleted.  It might
 * get deleted if the Partition object is deleted.
 * (this is an example of a place where a systematically used memory
 * management system would be helpful)
 **********************************************************/
int  impl_Hypre_PartitionBuilder_Start
(Hypre_PartitionBuilder this, array1int partitioning)
{
   int ierr = 0;
   struct Hypre_Partition_private_type *Pp;
/* normally one has
   MPI_Comm_size( *MCp, &num_procs );
   partitioning.lower[0] = 0;
   partitioning.upper[0] = num_procs;
*/

   if ( this->Hypre_PartitionBuilder_data->partition != NULL )
      Hypre_Partition_deleteReference( this->Hypre_PartitionBuilder_data->partition );
   this->Hypre_PartitionBuilder_data->partition = Hypre_Partition_New();
   this->Hypre_PartitionBuilder_data->partition_good = 0;
   Hypre_Partition_addReference( this->Hypre_PartitionBuilder_data->partition );

   Pp = this->Hypre_PartitionBuilder_data->partition->Hypre_Partition_data;
   Pp->partition = partitioning.data;
   Pp->lower = (partitioning.lower)[0];
   Pp->upper = (partitioning.upper)[0];
   this->Hypre_PartitionBuilder_data->partition_good = 1;

} /* end impl_Hypre_PartitionBuilder_Start */

/* ********************************************************
 * impl_Hypre_PartitionBuilder_Setup
 **********************************************************/
int  impl_Hypre_PartitionBuilder_Setup(Hypre_PartitionBuilder this) {
} /* end impl_Hypre_PartitionBuilder_Setup */

/* ********************************************************
 * impl_Hypre_PartitionBuilder_GetConstructedObject
 **********************************************************/
int  impl_Hypre_PartitionBuilder_GetConstructedObject
( Hypre_PartitionBuilder this, Hypre_Map* obj ) {
   Hypre_Partition part = this->Hypre_PartitionBuilder_data->partition;
   if ( part==NULL || this->Hypre_PartitionBuilder_data->partition_good==0 ) {
      printf( "Hypre_PartitionBuilder: object not constructed yet\n" );
      *obj = (Hypre_Map) NULL;
      return 1;
   };
   *obj = (Hypre_Map) Hypre_Partition_castTo( part, "Hypre.Map" );
   return 0;
} /* end impl_Hypre_PartitionBuilder_GetConstructedObject */

/* ********************************************************
 * impl_Hypre_PartitionBuilder_SetPartitioning
 **********************************************************/
int  impl_Hypre_PartitionBuilder_SetPartitioning
( Hypre_PartitionBuilder this, array1int partitioning ) {

   struct Hypre_Partition_private_type *Pp =
      this->Hypre_PartitionBuilder_data->partition->Hypre_Partition_data;
   Pp->partition = partitioning.data;
   Pp->lower = (partitioning.lower)[0];
   Pp->upper = (partitioning.upper)[0];
   this->Hypre_PartitionBuilder_data->partition_good = 1;
   return 0;
} /* end impl_Hypre_PartitionBuilder_SetPartitioning */

/* ********************************************************
 * impl_Hypre_PartitionBuilder_GetPartitioning
 **********************************************************/
int  impl_Hypre_PartitionBuilder_GetPartitioning
( Hypre_PartitionBuilder this, array1int* partitioning ) {
   struct Hypre_Partition_private_type *Pp =
      this->Hypre_PartitionBuilder_data->partition->Hypre_Partition_data;

   (*partitioning).data = Pp->partition;
   ((*partitioning).lower)[0] = Pp->lower;
   ((*partitioning).upper)[0] = Pp->upper;
   return 0;
} /* end impl_Hypre_PartitionBuilder_GetPartitioning */

