/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_ParVector
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm	 comm;

   int      	 global_size;
   int      	 first_index;
   hypre_Vector	*local_vector; 

   hypre_VectorCommPkg *vector_comm_pkg;

   /* Does the Vector create/destroy `data'? */
   int      	 owns_data;

} hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParVectorComm(vector)  	   ((vector) -> comm)
#define hypre_ParVectorGlobalSize(vector)  ((vector) -> global_size)
#define hypre_ParVectorFirstIndex(vector)  ((vector) -> first_index)
#define hypre_ParVectorLocalVector(vector) ((vector) -> local_vector)
#define hypre_ParVectorCommPkg(vector)     ((vector) -> vector_comm_pkg)
#define hypre_ParVectorOwnsData(vector)    ((vector) -> owns_data)

#endif
