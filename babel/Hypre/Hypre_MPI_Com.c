
/******************************************************
 *
 *  File:  Hypre_MPI_Com.c
 *
 *********************************************************/

#include "Hypre_MPI_Com_Skel.h" 
#include "Hypre_MPI_Com_Data.h" 

/* JfP:  There's not much to do here.  A Hypre_MPI_Com points to a MPI_Comm.
 A MPI_Comm is either an MPI handle, or else for sequential code it is a pointer
 to a dummy struct hypre_MPI_Comm (its only member is an int named "dummy").
 MPI_Comm is defined through file utilities/HYPRE_utilities.h: for parallel
 code because that file has a '#include "mpi.h"' and for sequential code
 MPI_Comm is defined in line 43 as a pointer to a hypre_MPI_Comm, which in turn
 is defined in utilities/mpistubs.h line 92 .
 */

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_MPI_Com_constructor(Hypre_MPI_Com this) {
/* Allocates Memory.  This code is written generally, but really
   all we have to allocate is a short chain of pointers, and maybe an
   int MPI handle. */

   this->d_table = (struct Hypre_MPI_Com_private_type *)
     malloc(sizeof(struct Hypre_MPI_Com_private_type)) ;

   this->d_table->hcom = (MPI_Comm *)
      (malloc(sizeof(MPI_Comm)));
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_MPI_Com_destructor(Hypre_MPI_Com this) {

   struct Hypre_MPI_Com_private_type * HMCp = this->d_table;
   MPI_Comm *C = HMCp->hcom; /* gkk: HMCp was CP??? */

   free(C);
   free(HMCp);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_MPI_ComNew
 **********************************************************/
void  impl_Hypre_MPI_Com_New(Hypre_MPI_Com this, int comm) {
/* For multiprocessing code, the int comm should really be an
   MPI handle such as MPI_COMM_WORLD.  For sequential code it
   should really be the appropriate pointer to a dummy struct
   (or, maybe, comm=0) */
   MPI_Comm * MCp = this->d_table->hcom;
   *MCp = (MPI_Comm) comm;
} /* end impl_Hypre_MPI_ComNew */

/* ********************************************************
 * impl_Hypre_MPI_ComConstructor
 **********************************************************/
Hypre_MPI_Com  impl_Hypre_MPI_Com_Constructor(int comm) {
   /* declared static; just combines the new and New functions */
   Hypre_MPI_Com C = Hypre_MPI_Com_new();
   Hypre_MPI_Com_New( C, comm );
   return C;
} /* end impl_Hypre_MPI_ComConstructor */

