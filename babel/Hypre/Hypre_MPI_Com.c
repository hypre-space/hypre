/*#*****************************************************
#
#	File:  Hypre_MPI_Com.c
#
#********************************************************/

#include "Hypre_MPI_Com_Skel.h" 

/* JfP:  There's not much to do here.  A Hypre_MPI_Com points to a MPI_Comm.
 A MPI_Comm is either an MPI handle, or else for sequential code it is a pointer
 to a dummy struct hypre_MPI_Comm (its only member is an int named "dummy").
 MPI_Comm is defined through file utilities/HYPRE_utilities.h: for parallel
 code because that file has a '#include "mpi.h"' and for sequential code
 MPI_Comm is defined in line 43 as a pointer to a hypre_MPI_Comm, which in turn
 is defined in utilities/mpistubs.h line 92 .
 */
      
/*#************************************************
#	Constructor
#**************************************************/

void Hypre_MPI_Com_constructor(Hypre_MPI_Com this) {

/* JFP: Allocates Memory.  This code is written generally, but really
   all we have to allocate is a short chain of pointers, and maybe an
   int MPI handle. */

#ifdef HYPRE_SEQUENTIAL
   typedef struct {int dummy;}  hypre_MPI_Comm;
   typedef struct hypre_MPI_Comm *MPI_Comm;
   MPI_Comm * MCp;
   hypre_MPI_Comm * MC;
#endif

   struct Hypre_MPI_Com_private * HMCp;
   HMCp = (struct Hypre_MPI_Com_private *)
      ( malloc(sizeof(struct Hypre_MPI_Com_private)) );
   this->d_table = (Hypre_MPI_Com_Private) HMCp;
   
   this->d_table->hcom = (MPI_Comm *)
      (malloc(sizeof(MPI_Comm)));

#ifdef HYPRE_SEQUENTIAL
   /* initialize dummy data to please Purify */
   MCp = this->d_table->hcom;
   *MCp = (MPI_Comm) malloc( sizeof(hypre_MPI_Comm) );
   MC = (hypre_MPI_Comm *) *MCp;
   MC ->dummy = 0;
   printf( "size of MPI_Comm=%i\n", sizeof(MPI_Comm) );
   
#endif

}


/*#************************************************
#	Destructor
#**************************************************/

void Hypre_MPI_Com_destructor(Hypre_MPI_Com this) {

   Hypre_MPI_Com_Private CP = this->d_table;
   struct Hypre_MPI_Com_private * HMCp = CP;
   MPI_Comm *C = CP->hcom;

   free(C);
   free(HMCp);

}


