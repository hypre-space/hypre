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
 *  Fake mpi stubs to generate serial codes without mpi
 *
 *****************************************************************************/

#ifndef hypre_thread_MPISTUBS
#define hypre_thread_MPISTUBS

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MPI_DEFINED

#define MPI_Init           thread_MPI_Init             
#define MPI_Wtime          thread_MPI_Wtime            
#define MPI_Wtick          thread_MPI_Wtick            
#define MPI_Barrier        thread_MPI_Barrier          
#define MPI_Finalize       thread_MPI_Finalize         
#define MPI_Comm_group     thread_MPI_Comm_group       
#define MPI_Comm_dup       thread_MPI_Comm_dup         
#define MPI_Group_incl     thread_MPI_Group_incl       
#define MPI_Comm_create    thread_MPI_Comm_create      
#define MPI_Allgather      thread_MPI_Allgather        
#define MPI_Allgatherv     thread_MPI_Allgatherv       
#define MPI_Bcast          thread_MPI_Bcast            
#define MPI_Send           thread_MPI_Send             
#define MPI_Recv           thread_MPI_Recv             

#define MPI_Isend          thread_MPI_Isend            
#define MPI_Irecv          thread_MPI_Irecv            
#define MPI_Wait           thread_MPI_Wait             
#define MPI_Waitall        thread_MPI_Waitall          
#define MPI_Waitany        thread_MPI_Waitany          
#define MPI_Comm_size      thread_MPI_Comm_size        
#define MPI_Comm_rank      thread_MPI_Comm_rank        
#define MPI_Allreduce      thread_MPI_Allreduce        
#define MPI_Type_hvector   thread_MPI_Type_hvector     
#define MPI_Type_struct    thread_MPI_Type_struct      
#define MPI_Type_free      thread_MPI_Type_free        
#define MPI_Type_commit    thread_MPI_Type_commit        

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* mpistubs.c */
int MPI_Init P((int *argc , char ***argv ));
double MPI_Wtime P((void ));
double MPI_Wtick P((void ));
int MPI_Barrier P((MPI_Comm comm ));
int MPI_Finalize P((void ));
int MPI_Comm_group P((MPI_Comm comm , MPI_Group *group ));
int MPI_Comm_dup P((MPI_Comm comm , MPI_Comm *newcomm ));
int MPI_Group_incl P((MPI_Group group , int n , int *ranks , MPI_Group *newgroup ));
int MPI_Comm_create P((MPI_Comm comm , MPI_Group group , MPI_Comm *newcomm ));
int MPI_Allgather P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm ));
int MPI_Allgatherv P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int *recvcounts , int *displs , MPI_Datatype recvtype , MPI_Comm comm ));
int MPI_Bcast P((void *buffer , int count , MPI_Datatype datatype , int root , MPI_Comm comm ));
int MPI_Send P((void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm ));
int MPI_Recv P((void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Status *status ));
int MPI_Isend P((void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request ));
int MPI_Irecv P((void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Request *request ));
int MPI_Wait P((MPI_Request *request , MPI_Status *status ));
int MPI_Waitall P((int count , MPI_Request *array_of_requests , MPI_Status *array_of_statuses ));
int MPI_Waitany P((int count , MPI_Request *array_of_requests , int *index , MPI_Status *status ));
int MPI_Comm_size P((MPI_Comm comm , int *size ));
int MPI_Comm_rank P((MPI_Comm comm , int *rank ));
int MPI_Allreduce P((void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm ));
int MPI_Type_hvector P((int count , int blocklength , MPI_Aint stride , MPI_Datatype oldtype , MPI_Datatype *newtype ));
int MPI_Type_struct P((int count , int *array_of_blocklengths , MPI_Aint *array_of_displacements , MPI_Datatype *array_of_types , MPI_Datatype *newtype ));
int MPI_Type_free P((MPI_Datatype *datatype ));
int MPI_Type_commit P((MPI_Datatype *datatype ));

#ifdef __cplusplus
}
#endif

#endif


