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
/*just a test comment*/
#ifndef hypre_thread_MPISTUBS
#define hypre_thread_MPISTUBS

#ifdef HYPRE_USE_PTHREADS

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HYPRE_USING_THREAD_MPISTUBS

#define MPI_Init           hypre_thread_MPI_Init             
#define MPI_Wtime          hypre_thread_MPI_Wtime            
#define MPI_Wtick          hypre_thread_MPI_Wtick            
#define MPI_Barrier        hypre_thread_MPI_Barrier          
#define MPI_Finalize       hypre_thread_MPI_Finalize         
#define MPI_Comm_group     hypre_thread_MPI_Comm_group       
#define MPI_Comm_dup       hypre_thread_MPI_Comm_dup         
#define MPI_Group_incl     hypre_thread_MPI_Group_incl       
#define MPI_Comm_create    hypre_thread_MPI_Comm_create      
#define MPI_Allgather      hypre_thread_MPI_Allgather        
#define MPI_Allgatherv     hypre_thread_MPI_Allgatherv       
#define MPI_Bcast          hypre_thread_MPI_Bcast            
#define MPI_Send           hypre_thread_MPI_Send             
#define MPI_Recv           hypre_thread_MPI_Recv             

#define MPI_Isend          hypre_thread_MPI_Isend            
#define MPI_Irecv          hypre_thread_MPI_Irecv            
#define MPI_Wait           hypre_thread_MPI_Wait             
#define MPI_Waitall        hypre_thread_MPI_Waitall          
#define MPI_Waitany        hypre_thread_MPI_Waitany          
#define MPI_Comm_size      hypre_thread_MPI_Comm_size        
#define MPI_Comm_rank      hypre_thread_MPI_Comm_rank        
#define MPI_Allreduce      hypre_thread_MPI_Allreduce        
#define MPI_Type_hvector   hypre_thread_MPI_Type_hvector     
#define MPI_Type_struct    hypre_thread_MPI_Type_struct      
#define MPI_Type_free      hypre_thread_MPI_Type_free        
#define MPI_Type_commit    hypre_thread_MPI_Type_commit        

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* mpistubs.c */
int MPI_Init( int *argc , char ***argv );
double MPI_Wtime( void );
double MPI_Wtick( void );
int MPI_Barrier( MPI_Comm comm );
int MPI_Finalize( void );
int MPI_Abort( MPI_Comm comm , int errorcode );
int MPI_Comm_group( MPI_Comm comm , MPI_Group *group );
int MPI_Comm_dup( MPI_Comm comm , MPI_Comm *newcomm );
int MPI_Group_incl( MPI_Group group , int n , int *ranks , MPI_Group *newgroup );
int MPI_Comm_create( MPI_Comm comm , MPI_Group group , MPI_Comm *newcomm );
int MPI_Get_count( MPI_Status *status , MPI_Datatype datatype , int *count );
int MPI_Alltoall( void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm );
int MPI_Allgather( void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm );
int MPI_Allgatherv( void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int *recvcounts , int *displs , MPI_Datatype recvtype , MPI_Comm comm );
int MPI_Gather( void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm );
int MPI_Scatter( void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm );
int MPI_Bcast( void *buffer , int count , MPI_Datatype datatype , int root , MPI_Comm comm );
int MPI_Send( void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm );
int MPI_Recv( void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Status *status );
int MPI_Isend( void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request );
int MPI_Irecv( void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Request *request );
int MPI_Wait( MPI_Request *request , MPI_Status *status );
int MPI_Waitall( int count , MPI_Request *array_of_requests , MPI_Status *array_of_statuses );
int MPI_Waitany( int count , MPI_Request *array_of_requests , int *index , MPI_Status *status );
int MPI_Comm_size( MPI_Comm comm , int *size );
int MPI_Comm_rank( MPI_Comm comm , int *rank );
int MPI_Allreduce( void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm );
int MPI_Address( void *location , MPI_Aint *address );
int MPI_Type_contiguous( int count , MPI_Datatype oldtype , MPI_Datatype *newtype );
int MPI_Type_vector( int count , int blocklength , int stride , MPI_Datatype oldtype , MPI_Datatype *newtype );
int MPI_Type_hvector( int count , int blocklength , MPI_Aint stride , MPI_Datatype oldtype , MPI_Datatype *newtype );
int MPI_Type_struct( int count , int *array_of_blocklengths , MPI_Aint *array_of_displacements , MPI_Datatype *array_of_types , MPI_Datatype *newtype );
int MPI_Type_free( MPI_Datatype *datatype );
int MPI_Type_commit( MPI_Datatype *datatype );
int MPI_Request_free( MPI_Request *request );
int MPI_Send_init( void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request );
int MPI_Recv_init( void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request );
int MPI_Startall( int count , MPI_Request *array_of_requests );
int MPI_Iprobe( int source , int tag , MPI_Comm comm , int *flag , MPI_Status *status );
int MPI_Probe( int source , int tag , MPI_Comm comm , MPI_Status *status );
int MPI_Irsend( void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request );

#ifdef __cplusplus
}
#endif

#endif

#endif
