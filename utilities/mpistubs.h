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

#ifndef hypre_MPISTUBS
#define hypre_MPISTUBS

#ifdef HYPRE_SEQUENTIAL

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Change all MPI names to hypre_MPI names to avoid link conflicts
 *
 * NOTE: MPI_Comm is the only MPI symbol in the HYPRE user interface,
 * and is defined in `HYPRE_utilities.h'.
 *--------------------------------------------------------------------------*/

#define MPI_Status          hypre_MPI_Status           
#define MPI_Request         hypre_MPI_Request          
#define MPI_Op              hypre_MPI_Op               
#define MPI_Datatype        hypre_MPI_Datatype         
#define MPI_Group           hypre_MPI_Group            
#define MPI_Aint            hypre_MPI_Aint             

#define MPI_COMM_WORLD      hypre_MPI_COMM_WORLD       

#define MPI_BOTTOM  	    hypre_MPI_BOTTOM

#define MPI_DOUBLE          hypre_MPI_DOUBLE           
#define MPI_INT             hypre_MPI_INT              
#define MPI_CHAR            hypre_MPI_CHAR             

#define MPI_SUM             hypre_MPI_SUM              
#define MPI_MIN             hypre_MPI_MIN              
#define MPI_MAX             hypre_MPI_MAX              
#define MPI_LOR             hypre_MPI_LOR              

#define MPI_UNDEFINED       hypre_MPI_UNDEFINED        

#define MPI_Init            hypre_MPI_Init             
#define MPI_Wtime           hypre_MPI_Wtime            
#define MPI_Wtick           hypre_MPI_Wtick            
#define MPI_Barrier         hypre_MPI_Barrier          
#define MPI_Finalize        hypre_MPI_Finalize         
#define MPI_Comm_group      hypre_MPI_Comm_group       
#define MPI_Comm_dup        hypre_MPI_Comm_dup         
#define MPI_Group_incl      hypre_MPI_Group_incl       
#define MPI_Comm_create     hypre_MPI_Comm_create      
#define MPI_Allgather       hypre_MPI_Allgather        
#define MPI_Allgatherv      hypre_MPI_Allgatherv       
#define MPI_Gather          hypre_MPI_Gather       
#define MPI_Scatter         hypre_MPI_Scatter       
#define MPI_Bcast           hypre_MPI_Bcast            
#define MPI_Send            hypre_MPI_Send             
#define MPI_Recv            hypre_MPI_Recv             
#define MPI_Isend           hypre_MPI_Isend            
#define MPI_Irecv           hypre_MPI_Irecv            
#define MPI_Wait            hypre_MPI_Wait             
#define MPI_Waitall         hypre_MPI_Waitall          
#define MPI_Waitany         hypre_MPI_Waitany          
#define MPI_Comm_size       hypre_MPI_Comm_size        
#define MPI_Comm_rank       hypre_MPI_Comm_rank        
#define MPI_Allreduce       hypre_MPI_Allreduce        
#define MPI_Address         hypre_MPI_Address        
#define MPI_Type_contiguous hypre_MPI_Type_contiguous     
#define MPI_Type_hvector    hypre_MPI_Type_hvector     
#define MPI_Type_struct     hypre_MPI_Type_struct      
#define MPI_Type_free       hypre_MPI_Type_free        
#define MPI_Type_commit     hypre_MPI_Type_commit        

/*--------------------------------------------------------------------------
 * Types, etc.
 *--------------------------------------------------------------------------*/

typedef struct {int dummy;}  hypre_MPI_Comm;

typedef int  hypre_MPI_Status;
typedef int  hypre_MPI_Request;
typedef int  hypre_MPI_Op;
typedef int  hypre_MPI_Datatype;
typedef int  hypre_MPI_Group;
typedef int  hypre_MPI_Aint;

#define  hypre_MPI_COMM_WORLD 0

#define  hypre_MPI_BOTTOM  0x0

#define  hypre_MPI_DOUBLE 0
#define  hypre_MPI_INT 1
#define  hypre_MPI_CHAR 2

#define  hypre_MPI_SUM 0
#define  hypre_MPI_MIN 1
#define  hypre_MPI_MAX 2
#define  hypre_MPI_LOR 3

#define  hypre_MPI_UNDEFINED -9999

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

# define        P(s) s

/* mpistubs.c */
int MPI_Init P((int *argc , char ***argv ));
double MPI_Wtime P((void ));
double MPI_Wtick P((void ));
int MPI_Barrier P((MPI_Comm comm ));
int MPI_Finalize P((void ));
int MPI_Abort P((MPI_Comm comm , int errorcode ));
int MPI_Comm_group P((MPI_Comm comm , MPI_Group *group ));
int MPI_Comm_dup P((MPI_Comm comm , MPI_Comm *newcomm ));
int MPI_Group_incl P((MPI_Group group , int n , int *ranks , MPI_Group *newgroup ));
int MPI_Comm_create P((MPI_Comm comm , MPI_Group group , MPI_Comm *newcomm ));
int MPI_Get_count P((MPI_Status status , MPI_Datatype datatype , int *count ));
int MPI_Alltoall P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm ));
int MPI_Allgather P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm ));
int MPI_Allgatherv P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int *recvcounts , int *displs , MPI_Datatype recvtype , MPI_Comm comm ));
int MPI_Gather P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm ));
int MPI_Scatter P((void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm ));
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
int MPI_Address P((void *location , MPI_Aint *address ));
int MPI_Type_contiguous P((int count , MPI_Datatype oldtype , MPI_Datatype *newtype ));
int MPI_Type_vector P((int count , int blocklength , int stride , MPI_Datatype oldtype , MPI_Datatype *newtype ));
int MPI_Type_hvector P((int count , int blocklength , MPI_Aint stride , MPI_Datatype oldtype , MPI_Datatype *newtype ));
int MPI_Type_struct P((int count , int *array_of_blocklengths , MPI_Aint *array_of_displacements , MPI_Datatype *array_of_types , MPI_Datatype *newtype ));
int MPI_Type_free P((MPI_Datatype *datatype ));
int MPI_Type_commit P((MPI_Datatype *datatype ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

#endif
