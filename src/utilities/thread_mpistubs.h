/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/


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
HYPRE_Int hypre_thread_MPI_Init( HYPRE_Int *argc , char ***argv );
double hypre_thread_MPI_Wtime( void );
double hypre_thread_MPI_Wtick( void );
HYPRE_Int hypre_thread_MPI_Barrier( MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Finalize( void );
HYPRE_Int hypre_thread_MPI_Abort( MPI_Comm comm , HYPRE_Int errorcode );
HYPRE_Int hypre_thread_MPI_Comm_group( MPI_Comm comm , MPI_Group *group );
HYPRE_Int hypre_thread_MPI_Comm_dup( MPI_Comm comm , MPI_Comm *newcomm );
HYPRE_Int hypre_thread_MPI_Group_incl( MPI_Group group , HYPRE_Int n , HYPRE_Int *ranks , MPI_Group *newgroup );
HYPRE_Int hypre_thread_MPI_Comm_create( MPI_Comm comm , MPI_Group group , MPI_Comm *newcomm );
HYPRE_Int hypre_thread_MPI_Get_count( MPI_Status *status , MPI_Datatype datatype , HYPRE_Int *count );
HYPRE_Int hypre_thread_MPI_Alltoall( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Allgather( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Allgatherv( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int *recvcounts , HYPRE_Int *displs , MPI_Datatype recvtype , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Gather( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , HYPRE_Int root , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Scatter( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , HYPRE_Int root , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Bcast( void *buffer , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int root , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Send( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Recv( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , MPI_Status *status );
HYPRE_Int hypre_thread_MPI_Isend( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int hypre_thread_MPI_Irecv( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int hypre_thread_MPI_Wait( MPI_Request *request , MPI_Status *status );
HYPRE_Int hypre_thread_MPI_Waitall( HYPRE_Int count , MPI_Request *array_of_requests , MPI_Status *array_of_statuses );
HYPRE_Int hypre_thread_MPI_Waitany( HYPRE_Int count , MPI_Request *array_of_requests , HYPRE_Int *index , MPI_Status *status );
HYPRE_Int hypre_thread_MPI_Comm_size( MPI_Comm comm , HYPRE_Int *size );
HYPRE_Int hypre_thread_MPI_Comm_rank( MPI_Comm comm , HYPRE_Int *rank );
HYPRE_Int hypre_thread_MPI_Allreduce( void *sendbuf , void *recvbuf , HYPRE_Int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm );
HYPRE_Int hypre_thread_MPI_Address( void *location , MPI_Aint *address );
HYPRE_Int hypre_thread_MPI_Type_contiguous( HYPRE_Int count , MPI_Datatype oldtype , MPI_Datatype *newtype );
HYPRE_Int hypre_thread_MPI_Type_vector( HYPRE_Int count , HYPRE_Int blocklength , HYPRE_Int stride , MPI_Datatype oldtype , MPI_Datatype *newtype );
HYPRE_Int hypre_thread_MPI_Type_hvector( HYPRE_Int count , HYPRE_Int blocklength , MPI_Aint stride , MPI_Datatype oldtype , MPI_Datatype *newtype );
HYPRE_Int hypre_thread_MPI_Type_struct( HYPRE_Int count , HYPRE_Int *array_of_blocklengths , MPI_Aint *array_of_displacements , MPI_Datatype *array_of_types , MPI_Datatype *newtype );
HYPRE_Int hypre_thread_MPI_Type_free( MPI_Datatype *datatype );
HYPRE_Int hypre_thread_MPI_Type_commit( MPI_Datatype *datatype );
HYPRE_Int hypre_thread_MPI_Request_free( MPI_Request *request );
HYPRE_Int hypre_thread_MPI_Send_init( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int hypre_thread_MPI_Recv_init( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int hypre_thread_MPI_Startall( HYPRE_Int count , MPI_Request *array_of_requests );
HYPRE_Int hypre_thread_MPI_Iprobe( HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , HYPRE_Int *flag , MPI_Status *status );
HYPRE_Int hypre_thread_MPI_Probe( HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , MPI_Status *status );
HYPRE_Int hypre_thread_MPI_Irsend( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );

#ifdef __cplusplus
}
#endif

#endif

#endif
