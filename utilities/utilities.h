#include "HYPRE_utilities.h"

#ifndef hypre_UTILITIES_HEADER
#define hypre_UTILITIES_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * General structures and values
 *
 *****************************************************************************/

#ifndef hypre_GENERAL_HEADER
#define hypre_GENERAL_HEADER

/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#ifndef hypre_max
#define hypre_max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef hypre_min
#define hypre_min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef hypre_round
#define hypre_round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

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

#define MPI_Comm            hypre_MPI_Comm            
#define MPI_Group           hypre_MPI_Group            
#define MPI_Request         hypre_MPI_Request          
#define MPI_Datatype        hypre_MPI_Datatype         
#define MPI_Status          hypre_MPI_Status           
#define MPI_Op              hypre_MPI_Op               
#define MPI_Aint            hypre_MPI_Aint             

#define MPI_COMM_WORLD      hypre_MPI_COMM_WORLD       
#define MPI_COMM_NULL       hypre_MPI_COMM_NULL

#define MPI_BOTTOM  	    hypre_MPI_BOTTOM

#define MPI_DOUBLE          hypre_MPI_DOUBLE           
#define MPI_INT             hypre_MPI_INT              
#define MPI_CHAR            hypre_MPI_CHAR             
#define MPI_LONG            hypre_MPI_LONG             
#define MPI_BYTE            hypre_MPI_BYTE             

#define MPI_SUM             hypre_MPI_SUM              
#define MPI_MIN             hypre_MPI_MIN              
#define MPI_MAX             hypre_MPI_MAX              
#define MPI_LOR             hypre_MPI_LOR              

#define MPI_UNDEFINED       hypre_MPI_UNDEFINED        
#define MPI_REQUEST_NULL    hypre_MPI_REQUEST_NULL        
#define MPI_ANY_SOURCE      hypre_MPI_ANY_SOURCE        
#define MPI_ANY_TAG         hypre_MPI_ANY_TAG

#define MPI_Init            hypre_MPI_Init             
#define MPI_Finalize        hypre_MPI_Finalize         
#define MPI_Abort           hypre_MPI_Abort         
#define MPI_Wtime           hypre_MPI_Wtime            
#define MPI_Wtick           hypre_MPI_Wtick            
#define MPI_Barrier         hypre_MPI_Barrier          
#define MPI_Comm_create     hypre_MPI_Comm_create      
#define MPI_Comm_dup        hypre_MPI_Comm_dup         
#define MPI_Comm_group      hypre_MPI_Comm_group       
#define MPI_Comm_size       hypre_MPI_Comm_size        
#define MPI_Comm_rank       hypre_MPI_Comm_rank        
#define MPI_Comm_free       hypre_MPI_Comm_free        
#define MPI_Comm_split      hypre_MPI_Comm_split        
#define MPI_Group_incl      hypre_MPI_Group_incl       
#define MPI_Group_free      hypre_MPI_Group_free        
#define MPI_Address         hypre_MPI_Address        
#define MPI_Get_count       hypre_MPI_Get_count        
#define MPI_Alltoall        hypre_MPI_Alltoall        
#define MPI_Allgather       hypre_MPI_Allgather        
#define MPI_Allgatherv      hypre_MPI_Allgatherv       
#define MPI_Gather          hypre_MPI_Gather       
#define MPI_Scatter         hypre_MPI_Scatter       
#define MPI_Bcast           hypre_MPI_Bcast            
#define MPI_Send            hypre_MPI_Send             
#define MPI_Recv            hypre_MPI_Recv             
#define MPI_Isend           hypre_MPI_Isend            
#define MPI_Irecv           hypre_MPI_Irecv            
#define MPI_Send_init       hypre_MPI_Send_init             
#define MPI_Recv_init       hypre_MPI_Recv_init             
#define MPI_Irsend          hypre_MPI_Irsend             
#define MPI_Startall        hypre_MPI_Startall             
#define MPI_Probe           hypre_MPI_Probe             
#define MPI_Iprobe          hypre_MPI_Iprobe             
#define MPI_Test            hypre_MPI_Test             
#define MPI_Testall         hypre_MPI_Testall
#define MPI_Wait            hypre_MPI_Wait             
#define MPI_Waitall         hypre_MPI_Waitall          
#define MPI_Waitany         hypre_MPI_Waitany          
#define MPI_Allreduce       hypre_MPI_Allreduce        
#define MPI_Reduce          hypre_MPI_Reduce        
#define MPI_Request_free    hypre_MPI_Request_free        
#define MPI_Type_contiguous hypre_MPI_Type_contiguous     
#define MPI_Type_vector     hypre_MPI_Type_vector     
#define MPI_Type_hvector    hypre_MPI_Type_hvector     
#define MPI_Type_struct     hypre_MPI_Type_struct      
#define MPI_Type_commit     hypre_MPI_Type_commit
#define MPI_Type_free       hypre_MPI_Type_free        

/*--------------------------------------------------------------------------
 * Types, etc.
 *--------------------------------------------------------------------------*/

/* These types have associated creation and destruction routines */
typedef int hypre_MPI_Comm;
typedef int hypre_MPI_Group;
typedef int hypre_MPI_Request;
typedef int hypre_MPI_Datatype;

typedef struct { int MPI_SOURCE; int MPI_TAG; } hypre_MPI_Status;
typedef int  hypre_MPI_Op;
typedef int  hypre_MPI_Aint;

#define  hypre_MPI_COMM_WORLD 0
#define  hypre_MPI_COMM_NULL  -1

#define  hypre_MPI_BOTTOM  0x0

#define  hypre_MPI_DOUBLE 0
#define  hypre_MPI_INT 1
#define  hypre_MPI_CHAR 2
#define  hypre_MPI_LONG 3
#define  hypre_MPI_BYTE 4

#define  hypre_MPI_SUM 0
#define  hypre_MPI_MIN 1
#define  hypre_MPI_MAX 2
#define  hypre_MPI_LOR 3

#define  hypre_MPI_UNDEFINED -9999
#define  hypre_MPI_REQUEST_NULL  0
#define  hypre_MPI_ANY_SOURCE    1
#define  hypre_MPI_ANY_TAG       1

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* mpistubs.c */
int hypre_MPI_Init( int *argc , char ***argv );
int hypre_MPI_Finalize( void );
int hypre_MPI_Abort( hypre_MPI_Comm comm , int errorcode );
double hypre_MPI_Wtime( void );
double hypre_MPI_Wtick( void );
int hypre_MPI_Barrier( hypre_MPI_Comm comm );
int hypre_MPI_Comm_create( hypre_MPI_Comm comm , hypre_MPI_Group group , hypre_MPI_Comm *newcomm );
int hypre_MPI_Comm_dup( hypre_MPI_Comm comm , hypre_MPI_Comm *newcomm );
int hypre_MPI_Comm_size( hypre_MPI_Comm comm , int *size );
int hypre_MPI_Comm_rank( hypre_MPI_Comm comm , int *rank );
int hypre_MPI_Comm_free( hypre_MPI_Comm *comm );
int hypre_MPI_Comm_group( hypre_MPI_Comm comm , hypre_MPI_Group *group );
int hypre_MPI_Comm_split( hypre_MPI_Comm comm, int n, int m, hypre_MPI_Comm * comms );
int hypre_MPI_Group_incl( hypre_MPI_Group group , int n , int *ranks , hypre_MPI_Group *newgroup );
int hypre_MPI_Group_free( hypre_MPI_Group *group );
int hypre_MPI_Address( void *location , hypre_MPI_Aint *address );
int hypre_MPI_Get_count( hypre_MPI_Status *status , hypre_MPI_Datatype datatype , int *count );
int hypre_MPI_Alltoall( void *sendbuf , int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , int recvcount , hypre_MPI_Datatype recvtype , hypre_MPI_Comm comm );
int hypre_MPI_Allgather( void *sendbuf , int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , int recvcount , hypre_MPI_Datatype recvtype , hypre_MPI_Comm comm );
int hypre_MPI_Allgatherv( void *sendbuf , int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , int *recvcounts , int *displs , hypre_MPI_Datatype recvtype , hypre_MPI_Comm comm );
int hypre_MPI_Gather( void *sendbuf , int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , int recvcount , hypre_MPI_Datatype recvtype , int root , hypre_MPI_Comm comm );
int hypre_MPI_Scatter( void *sendbuf , int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , int recvcount , hypre_MPI_Datatype recvtype , int root , hypre_MPI_Comm comm );
int hypre_MPI_Bcast( void *buffer , int count , hypre_MPI_Datatype datatype , int root , hypre_MPI_Comm comm );
int hypre_MPI_Send( void *buf , int count , hypre_MPI_Datatype datatype , int dest , int tag , hypre_MPI_Comm comm );
int hypre_MPI_Recv( void *buf , int count , hypre_MPI_Datatype datatype , int source , int tag , hypre_MPI_Comm comm , hypre_MPI_Status *status );
int hypre_MPI_Isend( void *buf , int count , hypre_MPI_Datatype datatype , int dest , int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
int hypre_MPI_Irecv( void *buf , int count , hypre_MPI_Datatype datatype , int source , int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
int hypre_MPI_Send_init( void *buf , int count , hypre_MPI_Datatype datatype , int dest , int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
int hypre_MPI_Recv_init( void *buf , int count , hypre_MPI_Datatype datatype , int dest , int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
int hypre_MPI_Irsend( void *buf , int count , hypre_MPI_Datatype datatype , int dest , int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
int hypre_MPI_Startall( int count , hypre_MPI_Request *array_of_requests );
int hypre_MPI_Probe( int source , int tag , hypre_MPI_Comm comm , hypre_MPI_Status *status );
int hypre_MPI_Iprobe( int source , int tag , hypre_MPI_Comm comm , int *flag , hypre_MPI_Status *status );
int hypre_MPI_Test( hypre_MPI_Request *request , int *flag , hypre_MPI_Status *status );
int hypre_MPI_Testall( int count , hypre_MPI_Request *array_of_requests , int *flag , hypre_MPI_Status *array_of_statuses );
int hypre_MPI_Wait( hypre_MPI_Request *request , hypre_MPI_Status *status );
int hypre_MPI_Waitall( int count , hypre_MPI_Request *array_of_requests , hypre_MPI_Status *array_of_statuses );
int hypre_MPI_Waitany( int count , hypre_MPI_Request *array_of_requests , int *index , hypre_MPI_Status *status );
int hypre_MPI_Allreduce( void *sendbuf , void *recvbuf , int count , hypre_MPI_Datatype datatype , hypre_MPI_Op op , hypre_MPI_Comm comm );
int hypre_MPI_Reduce( void *sendbuf , void *recvbuf , int count , hypre_MPI_Datatype datatype , hypre_MPI_Op op , int root , hypre_MPI_Comm comm );
int hypre_MPI_Request_free( hypre_MPI_Request *request );
int hypre_MPI_Type_contiguous( int count , hypre_MPI_Datatype oldtype , hypre_MPI_Datatype *newtype );
int hypre_MPI_Type_vector( int count , int blocklength , int stride , hypre_MPI_Datatype oldtype , hypre_MPI_Datatype *newtype );
int hypre_MPI_Type_hvector( int count , int blocklength , hypre_MPI_Aint stride , hypre_MPI_Datatype oldtype , hypre_MPI_Datatype *newtype );
int hypre_MPI_Type_struct( int count , int *array_of_blocklengths , hypre_MPI_Aint *array_of_displacements , hypre_MPI_Datatype *array_of_types , hypre_MPI_Datatype *newtype );
int hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype );
int hypre_MPI_Type_free( hypre_MPI_Datatype *datatype );

#ifdef __cplusplus
}
#endif

#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header file for memory management utilities
 *
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Use "Debug Malloc Library", dmalloc
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_MEMORY_DMALLOC

#define hypre_InitMemoryDebug(id)    hypre_InitMemoryDebugDML(id)
#define hypre_FinalizeMemoryDebug()  hypre_FinalizeMemoryDebugDML()

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAllocDML((unsigned int)(sizeof(type) * (count)),\
                          __FILE__, __LINE__) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAllocDML((unsigned int)(count), (unsigned int)sizeof(type),\
                          __FILE__, __LINE__) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAllocDML((char *)ptr,\
                           (unsigned int)(sizeof(type) * (count)),\
                           __FILE__, __LINE__) )

#define hypre_TFree(ptr) \
( hypre_FreeDML((char *)ptr, __FILE__, __LINE__), ptr = NULL )

/*--------------------------------------------------------------------------
 * Use standard memory routines
 *--------------------------------------------------------------------------*/

#else

#define hypre_InitMemoryDebug(id)
#define hypre_FinalizeMemoryDebug()  

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAlloc((unsigned int)(sizeof(type) * (count))) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAlloc((unsigned int)(count), (unsigned int)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAlloc((char *)ptr, (unsigned int)(sizeof(type) * (count))) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

#endif


#ifdef HYPRE_USE_PTHREADS

#define hypre_SharedTAlloc(type, count) \
( (type *)hypre_SharedMAlloc((unsigned int)(sizeof(type) * (count))) )


#define hypre_SharedCTAlloc(type, count) \
( (type *)hypre_SharedCAlloc((unsigned int)(count),\
                             (unsigned int)sizeof(type)) )

#define hypre_SharedTReAlloc(ptr, type, count) \
( (type *)hypre_SharedReAlloc((char *)ptr,\
                              (unsigned int)(sizeof(type) * (count))) )

#define hypre_SharedTFree(ptr) \
( hypre_SharedFree((char *)ptr), ptr = NULL )

#else

#define hypre_SharedTAlloc(type, count) hypre_TAlloc(type, (count))
#define hypre_SharedCTAlloc(type, count) hypre_CTAlloc(type, (count))
#define hypre_SharedTReAlloc(type, count) hypre_TReAlloc(type, (count))
#define hypre_SharedTFree(ptr) hypre_TFree(ptr)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* hypre_memory.c */
int hypre_OutOfMemory( int size );
char *hypre_MAlloc( int size );
char *hypre_CAlloc( int count , int elt_size );
char *hypre_ReAlloc( char *ptr , int size );
void hypre_Free( char *ptr );
char *hypre_SharedMAlloc( int size );
char *hypre_SharedCAlloc( int count , int elt_size );
char *hypre_SharedReAlloc( char *ptr , int size );
void hypre_SharedFree( char *ptr );
double *hypre_IncrementSharedDataPtr( double *ptr , int size );

/* memory_dmalloc.c */
int hypre_InitMemoryDebugDML( int id );
int hypre_FinalizeMemoryDebugDML( void );
char *hypre_MAllocDML( int size , char *file , int line );
char *hypre_CAllocDML( int count , int elt_size , char *file , int line );
char *hypre_ReAllocDML( char *ptr , int size , char *file , int line );
void hypre_FreeDML( char *ptr , char *file , int line );

#ifdef __cplusplus
}
#endif

#endif

/* random.c */
void hypre_SeedRand ( int seed );
double hypre_Rand ( void );
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


#ifndef hypre_THREADING_HEADER
#define hypre_THREADING_HEADER

#if defined(HYPRE_USING_OPENMP) || defined (HYPRE_USING_PGCC_SMP)

int hypre_NumThreads( void );

#else

#define hypre_NumThreads() 1

#endif


/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/* The pthreads stuff needs to be reworked */

#ifdef HYPRE_USE_PTHREADS

#ifndef MAX_QUEUE
#define MAX_QUEUE 256
#endif

#include <pthread.h>

/* hypre_work_proc_t typedef'd to be a pointer to a function with a void*
   argument and a void return type */
typedef void (*hypre_work_proc_t)(void *);

typedef struct hypre_workqueue_struct {
   pthread_mutex_t lock;
   pthread_cond_t work_wait;
   pthread_cond_t finish_wait;
   hypre_work_proc_t worker_proc_queue[MAX_QUEUE];
   int n_working;
   int n_waiting;
   int n_queue;
   int inp;
   int outp;
   void *argqueue[MAX_QUEUE];
} *hypre_workqueue_t;

void hypre_work_put( hypre_work_proc_t funcptr, void *argptr );
void hypre_work_wait( void );
int HYPRE_InitPthreads( int num_threads );
void HYPRE_DestroyPthreads( void );
void hypre_pthread_worker( int threadid );
int ifetchadd( int *w, pthread_mutex_t *mutex_fetchadd );
int hypre_fetch_and_add( int *w );
void hypre_barrier(pthread_mutex_t *mpi_mtx, int unthreaded);
int hypre_GetThreadID( void );

pthread_t initial_thread;
pthread_t hypre_thread[hypre_MAX_THREADS];
pthread_mutex_t hypre_mutex_boxloops;
pthread_mutex_t talloc_mtx;
pthread_mutex_t worker_mtx;
hypre_workqueue_t hypre_qptr;
pthread_mutex_t mpi_mtx;
pthread_mutex_t time_mtx;
volatile int hypre_thread_release;

#ifdef HYPRE_THREAD_GLOBALS
int hypre_NumThreads = 4;
#else
extern int hypre_NumThreads;
#endif

#endif
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header file for doing timing
 *
 *****************************************************************************/

#ifndef HYPRE_TIMING_HEADER
#define HYPRE_TIMING_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Prototypes for low-level timing routines
 *--------------------------------------------------------------------------*/

/* timer.c */
double time_getWallclockSeconds( void );
double time_getCPUSeconds( void );
double time_get_wallclock_seconds_( void );
double time_get_cpu_seconds_( void );

/*--------------------------------------------------------------------------
 * With timing off
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_TIMING

#define hypre_InitializeTiming(name) 0
#define hypre_IncFLOPCount(inc)
#define hypre_BeginTiming(i)
#define hypre_EndTiming(i)
#define hypre_PrintTiming(heading, comm)
#define hypre_FinalizeTiming(index)

/*--------------------------------------------------------------------------
 * With timing on
 *--------------------------------------------------------------------------*/

#else

/*-------------------------------------------------------
 * Global timing structure
 *-------------------------------------------------------*/

typedef struct
{
   double  *wall_time;
   double  *cpu_time;
   double  *flops;
   char   **name;
   int     *state;     /* boolean flag to allow for recursive timing */
   int     *num_regs;  /* count of how many times a name is registered */

   int      num_names;
   int      size;

   double   wall_count;
   double   CPU_count;
   double   FLOP_count;

} hypre_TimingType;

#ifdef HYPRE_TIMING_GLOBALS
hypre_TimingType *hypre_global_timing = NULL;
#else
extern hypre_TimingType *hypre_global_timing;
#endif

/*-------------------------------------------------------
 * Accessor functions
 *-------------------------------------------------------*/

#ifndef HYPRE_USE_PTHREADS
#define hypre_TimingWallTime(i) (hypre_global_timing -> wall_time[(i)])
#define hypre_TimingCPUTime(i)  (hypre_global_timing -> cpu_time[(i)])
#define hypre_TimingFLOPS(i)    (hypre_global_timing -> flops[(i)])
#define hypre_TimingName(i)     (hypre_global_timing -> name[(i)])
#define hypre_TimingState(i)    (hypre_global_timing -> state[(i)])
#define hypre_TimingNumRegs(i)  (hypre_global_timing -> num_regs[(i)])
#define hypre_TimingWallCount   (hypre_global_timing -> wall_count)
#define hypre_TimingCPUCount    (hypre_global_timing -> CPU_count)
#define hypre_TimingFLOPCount   (hypre_global_timing -> FLOP_count)
#else
#define hypre_TimingWallTime(i) (hypre_global_timing[threadid].wall_time[(i)])
#define hypre_TimingCPUTime(i)  (hypre_global_timing[threadid].cpu_time[(i)])
#define hypre_TimingFLOPS(i)    (hypre_global_timing[threadid].flops[(i)])
#define hypre_TimingName(i)     (hypre_global_timing[threadid].name[(i)])
#define hypre_TimingState(i)    (hypre_global_timing[threadid].state[(i)])
#define hypre_TimingNumRegs(i)  (hypre_global_timing[threadid].num_regs[(i)])
#define hypre_TimingWallCount   (hypre_global_timing[threadid].wall_count)
#define hypre_TimingCPUCount    (hypre_global_timing[threadid].CPU_count)
#define hypre_TimingFLOPCount   (hypre_global_timing[threadid].FLOP_count)
#define hypre_TimingAllFLOPS    (hypre_global_timing[hypre_NumThreads].FLOP_count)
#endif

/*-------------------------------------------------------
 * Prototypes
 *-------------------------------------------------------*/

/* timing.c */
int hypre_InitializeTiming( const char *name );
int hypre_FinalizeTiming( int time_index );
int hypre_IncFLOPCount( int inc );
int hypre_BeginTiming( int time_index );
int hypre_EndTiming( int time_index );
int hypre_ClearTiming( void );
int hypre_PrintTiming( const char *heading , MPI_Comm comm );

#endif

#ifdef __cplusplus
}
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header file link lists
 *
 *****************************************************************************/

#ifndef HYPRE_LINKLIST_HEADER
#define HYPRE_LINKLIST_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LIST_HEAD -1
#define LIST_TAIL -2

struct double_linked_list
{
       int                        data;
       struct double_linked_list *next_elt;
       struct double_linked_list *prev_elt;
       int                        head;
       int                        tail;
};

typedef struct double_linked_list hypre_ListElement;
typedef hypre_ListElement  *hypre_LinkList;  

#ifdef __cplusplus
}
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef hypre_EXCHANGE_DATA_HEADER
#define hypre_EXCHANGE_DATA_HEADER

#define hypre_BinaryTreeParentId(tree)      (tree->parent_id)
#define hypre_BinaryTreeNumChild(tree)      (tree->num_child)
#define hypre_BinaryTreeChildIds(tree)      (tree->child_id)
#define hypre_BinaryTreeChildId(tree, i)    (tree->child_id[i])


typedef struct
{
   int                   parent_id;
   int                   num_child;
   int		        *child_id;
} hypre_BinaryTree;



/* In the fill_response() function the user needs to set the recv__buf
   and the response_message_size.  Memory of size send_response_storage has been
   alllocated for the send_buf (in exchange_data) - if more is needed, then
   realloc and adjust
   the send_response_storage.  The realloc amount should be storage+overhead. 
   If the response is an empty "confirmation" message, then set
   response_message_size =0 (and do not modify the send_buf) */


typedef struct
{
   int    (*fill_response)(void* recv_buf, int contact_size, 
                           int contact_proc, void* response_obj, 
                           MPI_Comm comm, void** response_buf, 
                           int* response_message_size);
   int     send_response_overhead; /*set by exchange data */
   int     send_response_storage;  /*storage allocated for send_response_buf*/
   void    *data1;                 /*data fields user may want to access in fill_response */
   void    *data2;
   
} hypre_DataExchangeResponse;


int hypre_CreateBinaryTree(int, int, hypre_BinaryTree*);
int hypre_DestroyBinaryTree(hypre_BinaryTree*);


int hypre_DataExchangeList(int num_contacts, 
		     int *contact_proc_list, void *contact_send_buf, 
		     int *contact_send_buf_starts, int contact_obj_size, 
                     int response_obj_size,
		     hypre_DataExchangeResponse *response_obj, int max_response_size, 
                     int rnum, MPI_Comm comm,  void **p_response_recv_buf, 
                     int **p_response_recv_buf_starts);


#endif /* end of header */

/* amg_linklist.c */
void dispose_elt ( hypre_LinkList element_ptr );
void remove_point ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , int measure , int index , int *lists , int *where );
hypre_LinkList create_elt ( int Item );
void enter_on_lists ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , int measure , int index , int *lists , int *where );

/* binsearch.c */
int hypre_BinarySearch ( int *list , int value , int list_length );

/* qsplit.c */
int hypre_DoubleQuickSplit ( double *values , int *indices , int list_length , int NumberKept );

/* hypre_qsort.c */
void swap ( int *v , int i , int j );
void swap2 ( int *v , double *w , int i , int j );
void hypre_swap2i ( int *v , int *w , int i , int j );
void hypre_swap3i ( int *v , int *w , int *z , int i , int j );
void qsort0 ( int *v , int left , int right );
void qsort1 ( int *v , double *w , int left , int right );
void hypre_qsort2i ( int *v , int *w , int left , int right );
void hypre_qsort2 ( int *v , double *w , int left , int right );
void hypre_qsort3i ( int *v , int *w , int *z , int left , int right );
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


#ifndef hypre_ERROR_HEADER
#define hypre_ERROR_HEADER

/*--------------------------------------------------------------------------
 * HYPRE error codes
 *--------------------------------------------------------------------------*/

#define HYPRE_ERROR_GENERIC         1   /* generic error */
#define HYPRE_ERROR_MEMORY          2   /* unable to allocate memory */
#define HYPRE_ERROR_ARG             4   /* argument error */
/* bits 4-8 are reserved for the index of the argument error */
#define HYPRE_ERROR_CONV          256   /* method did not converge as expected */

/*--------------------------------------------------------------------------
 * Global variable used in hypre error checking
 *--------------------------------------------------------------------------*/

extern int hypre__global_error;
#define hypre_error_flag  hypre__global_error

/*--------------------------------------------------------------------------
 * HYPRE error macros
 *--------------------------------------------------------------------------*/

void hypre_error_handler(char *filename, int line, int ierr);
#define hypre_error(IERR)  hypre_error_handler(__FILE__, __LINE__, IERR)
#define hypre_error_in_arg(IARG)  hypre_error(HYPRE_ERROR_ARG | IARG<<3)
#ifdef NDEBUG
#define hypre_assert(EX)
#else
#define hypre_assert(EX) if (!(EX)) {fprintf(stderr,"hypre_assert failed: %s\n", #EX); hypre_error(1);}
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

/* F2C_INTEGER will normally be `int' but would be `long' on 16-bit systems */
/* we assume short, float are OK */

/* integer changed to int - edmond 1/12/00 */

typedef int integer;
typedef unsigned long int /* long */ uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef long int /* long int */ logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;
/* integer*8 support from f2c not currently supported: */
#if 0
typedef @F2C_LONGINT@ /* long long */ longint; /* system-dependent */
typedef unsigned @F2C_LONGINT@ ulongint;	/* system-dependent */
#define qbit_clear(a,b)	((a) & ~((ulongint)1 << (b)))
#define qbit_set(a,b)	((a) |  ((ulongint)1 << (b)))
#endif
/* typedef long long int longint; */ /* RDF: removed */

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

#ifdef f2c_i2
  #error f2c_i2 will not work with g77!!!!
/* for -i2 */
typedef short flag;
typedef short ftnlen;
typedef short ftnint;
#else
typedef long int /* int or long int */ flag;
typedef int /* int or long int */ ftnlen; /* changed by edmond */
typedef long int /* int or long int */ ftnint;
#endif

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

/*typedef long int Long;*/	/* No longer used; formerly in Namelist */

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (doublereal)abs(x)
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (doublereal)min(a,b)
#define dmax(a,b) (doublereal)max(a,b)
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef int /* Unknown procedure type */ (*U_fp)(...);
typedef shortint (*J_fp)(...);
typedef integer (*I_fp)(...);
typedef real (*R_fp)(...);
typedef doublereal (*D_fp)(...), (*E_fp)(...);
typedef /* Complex */ VOID (*C_fp)(...);
typedef /* Double Complex */ VOID (*Z_fp)(...);
typedef logical (*L_fp)(...);
typedef shortlogical (*K_fp)(...);
typedef /* Character */ VOID (*H_fp)(...);
typedef /* Subroutine */ int (*S_fp)(...);
#else
typedef int /* Unknown procedure type */ (*U_fp)();
typedef shortint (*J_fp)();
typedef integer (*I_fp)();
typedef real (*R_fp)();
typedef doublereal (*D_fp)(), (*E_fp)();
typedef /* Complex */ VOID (*C_fp)();
typedef /* Double Complex */ VOID (*Z_fp)();
typedef logical (*L_fp)();
typedef shortlogical (*K_fp)();
typedef /* Character */ VOID (*H_fp)();
typedef /* Subroutine */ int (*S_fp)();
#endif
/* E_fp is for real functions when -R is not specified */
typedef VOID C_f;	/* complex function */
typedef VOID H_f;	/* character function */
typedef VOID Z_f;	/* double complex function */
typedef doublereal E_f;	/* real function with -R not specified */

/* undef any lower-case symbols that your C compiler predefines, e.g.: */

#ifndef Skip_f2c_Undefs
/* (No such symbols should be defined in a strict ANSI C compiler.
   We can avoid trouble with f2c-translated code by using
   gcc -ansi [-traditional].) */
#undef cray
#undef gcos
#undef mc68010
#undef mc68020
#undef mips
#undef pdp11
#undef sgi
#undef sparc
#undef sun
#undef sun2
#undef sun3
#undef sun4
#undef u370
#undef u3b
#undef u3b2
#undef u3b5
#undef unix
#undef vax
#endif
#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/* hypre_blas.h  --  Contains BLAS prototypes needed by Hypre */

#ifndef HYPRE_BLAS_H
#define HYPRE_BLAS_H

/* --------------------------------------------------------------------------
 *   Change all names to hypre_ to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dasum_   hypre_dasum
#define daxpy_   hypre_daxpy
#define dcopy_   hypre_dcopy
#define ddot_    hypre_ddot
#define dgemm_   hypre_dgemm
#define dgemv_   hypre_dgemv
#define dger_    hypre_dger
#define dnrm2_   hypre_dnrm2
#define drot_    hypre_drot
#define dscal_   hypre_dscal
#define dswap_   hypre_dswap
#define dsymm_   hypre_dsymm
#define dsymv_   hypre_dsymv
#define dsyr2_   hypre_dsyr2
#define dsyr2k_  hypre_dsyr2k
#define dsyrk_   hypre_dsyrk
#define dtrmm_   hypre_dtrmm
#define dtrmv_   hypre_dtrmv
#define dtrsm_   hypre_dtrsm
#define dtrsv_   hypre_dtrsv
#define idamax_  hypre_idamax

/* blas_utils.c */
logical lsame_ ( char *ca , char *cb );
int xerbla_ ( char *srname , integer *info );
integer s_cmp ( char *a0 , char *b0 , ftnlen la , ftnlen lb );
VOID s_copy ( char *a , char *b , ftnlen la , ftnlen lb );

/* dasum.c */
doublereal dasum_ ( integer *n , doublereal *dx , integer *incx );

/* daxpy.c */
int daxpy_ ( integer *n , doublereal *da , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dcopy.c */
int dcopy_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* ddot.c */
doublereal ddot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dgemm.c */
int dgemm_ ( char *transa , char *transb , integer *m , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c , integer *ldc );

/* dgemv.c */
int dgemv_ ( char *trans , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dger.c */
int dger_ ( integer *m , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dnrm2.c */
doublereal dnrm2_ ( integer *n , doublereal *dx , integer *incx );

/* drot.c */
int drot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy , doublereal *c , doublereal *s );

/* dscal.c */
int dscal_ ( integer *n , doublereal *da , doublereal *dx , integer *incx );

/* dswap.c */
int dswap_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dsymm.c */
int dsymm_ ( char *side , char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsymv.c */
int dsymv_ ( char *uplo , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dsyr2.c */
int dsyr2_ ( char *uplo , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dsyr2k.c */
int dsyr2k_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsyrk.c */
int dsyrk_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *beta , doublereal *c , integer *ldc );

/* dtrmm.c */
int dtrmm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrmv.c */
int dtrmv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* dtrsm.c */
int dtrsm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrsv.c */
int dtrsv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* idamax.c */
integer idamax_ ( integer *n , doublereal *dx , integer *incx );

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/* hypre_lapack.h  --  Contains LAPACK prototypes needed by Hypre */

#ifndef HYPRE_LAPACK_H
#define HYPRE_LAPACK_H

/* --------------------------------------------------------------------------
 *  Change all names to hypre_ to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dgebd2_  hypre_dgebd2
#define dgebrd_  hypre_dgebrd
#define dgelq2_  hypre_dgelq2
#define dgelqf_  hypre_dgelqf
#define dgels_   hypre_dgels
#define dgeqr2_  hypre_dgeqr2
#define dgeqrf_  hypre_dgeqrf
#define dgesvd_  hypre_dgesvd
#define dgetf2_  hypre_dgetf2
#define dgetrf_  hypre_dgetrf
#define dgetrs_  hypre_dgetrs
#define dlabad_  hypre_dlabad
#define dlabrd_  hypre_dlabrd
#define dlae2_   hypre_dlae2
#define dlaev2_  hypre_dlaev2
#define dlamch_  hypre_dlamch
#define dlamc1_  hypre_dlamc1
#define dlamc2_  hypre_dlamc2
#define dlamc3_  hypre_dlamc3
#define dlamc4_  hypre_dlamc4
#define dlamc5_  hypre_dlamc5
#define dlaswp_  hypre_dlaswp
#define dlange_  hypre_dlange
#define dlanst_  hypre_dlanst
#define dlansy_  hypre_dlansy
#define dlapy2_  hypre_dlapy2
#define dlarf_   hypre_dlarf
#define dlarfb_  hypre_dlarfb
#define dlarfg_  hypre_dlarfg
#define dlarft_  hypre_dlarft
#define dlartg_  hypre_dlartg
#define dlascl_  hypre_dlascl
#define dlaset_  hypre_dlaset
#define dlasr_   hypre_dlasr
#define dlasrt_  hypre_dlasrt
#define dlassq_  hypre_dlassq
#define dlatrd_  hypre_dlatrd
#define dorg2l_  hypre_dorg2l
#define dorg2r_  hypre_dorg2r
#define dorgql_  hypre_dorgql
#define dorgqr_  hypre_dorgqr
#define dorgtr_  hypre_dorgtr
#define dorm2r_  hypre_dorm2r
#define dorml2_  hypre_dorml2
#define dormlq_  hypre_dormlq
#define dormqr_  hypre_dormqr
#define dpotf2_  hypre_dpotf2
#define dpotrf_  hypre_dpotrf
#define dpotrs_  hypre_dpotrs
#define dsteqr_  hypre_dsteqr
#define dsterf_  hypre_dsterf
#define dsyev_   hypre_dsyev
#define dsygst_  hypre_dsygst
#define dsygv_   hypre_dsygv
#define dsytd2_  hypre_dsytd2
#define dsytrd_  hypre_dsytrd
#define ieeeck_  hypre_ieeeck
#define ilaenv_  hypre_ilaenv
#define d_lg10_  hypre_d_lg10
#define d_sign_  hypre_d_sign
#define pow_di_  hypre_pow_di
#define pow_dd_  hypre_pow_dd
#define s_cat_   hypre_s_cat
#define lsame_   hypre_lsame
#define xerbla_  hypre_xerbla
#define dbdsqr_  hypre_dbdsqr
#define dorgbr_  hypre_dorgbr
#define dsygs2_  hypre_dsygs2
#define dorglq_  hypre_dorglq
#define dlacpy_  hypre_dlacpy
#define dormbr_  hypre_dormbr
#define dlasq1_  hypre_dlasq1
#define dlas2_   hypre_dlas2
#define dlasv2_  hypre_dlasv2
#define dorgl2_  hypre_dorgl2
#define dlasq2_  hypre_dlasq2
#define dlasq3_  hypre_dlasq3
#define dlasq4_  hypre_dlasq4
#define dlasq5_  hypre_dlasq5
#define dlasq6_  hypre_dlasq6

/* --------------------------------------------------------------------------
 *           Prototypes
 * --------------------------------------------------------------------------*/

/* dgebd2.c */
int dgebd2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *work , integer *info );

/* dgebrd.c */
int dgebrd_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *work , integer *lwork , integer *info );

/* dgelq2.c */
int dgelq2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dgelqf.c */
int dgelqf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dgels.c */
int dgels_ ( char *trans , integer *m , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *work , integer *lwork , integer *info );

/* dgeqr2.c */
int dgeqr2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dgeqrf.c */
int dgeqrf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dgesvd.c */
int dgesvd_ ( char *jobu , char *jobvt , integer *m , integer *n , doublereal *a , integer *lda , doublereal *s , doublereal *u , integer *ldu , doublereal *vt , integer *ldvt , doublereal *work , integer *lwork , integer *info );


/* dgetf2.c */
int dgetf2_( integer *m , integer *n , doublereal *a , integer *lda , integer *ipiv , integer *info );

/* dgetrf.c */
int dgetrf_( integer *m , integer *n , doublereal *a , integer *lda , integer *ipiv , integer *info );

/* dgetrs.c */
int dgetrs_( char *trans , integer *n , integer *nrhs , doublereal *a , integer *lda , integer *ipiv , doublereal *b , integer *ldb , integer *info );

/* dlabad.c */
int dlabad_ ( doublereal *small , doublereal *large );

/* dlabrd.c */
int dlabrd_ ( integer *m , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *x , integer *ldx , doublereal *y , integer *ldy );

/* dlae2.c */
int dlae2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 );

/* dlaev2.c */
int dlaev2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 , doublereal *cs1 , doublereal *sn1 );

/* dlamch.c */
doublereal dlamch_ ( char *cmach );
int dlamc1_ ( integer *beta , integer *t , logical *rnd , logical *ieee1 );
int dlamc2_ ( integer *beta , integer *t , logical *rnd , doublereal *eps , integer *emin , doublereal *rmin , integer *emax , doublereal *rmax );
doublereal dlamc3_ ( doublereal *a , doublereal *b );
int dlamc4_ ( integer *emin , doublereal *start , integer *base );
int dlamc5_ ( integer *beta , integer *p , integer *emin , logical *ieee , integer *emax , doublereal *rmax );


/* dlaswp.c */
int dlaswp_( integer *n , doublereal *a , integer *lda , integer *k1 , integer *k2 , integer *ipiv , integer *incx );

/* dlange.c */
doublereal dlange_ ( char *norm , integer *m , integer *n , doublereal *a , integer *lda , doublereal *work );

/* dlanst.c */
doublereal dlanst_ ( char *norm , integer *n , doublereal *d__ , doublereal *e );

/* dlansy.c */
doublereal dlansy_ ( char *norm , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *work );

/* dlapy2.c */
doublereal dlapy2_ ( doublereal *x , doublereal *y );

/* dlarf.c */
int dlarf_ ( char *side , integer *m , integer *n , doublereal *v , integer *incv , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work );

/* dlarfb.c */
int dlarfb_ ( char *side , char *trans , char *direct , char *storev , integer *m , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *t , integer *ldt , doublereal *c__ , integer *ldc , doublereal *work , integer *ldwork );

/* dlarfg.c */
int dlarfg_ ( integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *tau );

/* dlarft.c */
int dlarft_ ( char *direct , char *storev , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *tau , doublereal *t , integer *ldt );

/* dlartg.c */
int dlartg_ ( doublereal *f , doublereal *g , doublereal *cs , doublereal *sn , doublereal *r__ );

/* dlascl.c */
int dlascl_ ( char *type__ , integer *kl , integer *ku , doublereal *cfrom , doublereal *cto , integer *m , integer *n , doublereal *a , integer *lda , integer *info );

/* dlaset.c */
int dlaset_ ( char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *beta , doublereal *a , integer *lda );

/* dlasr.c */
int dlasr_ ( char *side , char *pivot , char *direct , integer *m , integer *n , doublereal *c__ , doublereal *s , doublereal *a , integer *lda );

/* dlasrt.c */
int dlasrt_ ( char *id , integer *n , doublereal *d__ , integer *info );

/* dlassq.c */
int dlassq_ ( integer *n , doublereal *x , integer *incx , doublereal *scale , doublereal *sumsq );

/* dlatrd.c */
int dlatrd_ ( char *uplo , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *e , doublereal *tau , doublereal *w , integer *ldw );

/* dorg2l.c */
int dorg2l_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dorg2r.c */
int dorg2r_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dorgql.c */
int dorgql_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorgqr.c */
int dorgqr_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorgtr.c */
int dorgtr_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorm2r.c */
int dorm2r_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dorml2.c */
int dorml2_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dormlq.c */
int dormlq_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dormqr.c */
int dormqr_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dpotf2.c */
int dpotf2_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );

/* dpotrf.c */
int dpotrf_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );

/* dpotrs.c */
int dpotrs_ ( char *uplo , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dsteqr.c */
int dsteqr_ ( char *compz , integer *n , doublereal *d__ , doublereal *e , doublereal *z__ , integer *ldz , doublereal *work , integer *info );

/* dsterf.c */
int dsterf_ ( integer *n , doublereal *d__ , doublereal *e , integer *info );

/* dsyev.c */
int dsyev_ ( char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsygst.c */
int dsygst_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dsygv.c */
int dsygv_ ( integer *itype , char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsytd2.c */
int dsytd2_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , integer *info );

/* dsytrd.c */
int dsytrd_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* ieeeck.c */
integer ieeeck_ ( integer *ispec , real *zero , real *one );

/* ilaenv.c */
integer ilaenv_ ( integer *ispec , char *name__ , char *opts , integer *n1 , integer *n2 , integer *n3 , integer *n4 , ftnlen name_len , ftnlen opts_len );

/* lapack_utils.c */
double d_lg10 ( doublereal *x );
double d_sign ( doublereal *a , doublereal *b );
double pow_di ( doublereal *ap , integer *bp );
double pow_dd ( doublereal *ap , doublereal *bp );
int s_cat ( char *lp , char *rpp [], ftnlen rnp [], ftnlen *np , ftnlen ll );

/* lsame.c */
logical lsame_ ( char *ca , char *cb );

/* xerbla.c */
int xerbla_ ( char *srname , integer *info );

/* dbdsqr.c */
int dbdsqr_ ( char *uplo , integer *n , integer *ncvt , integer *nru , integer *ncc , doublereal *d__ , doublereal *e , doublereal *vt , integer *ldvt , doublereal *u , integer *ldu , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dorgbr.c */
int dorgbr_ ( char *vect , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dsygs2.c */
int dsygs2_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dorglq.c */
int dorglq_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dlacpy.c */
int dlacpy_ ( char *uplo , integer *m , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dormbr.c */
int dormbr_ ( char *vect , char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dlasq1.c */
int dlasq1_ ( integer *n , doublereal *d__ , doublereal *e , doublereal *work , integer *info );

/* dlas2.c */
int dlas2_ ( doublereal *f , doublereal *g , doublereal *h__ , doublereal *ssmin , doublereal *ssmax );

/* dlasv2.c */
int dlasv2_ ( doublereal *f , doublereal *g , doublereal *h__ , doublereal *ssmin , doublereal *ssmax , doublereal *snr , doublereal *csr , doublereal *snl , doublereal *csl );

/* dorgl2.c */
int dorgl2_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dlasq2.c */
int dlasq2_ ( integer *n , doublereal *z__ , integer *info );

/* dlasq3.c */
int dlasq3_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *dmin__ , doublereal *sigma , doublereal *desig , doublereal *qmax , integer *nfail , integer *iter , integer *ndiv , logical *ieee );

/* dlasq4.c */
int dlasq4_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , integer *n0in , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dn1 , doublereal *dn2 , doublereal *tau , integer *ttype );

/* dlasq5.c */
int dlasq5_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *tau , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dnm1 , doublereal *dnm2 , logical *ieee );

/* dlasq6.c */
int dlasq6_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dnm1 , doublereal *dnm2 );

#endif

#ifdef __cplusplus
}
#endif

#endif

