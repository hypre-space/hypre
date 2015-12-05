/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/

#ifndef hypre_UTILITIES_HEADER
#define hypre_UTILITIES_HEADER

#include "HYPRE_utilities.h"

/* This allows us to consistently avoid 'int' throughout hypre */
typedef int               hypre_int;
typedef long int          hypre_longint;
typedef unsigned int      hypre_uint;
typedef unsigned long int hypre_ulongint;

#ifdef HYPRE_USE_PTHREADS
#ifndef hypre_MAX_THREADS
#define hypre_MAX_THREADS 128
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
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
#define hypre_round(x)  ( ((x) < 0.0) ? ((HYPRE_Int)(x - 0.5)) : ((HYPRE_Int)(x + 0.5)) )
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 *  Fake mpi stubs to generate serial codes without mpi
 *
 *****************************************************************************/

#ifndef hypre_MPISTUBS
#define hypre_MPISTUBS

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HYPRE_SEQUENTIAL

/******************************************************************************
 * MPI stubs to generate serial codes without mpi
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Change all MPI names to hypre_MPI names to avoid link conflicts.
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
#define MPI_COMM_SELF       hypre_MPI_COMM_SELF

#define MPI_BOTTOM  	    hypre_MPI_BOTTOM

#define MPI_DOUBLE          hypre_MPI_DOUBLE           
#define MPI_INT             hypre_MPI_INT              
#define MPI_LONG_LONG_INT   hypre_MPI_INT              
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
#define MPI_SOURCE          hypre_MPI_SOURCE
#define MPI_TAG             hypre_MPI_TAG

#define MPI_Init            hypre_MPI_Init             
#define MPI_Finalize        hypre_MPI_Finalize         
#define MPI_Abort           hypre_MPI_Abort         
#define MPI_Wtime           hypre_MPI_Wtime            
#define MPI_Wtick           hypre_MPI_Wtick            
#define MPI_Barrier         hypre_MPI_Barrier          
#define MPI_Comm_create     hypre_MPI_Comm_create      
#define MPI_Comm_dup        hypre_MPI_Comm_dup         
#define MPI_Comm_f2c        hypre_MPI_Comm_f2c
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
#define MPI_Scan            hypre_MPI_Scan        
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
typedef HYPRE_Int hypre_MPI_Comm;
typedef HYPRE_Int hypre_MPI_Group;
typedef HYPRE_Int hypre_MPI_Request;
typedef HYPRE_Int hypre_MPI_Datatype;

typedef struct
{
   HYPRE_Int hypre_MPI_SOURCE;
   HYPRE_Int hypre_MPI_TAG;
} hypre_MPI_Status;
typedef HYPRE_Int  hypre_MPI_Op;
typedef HYPRE_Int  hypre_MPI_Aint;

#define  hypre_MPI_COMM_SELF 1
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

#else

/******************************************************************************
 * MPI stubs to do casting of HYPRE_Int and hypre_int correctly
 *****************************************************************************/

typedef MPI_Comm     hypre_MPI_Comm;
typedef MPI_Group    hypre_MPI_Group;
typedef MPI_Request  hypre_MPI_Request;
typedef MPI_Datatype hypre_MPI_Datatype;
typedef MPI_Status   hypre_MPI_Status;
typedef MPI_Op       hypre_MPI_Op;
typedef MPI_Aint     hypre_MPI_Aint;

#define  hypre_MPI_COMM_WORLD MPI_COMM_WORLD
#define  hypre_MPI_COMM_NULL  MPI_COMM_NULL
#define  hypre_MPI_BOTTOM     MPI_BOTTOM
#define  hypre_MPI_COMM_SELF  MPI_COMM_SELF

#define  hypre_MPI_DOUBLE MPI_DOUBLE
/* HYPRE_MPI_INT is defined in HYPRE_utilities.h */
#define  hypre_MPI_CHAR   MPI_CHAR
#define  hypre_MPI_LONG   MPI_LONG
#define  hypre_MPI_BYTE   MPI_BYTE

#define  hypre_MPI_SUM MPI_SUM
#define  hypre_MPI_MIN MPI_MIN
#define  hypre_MPI_MAX MPI_MAX
#define  hypre_MPI_LOR MPI_LOR

#define  hypre_MPI_UNDEFINED       MPI_UNDEFINED   
#define  hypre_MPI_REQUEST_NULL    MPI_REQUEST_NULL
#define  hypre_MPI_ANY_SOURCE      MPI_ANY_SOURCE  
#define  hypre_MPI_ANY_TAG         MPI_ANY_TAG
#define  hypre_MPI_SOURCE          MPI_SOURCE
#define  hypre_MPI_TAG             MPI_TAG
#define  hypre_MPI_STATUSES_IGNORE MPI_STATUSES_IGNORE
#define  hypre_MPI_LAND            MPI_LAND

#endif

/******************************************************************************
 * Everything below this applies to both ifdef cases above
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* mpistubs.c */
HYPRE_Int hypre_MPI_Init( hypre_int *argc , char ***argv );
HYPRE_Int hypre_MPI_Finalize( void );
HYPRE_Int hypre_MPI_Abort( hypre_MPI_Comm comm , HYPRE_Int errorcode );
double hypre_MPI_Wtime( void );
double hypre_MPI_Wtick( void );
HYPRE_Int hypre_MPI_Barrier( hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Comm_create( hypre_MPI_Comm comm , hypre_MPI_Group group , hypre_MPI_Comm *newcomm );
HYPRE_Int hypre_MPI_Comm_dup( hypre_MPI_Comm comm , hypre_MPI_Comm *newcomm );
hypre_MPI_Comm hypre_MPI_Comm_f2c( hypre_int comm );
HYPRE_Int hypre_MPI_Comm_size( hypre_MPI_Comm comm , HYPRE_Int *size );
HYPRE_Int hypre_MPI_Comm_rank( hypre_MPI_Comm comm , HYPRE_Int *rank );
HYPRE_Int hypre_MPI_Comm_free( hypre_MPI_Comm *comm );
HYPRE_Int hypre_MPI_Comm_group( hypre_MPI_Comm comm , hypre_MPI_Group *group );
HYPRE_Int hypre_MPI_Comm_split( hypre_MPI_Comm comm, HYPRE_Int n, HYPRE_Int m, hypre_MPI_Comm * comms );
HYPRE_Int hypre_MPI_Group_incl( hypre_MPI_Group group , HYPRE_Int n , HYPRE_Int *ranks , hypre_MPI_Group *newgroup );
HYPRE_Int hypre_MPI_Group_free( hypre_MPI_Group *group );
HYPRE_Int hypre_MPI_Address( void *location , hypre_MPI_Aint *address );
HYPRE_Int hypre_MPI_Get_count( hypre_MPI_Status *status , hypre_MPI_Datatype datatype , HYPRE_Int *count );
HYPRE_Int hypre_MPI_Alltoall( void *sendbuf , HYPRE_Int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , hypre_MPI_Datatype recvtype , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Allgather( void *sendbuf , HYPRE_Int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , hypre_MPI_Datatype recvtype , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Allgatherv( void *sendbuf , HYPRE_Int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int *recvcounts , HYPRE_Int *displs , hypre_MPI_Datatype recvtype , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Gather( void *sendbuf , HYPRE_Int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , hypre_MPI_Datatype recvtype , HYPRE_Int root , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Scatter( void *sendbuf , HYPRE_Int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , hypre_MPI_Datatype recvtype , HYPRE_Int root , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Bcast( void *buffer , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int root , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Send( void *buf , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Recv( void *buf , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int source , HYPRE_Int tag , hypre_MPI_Comm comm , hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Isend( void *buf , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Irecv( void *buf , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int source , HYPRE_Int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Send_init( void *buf , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Recv_init( void *buf , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Irsend( void *buf , HYPRE_Int count , hypre_MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , hypre_MPI_Comm comm , hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Startall( HYPRE_Int count , hypre_MPI_Request *array_of_requests );
HYPRE_Int hypre_MPI_Probe( HYPRE_Int source , HYPRE_Int tag , hypre_MPI_Comm comm , hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Iprobe( HYPRE_Int source , HYPRE_Int tag , hypre_MPI_Comm comm , HYPRE_Int *flag , hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Test( hypre_MPI_Request *request , HYPRE_Int *flag , hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Testall( HYPRE_Int count , hypre_MPI_Request *array_of_requests , HYPRE_Int *flag , hypre_MPI_Status *array_of_statuses );
HYPRE_Int hypre_MPI_Wait( hypre_MPI_Request *request , hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Waitall( HYPRE_Int count , hypre_MPI_Request *array_of_requests , hypre_MPI_Status *array_of_statuses );
HYPRE_Int hypre_MPI_Waitany( HYPRE_Int count , hypre_MPI_Request *array_of_requests , HYPRE_Int *index , hypre_MPI_Status *status );
HYPRE_Int hypre_MPI_Allreduce( void *sendbuf , void *recvbuf , HYPRE_Int count , hypre_MPI_Datatype datatype , hypre_MPI_Op op , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Reduce( void *sendbuf , void *recvbuf , HYPRE_Int count , hypre_MPI_Datatype datatype , hypre_MPI_Op op , HYPRE_Int root , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Scan( void *sendbuf , void *recvbuf , HYPRE_Int count , hypre_MPI_Datatype datatype , hypre_MPI_Op op , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Request_free( hypre_MPI_Request *request );
HYPRE_Int hypre_MPI_Type_contiguous( HYPRE_Int count , hypre_MPI_Datatype oldtype , hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_vector( HYPRE_Int count , HYPRE_Int blocklength , HYPRE_Int stride , hypre_MPI_Datatype oldtype , hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_hvector( HYPRE_Int count , HYPRE_Int blocklength , hypre_MPI_Aint stride , hypre_MPI_Datatype oldtype , hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_struct( HYPRE_Int count , HYPRE_Int *array_of_blocklengths , hypre_MPI_Aint *array_of_displacements , hypre_MPI_Datatype *array_of_types , hypre_MPI_Datatype *newtype );
HYPRE_Int hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype );
HYPRE_Int hypre_MPI_Type_free( hypre_MPI_Datatype *datatype );

#ifdef __cplusplus
}
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header file for memory management utilities
 *
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#include <stdio.h>
#include <stdlib.h>

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
( (type *)hypre_MAllocDML((size_t)(sizeof(type) * (count)),\
                          __FILE__, __LINE__) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAllocDML((size_t)(count), (size_t)sizeof(type),\
                          __FILE__, __LINE__) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAllocDML((char *)ptr,\
                           (size_t)(sizeof(type) * (count)),\
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
( (type *)hypre_MAlloc((size_t)(sizeof(type) * (count))) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAlloc((size_t)(count), (size_t)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count))) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

#endif


#ifdef HYPRE_USE_PTHREADS

#define hypre_SharedTAlloc(type, count) \
( (type *)hypre_SharedMAlloc((size_t)(sizeof(type) * (count))) )


#define hypre_SharedCTAlloc(type, count) \
( (type *)hypre_SharedCAlloc((size_t)(count),\
                             (size_t)sizeof(type)) )

#define hypre_SharedTReAlloc(ptr, type, count) \
( (type *)hypre_SharedReAlloc((char *)ptr,\
                              (size_t)(sizeof(type) * (count))) )

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
HYPRE_Int hypre_OutOfMemory ( size_t size );
char *hypre_MAlloc ( size_t size );
char *hypre_CAlloc ( size_t count , size_t elt_size );
char *hypre_ReAlloc ( char *ptr , size_t size );
void hypre_Free ( char *ptr );
char *hypre_SharedMAlloc ( size_t size );
char *hypre_SharedCAlloc ( size_t count , size_t elt_size );
char *hypre_SharedReAlloc ( char *ptr , size_t size );
void hypre_SharedFree ( char *ptr );
double *hypre_IncrementSharedDataPtr ( double *ptr , size_t size );

/* memory_dmalloc.c */
HYPRE_Int hypre_InitMemoryDebugDML( HYPRE_Int id );
HYPRE_Int hypre_FinalizeMemoryDebugDML( void );
char *hypre_MAllocDML( HYPRE_Int size , char *file , HYPRE_Int line );
char *hypre_CAllocDML( HYPRE_Int count , HYPRE_Int elt_size , char *file , HYPRE_Int line );
char *hypre_ReAllocDML( char *ptr , HYPRE_Int size , char *file , HYPRE_Int line );
void hypre_FreeDML( char *ptr , char *file , HYPRE_Int line );

#ifdef __cplusplus
}
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
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
HYPRE_Int MPI_Init( HYPRE_Int *argc , char ***argv );
double MPI_Wtime( void );
double MPI_Wtick( void );
HYPRE_Int MPI_Barrier( MPI_Comm comm );
HYPRE_Int MPI_Finalize( void );
HYPRE_Int MPI_Abort( MPI_Comm comm , HYPRE_Int errorcode );
HYPRE_Int MPI_Comm_group( MPI_Comm comm , MPI_Group *group );
HYPRE_Int MPI_Comm_dup( MPI_Comm comm , MPI_Comm *newcomm );
HYPRE_Int MPI_Group_incl( MPI_Group group , HYPRE_Int n , HYPRE_Int *ranks , MPI_Group *newgroup );
HYPRE_Int MPI_Comm_create( MPI_Comm comm , MPI_Group group , MPI_Comm *newcomm );
HYPRE_Int MPI_Get_count( MPI_Status *status , MPI_Datatype datatype , HYPRE_Int *count );
HYPRE_Int MPI_Alltoall( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , MPI_Comm comm );
HYPRE_Int MPI_Allgather( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , MPI_Comm comm );
HYPRE_Int MPI_Allgatherv( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int *recvcounts , HYPRE_Int *displs , MPI_Datatype recvtype , MPI_Comm comm );
HYPRE_Int MPI_Gather( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , HYPRE_Int root , MPI_Comm comm );
HYPRE_Int MPI_Scatter( void *sendbuf , HYPRE_Int sendcount , MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , MPI_Datatype recvtype , HYPRE_Int root , MPI_Comm comm );
HYPRE_Int MPI_Bcast( void *buffer , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int root , MPI_Comm comm );
HYPRE_Int MPI_Send( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm );
HYPRE_Int MPI_Recv( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , MPI_Status *status );
HYPRE_Int MPI_Isend( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int MPI_Irecv( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int MPI_Wait( MPI_Request *request , MPI_Status *status );
HYPRE_Int MPI_Waitall( HYPRE_Int count , MPI_Request *array_of_requests , MPI_Status *array_of_statuses );
HYPRE_Int MPI_Waitany( HYPRE_Int count , MPI_Request *array_of_requests , HYPRE_Int *index , MPI_Status *status );
HYPRE_Int MPI_Comm_size( MPI_Comm comm , HYPRE_Int *size );
HYPRE_Int MPI_Comm_rank( MPI_Comm comm , HYPRE_Int *rank );
HYPRE_Int MPI_Allreduce( void *sendbuf , void *recvbuf , HYPRE_Int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm );
HYPRE_Int MPI_Address( void *location , MPI_Aint *address );
HYPRE_Int MPI_Type_contiguous( HYPRE_Int count , MPI_Datatype oldtype , MPI_Datatype *newtype );
HYPRE_Int MPI_Type_vector( HYPRE_Int count , HYPRE_Int blocklength , HYPRE_Int stride , MPI_Datatype oldtype , MPI_Datatype *newtype );
HYPRE_Int MPI_Type_hvector( HYPRE_Int count , HYPRE_Int blocklength , MPI_Aint stride , MPI_Datatype oldtype , MPI_Datatype *newtype );
HYPRE_Int MPI_Type_struct( HYPRE_Int count , HYPRE_Int *array_of_blocklengths , MPI_Aint *array_of_displacements , MPI_Datatype *array_of_types , MPI_Datatype *newtype );
HYPRE_Int MPI_Type_free( MPI_Datatype *datatype );
HYPRE_Int MPI_Type_commit( MPI_Datatype *datatype );
HYPRE_Int MPI_Request_free( MPI_Request *request );
HYPRE_Int MPI_Send_init( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int MPI_Recv_init( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );
HYPRE_Int MPI_Startall( HYPRE_Int count , MPI_Request *array_of_requests );
HYPRE_Int MPI_Iprobe( HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , HYPRE_Int *flag , MPI_Status *status );
HYPRE_Int MPI_Probe( HYPRE_Int source , HYPRE_Int tag , MPI_Comm comm , MPI_Status *status );
HYPRE_Int MPI_Irsend( void *buf , HYPRE_Int count , MPI_Datatype datatype , HYPRE_Int dest , HYPRE_Int tag , MPI_Comm comm , MPI_Request *request );

#ifdef __cplusplus
}
#endif

#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/



#ifndef hypre_THREADING_HEADER
#define hypre_THREADING_HEADER

#if defined(HYPRE_USING_OPENMP) || defined (HYPRE_USING_PGCC_SMP)

HYPRE_Int hypre_NumThreads( void );
HYPRE_Int hypre_GetThreadNum( void );

#else

#define hypre_NumThreads() 1
#define hypre_GetThreadNum() 0

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
   HYPRE_Int n_working;
   HYPRE_Int n_waiting;
   HYPRE_Int n_queue;
   HYPRE_Int inp;
   HYPRE_Int outp;
   void *argqueue[MAX_QUEUE];
} *hypre_workqueue_t;

void hypre_work_put( hypre_work_proc_t funcptr, void *argptr );
void hypre_work_wait( void );
HYPRE_Int HYPRE_InitPthreads( HYPRE_Int num_threads );
void HYPRE_DestroyPthreads( void );
void hypre_pthread_worker( HYPRE_Int threadid );
HYPRE_Int ifetchadd( HYPRE_Int *w, pthread_mutex_t *mutex_fetchadd );
HYPRE_Int hypre_fetch_and_add( HYPRE_Int *w );
void hypre_barrier(pthread_mutex_t *mpi_mtx, HYPRE_Int unthreaded);
HYPRE_Int hypre_GetThreadID( void );

pthread_t initial_thread;
pthread_t hypre_thread[hypre_MAX_THREADS];
pthread_mutex_t hypre_mutex_boxloops;
pthread_mutex_t talloc_mtx;
pthread_mutex_t worker_mtx;
hypre_workqueue_t hypre_qptr;
pthread_mutex_t mpi_mtx;
pthread_mutex_t time_mtx;
volatile HYPRE_Int hypre_thread_release;

#ifdef HYPRE_THREAD_GLOBALS
HYPRE_Int hypre_NumThreads = 4;
#else
extern HYPRE_Int hypre_NumThreads;
#endif

#endif
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
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
   HYPRE_Int     *state;     /* boolean flag to allow for recursive timing */
   HYPRE_Int     *num_regs;  /* count of how many times a name is registered */

   HYPRE_Int      num_names;
   HYPRE_Int      size;

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
HYPRE_Int hypre_InitializeTiming( const char *name );
HYPRE_Int hypre_FinalizeTiming( HYPRE_Int time_index );
HYPRE_Int hypre_IncFLOPCount( HYPRE_Int inc );
HYPRE_Int hypre_BeginTiming( HYPRE_Int time_index );
HYPRE_Int hypre_EndTiming( HYPRE_Int time_index );
HYPRE_Int hypre_ClearTiming( void );
HYPRE_Int hypre_PrintTiming( const char *heading , MPI_Comm comm );

#endif

#ifdef __cplusplus
}
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
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
       HYPRE_Int                        data;
       struct double_linked_list *next_elt;
       struct double_linked_list *prev_elt;
       HYPRE_Int                        head;
       HYPRE_Int                        tail;
};

typedef struct double_linked_list hypre_ListElement;
typedef hypre_ListElement  *hypre_LinkList;  

#ifdef __cplusplus
}
#endif

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/


#ifndef hypre_EXCHANGE_DATA_HEADER
#define hypre_EXCHANGE_DATA_HEADER

#define hypre_BinaryTreeParentId(tree)      (tree->parent_id)
#define hypre_BinaryTreeNumChild(tree)      (tree->num_child)
#define hypre_BinaryTreeChildIds(tree)      (tree->child_id)
#define hypre_BinaryTreeChildId(tree, i)    (tree->child_id[i])


typedef struct
{
   HYPRE_Int                   parent_id;
   HYPRE_Int                   num_child;
   HYPRE_Int		        *child_id;
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
   HYPRE_Int    (*fill_response)(void* recv_buf, HYPRE_Int contact_size, 
                           HYPRE_Int contact_proc, void* response_obj, 
                           MPI_Comm comm, void** response_buf, 
                           HYPRE_Int* response_message_size);
   HYPRE_Int     send_response_overhead; /*set by exchange data */
   HYPRE_Int     send_response_storage;  /*storage allocated for send_response_buf*/
   void    *data1;                 /*data fields user may want to access in fill_response */
   void    *data2;
   
} hypre_DataExchangeResponse;


HYPRE_Int hypre_CreateBinaryTree(HYPRE_Int, HYPRE_Int, hypre_BinaryTree*);
HYPRE_Int hypre_DestroyBinaryTree(hypre_BinaryTree*);


HYPRE_Int hypre_DataExchangeList(HYPRE_Int num_contacts, 
		     HYPRE_Int *contact_proc_list, void *contact_send_buf, 
		     HYPRE_Int *contact_send_buf_starts, HYPRE_Int contact_obj_size, 
                     HYPRE_Int response_obj_size,
		     hypre_DataExchangeResponse *response_obj, HYPRE_Int max_response_size, 
                     HYPRE_Int rnum, MPI_Comm comm,  void **p_response_recv_buf, 
                     HYPRE_Int **p_response_recv_buf_starts);


#endif /* end of header */
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/



#ifndef hypre_ERROR_HEADER
#define hypre_ERROR_HEADER

/*--------------------------------------------------------------------------
 * Global variable used in hypre error checking
 *--------------------------------------------------------------------------*/

extern HYPRE_Int hypre__global_error;
#define hypre_error_flag  hypre__global_error

/*--------------------------------------------------------------------------
 * HYPRE error macros
 *--------------------------------------------------------------------------*/

void hypre_error_handler(const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg);
#define hypre_error(IERR)  hypre_error_handler(__FILE__, __LINE__, IERR, NULL)
#define hypre_error_w_msg(IERR, msg)  hypre_error_handler(__FILE__, __LINE__, IERR, msg)
#define hypre_error_in_arg(IARG)  hypre_error(HYPRE_ERROR_ARG | IARG<<3)
#ifdef NDEBUG
#define hypre_assert(EX)
#else
#define hypre_assert(EX) if (!(EX)) {hypre_fprintf(stderr,"hypre_assert failed: %s\n", #EX); hypre_error(1);}
#endif

#endif

/* amg_linklist.c */
void dispose_elt ( hypre_LinkList element_ptr );
void remove_point ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , HYPRE_Int measure , HYPRE_Int index , HYPRE_Int *lists , HYPRE_Int *where );
hypre_LinkList create_elt ( HYPRE_Int Item );
void enter_on_lists ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , HYPRE_Int measure , HYPRE_Int index , HYPRE_Int *lists , HYPRE_Int *where );

/* binsearch.c */
HYPRE_Int hypre_BinarySearch ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int list_length );
HYPRE_Int hypre_BinarySearch2 ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int low , HYPRE_Int high , HYPRE_Int *spot );

/* hypre_printf.c */
#ifdef HYPRE_BIGINT
HYPRE_Int hypre_printf( const char *format , ... );
HYPRE_Int hypre_fprintf( FILE *stream , const char *format, ... );
HYPRE_Int hypre_sprintf( char *s , const char *format, ... );
HYPRE_Int hypre_scanf( const char *format , ... );
HYPRE_Int hypre_fscanf( FILE *stream , const char *format, ... );
HYPRE_Int hypre_sscanf( char *s , const char *format, ... );
#else
#define hypre_printf  printf
#define hypre_fprintf fprintf
#define hypre_sprintf sprintf
#define hypre_scanf   scanf
#define hypre_fscanf  fscanf
#define hypre_sscanf  sscanf
#endif

/* hypre_qsort.c */
void swap ( HYPRE_Int *v , HYPRE_Int i , HYPRE_Int j );
void swap2 ( HYPRE_Int *v , double *w , HYPRE_Int i , HYPRE_Int j );
void hypre_swap2i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int i , HYPRE_Int j );
void hypre_swap3i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int i , HYPRE_Int j );
void hypre_swap3_d ( double *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int i , HYPRE_Int j );
void hypre_swap4_d ( double *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int *y , HYPRE_Int i , HYPRE_Int j );
void hypre_swap_d ( double *v , HYPRE_Int i , HYPRE_Int j );
void qsort0 ( HYPRE_Int *v , HYPRE_Int left , HYPRE_Int right );
void qsort1 ( HYPRE_Int *v , double *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort2i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort2 ( HYPRE_Int *v , double *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort3i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort3_abs ( double *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort4_abs ( double *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int *y , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort_abs ( double *w , HYPRE_Int left , HYPRE_Int right );

/* qsplit.c */
HYPRE_Int hypre_DoubleQuickSplit ( double *values , HYPRE_Int *indices , HYPRE_Int list_length , HYPRE_Int NumberKept );

/* random.c */
void hypre_SeedRand ( HYPRE_Int seed );
double hypre_Rand ( void );

#ifdef __cplusplus
}
#endif

#endif

