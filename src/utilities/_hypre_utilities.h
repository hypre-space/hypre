/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef hypre_UTILITIES_HEADER
#define hypre_UTILITIES_HEADER

#include "HYPRE_utilities.h"

#ifdef HYPRE_USING_OPENMP
#include <omp.h>
#endif

/* This allows us to consistently avoid 'int' throughout hypre */
typedef int               hypre_int;
typedef long int          hypre_longint;
typedef unsigned int      hypre_uint;
typedef unsigned long int hypre_ulongint;

/* This allows us to consistently avoid 'double' throughout hypre */
typedef double            hypre_double;

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

#ifndef hypre_abs
#define hypre_abs(a)  (((a)>0) ? (a) : -(a))
#endif

#ifndef hypre_round
#define hypre_round(x)  ( ((x) < 0.0) ? ((HYPRE_Int)(x - 0.5)) : ((HYPRE_Int)(x + 0.5)) )
#endif

#ifndef hypre_pow2
#define hypre_pow2(i)  ( 1 << (i) )
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
 * $Revision$
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
#define MPI_C_DOUBLE_COMPLEX hypre_MPI_COMPLEX

#define MPI_SUM             hypre_MPI_SUM              
#define MPI_MIN             hypre_MPI_MIN              
#define MPI_MAX             hypre_MPI_MAX              
#define MPI_LOR             hypre_MPI_LOR              
#define MPI_SUCCESS         hypre_MPI_SUCCESS
#define MPI_STATUSES_IGNORE hypre_MPI_STATUSES_IGNORE

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
#define MPI_Gatherv         hypre_MPI_Gatherv       
#define MPI_Scatter         hypre_MPI_Scatter       
#define MPI_Scatterv        hypre_MPI_Scatterv       
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
#define MPI_Op_free         hypre_MPI_Op_free        
#define MPI_Op_create       hypre_MPI_Op_create
#define MPI_User_function   hypre_MPI_User_function

/*--------------------------------------------------------------------------
 * Types, etc.
 *--------------------------------------------------------------------------*/

/* These types have associated creation and destruction routines */
typedef HYPRE_Int hypre_MPI_Comm;
typedef HYPRE_Int hypre_MPI_Group;
typedef HYPRE_Int hypre_MPI_Request;
typedef HYPRE_Int hypre_MPI_Datatype;
typedef void (hypre_MPI_User_function) ();

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
#define  hypre_MPI_REAL 5
#define  hypre_MPI_COMPLEX 6

#define  hypre_MPI_SUM 0
#define  hypre_MPI_MIN 1
#define  hypre_MPI_MAX 2
#define  hypre_MPI_LOR 3
#define  hypre_MPI_SUCCESS 0
#define  hypre_MPI_STATUSES_IGNORE 0

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
typedef MPI_User_function    hypre_MPI_User_function;

#define  hypre_MPI_COMM_WORLD MPI_COMM_WORLD
#define  hypre_MPI_COMM_NULL  MPI_COMM_NULL
#define  hypre_MPI_BOTTOM     MPI_BOTTOM
#define  hypre_MPI_COMM_SELF  MPI_COMM_SELF

#define  hypre_MPI_DOUBLE  MPI_DOUBLE
/* HYPRE_MPI_INT is defined in HYPRE_utilities.h */
#define  hypre_MPI_INT     HYPRE_MPI_INT
#define  hypre_MPI_CHAR    MPI_CHAR
#define  hypre_MPI_LONG    MPI_LONG
#define  hypre_MPI_BYTE    MPI_BYTE
/* HYPRE_MPI_REAL is defined in HYPRE_utilities.h */
#define  hypre_MPI_REAL    HYPRE_MPI_REAL
/* HYPRE_MPI_COMPLEX is defined in HYPRE_utilities.h */
#define  hypre_MPI_COMPLEX HYPRE_MPI_COMPLEX

#define  hypre_MPI_SUM MPI_SUM
#define  hypre_MPI_MIN MPI_MIN
#define  hypre_MPI_MAX MPI_MAX
#define  hypre_MPI_LOR MPI_LOR
#define  hypre_MPI_SUCCESS MPI_SUCCESS

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
HYPRE_Real hypre_MPI_Wtime( void );
HYPRE_Real hypre_MPI_Wtick( void );
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
HYPRE_Int hypre_MPI_Gatherv( void *sendbuf , HYPRE_Int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int *recvcounts , HYPRE_Int *displs , hypre_MPI_Datatype recvtype , HYPRE_Int root , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Scatter( void *sendbuf , HYPRE_Int sendcount , hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , hypre_MPI_Datatype recvtype , HYPRE_Int root , hypre_MPI_Comm comm );
HYPRE_Int hypre_MPI_Scatterv( void *sendbuf , HYPRE_Int *sendcounts , HYPRE_Int *displs, hypre_MPI_Datatype sendtype , void *recvbuf , HYPRE_Int recvcount , hypre_MPI_Datatype recvtype , HYPRE_Int root , hypre_MPI_Comm comm );
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
HYPRE_Int hypre_MPI_Op_free( hypre_MPI_Op *op );
HYPRE_Int hypre_MPI_Op_create( hypre_MPI_User_function *function , hypre_int commute , hypre_MPI_Op *op );

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
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef HYPRE_SMP_HEADER
#define HYPRE_SMP_HEADER
#endif

#define HYPRE_SMP_SCHEDULE schedule(static)

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
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

#define hypre_SharedTAlloc(type, count) hypre_TAlloc(type, (count))
#define hypre_SharedCTAlloc(type, count) hypre_CTAlloc(type, (count))
#define hypre_SharedTReAlloc(type, count) hypre_TReAlloc(type, (count))
#define hypre_SharedTFree(ptr) hypre_TFree(ptr)

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
HYPRE_Real *hypre_IncrementSharedDataPtr ( HYPRE_Real *ptr , size_t size );

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
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef hypre_THREADING_HEADER
#define hypre_THREADING_HEADER

#ifdef HYPRE_USING_OPENMP

HYPRE_Int hypre_NumThreads( void );
HYPRE_Int hypre_NumActiveThreads( void );
HYPRE_Int hypre_GetThreadNum( void );

#else

#define hypre_NumThreads() 1
#define hypre_NumActiveThreads() 1
#define hypre_GetThreadNum() 0

#endif

void hypre_GetSimpleThreadPartition( HYPRE_Int *begin, HYPRE_Int *end, HYPRE_Int n );

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
HYPRE_Real time_getWallclockSeconds( void );
HYPRE_Real time_getCPUSeconds( void );
HYPRE_Real time_get_wallclock_seconds_( void );
HYPRE_Real time_get_cpu_seconds_( void );

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
   HYPRE_Real  *wall_time;
   HYPRE_Real  *cpu_time;
   HYPRE_Real  *flops;
   char   **name;
   HYPRE_Int     *state;     /* boolean flag to allow for recursive timing */
   HYPRE_Int     *num_regs;  /* count of how many times a name is registered */

   HYPRE_Int      num_names;
   HYPRE_Int      size;

   HYPRE_Real   wall_count;
   HYPRE_Real   CPU_count;
   HYPRE_Real   FLOP_count;

} hypre_TimingType;

#ifdef HYPRE_TIMING_GLOBALS
hypre_TimingType *hypre_global_timing = NULL;
#else
extern hypre_TimingType *hypre_global_timing;
#endif

/*-------------------------------------------------------
 * Accessor functions
 *-------------------------------------------------------*/

#define hypre_TimingWallTime(i) (hypre_global_timing -> wall_time[(i)])
#define hypre_TimingCPUTime(i)  (hypre_global_timing -> cpu_time[(i)])
#define hypre_TimingFLOPS(i)    (hypre_global_timing -> flops[(i)])
#define hypre_TimingName(i)     (hypre_global_timing -> name[(i)])
#define hypre_TimingState(i)    (hypre_global_timing -> state[(i)])
#define hypre_TimingNumRegs(i)  (hypre_global_timing -> num_regs[(i)])
#define hypre_TimingWallCount   (hypre_global_timing -> wall_count)
#define hypre_TimingCPUCount    (hypre_global_timing -> CPU_count)
#define hypre_TimingFLOPCount   (hypre_global_timing -> FLOP_count)

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
 * $Revision$
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

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header file for Caliper instrumentation macros
 *
 *****************************************************************************/

#ifndef CALIPER_INSTRUMENTATION_HEADER
#define CALIPER_INSTRUMENTATION_HEADER

#include "HYPRE_config.h"

#ifdef HYPRE_USING_CALIPER

#include <caliper/cali.h>

#define HYPRE_ANNOTATION_BEGIN( str ) cali_begin_string_byname("hypre.kernel", str)
#define HYPRE_ANNOTATION_END( str ) cali_end_byname("hypre.kernel")

#else

#define HYPRE_ANNOTATION_BEGIN( str ) 
#define HYPRE_ANNOTATION_END( str ) 

#endif

#endif /* CALIPER_INSTRUMENTATION_HEADER */

/*--------------------------------------------------------------------------
 * Other prototypes
 *--------------------------------------------------------------------------*/

/* amg_linklist.c */
void hypre_dispose_elt ( hypre_LinkList element_ptr );
void hypre_remove_point ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , HYPRE_Int measure , HYPRE_Int index , HYPRE_Int *lists , HYPRE_Int *where );
hypre_LinkList hypre_create_elt ( HYPRE_Int Item );
void hypre_enter_on_lists ( hypre_LinkList *LoL_head_ptr , hypre_LinkList *LoL_tail_ptr , HYPRE_Int measure , HYPRE_Int index , HYPRE_Int *lists , HYPRE_Int *where );

/* binsearch.c */
HYPRE_Int hypre_BinarySearch ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int list_length );
HYPRE_Int hypre_BinarySearch2 ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int low , HYPRE_Int high , HYPRE_Int *spot );
HYPRE_Int *hypre_LowerBound( HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int value );

/* hypre_complex.c */
#ifdef HYPRE_COMPLEX
HYPRE_Complex hypre_conj( HYPRE_Complex value );
HYPRE_Real    hypre_cabs( HYPRE_Complex value );
HYPRE_Real    hypre_creal( HYPRE_Complex value );
HYPRE_Real    hypre_cimag( HYPRE_Complex value );
#else
#define hypre_conj(value)  value
#define hypre_cabs(value)  fabs(value)
#define hypre_creal(value) value
#define hypre_cimag(value) 0.0
#endif

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
void hypre_swap ( HYPRE_Int *v , HYPRE_Int i , HYPRE_Int j );
void hypre_swap2 ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int i , HYPRE_Int j );
void hypre_swap2i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int i , HYPRE_Int j );
void hypre_swap3i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int i , HYPRE_Int j );
void hypre_swap3_d ( HYPRE_Real *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int i , HYPRE_Int j );
void hypre_swap4_d ( HYPRE_Real *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int *y , HYPRE_Int i , HYPRE_Int j );
void hypre_swap_d ( HYPRE_Real *v , HYPRE_Int i , HYPRE_Int j );
void hypre_qsort0 ( HYPRE_Int *v , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort1 ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort2i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort2 ( HYPRE_Int *v , HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort3i ( HYPRE_Int *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort3_abs ( HYPRE_Real *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort4_abs ( HYPRE_Real *v , HYPRE_Int *w , HYPRE_Int *z , HYPRE_Int *y , HYPRE_Int left , HYPRE_Int right );
void hypre_qsort_abs ( HYPRE_Real *w , HYPRE_Int left , HYPRE_Int right );

/* qsplit.c */
HYPRE_Int hypre_DoubleQuickSplit ( HYPRE_Real *values , HYPRE_Int *indices , HYPRE_Int list_length , HYPRE_Int NumberKept );

/* random.c */
void hypre_SeedRand ( HYPRE_Int seed );
HYPRE_Real hypre_Rand ( void );

/* hypre_prefix_sum.c */
/**
 * Assumed to be called within an omp region.
 * Let x_i be the input of ith thread.
 * The output of ith thread y_i = x_0 + x_1 + ... + x_{i-1}
 * Additionally, sum = x_0 + x_1 + ... + x_{nthreads - 1}
 * Note that always y_0 = 0
 *
 * @param workspace at least with length (nthreads+1)
 *                  workspace[tid] will contain result for tid
 *                  workspace[nthreads] will contain sum
 */
void hypre_prefix_sum(HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int *workspace);
/**
 * This version does prefix sum in pair.
 * Useful when we prefix sum of diag and offd in tandem.
 *
 * @param worksapce at least with length 2*(nthreads+1)
 *                  workspace[2*tid] and workspace[2*tid+1] will contain results for tid
 *                  workspace[3*nthreads] and workspace[3*nthreads + 1] will contain sums
 */
void hypre_prefix_sum_pair(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2, HYPRE_Int *workspace);
/**
 * @param workspace at least with length 3*(nthreads+1)
 *                  workspace[3*tid:3*tid+3) will contain results for tid
 */
void hypre_prefix_sum_triple(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2, HYPRE_Int *in_out3, HYPRE_Int *sum3, HYPRE_Int *workspace);

/**
 * n prefix-sums together.
 * workspace[n*tid:n*(tid+1)) will contain results for tid
 * workspace[nthreads*tid:nthreads*(tid+1)) will contain sums
 *
 * @param workspace at least with length n*(nthreads+1)
 */
void hypre_prefix_sum_multiple(HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int n, HYPRE_Int *workspace);

/* hypre_merge_sort.c */
/**
 * Why merge sort?
 * 1) Merge sort can take advantage of eliminating duplicates.
 * 2) Merge sort is more efficiently parallelizable than qsort
 */

/**
 * Out of place merge sort with duplicate elimination
 * @ret number of unique elements
 */
HYPRE_Int hypre_merge_sort_unique(HYPRE_Int *in, HYPRE_Int *out, HYPRE_Int len);
/**
 * Out of place merge sort with duplicate elimination
 *
 * @param out pointer to output can be in or temp
 * @ret number of unique elements
 */
HYPRE_Int hypre_merge_sort_unique2(HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **out);

void hypre_merge_sort(HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **sorted);

/* hypre_hopscotch_hash.c */

#ifdef HYPRE_USING_OPENMP

/* Check if atomic operations are available to use concurrent hopscotch hash table */
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
#define HYPRE_USING_ATOMIC 
//#elif defined _MSC_VER // JSP: haven't tested, so comment out for now
//#define HYPRE_USING_ATOMIC
//#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
// JSP: not many compilers have implemented this, so comment out for now
//#define HYPRE_USING_ATOMIC
//#include <stdatomic.h>
#endif

#endif // HYPRE_USING_OPENMP

#ifdef HYPRE_HOPSCOTCH
#ifdef HYPRE_USING_ATOMIC
// concurrent hopscotch hashing is possible only with atomic supports
#define HYPRE_CONCURRENT_HOPSCOTCH 
#endif 
#endif 

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
typedef struct {
  HYPRE_Int volatile timestamp;
  omp_lock_t         lock;
} hypre_HopscotchSegment;
#endif

/**
 * The current typical use case of unordered set is putting input sequence
 * with lots of duplication (putting all colidx received from other ranks),
 * followed by one sweep of enumeration.
 * Since the capacity is set to the number of inputs, which is much larger
 * than the number of unique elements, we optimize for initialization and
 * enumeration whose time is proportional to the capacity.
 * For initialization and enumeration, structure of array (SoA) is better
 * for vectorization, cache line utilization, and so on.
 */
typedef struct
{
	HYPRE_Int  volatile              segmentMask;
	HYPRE_Int  volatile              bucketMask;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
	hypre_HopscotchSegment* volatile segments;
#endif
  HYPRE_Int *volatile              key;
  hypre_uint *volatile             hopInfo;
	HYPRE_Int *volatile	             hash;
} hypre_UnorderedIntSet;

typedef struct
{
  hypre_uint volatile hopInfo;
  HYPRE_Int  volatile hash;
  HYPRE_Int  volatile key;
  HYPRE_Int  volatile data;
} hypre_HopscotchBucket;

/**
 * The current typical use case of unoredered map is putting input sequence
 * with no duplication (inverse map of a bijective mapping) followed by
 * lots of lookups.
 * For lookup, array of structure (AoS) gives better cache line utilization.
 */
typedef struct
{
	HYPRE_Int  volatile              segmentMask;
	HYPRE_Int  volatile              bucketMask;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
	hypre_HopscotchSegment*	volatile segments;
#endif
	hypre_HopscotchBucket* volatile	 table;
} hypre_UnorderedIntMap;

/**
 * Sort array "in" with length len and put result in array "out"
 * "in" will be deallocated unless in == *out
 * inverse_map is an inverse hash table s.t. inverse_map[i] = j iff (*out)[j] = i
 */
void hypre_sort_and_create_inverse_map(
  HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out, hypre_UnorderedIntMap *inverse_map);

#ifdef __cplusplus
}
#endif

/*#include "hypre_hopscotch_hash.h"*/

#endif

