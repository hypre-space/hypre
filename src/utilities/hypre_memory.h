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

#define HYPRE_MEMORY_DEVICE ( 0)
#define HYPRE_MEMORY_HOST   ( 1)
#define HYPRE_MEMORY_SHARED ( 2)
#define HYPRE_MEMORY_UNSET  (-1)

#define HYPRE_CUDA_GLOBAL

#define hypre_InitMemoryDebug(id)
#define hypre_FinalizeMemoryDebug()

#define hypre_TAlloc(type, count, location) \
( (type *)hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define hypre_CTAlloc(type, count, location) \
( (type *)hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location) )

#define hypre_TReAlloc(ptr, type, count, location) \
( (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location) )

#define hypre_TFree(ptr,location) \
( hypre_Free((char *)ptr, location), ptr = NULL )

#define hypre_DataCopyToData(ptrH,ptrD,type,count) memcpy(ptrD, ptrH, sizeof(type)*(count))
#define hypre_DataCopyFromData(ptrH,ptrD,type,count) memcpy(ptrH, ptrD, sizeof(type)*(count))
#define hypre_DeviceMemset(ptr,value,type,count)	memset(ptr,value,count*sizeof(type))
  
#define hypre_PinnedTAlloc(type, count)\
( (type *)hypre_MAllocPinned((size_t)(sizeof(type) * (count))) )

/*	
#define  hypre_TAlloc(type,  count, HYPRE_MEMORY_HOST) \
( (type *)hypre_MAllocHost((size_t)(sizeof(type) * (count))) )

#define  hypre_CTAlloc(type,  count, HYPRE_MEMORY_HOST) \
( (type *)hypre_CAllocHost((size_t)(count), (size_t)sizeof(type)) )

#define  hypre_TReAlloc(ptr,  type,  count, HYPRE_MEMORY_HOST) \
( (type *)hypre_ReAllocHost((char *)ptr, (size_t)(sizeof(type) * (count))) )

#define  hypre_TFree(ptr, HYPRE_MEMORY_HOST) \
( hypre_FreeHost((char *)ptr), ptr = NULL )
*/
/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* hypre_memory.c */
HYPRE_Int hypre_OutOfMemory ( size_t size );
char *hypre_MAlloc( size_t size , HYPRE_Int location );
char *hypre_CAlloc( size_t count ,  size_t elt_size , HYPRE_Int location);
char *hypre_MAllocPinned( size_t size );
char *hypre_ReAlloc( char *ptr ,  size_t size , HYPRE_Int location);
void hypre_Free( char *ptr , HYPRE_Int location );
char *hypre_CAllocHost( size_t count,size_t elt_size );
char *hypre_MAllocHost( size_t size );
char *hypre_ReAllocHost( char   *ptr,size_t  size );
void hypre_FreeHost( char *ptr );
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

