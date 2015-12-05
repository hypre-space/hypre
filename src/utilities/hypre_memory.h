/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
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
