/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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

/* memory.c */
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
