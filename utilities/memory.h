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

#ifndef zzz_MEMORY_HEADER
#define zzz_MEMORY_HEADER

#include <stdlib.h>
#include <stdio.h>

#ifndef ZZZ_MEMORY_CHECK_SIZE
#define ZZZ_MEMORY_CHECK_SIZE 10000
#endif

/*--------------------------------------------------------------------------
 * Check memory allocation
 *--------------------------------------------------------------------------*/

#ifdef ZZZ_MEMORY_CHECK

#define zzz_TAlloc(type, count) \
( (type *) zzz_MAllocCheck((unsigned int)(sizeof(type) * (count)),\
                           __FILE__, __LINE__) )

#define zzz_CTAlloc(type, count) \
( (type *) zzz_CAllocCheck((unsigned int)(count), (unsigned int)sizeof(type),\
                           __FILE__, __LINE__) )

#define zzz_TReAlloc(ptr, type, count) \
( (type *) zzz_ReAllocCheck((char *)ptr,\
                            (unsigned int)(sizeof(type) * (count)),\
                            __FILE__, __LINE__) )

#define zzz_TFree(ptr) \
( zzz_Free((char *)ptr), ptr = NULL )

/*--------------------------------------------------------------------------
 * Do not check memory allocation
 *--------------------------------------------------------------------------*/

#else

#define zzz_TAlloc(type, count) \
( (type *) zzz_MAlloc((unsigned int)(sizeof(type) * (count))) )

#define zzz_CTAlloc(type, count) \
( (type *) zzz_CAlloc((unsigned int)(count), (unsigned int)sizeof(type)) )

#define zzz_TReAlloc(ptr, type, count) \
( (type *) zzz_ReAlloc((char *)ptr, (unsigned int)(sizeof(type) * (count))) )

#define zzz_TFree(ptr) \
( zzz_Free((char *)ptr), ptr = NULL )

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* memory.c */
char *zzz_MAlloc P((int size ));
char *zzz_CAlloc P((int count , int elt_size ));
char *zzz_ReAlloc P((char *ptr , int size ));
void zzz_Free P((char *ptr ));
char *zzz_MAllocCheck P((int size , char *file , int line ));
char *zzz_CAllocCheck P((int count , int elt_size , char *file , int line ));
char *zzz_ReAllocCheck P((char *ptr , int size , char *file , int line ));
 
#undef P

#endif
