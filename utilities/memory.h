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

#include <stdlib.h>
#include <stdio.h>

#ifndef HYPRE_MEMORY_CHECK_SIZE
#define HYPRE_MEMORY_CHECK_SIZE 10000
#endif

/*--------------------------------------------------------------------------
 * Check memory allocation
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_MEMORY_CHECK

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAllocCheck((unsigned int)(sizeof(type) * (count)),\
                             __FILE__, __LINE__) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAllocCheck((unsigned int)(count), (unsigned int)sizeof(type),\
                            __FILE__, __LINE__) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAllocCheck((char *)ptr,\
                             (unsigned int)(sizeof(type) * (count)),\
                             __FILE__, __LINE__) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

/*--------------------------------------------------------------------------
 * Do not check memory allocation
 *--------------------------------------------------------------------------*/

#else

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAlloc((unsigned int)(sizeof(type) * (count))) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAlloc((unsigned int)(count), (unsigned int)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAlloc((char *)ptr, (unsigned int)(sizeof(type) * (count))) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

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
char *hypre_MAlloc P((int size ));
char *hypre_CAlloc P((int count , int elt_size ));
char *hypre_ReAlloc P((char *ptr , int size ));
void hypre_Free P((char *ptr ));
char *hypre_MAllocCheck P((int size , char *file , int line ));
char *hypre_CAllocCheck P((int count , int elt_size , char *file , int line ));
char *hypre_ReAllocCheck P((char *ptr , int size , char *file , int line ));
 
#undef P

#endif
