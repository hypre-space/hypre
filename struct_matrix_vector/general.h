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
 * General structures and values
 *
 *****************************************************************************/

#ifndef zzz_GENERAL_HEADER
#define zzz_GENERAL_HEADER


/*--------------------------------------------------------------------------
 * Define memory allocation routines
 *--------------------------------------------------------------------------*/

#define zzz_TAlloc(type, count) \
((count) ? (type *) malloc((unsigned int)(sizeof(type) * (count))) : NULL)

#define zzz_CTAlloc(type, count) \
((count) ? (type *) calloc((unsigned int)(count),\
                           (unsigned int)sizeof(type)) : NULL)

#define zzz_TRealloc(ptr, type, count) \
((count) ? (type *) realloc((char *)ptr,\
                            (unsigned int)(sizeof(type) * (count))) : NULL)

/* note: the `else' is required to guarantee termination of the `if' */
#define zzz_TFree(ptr) if (ptr) free(ptr); else

/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef min
#define min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef round
#define round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
#endif


#endif
