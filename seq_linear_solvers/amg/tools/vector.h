/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Vector data structures
 *
 *****************************************************************************/

#ifndef _VECTOR_HEADER
#define _VECTOR_HEADER


/*--------------------------------------------------------------------------
 * Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int      size;

} Vector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define VectorData(vector)      ((vector) -> data)
#define VectorSize(vector)      ((vector) -> size)

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* vector.c */
Vector *NewVector P((double *data , int size ));
void FreeVector P((Vector *vector ));

#undef P


#endif
