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


typedef struct
{
   int     *data;
   int      size;

} VectorInt;

/*--------------------------------------------------------------------------
 * Accessor functions for the VectorInt structure
 *--------------------------------------------------------------------------*/

#define VectorIntData(vector)      ((vector) -> data)
#define VectorIntSize(vector)      ((vector) -> size)

#endif
