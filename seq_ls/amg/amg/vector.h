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
 * Header info for hypre_Vector data structures
 *
 *****************************************************************************/

#ifndef HYPRE_VECTOR_HEADER
#define HYPRE_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * hypre_Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int      size;

} hypre_Vector;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorData(vector)      ((vector) -> data)
#define hypre_VectorSize(vector)      ((vector) -> size)


typedef struct
{
   int     *data;
   int      size;

} hypre_VectorInt;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_VectorInt structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorIntData(vector)      ((vector) -> data)
#define hypre_VectorIntSize(vector)      ((vector) -> size)

#endif
