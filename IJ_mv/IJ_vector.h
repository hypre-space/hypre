/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_VECTOR_HEADER
#define hypre_IJ_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJVector:
 *--------------------------------------------------------------------------*/

typedef struct hypre_IJVector_struct
{
   MPI_Comm      comm;

   int 		*partitioning;      /* Indicates partitioning over tasks */

   int           object_type;       /* Indicates the type of "local storage" */

   void         *object;            /* Structure for storing local portion */

} hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJVector
 *--------------------------------------------------------------------------*/

#define hypre_IJVectorComm(vector)           ((vector) -> comm)

#define hypre_IJVectorPartitioning(vector)   ((vector) -> partitioning)

#define hypre_IJVectorObjectType(vector)     ((vector) -> object_type)

#define hypre_IJVectorObject(vector)         ((vector) -> object)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif
