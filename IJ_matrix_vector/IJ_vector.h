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

#include "../utilities/general.h"
#include "../utilities/utilities.h"

#include "../HYPRE.h"

/* #include "./HYPRE_IJ_vector_types.h" */

/*--------------------------------------------------------------------------
 * hypre_IJVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   int N;                                  /* number of rows in column vector */


   void         *local_storage;            /* Structure for storing local portio
n */
   int           local_storage_type;       /* Indicates the type of "local stora
ge" */
   void         *translator;               /* optional storage_type specfic stru
cture
                                              for holding additional local info
*/

   int           insertion_semantics;      /* Flag that indicates for the curren
t
                                              object to what extent values can b
e set
                                              from different processors than the
 one that
                                              stores the row. */
                                           /* 0: minimum definition, values can
only be set on-processor. */
   int           ref_count;                /* reference count for memory managem
ent */
} hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJVector
 *--------------------------------------------------------------------------*/

#define hypre_IJVectorContext(vector)              ((vector) -> context)
#define hypre_IJVectorN(vector)                    ((vector) -> N)

#define hypre_IJVectorLocalStorageType(vector)     ((vector) -> local_storage_type)
#define hypre_IJVectorTranslator(vector)           ((vector) -> translator)
#define hypre_IJVectorLocalStorage(vector)         ((vector) -> local_storage)

#define hypre_IJVectorInsertionSemantics(vector)   ((vector) -> insertion_semantics)
#define hypre_IJVectorReferenceCount(vector)       ((vector) -> ref_count)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif
