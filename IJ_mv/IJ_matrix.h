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

#ifndef hypre_IJ_MATRIX_HEADER
#define hypre_IJ_MATRIX_HEADER

#include "../utilities/general.h"
#include "../utilities/utilities.h"

#include "../HYPRE.h"

/* #include "./HYPRE_IJ_matrix_types.h" */

/*--------------------------------------------------------------------------
 * hypre_IJMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   int M, N;                               /* number of rows and cols in matrix */


   void         *local_storage;            /* Structure for storing local portion */
   int      	 local_storage_type;       /* Indicates the type of "local storage" */
   void         *translator;               /* optional storage_type specfic structure
                                              for holding additional local info */

   int           insertion_semantics;      /* Flag that indicates for the current
                                              object to what extent values can be set
                                              from different processors than the one that
                                              stores the row. */
                                           /* 0: minimum definition, values can only be set on-processor. */
   int           ref_count;                /* reference count for memory management */
} hypre_IJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJMatrix
 *--------------------------------------------------------------------------*/

#define hypre_IJMatrixContext(matrix)              ((matrix) -> context)
#define hypre_IJMatrixM(matrix)                    ((matrix) -> M)
#define hypre_IJMatrixN(matrix)                    ((matrix) -> N)

#define hypre_IJMatrixLocalStorageType(matrix)     ((matrix) -> local_storage_type)
#define hypre_IJMatrixTranslator(matrix)           ((matrix) -> translator)
#define hypre_IJMatrixLocalStorage(matrix)         ((matrix) -> local_storage)

#define hypre_IJMatrixInsertionSemantics(matrix)   ((matrix) -> insertion_semantics)
#define hypre_IJMatrixReferenceCount(matrix)       ((matrix) -> ref_count)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/

#ifdef PETSC_AVAILABLE
/* IJMatrix_petsc.c */
int
hypre_GetIJMatrixParCSRMatrix( HYPRE_IJMatrix IJmatrix, Mat *reference )
#endif
  
#ifdef ISIS_AVAILABLE
/* IJMatrix_isis.c */
int
hypre_GetIJMatrixISISMatrix( HYPRE_IJMatrix IJmatrix, RowMatrix *reference )
#endif

/* #include "./internal_protos.h" */

#endif
